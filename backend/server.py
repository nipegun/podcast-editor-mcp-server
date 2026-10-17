"""The MCP server.

Transport is **stdio**: the client starts this process and speaks JSON-RPC over
its standard streams. That is what makes it usable from Codex and any other
stdio-capable MCP client without opening a port.

stdout belongs to the protocol. Everything this process logs goes to stderr or
to a file; the SDK additionally redirects fd 1 to stderr while the stdio
transport is running, so even a stray ``print`` in a dependency cannot corrupt
the stream.

HTTP is not enabled and no socket is opened. The internal structure - build the
server object, register tools, then run a transport - is already the shape a
Streamable HTTP transport would need, so adding one later is a change in
``fMain``, not a rewrite.
"""

from __future__ import annotations

import argparse
import os
import sys

from mcp.server import MCPServer

from podcast_editor_mcp_server.config import fLoadRuntimeConfig
from podcast_editor_mcp_server.errors import PodcastEditorMcpServerError
from podcast_editor_mcp_server.logging_setup import fGetLogger, fSetupLogging
from podcast_editor_mcp_server.mcp.prompts import fRegisterPrompts
from podcast_editor_mcp_server.mcp.resources import fRegisterResources
from podcast_editor_mcp_server.mcp.state import ServerState
from podcast_editor_mcp_server.mcp.tools import fRegisterTools
from podcast_editor_mcp_server.version import cProductName, cProductTitle, cVersion

vLog = fGetLogger("server")

cInstructions: str = """\
podcast-editor-mcp-server edits and masters podcast recordings locally.

Start with `get_server_info`: it reports which optional pieces (local
transcription, Audacity) are usable on this machine and which directories you
may read from and write to.

Three ways to work:

* `prepare_episode_edit`, then `finish_episode_edit`: the server edits the
  episode itself - retakes (the last take is kept), talk about the recording,
  pauses (with a tight profile, every gap between words), every join checked
  by listening to it - and leaves you only the `review_items` it cannot
  decide: read each one and pass `cut` or `keep`. Read the transcript too,
  and pass what no detector finds (a fragment said again with other first
  words, a restarted word, a self-correction) as `extra_cuts`. To keep
  editing and mastering apart, pass `mastering=false`: the episode comes out
  as recorded, only cut (`<name>.wav` and the Audacity project), and
  `master_audio` cleans it up and masters it afterwards (`<name>.master.wav`
  and the other delivery formats).
* `process_podcast` does everything with the active profile in one call,
  following the proposed action of every review item.
* The fine-grained tools let you take every decision yourself: analyse,
  transcribe, look at the candidates, build an EditPlan, validate it, simulate
  it, then execute it.

Things worth knowing:

* Filler words, repetitions and false starts are reported as **candidates**.
  Nothing is removed on their account unless the profile says so or you put the
  operation in a plan. Read the transcript around a candidate before accepting
  it.
* An earlier take that says something the later take does not is never cut
  without a decision: removing it removes content, so ask the user.
* Every timestamp in an EditPlan is a coordinate of the *original* file, never
  of a partially edited one. Do not compensate for earlier cuts.
* Plans are always validated before execution, and a plan is bound to one exact
  file through its SHA-256.
* The source recording is opened read-only and is never modified.
* Nothing leaves this machine.
"""


def fBuildServer(pState: ServerState | None = None) -> MCPServer:
  """Construct the MCP server with every tool, resource and prompt registered.

  Nothing is read from disk and no directory is created until a tool actually
  runs, so building the server is cheap and side-effect free.
  """
  vState = pState if pState is not None else ServerState()
  vServer: MCPServer = MCPServer(
    name=cProductName,
    title=cProductTitle,
    version=cVersion,
    instructions=cInstructions,
    website_url="https://github.com/nipegun/podcast-editor-mcp-server",
  )
  fRegisterTools(vServer, vState)
  fRegisterResources(vServer, vState)
  fRegisterPrompts(vServer, vState)
  return vServer


def fConfigureLogging(pDebug: bool = False, pLogFile: str | None = None) -> None:
  """Set up logging for a server process, never touching stdout."""
  vConfig = None
  try:
    vConfig = fLoadRuntimeConfig()
  except PodcastEditorMcpServerError as vError:
    # A broken configuration file must still produce a readable error on
    # stderr rather than a silent exit.
    sys.stderr.write(f"{cProductName}: configuration problem: {vError}\n")

  vLevel = "DEBUG" if pDebug else (vConfig.logging.level if vConfig else "INFO")
  vTarget = None
  if pLogFile:
    from pathlib import Path

    vTarget = Path(pLogFile).expanduser()
  elif vConfig is not None:
    try:
      from podcast_editor_mcp_server.config import Workspace

      vWorkspace = Workspace(vConfig)
      vWorkspace.logs.mkdir(parents=True, exist_ok=True)
      vTarget = vWorkspace.log_file
    except OSError:
      vTarget = None

  fSetupLogging(
    pLevel=vLevel,
    pLogFile=vTarget,
    pJson=bool(vConfig.logging.json_format) if vConfig else False,
    pForce=True,
  )


def fBuildArgumentParser() -> argparse.ArgumentParser:
  vParser = argparse.ArgumentParser(
    prog=cProductName,
    description="MCP server for podcast editing and post-production (stdio transport).",
    epilog="Register this command in your MCP client; it is not meant to be run interactively.",
  )
  vParser.add_argument("--version", action="version", version=f"{cProductName} {cVersion}")
  vParser.add_argument("--debug", action="store_true", help="log at DEBUG level")
  vParser.add_argument("--log-file", metavar="PATH", help="write the log to this file as well")
  vParser.add_argument(
    "--config",
    metavar="PATH",
    help="use this runtime configuration file instead of the default search path",
  )
  vParser.add_argument(
    "--workspace",
    metavar="PATH",
    help="override the workspace root (default: /opt/podcast-editor-mcp-server/var/workspace/<user>)",
  )
  vParser.add_argument(
    "--profile",
    metavar="NAME",
    help="make this the active profile for the session",
  )
  vParser.add_argument(
    "--print-tools",
    action="store_true",
    help="list the registered tools and exit, without serving",
  )
  return vParser


def fMain(pArgv: list[str] | None = None) -> int:
  """Entry point for the ``podcast-editor-mcp-server`` command."""
  vParser = fBuildArgumentParser()
  vArguments = vParser.parse_args(pArgv)

  # Command-line overrides are applied through the environment so that every
  # lazily-constructed component sees them, including background threads.
  if vArguments.config:
    os.environ["PODCAST_EDITOR_MCP_SERVER_CONFIG"] = vArguments.config
  if vArguments.workspace:
    os.environ["PODCAST_EDITOR_MCP_SERVER_WORKSPACE"] = vArguments.workspace
  if vArguments.profile:
    os.environ["PODCAST_EDITOR_MCP_SERVER_PROFILE"] = vArguments.profile

  fConfigureLogging(vArguments.debug, vArguments.log_file)

  if os.getuid() == 0:
    vLog.warning(
      "Running as root. Podcast processing should run as an ordinary user, "
      "and Audacity will refuse to start."
    )

  vState = ServerState()
  vServer = fBuildServer(pState=vState)

  if vArguments.print_tools:
    import anyio

    lTools = anyio.run(vServer.list_tools)
    for vTool in sorted(lTools, key=lambda pItem: pItem.name):
      sys.stdout.write(f"{vTool.name}\n")
    return 0

  vLog.info("Starting MCP server", extra={"version": cVersion, "transport": "stdio"})
  try:
    vServer.run(transport="stdio")
  except KeyboardInterrupt:  # pragma: no cover - interactive interruption
    vLog.info("Interrupted; shutting down")
  finally:
    vState.mShutdown()
  return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
  raise SystemExit(fMain())
