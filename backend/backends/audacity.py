"""Audacity integration, driven through ``mod-script-pipe``.

What this talks to
------------------

Audacity's scripting module creates two named pipes in ``/tmp``:

    /tmp/audacity_script_pipe.to.<uid>     we write commands here
    /tmp/audacity_script_pipe.from.<uid>   we read replies here

A command is one line, ``Name: Param=Value ...``, terminated by ``\\n``. The
reply is a run of lines ending in ``BatchCommand finished: OK`` (or
``... Failed!``) followed by a blank line.

Safety model
------------

**No caller-supplied string ever becomes an Audacity command.** The backend
exposes named operations; each one builds its own command from a whitelist,
with per-parameter validation and quoting. There is deliberately no
``run_command(text)`` anywhere in this file, and nothing in the MCP surface can
reach one.

Robustness
----------

* The "from" pipe is opened ``O_RDWR`` so a momentarily absent writer reads as
  "no data yet" rather than a tight EOF loop; the "to" pipe is opened
  ``O_WRONLY | O_NONBLOCK`` so an absent Audacity fails immediately with ENXIO
  instead of hanging forever.
* Every read and write waits on a selector with a deadline.
* A single lock serialises commands inside the process, and an advisory file
  lock serialises them between processes, so two jobs can never interleave
  edits on one project.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import re
import selectors
import shutil
import signal
import stat
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from podcast_editor_mcp_server.backends.base import BackendStatus
from podcast_editor_mcp_server.config import AudacitySettings
from podcast_editor_mcp_server.errors import AudacityCommandError, AudacityUnavailableError
from podcast_editor_mcp_server.logging_setup import fGetLogger
from podcast_editor_mcp_server.util.atomic import fBackupFile, fWriteTextAtomic
from podcast_editor_mcp_server.util.proc import fRunProcess, fWhich
from podcast_editor_mcp_server.util.timeline import KeepSegment

vLog = fGetLogger("backends.audacity")

cPipeToTemplate: str = "audacity_script_pipe.to.{uid}"
cPipeFromTemplate: str = "audacity_script_pipe.from.{uid}"
cResponseTerminator: str = "BatchCommand finished:"
cReadChunkBytes: int = 65_536
cLockFileName: str = "podcast-editor-mcp-server-audacity.lock"
# How long to wait for the FIFOs to come back after a client disconnects.
cPipeReappearTimeout: float = 5.0
# Bounds on what is read from the reply pipe: the leftovers discarded when
# connecting, and the size of one reply.
cMaxDrainBytes: int = 1_048_576
cMaxDrainSeconds: float = 2.0
cMaxResponseChars: int = 16 * 1_048_576

# Audacity's ModulePrefs status values, from src/prefs/ModulePrefs.cpp.
cModuleDisabled: int = 0
cModuleEnabled: int = 1
cModuleAsk: int = 2
cModuleFailed: int = 3
cModuleNew: int = 4
cModuleStatusNames: dict[int, str] = {
  cModuleDisabled: "disabled",
  cModuleEnabled: "enabled",
  cModuleAsk: "ask",
  cModuleFailed: "failed",
  cModuleNew: "new (never enabled)",
}

cModuleLibraryPath: Path = Path("/usr/lib/audacity/modules/mod-script-pipe.so")
cModuleName: str = "mod-script-pipe"


# --------------------------------------------------------------------------- #
# Command whitelist
# --------------------------------------------------------------------------- #

cCommandNamePattern = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,40}$")
cParameterNamePattern = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,40}$")

# Only these commands may ever be sent. The value is the set of parameter names
# each command accepts; anything else is rejected before a byte is written.
cAllowedCommands: dict[str, frozenset[str]] = {
  "Help": frozenset({"Command", "Format"}),
  "GetInfo": frozenset({"Type", "Format"}),
  "New": frozenset(),
  "SelectNone": frozenset(),
  "SelectTime": frozenset({"Start", "End", "RelativeTo"}),
  "SelectTracks": frozenset({"Track", "TrackCount", "Mode"}),
  "Delete": frozenset(),
  "Import2": frozenset({"Filename"}),
  "SaveProject2": frozenset({"Filename", "AddToHistory"}),
  "FitInWindow": frozenset(),
}

# Every entry above was verified against Audacity 3.7.3 with
# "GetInfo: Type=Commands" and "GetInfo: Type=Menus"; nothing here is guessed.

cFloatParameters: frozenset[str] = frozenset({"Start", "End"})
cIntegerParameters: frozenset[str] = frozenset({"Track", "TrackCount", "AddToHistory"})
cEnumParameters: dict[str, frozenset[str]] = {
  "RelativeTo": frozenset({"ProjectStart", "Project", "ProjectEnd", "SelectionStart", "Selection", "SelectionEnd"}),
  "Mode": frozenset({"Set", "Add", "Remove"}),
  "Format": frozenset({"JSON", "LISP", "Brief"}),
  "Type": frozenset({"Commands", "Menus", "Preferences", "Tracks", "Clips", "Envelopes", "Labels", "Boxes"}),
}
cMaxFilenameLength: int = 1000


class AudacityCommand:
  """A validated command, built only from whitelisted parts."""

  def __init__(self, pName: str, pParameters: dict[str, Any] | None = None) -> None:
    if not cCommandNamePattern.match(pName):
      raise AudacityCommandError("Malformed Audacity command name", {"command": pName})
    if pName not in cAllowedCommands:
      raise AudacityCommandError(
        "Command is not on the allow list",
        {"command": pName, "allowed": sorted(cAllowedCommands)},
      )
    self.name = pName
    self.parameters: dict[str, str] = {}
    sAllowed = cAllowedCommands[pName]
    for vKey, vValue in (pParameters or {}).items():
      if not cParameterNamePattern.match(vKey):
        raise AudacityCommandError("Malformed parameter name", {"command": pName, "parameter": vKey})
      if vKey not in sAllowed:
        raise AudacityCommandError(
          "Parameter is not allowed for this command",
          {"command": pName, "parameter": vKey, "allowed": sorted(sAllowed)},
        )
      self.parameters[vKey] = fEncodeParameter(vKey, vValue)

  def mRender(self) -> str:
    """Return the exact line that will be written to the pipe."""
    if not self.parameters:
      return f"{self.name}:"
    lParts = [f"{vKey}={vValue}" for vKey, vValue in sorted(self.parameters.items())]
    return f"{self.name}: {' '.join(lParts)}"

  def __repr__(self) -> str:  # pragma: no cover - debugging aid
    return f"<AudacityCommand {self.mRender()!r}>"


def fEncodeParameter(pName: str, pValue: Any) -> str:
  """Validate and encode one parameter value."""
  if pName in cFloatParameters:
    vNumber = float(pValue)
    if vNumber != vNumber or vNumber in (float("inf"), float("-inf")):
      raise AudacityCommandError("Parameter must be a finite number", {"parameter": pName})
    if vNumber < 0.0:
      raise AudacityCommandError("Time parameters must not be negative", {"parameter": pName})
    return f"{vNumber:.6f}"
  if pName in cIntegerParameters:
    vInteger = int(pValue)
    if not -1_000_000 <= vInteger <= 1_000_000:
      raise AudacityCommandError("Integer parameter out of range", {"parameter": pName})
    return str(vInteger)
  if pName in cEnumParameters:
    vText = str(pValue)
    if vText not in cEnumParameters[pName]:
      raise AudacityCommandError(
        "Parameter value is not one of the allowed options",
        {"parameter": pName, "value": vText, "allowed": sorted(cEnumParameters[pName])},
      )
    return vText
  if pName in {"Filename", "Command"}:
    return fEncodeStringParameter(pName, pValue)
  raise AudacityCommandError("Unknown parameter", {"parameter": pName})


def fEncodeStringParameter(pName: str, pValue: Any) -> str:
  """Quote a string parameter, refusing anything that could break the line."""
  vText = str(pValue)
  if len(vText) > cMaxFilenameLength:
    raise AudacityCommandError("String parameter is too long", {"parameter": pName})
  if any(vCharacter in vText for vCharacter in ("\n", "\r", "\x00")):
    raise AudacityCommandError(
      "String parameter contains a control character",
      {"parameter": pName},
    )
  if '"' in vText:
    raise AudacityCommandError(
      "String parameter contains a double quote, which Audacity's parser cannot escape",
      {"parameter": pName},
    )
  return f'"{vText}"'


# --------------------------------------------------------------------------- #
# Pipe client
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class AudacityResponse:
  """One reply from Audacity."""

  body: str
  status: str

  def mGetOk(self) -> bool:
    return self.status == "OK"

  ok = property(mGetOk)


class AudacityPipeClient:
  """Low-level, timeout-aware access to ``mod-script-pipe``."""

  def __init__(self, pPipeDirectory: Path | None = None, pCommandTimeout: float = 180.0) -> None:
    self.pipe_directory = pPipeDirectory or Path("/tmp")  # noqa: S108 - Audacity hard-codes this
    self.command_timeout = pCommandTimeout
    self._vToDescriptor: int | None = None
    self._vFromDescriptor: int | None = None
    self._vLock = threading.RLock()
    self._vLockFile: int | None = None

  # ------------------------------------------------------------- discovery --

  def mGetToPipe(self) -> Path:
    return self.pipe_directory / cPipeToTemplate.format(uid=os.getuid())

  to_pipe = property(mGetToPipe)

  def mGetFromPipe(self) -> Path:
    return self.pipe_directory / cPipeFromTemplate.format(uid=os.getuid())

  from_pipe = property(mGetFromPipe)

  def mPipesExist(self) -> bool:
    return self.to_pipe.exists() and self.from_pipe.exists()

  def _mCheckPipe(self, pPath: Path) -> os.stat_result:
    """Make sure a script pipe is a private FIFO of this user, and return its identity.

    The names are predictable and live in a shared directory, so another
    account could create them first and read our commands or answer them.
    Audacity creates them owned by the user and closed to everybody else, and
    that is the only kind accepted.
    """
    try:
      vStat = os.lstat(pPath)
    except OSError as vError:
      raise AudacityUnavailableError(
        "Could not inspect Audacity's script pipe",
        {"path": str(pPath), "reason": str(vError)},
      ) from vError
    if not stat.S_ISFIFO(vStat.st_mode):
      raise AudacityUnavailableError(
        "The script pipe paths exist but are not FIFOs; refusing to use them",
        {"path": str(pPath)},
      )
    if vStat.st_uid != os.getuid() or vStat.st_mode & 0o077:
      raise AudacityUnavailableError(
        "The script pipe belongs to another user or is open to other users; refusing to use it",
        {"path": str(pPath), "owner_uid": vStat.st_uid, "mode": oct(stat.S_IMODE(vStat.st_mode))},
      )
    return vStat

  def _mCheckOpened(self, pDescriptor: int, pExpected: os.stat_result, pPath: Path) -> None:
    """Make sure the descriptor is the very FIFO that was inspected, not a replacement."""
    vStat = os.fstat(pDescriptor)
    if (vStat.st_dev, vStat.st_ino) != (pExpected.st_dev, pExpected.st_ino) or not stat.S_ISFIFO(vStat.st_mode):
      raise AudacityUnavailableError(
        "The script pipe was replaced while it was being opened; refusing to use it",
        {"path": str(pPath)},
      )

  def mWaitForPipes(self, pTimeout: float) -> bool:
    """Block until both pipes appear, or the deadline passes."""
    vDeadline = time.monotonic() + pTimeout
    while time.monotonic() < vDeadline:
      if self.mPipesExist():
        return True
      time.sleep(0.25)
    return self.mPipesExist()

  def mGetConnected(self) -> bool:
    return self._vToDescriptor is not None and self._vFromDescriptor is not None

  connected = property(mGetConnected)

  # ------------------------------------------------------------ connection --

  def mConnect(self, pTimeout: float = 10.0) -> None:
    """Open both pipes, raising :class:`AudacityUnavailableError` on failure."""
    with self._vLock:
      if self.connected:
        return
      # mod-script-pipe tears the FIFOs down and recreates them when a client
      # disconnects, so two operations in a row can find them momentarily
      # absent. Wait for them when Audacity is alive; fail at once when it is
      # not, so a diagnostic does not stall for the full timeout.
      if not self.mPipesExist() and fAudacityIsRunning():
        vLog.debug("Script pipes are missing but Audacity is alive; waiting for them to reappear")
        self.mWaitForPipes(min(pTimeout, cPipeReappearTimeout))

      if not self.mPipesExist():
        raise AudacityUnavailableError(
          "Audacity's script pipes do not exist",
          {
            "expected": [str(self.to_pipe), str(self.from_pipe)],
            "hint": "Start Audacity with mod-script-pipe enabled, or run 'podcast-editor-mcp-server-cli audacity-status'",
          },
        )
      vToStat = self._mCheckPipe(self.to_pipe)
      vFromStat = self._mCheckPipe(self.from_pipe)

      vFromDescriptor: int | None = None
      vToDescriptor: int | None = None
      try:
        # O_RDWR keeps a writer attached so an idle pipe reads as "no data"
        # rather than a permanent EOF.
        vFromDescriptor = os.open(str(self.from_pipe), os.O_RDWR | os.O_NONBLOCK | os.O_NOFOLLOW)
        self._mCheckOpened(vFromDescriptor, vFromStat, self.from_pipe)
        vDeadline = time.monotonic() + pTimeout
        vLastError: OSError | None = None
        while time.monotonic() < vDeadline:
          try:
            vToDescriptor = os.open(str(self.to_pipe), os.O_WRONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
            break
          except OSError as vError:
            if vError.errno not in (errno.ENXIO, errno.EAGAIN):
              raise
            vLastError = vError
            time.sleep(0.2)
        if vToDescriptor is None:
          raise AudacityUnavailableError(
            "Audacity is not reading its command pipe",
            {
              "path": str(self.to_pipe),
              "errno": vLastError.errno if vLastError else None,
              "hint": "Audacity may have exited, or mod-script-pipe is not enabled",
            },
          )
        self._mCheckOpened(vToDescriptor, vToStat, self.to_pipe)
      except AudacityUnavailableError:
        if vFromDescriptor is not None:
          os.close(vFromDescriptor)
        if vToDescriptor is not None:
          os.close(vToDescriptor)
        raise
      except OSError as vError:
        if vFromDescriptor is not None:
          os.close(vFromDescriptor)
        if vToDescriptor is not None:
          os.close(vToDescriptor)
        raise AudacityUnavailableError(
          "Could not open Audacity's script pipes",
          {"reason": str(vError), "errno": vError.errno},
        ) from vError

      self._vFromDescriptor = vFromDescriptor
      self._vToDescriptor = vToDescriptor
      try:
        self._mDrain()
      except AudacityUnavailableError:
        self.mClose()
        raise
      vLog.info("Connected to Audacity", extra={"to": str(self.to_pipe), "from": str(self.from_pipe)})

  def mClose(self) -> None:
    """Close both descriptors. Safe to call more than once."""
    with self._vLock:
      for vAttribute in ("_vToDescriptor", "_vFromDescriptor"):
        vDescriptor = getattr(self, vAttribute)
        if vDescriptor is not None:
          try:
            os.close(vDescriptor)
          except OSError as vError:
            vLog.debug("Error closing a pipe descriptor", extra={"reason": str(vError)})
          setattr(self, vAttribute, None)
      self._mReleaseProcessLock()

  # ---------------------------------------------------------- process lock --

  def mAcquireProcessLock(self, pTimeout: float = 30.0) -> None:
    """Take an advisory lock so two processes cannot drive Audacity at once."""
    if self._vLockFile is not None:
      return
    vLockPath = self.pipe_directory / f"{cLockFileName}.{os.getuid()}"
    try:
      vDescriptor = os.open(str(vLockPath), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    except OSError as vError:
      raise AudacityUnavailableError(
        "Could not open the Audacity lock file",
        {"lock_file": str(vLockPath), "reason": str(vError)},
      ) from vError
    # In a shared directory another account could have created it first.
    if os.fstat(vDescriptor).st_uid != os.getuid():
      os.close(vDescriptor)
      raise AudacityUnavailableError(
        "The Audacity lock file belongs to another user; refusing to use it",
        {"lock_file": str(vLockPath)},
      )
    vDeadline = time.monotonic() + pTimeout
    while True:
      try:
        fcntl.flock(vDescriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self._vLockFile = vDescriptor
        return
      except OSError:
        if time.monotonic() >= vDeadline:
          os.close(vDescriptor)
          raise AudacityUnavailableError(
            "Another podcast-editor-mcp-server process is currently driving Audacity",
            {"lock_file": str(vLockPath)},
          ) from None
        time.sleep(0.25)

  def _mReleaseProcessLock(self) -> None:
    if self._vLockFile is None:
      return
    try:
      fcntl.flock(self._vLockFile, fcntl.LOCK_UN)
      os.close(self._vLockFile)
    except OSError as vError:
      vLog.debug("Error releasing the Audacity lock", extra={"reason": str(vError)})
    self._vLockFile = None

  # ---------------------------------------------------------------- I / O --

  def _mDrain(self) -> None:
    """Discard anything left in the reply pipe from a previous session.

    Leftovers are a few lines at most. A pipe that keeps producing data past
    ``cMaxDrainBytes`` or ``cMaxDrainSeconds`` is not Audacity's and is refused.
    """
    if self._vFromDescriptor is None:
      return
    vDeadline = time.monotonic() + cMaxDrainSeconds
    vDrained = 0
    while vDrained <= cMaxDrainBytes and time.monotonic() < vDeadline:
      try:
        vChunk = os.read(self._vFromDescriptor, cReadChunkBytes)
      except BlockingIOError:
        return
      except OSError:
        return
      if not vChunk:
        return
      vDrained += len(vChunk)
    raise AudacityUnavailableError(
      "Audacity's reply pipe keeps producing data; refusing to use it",
      {"drained_bytes": vDrained},
    )

  def _mWrite(self, pText: str, pTimeout: float) -> None:
    if self._vToDescriptor is None:
      raise AudacityUnavailableError("Not connected to Audacity")
    vPayload = pText.encode("utf-8")
    vDeadline = time.monotonic() + pTimeout
    vSelector = selectors.DefaultSelector()
    vSelector.register(self._vToDescriptor, selectors.EVENT_WRITE)
    try:
      vOffset = 0
      while vOffset < len(vPayload):
        vRemaining = vDeadline - time.monotonic()
        if vRemaining <= 0.0:
          raise AudacityCommandError("Timed out writing to Audacity", {"timeout_seconds": pTimeout})
        if not vSelector.select(timeout=vRemaining):
          continue
        try:
          vOffset += os.write(self._vToDescriptor, vPayload[vOffset:])
        except BlockingIOError:
          continue
        except BrokenPipeError as vError:
          raise AudacityUnavailableError(
            "Audacity closed the command pipe",
            {"reason": str(vError)},
          ) from vError
    finally:
      vSelector.close()

  def _mReadResponse(self, pTimeout: float) -> tuple[str, str]:
    """Read until the terminator line, returning ``(body, status)``."""
    if self._vFromDescriptor is None:
      raise AudacityUnavailableError("Not connected to Audacity")
    vDeadline = time.monotonic() + pTimeout
    vBuffer = ""
    vSelector = selectors.DefaultSelector()
    vSelector.register(self._vFromDescriptor, selectors.EVENT_READ)
    try:
      while True:
        vStatus = fFindTerminator(vBuffer)
        if vStatus is not None:
          vBody, vStatusText = vStatus
          return (vBody, vStatusText)
        vRemaining = vDeadline - time.monotonic()
        if vRemaining <= 0.0:
          raise AudacityCommandError(
            "Timed out waiting for Audacity",
            {"timeout_seconds": pTimeout, "partial_response": vBuffer[-500:]},
          )
        if not vSelector.select(timeout=min(0.5, vRemaining)):
          continue
        try:
          vChunk = os.read(self._vFromDescriptor, cReadChunkBytes)
        except BlockingIOError:
          continue
        except OSError as vError:
          raise AudacityUnavailableError(
            "Error reading from Audacity",
            {"reason": str(vError), "errno": vError.errno},
          ) from vError
        if vChunk:
          vBuffer += vChunk.decode("utf-8", errors="replace")
          if len(vBuffer) > cMaxResponseChars:
            raise AudacityCommandError(
              "Audacity's reply is larger than the limit",
              {"limit_chars": cMaxResponseChars},
            )
    finally:
      vSelector.close()

  def mSend(self, pCommand: AudacityCommand, pTimeout: float | None = None) -> AudacityResponse:
    """Send one validated command and wait for its reply."""
    vTimeout = pTimeout if pTimeout is not None else self.command_timeout
    vLine = pCommand.mRender()
    with self._vLock:
      if not self.connected:
        raise AudacityUnavailableError("Not connected to Audacity")
      vLog.debug("Audacity command", extra={"command": vLine})
      self._mWrite(vLine + "\n", min(vTimeout, 30.0))
      vBody, vStatus = self._mReadResponse(vTimeout)

    vResponse = AudacityResponse(body=vBody, status=vStatus)
    if not vResponse.ok:
      raise AudacityCommandError(
        "Audacity rejected a command",
        {"command": pCommand.name, "status": vStatus, "output": vBody.strip()[-500:]},
      )
    return vResponse

  def mRun(self, pName: str, pParameters: dict[str, Any] | None = None, pTimeout: float | None = None) -> AudacityResponse:
    """Build, validate and send a command in one step."""
    return self.mSend(AudacityCommand(pName, pParameters), pTimeout)

  def mPing(self, pTimeout: float = 15.0) -> bool:
    """Round-trip a harmless command to prove the connection works."""
    try:
      self.mRun("Help", {"Command": "Help"}, pTimeout)
    except (AudacityCommandError, AudacityUnavailableError) as vError:
      vLog.debug("Audacity ping failed", extra={"reason": str(vError)})
      return False
    return True

  def mListCommands(self, pTimeout: float = 30.0) -> list[dict[str, Any]]:
    """Ask Audacity which commands this build actually supports."""
    vResponse = self.mRun("GetInfo", {"Type": "Commands", "Format": "JSON"}, pTimeout)
    vText = vResponse.body.strip()
    vStart = vText.find("[")
    vEnd = vText.rfind("]")
    if vStart == -1 or vEnd == -1:
      return []
    try:
      ldParsed = json.loads(vText[vStart : vEnd + 1])
    except json.JSONDecodeError:
      return []
    return [dItem for dItem in ldParsed if isinstance(dItem, dict)]


def fFindTerminator(pBuffer: str) -> tuple[str, str] | None:
  """Split a reply buffer at ``BatchCommand finished: <status>``."""
  vIndex = pBuffer.find(cResponseTerminator)
  if vIndex == -1:
    return None
  vLineEnd = pBuffer.find("\n", vIndex)
  if vLineEnd == -1:
    return None
  vStatusLine = pBuffer[vIndex + len(cResponseTerminator) : vLineEnd].strip()
  vStatus = "OK" if vStatusLine.upper().startswith("OK") else vStatusLine or "Failed"
  return (pBuffer[:vIndex], vStatus)


# --------------------------------------------------------------------------- #
# Module configuration
# --------------------------------------------------------------------------- #


def fAudacityConfigPath() -> Path:
  """Location of ``audacity.cfg`` for the current user."""
  vBase = os.environ.get("XDG_CONFIG_HOME")
  vRoot = Path(vBase).expanduser() if vBase else Path.home() / ".config"
  return vRoot / "audacity" / "audacity.cfg"


def fReadModuleStatus(pConfigPath: Path | None = None) -> int | None:
  """Return the ``[Module]`` status of ``mod-script-pipe``, or ``None``."""
  vPath = pConfigPath or fAudacityConfigPath()
  if not vPath.is_file():
    return None
  vInModuleSection = False
  try:
    for vLine in vPath.read_text(encoding="utf-8", errors="replace").splitlines():
      vStripped = vLine.strip()
      if vStripped.startswith("["):
        vInModuleSection = vStripped == "[Module]"
        continue
      if vInModuleSection and vStripped.startswith(f"{cModuleName}="):
        try:
          return int(vStripped.split("=", 1)[1].strip())
        except ValueError:
          return None
  except OSError:
    return None
  return None


def fEnableScriptPipeModule(pConfigPath: Path | None = None, pModulePath: Path | None = None) -> dict[str, Any]:
  """Set ``mod-script-pipe=1`` in ``audacity.cfg``.

  This writes exactly the key that Audacity's own Preferences > Modules page
  writes, so it is a supported configuration change rather than poking at
  internals. The file is only rewritten when something in it changes; it is
  backed up first and replaced atomically, and Audacity must not be running
  while it happens (it rewrites the file on exit).
  """
  vPath = pConfigPath or fAudacityConfigPath()
  vLibrary = pModulePath or cModuleLibraryPath
  dResult: dict[str, Any] = {"config": str(vPath), "changed": False}

  if not vLibrary.is_file():
    dResult["error"] = f"{cModuleName}.so was not found at {vLibrary}"
    return dResult

  vPath.parent.mkdir(parents=True, exist_ok=True)
  if not vPath.exists():
    fWriteTextAtomic(
      vPath,
      f"[Module]\n{cModuleName}={cModuleEnabled}\n[ModulePath]\n{cModuleName}={vLibrary}\n",
    )
    dResult["changed"] = True
    dResult["created"] = True
    return dResult

  lLines = vPath.read_text(encoding="utf-8", errors="replace").splitlines()

  lOutput: list[str] = []
  vCurrentSection = ""
  vWroteModule = False
  vWrotePath = False
  vSawModuleSection = False
  vSawPathSection = False

  for vLine in lLines:
    vStripped = vLine.strip()
    if vStripped.startswith("[") and vStripped.endswith("]"):
      # Close out a section we were supposed to add a key to.
      if vCurrentSection == "[Module]" and not vWroteModule:
        lOutput.append(f"{cModuleName}={cModuleEnabled}")
        vWroteModule = True
        dResult["changed"] = True
      if vCurrentSection == "[ModulePath]" and not vWrotePath:
        lOutput.append(f"{cModuleName}={vLibrary}")
        vWrotePath = True
        dResult["changed"] = True
      vCurrentSection = vStripped
      if vStripped == "[Module]":
        vSawModuleSection = True
      elif vStripped == "[ModulePath]":
        vSawPathSection = True
      lOutput.append(vLine)
      continue

    if vCurrentSection == "[Module]" and vStripped.startswith(f"{cModuleName}="):
      if vStripped != f"{cModuleName}={cModuleEnabled}":
        dResult["changed"] = True
        dResult["previous_status"] = vStripped.split("=", 1)[1].strip()
      lOutput.append(f"{cModuleName}={cModuleEnabled}")
      vWroteModule = True
      continue

    if vCurrentSection == "[ModulePath]" and vStripped.startswith(f"{cModuleName}="):
      lOutput.append(f"{cModuleName}={vLibrary}")
      vWrotePath = True
      continue

    lOutput.append(vLine)

  if vCurrentSection == "[Module]" and not vWroteModule:
    lOutput.append(f"{cModuleName}={cModuleEnabled}")
    vWroteModule = True
    dResult["changed"] = True
  if vCurrentSection == "[ModulePath]" and not vWrotePath:
    lOutput.append(f"{cModuleName}={vLibrary}")
    vWrotePath = True
    dResult["changed"] = True

  if not vSawModuleSection:
    lOutput.extend(["[Module]", f"{cModuleName}={cModuleEnabled}"])
    dResult["changed"] = True
  if not vSawPathSection:
    lOutput.extend(["[ModulePath]", f"{cModuleName}={vLibrary}"])
    dResult["changed"] = True

  dResult["status"] = cModuleStatusNames[cModuleEnabled]
  if lOutput == lLines:
    dResult["changed"] = False
    return dResult
  vBackup = fBackupFile(vPath)
  if vBackup is not None:
    dResult["backup"] = str(vBackup)
  fWriteTextAtomic(vPath, "\n".join(lOutput) + "\n")
  dResult["changed"] = True
  return dResult


def fAudacityIsRunning() -> bool:
  """Whether an Audacity process belonging to this user is alive."""
  vPgrep = fWhich("pgrep")
  if vPgrep is None:
    return False
  vResult = fRunProcess([vPgrep, "-u", str(os.getuid()), "-x", "audacity"], pTimeout=10.0)
  return vResult.returncode == 0 and bool(vResult.stdout.strip())


# --------------------------------------------------------------------------- #
# Launcher
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class LaunchedProcesses:
  """Handles for the processes the launcher started."""

  xvfb: subprocess.Popen[bytes] | None = None
  audacity: subprocess.Popen[bytes] | None = None


class AudacityLauncher:
  """Starts Xvfb and Audacity and waits until the script pipes are live.

  Audacity is a GUI application: it needs an X display even when it is only
  being driven by script. On a headless machine that display is provided by
  Xvfb. Nothing here ever runs as root.
  """

  def __init__(self, pSettings: AudacitySettings) -> None:
    self.settings = pSettings
    self.processes = LaunchedProcesses()

  def mXvfbAvailable(self) -> bool:
    return fWhich("Xvfb") is not None

  def mAudacityPath(self) -> str | None:
    vExecutable = self.settings.executable
    if "/" in vExecutable:
      return vExecutable if Path(vExecutable).is_file() else None
    return fWhich(vExecutable)

  def mFindFreeDisplay(self, pStart: int = 90, pEnd: int = 120) -> str:
    """Return a ``:N`` display number whose X11 socket is not in use."""
    for vNumber in range(pStart, pEnd):
      vLock = Path(f"/tmp/.X{vNumber}-lock")  # noqa: S108 - X11's fixed location
      vSocket = Path(f"/tmp/.X11-unix/X{vNumber}")  # noqa: S108 - X11's fixed location
      if not vLock.exists() and not vSocket.exists():
        return f":{vNumber}"
    raise AudacityUnavailableError("No free X display number is available", {"searched": f"{pStart}-{pEnd}"})

  def mStart(self, pWaitSeconds: float | None = None) -> dict[str, Any]:
    """Start the stack and return a description of what is running."""
    if os.getuid() == 0:
      raise AudacityUnavailableError(
        "Refusing to start Audacity as root",
        {"hint": "Run podcast-editor-mcp-server as the ordinary user that owns the podcast files"},
      )

    vAudacity = self.mAudacityPath()
    if vAudacity is None:
      raise AudacityUnavailableError(
        "Audacity is not installed",
        {"looked_for": self.settings.executable, "hint": "sudo apt-get install audacity"},
      )

    vTimeout = pWaitSeconds if pWaitSeconds is not None else self.settings.startup_timeout_seconds
    dEnvironment = dict(os.environ)

    vDisplay = self.settings.display or os.environ.get("DISPLAY")
    vWantXvfb = self.settings.use_xvfb == "always" or (self.settings.use_xvfb == "auto" and not vDisplay)
    if vWantXvfb:
      if not self.mXvfbAvailable():
        if not vDisplay:
          raise AudacityUnavailableError(
            "No X display is available and Xvfb is not installed",
            {"hint": "sudo apt-get install xvfb"},
          )
        vLog.warning("Xvfb is not installed; using the existing display instead")
      else:
        vDisplay = self._mStartXvfb()
    if not vDisplay:
      raise AudacityUnavailableError(
        "Audacity needs an X display",
        {"hint": "Set audacity.display, enable audacity.use_xvfb, or run inside a desktop session"},
      )
    dEnvironment["DISPLAY"] = vDisplay

    vClient = AudacityPipeClient(Path(self.settings.pipe_directory), self.settings.command_timeout_seconds)
    vLog.info("Starting Audacity", extra={"display": vDisplay, "executable": vAudacity})
    self.processes.audacity = subprocess.Popen(  # noqa: S603 - argument list, shell=False
      [vAudacity],
      env=dEnvironment,
      stdin=subprocess.DEVNULL,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
      start_new_session=True,
    )

    if not vClient.mWaitForPipes(vTimeout):
      self.mStop()
      raise AudacityUnavailableError(
        "Audacity started but never created its script pipes",
        {
          "timeout_seconds": vTimeout,
          "hint": "mod-script-pipe is probably not enabled; run 'podcast-editor-mcp-server-cli audacity-status --enable-module'",
        },
      )

    vDeadline = time.monotonic() + vTimeout
    while time.monotonic() < vDeadline:
      try:
        vClient.mConnect(pTimeout=2.0)
        if vClient.mPing():
          vClient.mClose()
          return {
            "display": vDisplay,
            "audacity_pid": self.processes.audacity.pid,
            "xvfb_pid": self.processes.xvfb.pid if self.processes.xvfb else None,
            "pipes": [str(vClient.to_pipe), str(vClient.from_pipe)],
          }
        vClient.mClose()
      except AudacityUnavailableError:
        pass
      time.sleep(0.5)

    self.mStop()
    raise AudacityUnavailableError(
      "Audacity did not respond on its script pipe",
      {"timeout_seconds": vTimeout},
    )

  def _mStartXvfb(self) -> str:
    vDisplay = self.mFindFreeDisplay()
    vXvfb = fWhich("Xvfb")
    if vXvfb is None:  # pragma: no cover - guarded by the caller
      raise AudacityUnavailableError("Xvfb is not installed")
    vLog.info("Starting Xvfb", extra={"display": vDisplay, "screen": self.settings.xvfb_screen})
    self.processes.xvfb = subprocess.Popen(  # noqa: S603 - argument list, shell=False
      [vXvfb, vDisplay, "-screen", "0", self.settings.xvfb_screen, "-nolisten", "tcp"],
      stdin=subprocess.DEVNULL,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
      start_new_session=True,
    )
    vSocket = Path(f"/tmp/.X11-unix/X{vDisplay.lstrip(':')}")  # noqa: S108 - X11's fixed location
    vDeadline = time.monotonic() + 20.0
    while time.monotonic() < vDeadline:
      if vSocket.exists():
        return vDisplay
      if self.processes.xvfb.poll() is not None:
        raise AudacityUnavailableError("Xvfb exited immediately", {"display": vDisplay})
      time.sleep(0.2)
    self.mStop()
    raise AudacityUnavailableError("Xvfb did not come up in time", {"display": vDisplay})

  def mStop(self) -> None:
    """Terminate whatever this launcher started, children included."""
    for vName in ("audacity", "xvfb"):
      vProcess = getattr(self.processes, vName)
      if vProcess is None:
        continue
      if vProcess.poll() is None:
        try:
          os.killpg(os.getpgid(vProcess.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
          vProcess.terminate()
        try:
          vProcess.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
          try:
            os.killpg(os.getpgid(vProcess.pid), signal.SIGKILL)
          except (ProcessLookupError, PermissionError):
            vProcess.kill()
          try:
            vProcess.wait(timeout=5.0)
          except subprocess.TimeoutExpired:
            vLog.warning("Process did not exit after SIGKILL", extra={"process": vName})
      setattr(self.processes, vName, None)


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class AudacityProjectResult:
  """The outcome of building an Audacity project."""

  project: Path
  mode: str
  imported: str
  warnings: list[str] = field(default_factory=list)


class AudacityBackend:
  """Drives Audacity to produce editable ``.aup3`` projects."""

  name = "audacity"

  def __init__(self, pSettings: AudacitySettings | None = None) -> None:
    self.settings = pSettings if pSettings is not None else AudacitySettings()
    self.client = AudacityPipeClient(
      Path(self.settings.pipe_directory),
      self.settings.command_timeout_seconds,
    )

  # ----------------------------------------------------------------- status --

  def mStatus(self) -> BackendStatus:
    dDetails: dict[str, Any] = {}
    if not self.settings.enabled:
      return BackendStatus(name=self.name, available=False, reason="Audacity integration is disabled in the configuration")

    vExecutable = shutil.which(self.settings.executable) if "/" not in self.settings.executable else self.settings.executable
    dDetails["executable"] = vExecutable or "<not found>"
    if not vExecutable:
      return BackendStatus(
        name=self.name,
        available=False,
        reason="Audacity is not installed",
        details=dDetails,
      )
    # Report the version even when the backend is unusable: it is the first
    # thing anybody troubleshooting wants to know.
    vVersion = self.mVersion()

    vModuleStatus = fReadModuleStatus()
    dDetails["module_status"] = cModuleStatusNames.get(vModuleStatus, "unknown") if vModuleStatus is not None else "not configured"
    dDetails["module_library"] = str(cModuleLibraryPath) if cModuleLibraryPath.is_file() else "<not found>"
    dDetails["running"] = fAudacityIsRunning()
    dDetails["pipes_present"] = self.client.mPipesExist()
    dDetails["to_pipe"] = str(self.client.to_pipe)
    dDetails["from_pipe"] = str(self.client.from_pipe)

    if not self.client.mPipesExist():
      vReason = "Audacity is not running with mod-script-pipe"
      if vModuleStatus is not None and vModuleStatus != cModuleEnabled:
        vReason = f"mod-script-pipe is {cModuleStatusNames.get(vModuleStatus, 'not enabled')} in audacity.cfg"
      return BackendStatus(name=self.name, available=False, reason=vReason, version=vVersion, details=dDetails)

    try:
      self.client.mConnect(self.settings.connect_timeout_seconds)
      vAlive = self.client.mPing()
    except AudacityUnavailableError as vError:
      return BackendStatus(name=self.name, available=False, reason=vError.message, version=vVersion, details=dDetails)
    finally:
      self.client.mClose()

    if not vAlive:
      return BackendStatus(
        name=self.name,
        available=False,
        reason="Audacity did not answer on the script pipe",
        version=vVersion,
        details=dDetails,
      )
    return BackendStatus(name=self.name, available=True, version=vVersion, details=dDetails)

  def mVersion(self) -> str | None:
    """Report the installed Audacity version from the package manager."""
    vDpkg = fWhich("dpkg-query")
    if vDpkg is not None:
      vResult = fRunProcess([vDpkg, "-W", "-f=${Version}", "audacity"], pTimeout=15.0)
      if vResult.ok and vResult.stdout.strip():
        return vResult.stdout.strip()
    return None

  # -------------------------------------------------------------- projects --

  def mCreateProject(
    self,
    pAudio: Path,
    pProject: Path,
    pMode: str = "master",
    pSourceAudio: Path | None = None,
    pRemovals: list[tuple[float, float]] | None = None,
  ) -> AudacityProjectResult:
    """Build an editable ``.aup3`` project.

    ``master`` imports the finished audio, which is fast and always correct.
    ``replay`` imports the original and performs the cuts inside Audacity, so
    the user gets an undo history and can revise individual edits.
    """
    if pProject.suffix != ".aup3":
      raise AudacityCommandError("Audacity projects must use the .aup3 extension", {"path": pProject.name})
    lWarnings: list[str] = []
    vMode = pMode if pMode in {"master", "replay"} else "master"
    vImport = pAudio
    if vMode == "replay":
      if pSourceAudio is None or pRemovals is None:
        lWarnings.append("Replay mode needs the original audio and the cut list; falling back to master mode")
        vMode = "master"
      else:
        vImport = pSourceAudio

    # Audacity refuses to overwrite an existing project through SaveProject2.
    if pProject.exists():
      pProject.unlink()
    for vSuffix in ("-wal", "-shm"):
      Path(str(pProject) + vSuffix).unlink(missing_ok=True)

    self.client.mAcquireProcessLock()
    try:
      self.client.mConnect(self.settings.connect_timeout_seconds)
      self.client.mRun("New")
      self.client.mRun("Import2", {"Filename": str(vImport)})
      self.client.mRun("SelectTracks", {"Track": 0, "TrackCount": 1, "Mode": "Set"})
      if vMode == "replay" and pRemovals:
        for vStart, vEnd in sorted(pRemovals, key=lambda pItem: pItem[0], reverse=True):
          self.client.mRun("SelectTime", {"Start": vStart, "End": vEnd, "RelativeTo": "ProjectStart"})
          self.client.mRun("Delete")
      self.client.mRun("SelectNone")
      self.client.mRun("SaveProject2", {"Filename": str(pProject), "AddToHistory": 0})
    finally:
      self.client.mClose()

    if not pProject.exists():
      raise AudacityCommandError("Audacity did not write the project file", {"path": pProject.name})

    return AudacityProjectResult(
      project=pProject,
      mode=vMode,
      imported=vImport.name,
      warnings=lWarnings,
    )


def fInvertKeepSegments(pKeep: list[KeepSegment], pDuration: float) -> list[tuple[float, float]]:
  """Return the spans that must be deleted to leave only ``pKeep``."""
  lRemovals: list[tuple[float, float]] = []
  vCursor = 0.0
  for vSegment in sorted(pKeep, key=lambda pItem: pItem.source_start):
    if vSegment.source_start - vCursor > 1e-6:
      lRemovals.append((vCursor, vSegment.source_start))
    vCursor = max(vCursor, vSegment.source_end)
  if pDuration - vCursor > 1e-6:
    lRemovals.append((vCursor, pDuration))
  return lRemovals
