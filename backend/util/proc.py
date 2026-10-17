"""Safe external process execution.

Everything in this product that shells out goes through :func:`fRunProcess`.
There is no ``shell=True`` anywhere: commands are always argument lists, which
removes quoting and injection from the threat model entirely.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

from podcast_editor_mcp_server.errors import PodcastEditorMcpServerError
from podcast_editor_mcp_server.logging_setup import fGetLogger

vLog = fGetLogger("util.proc")

cDefaultTimeout: float = 900.0
# Only this much of stdout/stderr is kept in memory and in error payloads.
cMaxCapturedChars: int = 64_000


@dataclass(slots=True)
class ProcessResult:
  """Outcome of an external command."""

  returncode: int
  stdout: str
  stderr: str
  timed_out: bool = False

  def mGetOk(self) -> bool:
    return self.returncode == 0 and not self.timed_out

  ok = property(mGetOk)

  def mTail(self, pLines: int = 20) -> str:
    """Return the last lines of stderr, the useful part of an ffmpeg failure."""
    lLines = [vLine for vLine in self.stderr.splitlines() if vLine.strip()]
    return "\n".join(lLines[-pLines:])


def fTruncate(pText: str, pLimit: int = cMaxCapturedChars) -> str:
  """Clamp captured output so a chatty tool cannot exhaust memory or logs."""
  if len(pText) <= pLimit:
    return pText
  vKeep = pLimit // 2
  return pText[:vKeep] + "\n...[truncated]...\n" + pText[-vKeep:]


def fRunProcess(
  pCommand: list[str],
  pTimeout: float | None = cDefaultTimeout,
) -> ProcessResult:
  """Run ``pCommand`` without a shell and capture its output."""
  if not pCommand:
    raise ValueError("fRunProcess requires a non-empty command list")
  for vArgument in pCommand:
    if not isinstance(vArgument, str):
      raise TypeError(f"Command arguments must be str, got {type(vArgument).__name__}")
    if "\x00" in vArgument:
      raise ValueError("Command arguments must not contain NUL bytes")

  vTimedOut = False
  vStdout = ""
  vStderr = ""
  vReturnCode = -1

  vLog.debug("Running command", extra={"command": pCommand, "timeout": pTimeout})
  try:
    vCompleted = subprocess.run(  # noqa: S603 - argument list, shell=False
      pCommand,
      capture_output=True,
      text=True,
      encoding="utf-8",
      errors="replace",
      timeout=pTimeout,
      check=False,
      shell=False,
    )
    vReturnCode = vCompleted.returncode
    vStdout = fTruncate(vCompleted.stdout or "")
    vStderr = fTruncate(vCompleted.stderr or "")
  except subprocess.TimeoutExpired as vError:
    vTimedOut = True
    vStdout = fTruncate(_fDecode(vError.stdout))
    vStderr = fTruncate(_fDecode(vError.stderr))
    vReturnCode = -9
  except FileNotFoundError as vError:
    raise PodcastEditorMcpServerError(
      f"Executable not found: {pCommand[0]}",
      {"command": pCommand[0], "reason": str(vError)},
    ) from vError
  except PermissionError as vError:
    raise PodcastEditorMcpServerError(
      f"Executable not runnable: {pCommand[0]}",
      {"command": pCommand[0], "reason": str(vError)},
    ) from vError

  if vTimedOut:
    vLog.error("Command timed out", extra={"command": pCommand[0], "timeout": pTimeout})

  return ProcessResult(returncode=vReturnCode, stdout=vStdout, stderr=vStderr, timed_out=vTimedOut)


def _fDecode(pValue: bytes | str | None) -> str:
  """Best-effort decoding of partial output captured from a timeout."""
  if pValue is None:
    return ""
  if isinstance(pValue, bytes):
    return pValue.decode("utf-8", errors="replace")
  return pValue


def fWhich(pExecutable: str) -> str | None:
  """Locate an executable on PATH, returning ``None`` when it is absent."""
  return shutil.which(pExecutable)

