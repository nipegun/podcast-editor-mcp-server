"""Structured logging.

Hard rule: stdout belongs to the MCP JSON-RPC stream. Nothing here ever writes
to ``sys.stdout``. Console output goes to stderr, everything else goes to
rotating files under the workspace log directory.

The module is named ``logging_setup`` rather than ``logging`` so that reading a
traceback never leaves any doubt about which module a frame belongs to.
"""

from __future__ import annotations

import contextlib
import json
import logging
import logging.handlers
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

cLoggerRoot: str = "podcast_editor_mcp_server"
cMaxLogBytes: int = 8 * 1024 * 1024
cLogBackupCount: int = 5

# Attributes that ``logging.LogRecord`` always carries; anything else a caller
# attaches through ``extra=`` is treated as structured context.
cReservedRecordKeys: frozenset[str] = frozenset(
  {
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs", "message", "msg", "name",
    "pathname", "process", "processName", "relativeCreated", "stack_info",
    "taskName", "thread", "threadName",
  }
)

_vConfigured: bool = False
_vConfigureLock = threading.Lock()


class StructuredFormatter(logging.Formatter):
  """Human readable line plus ``key=value`` pairs for any structured context."""

  def mFormat(self, pRecord: logging.LogRecord) -> str:
    vBase = super().format(pRecord)
    lExtras: list[str] = []
    for vKey, vValue in sorted(pRecord.__dict__.items()):
      if vKey in cReservedRecordKeys or vKey.startswith("_"):
        continue
      lExtras.append(f"{vKey}={vValue!r}")
    if lExtras:
      return vBase + " | " + " ".join(lExtras)
    return vBase

  format = mFormat


class JsonFormatter(logging.Formatter):
  """One JSON object per line, for machine consumption."""

  def mFormat(self, pRecord: logging.LogRecord) -> str:
    dPayload: dict[str, Any] = {
      "ts": self.formatTime(pRecord, "%Y-%m-%dT%H:%M:%S%z"),
      "level": pRecord.levelname,
      "logger": pRecord.name,
      "message": pRecord.getMessage(),
    }
    for vKey, vValue in pRecord.__dict__.items():
      if vKey in cReservedRecordKeys or vKey.startswith("_"):
        continue
      try:
        json.dumps(vValue)
        dPayload[vKey] = vValue
      except (TypeError, ValueError):
        dPayload[vKey] = repr(vValue)
    if pRecord.exc_info:
      dPayload["exception"] = self.formatException(pRecord.exc_info)
    return json.dumps(dPayload, ensure_ascii=False)

  format = mFormat


def fResolveLevel(pLevel: str) -> int:
  """Translate a level name into a logging level number."""
  vName = pLevel.strip().upper()
  vResolved = logging.getLevelNamesMapping().get(vName)
  if vResolved is None:
    return logging.INFO
  return vResolved


def fSetupLogging(
  pLevel: str,
  pLogFile: Path | None = None,
  pJson: bool = False,
  pForce: bool = False,
) -> logging.Logger:
  """Configure the ``podcast_editor_mcp_server`` logger tree.

  Idempotent: calling it twice keeps the first configuration unless ``pForce``
  is set. Console output always goes to stderr.
  """
  global _vConfigured
  with _vConfigureLock:
    vLogger = logging.getLogger(cLoggerRoot)
    if _vConfigured and not pForce:
      return vLogger
    for vHandler in list(vLogger.handlers):
      vLogger.removeHandler(vHandler)
      with contextlib.suppress(OSError, ValueError):
        vHandler.close()

    vLevel = fResolveLevel(pLevel)
    vLogger.setLevel(vLevel)
    # Never let records bubble up to the root logger: a third-party
    # ``basicConfig()`` could otherwise attach a stdout handler behind our back.
    vLogger.propagate = False

    vFormatter: logging.Formatter
    if pJson:
      vFormatter = JsonFormatter()
    else:
      vFormatter = StructuredFormatter(
        fmt="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
      )

    vConsoleHandler = logging.StreamHandler(stream=sys.stderr)
    vConsoleHandler.setFormatter(vFormatter)
    # The level lives on the handler as well as the logger, so a per-job
    # handler can lower the logger's level without making the console verbose.
    vConsoleHandler.setLevel(vLevel)
    vLogger.addHandler(vConsoleHandler)

    if pLogFile is not None:
      try:
        pLogFile.parent.mkdir(parents=True, exist_ok=True)
        vFileHandler = logging.handlers.RotatingFileHandler(
          filename=str(pLogFile),
          maxBytes=cMaxLogBytes,
          backupCount=cLogBackupCount,
          encoding="utf-8",
        )
        vFileHandler.setFormatter(JsonFormatter() if pJson else vFormatter)
        vFileHandler.setLevel(vLevel)
        vLogger.addHandler(vFileHandler)
      except OSError as vError:
        # A missing log directory must never take the server down; warn on
        # stderr and keep going with console logging only.
        vLogger.warning("Could not open log file %s: %s", pLogFile, vError)

    _vConfigured = True
    return vLogger


def fSafeExtra(pExtra: dict[str, Any]) -> dict[str, Any]:
  """Rename structured-context keys that collide with ``LogRecord`` attributes.

  ``logging.Logger.makeRecord`` raises ``KeyError`` when ``extra`` carries a
  reserved name such as ``message``, ``name`` or ``module``. Losing a log line
  would be bad enough; because these calls sit inside error handlers, the
  exception would also replace a precise error with a generic one. Colliding
  keys are prefixed instead.
  """
  dSafe: dict[str, Any] = {}
  for vKey, vValue in pExtra.items():
    dSafe[f"ctx_{vKey}" if vKey in cReservedRecordKeys else vKey] = vValue
  return dSafe


class SafeLoggerAdapter(logging.LoggerAdapter):
  """A logger whose ``extra=`` can never break ``makeRecord``."""

  def mProcess(self, pMsg: Any, pKwargs: Any) -> tuple[Any, Any]:
    dExtra = pKwargs.get("extra")
    if dExtra:
      pKwargs["extra"] = fSafeExtra(dExtra)
    return (pMsg, pKwargs)

  process = mProcess


def fGetLogger(pName: str) -> SafeLoggerAdapter:
  """Return a child logger of the product logger tree."""
  vFullName = pName if pName.startswith(cLoggerRoot) else f"{cLoggerRoot}.{pName}"
  return SafeLoggerAdapter(logging.getLogger(vFullName), {})


@contextmanager
def fJobLogHandler(pLogFile: Path, pLevel: str = "DEBUG") -> Iterator[Path]:
  """Temporarily tee the product logger into a per-job log file."""
  vLogger = logging.getLogger(cLoggerRoot)
  pLogFile.parent.mkdir(parents=True, exist_ok=True)
  vHandler = logging.FileHandler(str(pLogFile), encoding="utf-8")
  vHandler.setFormatter(JsonFormatter())
  vHandler.setLevel(fResolveLevel(pLevel))
  vPreviousLevel = vLogger.level
  if vHandler.level < vPreviousLevel:
    vLogger.setLevel(vHandler.level)
  vLogger.addHandler(vHandler)
  try:
    yield pLogFile
  finally:
    vLogger.removeHandler(vHandler)
    vLogger.setLevel(vPreviousLevel)
    with contextlib.suppress(OSError, ValueError):
      vHandler.close()
