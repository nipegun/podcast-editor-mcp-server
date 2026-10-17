"""Atomic file writes.

An interrupted process must never leave behind a JSON document that parses but
is truncated. Every important artifact is written to a temporary file in the
destination directory, fsync'ed, and then ``os.replace``d over the target,
which is atomic within a single filesystem. When the target must not be
replaced, the temporary file is hard-linked into place instead, which fails if
anything is already there.

Runs that write into the same place hold an exclusive lock for the whole of
their work: :func:`fAcquireLock` and :func:`fReleaseLock`.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from podcast_editor_mcp_server.errors import ExportError
from podcast_editor_mcp_server.logging_setup import fGetLogger

vLog = fGetLogger("util.atomic")


@contextmanager
def fAtomicPath(pTarget: Path, pSuffix: str = ".tmp", pReplace: bool = True) -> Iterator[Path]:
  """Yield a temporary path that becomes ``pTarget`` on clean exit.

  With ``pReplace`` false an existing ``pTarget`` is never replaced, not even
  one that appeared while the temporary file was being written.
  """
  pTarget.parent.mkdir(parents=True, exist_ok=True)
  vHandle, vTempName = tempfile.mkstemp(
    dir=str(pTarget.parent),
    prefix=f".{pTarget.name}.",
    suffix=pSuffix,
  )
  os.close(vHandle)
  vTempPath = Path(vTempName)
  try:
    yield vTempPath
    if pReplace:
      vTempPath.replace(pTarget)
    else:
      _fPublishWithoutReplacing(vTempPath, pTarget)
  except BaseException:
    vTempPath.unlink(missing_ok=True)
    raise


def _fPublishWithoutReplacing(pTempPath: Path, pTarget: Path) -> None:
  """Move a finished temporary file to ``pTarget`` only if nothing is there."""
  try:
    os.link(pTempPath, pTarget)
  except FileExistsError:
    raise ExportError(
      "An artifact with this name already exists",
      {"path": pTarget.name, "hint": "pass overwrite to replace it"},
    ) from None
  except OSError:
    # A filesystem without hard links: the check and the rename are separate.
    if pTarget.exists():
      raise ExportError(
        "An artifact with this name already exists",
        {"path": pTarget.name, "hint": "pass overwrite to replace it"},
      ) from None
    pTempPath.replace(pTarget)
    return
  pTempPath.unlink(missing_ok=True)


def fWriteBytesAtomic(pTarget: Path, pData: bytes, pReplace: bool = True) -> Path:
  """Write bytes atomically and return the target path."""
  with fAtomicPath(pTarget, pReplace=pReplace) as vTempPath, vTempPath.open("wb") as vHandle:
    vHandle.write(pData)
    vHandle.flush()
    os.fsync(vHandle.fileno())
  return pTarget


def fWriteTextAtomic(pTarget: Path, pText: str, pEncoding: str = "utf-8", pReplace: bool = True) -> Path:
  """Write text atomically and return the target path."""
  return fWriteBytesAtomic(pTarget, pText.encode(pEncoding), pReplace)


def fWriteJsonAtomic(pTarget: Path, pPayload: Any, pIndent: int | None = 2, pReplace: bool = True) -> Path:
  """Serialise ``pPayload`` as UTF-8 JSON and write it atomically."""
  vText = json.dumps(pPayload, indent=pIndent, ensure_ascii=False, default=str) + "\n"
  return fWriteTextAtomic(pTarget, vText, pReplace=pReplace)


def fAcquireLock(pLockPath: Path) -> int | None:
  """Take an exclusive lock on ``pLockPath`` without waiting.

  Returns the open descriptor that holds it, or ``None`` when somebody else
  holds it. The lock is an ``flock`` on an open file, so it covers threads and
  processes alike, and the kernel drops it when the process dies.
  """
  pLockPath.parent.mkdir(parents=True, exist_ok=True)
  while True:
    vDescriptor = os.open(str(pLockPath), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
      fcntl.flock(vDescriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
      os.close(vDescriptor)
      return None
    try:
      vOnDisk = os.stat(pLockPath)
    except FileNotFoundError:
      vOnDisk = None
    vOpened = os.fstat(vDescriptor)
    if vOnDisk is not None and (vOnDisk.st_dev, vOnDisk.st_ino) == (vOpened.st_dev, vOpened.st_ino):
      return vDescriptor
    # Its previous holder released it and removed the file between the open
    # and the lock: this descriptor locks nothing anybody else can see.
    os.close(vDescriptor)


def fReleaseLock(pLockPath: Path, pDescriptor: int) -> None:
  """Release a lock taken with :func:`fAcquireLock` and remove its file."""
  try:
    # Removed while still held, so nobody can lock the file being removed.
    pLockPath.unlink(missing_ok=True)
  finally:
    os.close(pDescriptor)


def fDefaultFileMode() -> int:
  """The mode a file created now would get from the process umask.

  Read from ``/proc`` rather than with ``os.umask``, which can only be read
  by changing it for every thread at once. 0o644 when it cannot be read.
  """
  try:
    for vLine in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
      if vLine.startswith("Umask:"):
        return 0o666 & ~int(vLine.split()[1], 8)
  except (OSError, ValueError, IndexError):
    pass
  return 0o644


def fBackupFile(pTarget: Path, pSuffix: str = ".bak") -> Path | None:
  """Copy ``pTarget`` aside before it is overwritten, returning the copy path."""
  if not pTarget.exists():
    return None
  vBackup = pTarget.with_name(pTarget.name + pSuffix)
  vCounter = 1
  while vBackup.exists():
    vBackup = pTarget.with_name(f"{pTarget.name}{pSuffix}.{vCounter}")
    vCounter += 1
    if vCounter > 100:
      raise OSError(f"Too many backups already exist next to {pTarget}")
  vBackup.write_bytes(pTarget.read_bytes())
  vLog.debug("Created backup", extra={"source": str(pTarget), "backup": str(vBackup)})
  return vBackup
