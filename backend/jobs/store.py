"""Job persistence.

One JSON document per job, written atomically. Files were chosen over SQLite
deliberately: a job record is read far more often by a human or an agent than
by the product, there are no cross-record queries worth a database, and a
half-written file can never corrupt the others.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from pydantic import ValidationError

from podcast_editor_mcp_server.errors import JobError as JobStoreError
from podcast_editor_mcp_server.jobs.models import Job, JobState
from podcast_editor_mcp_server.logging_setup import fGetLogger
from podcast_editor_mcp_server.util.atomic import fWriteJsonAtomic

vLog = fGetLogger("jobs.store")

cJobFileSuffix: str = ".job.json"
cMaxListed: int = 500


class JobStore:
  """A directory of job records."""

  def __init__(self, pDirectory: Path) -> None:
    self.directory = pDirectory

  def mPathFor(self, pJobId: str) -> Path:
    fValidateJobId(pJobId)
    return self.directory / f"{pJobId}{cJobFileSuffix}"

  def mSave(self, pJob: Job) -> Path:
    """Persist a job atomically and return its path."""
    self.directory.mkdir(parents=True, exist_ok=True)
    vPath = self.mPathFor(pJob.id)
    pJob.mTouch()
    fWriteJsonAtomic(vPath, pJob.model_dump(mode="json"))
    return vPath

  def mLoad(self, pJobId: str) -> Job:
    """Load one job, raising when it is missing or unreadable."""
    vPath = self.mPathFor(pJobId)
    if not vPath.is_file():
      raise JobStoreError("Job not found", {"job_id": pJobId})
    try:
      return Job.model_validate_json(vPath.read_text(encoding="utf-8"))
    except (ValidationError, ValueError) as vError:
      raise JobStoreError(
        "Job record is corrupt or does not match the current format",
        {"job_id": pJobId, "reason": str(vError)[:400]},
      ) from vError
    except OSError as vError:
      raise JobStoreError("Job record could not be read", {"job_id": pJobId, "reason": str(vError)}) from vError

  def mExists(self, pJobId: str) -> bool:
    try:
      return self.mPathFor(pJobId).is_file()
    except JobStoreError:
      return False

  def mList(self, pState: JobState | None = None, pLimit: int = 50) -> list[Job]:
    """List jobs newest first, optionally filtered by state."""
    if not self.directory.is_dir():
      return []
    lFiles = sorted(
      self.directory.glob(f"*{cJobFileSuffix}"),
      key=lambda pFile: pFile.stat().st_mtime,
      reverse=True,
    )[:cMaxListed]
    lJobs: list[Job] = []
    for vFile in lFiles:
      try:
        vJob = Job.model_validate_json(vFile.read_text(encoding="utf-8"))
      except (ValidationError, ValueError, OSError) as vError:
        vLog.warning("Skipping unreadable job record", extra={"path": vFile.name, "reason": str(vError)[:200]})
        continue
      if pState is not None and vJob.state != pState:
        continue
      lJobs.append(vJob)
      if len(lJobs) >= pLimit:
        break
    return lJobs

  def mPurgeOlderThan(self, pDays: int) -> int:
    """Delete terminal job records older than ``pDays``; returns the count."""
    if pDays <= 0 or not self.directory.is_dir():
      return 0
    vCutoff = dt.datetime.now(dt.UTC) - dt.timedelta(days=pDays)
    vRemoved = 0
    for vFile in self.directory.glob(f"*{cJobFileSuffix}"):
      try:
        vJob = Job.model_validate_json(vFile.read_text(encoding="utf-8"))
      except (ValidationError, ValueError, OSError):
        continue
      if not vJob.is_terminal:
        continue
      try:
        vFinished = dt.datetime.fromisoformat(vJob.finished_at or vJob.updated_at)
      except ValueError:
        continue
      if vFinished.tzinfo is None:
        vFinished = vFinished.replace(tzinfo=dt.UTC)
      if vFinished < vCutoff:
        vFile.unlink(missing_ok=True)
        vRemoved += 1
    return vRemoved


def fValidateJobId(pJobId: str) -> str:
  """Reject anything that is not a plain UUID.

  Job ids reach the filesystem, so they are checked as strictly as any other
  path component: no separators, no dots, no surprises.
  """
  vText = str(pJobId).strip()
  if not vText or len(vText) > 64:
    raise JobStoreError("Invalid job id", {"job_id": pJobId})
  for vCharacter in vText:
    if not (vCharacter.isalnum() or vCharacter == "-"):
      raise JobStoreError("Job ids may only contain letters, digits and hyphens", {"job_id": pJobId})
  return vText
