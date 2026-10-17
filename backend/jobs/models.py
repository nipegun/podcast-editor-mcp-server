"""Job model: the durable record of one processing run."""

from __future__ import annotations

import datetime as dt
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from podcast_editor_mcp_server.models.report import ArtifactRecord, EditReport
from podcast_editor_mcp_server.version import cJobSchemaVersion, cProductName, cVersion


class JobState(StrEnum):
  """Where a job is in its lifecycle."""

  QUEUED = "queued"
  PROBING = "probing"
  TRANSCRIBING = "transcribing"
  ANALYZING = "analyzing"
  PLANNING = "planning"
  VALIDATING = "validating"
  EDITING = "editing"
  MASTERING = "mastering"
  EXPORTING = "exporting"
  COMPLETED = "completed"
  FAILED = "failed"
  CANCELLED = "cancelled"


cTerminalStates: frozenset[JobState] = frozenset({JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED})

# Rough progress fraction reached on entering each state, for clients that show
# a bar. Transcription dominates the wall clock, so it owns the widest band.
cStateProgress: dict[JobState, float] = {
  JobState.QUEUED: 0.0,
  JobState.PROBING: 0.03,
  JobState.ANALYZING: 0.10,
  JobState.TRANSCRIBING: 0.25,
  JobState.PLANNING: 0.70,
  JobState.VALIDATING: 0.75,
  JobState.EDITING: 0.80,
  JobState.MASTERING: 0.90,
  JobState.EXPORTING: 0.95,
  JobState.COMPLETED: 1.0,
  JobState.FAILED: 1.0,
  JobState.CANCELLED: 1.0,
}


class JobMessage(BaseModel):
  """One timestamped log line attached to a job."""

  model_config = ConfigDict(extra="forbid")

  at: str = Field(default_factory=lambda: dt.datetime.now(dt.UTC).isoformat())
  level: str = "info"
  message: str
  details: dict[str, Any] = Field(default_factory=dict)


class JobError(BaseModel):
  """A failure recorded on a job."""

  model_config = ConfigDict(extra="forbid")

  at: str = Field(default_factory=lambda: dt.datetime.now(dt.UTC).isoformat())
  code: str
  message: str
  state: str
  details: dict[str, Any] = Field(default_factory=dict)


class Job(BaseModel):
  """Everything known about one processing run."""

  model_config = ConfigDict(extra="forbid")

  schema_version: int = cJobSchemaVersion
  id: str = Field(default_factory=lambda: str(uuid.uuid4()))
  generator: str = f"{cProductName}/{cVersion}"
  created_at: str = Field(default_factory=lambda: dt.datetime.now(dt.UTC).isoformat())
  updated_at: str = Field(default_factory=lambda: dt.datetime.now(dt.UTC).isoformat())
  finished_at: str | None = None

  state: JobState = JobState.QUEUED
  progress: float = Field(default=0.0, ge=0.0, le=1.0)

  source_path: str
  source_label: str
  source_sha256: str = ""
  source_duration_seconds: float = Field(default=0.0, ge=0.0)

  profile: str = ""
  profile_sha256: str = ""
  options: dict[str, Any] = Field(default_factory=dict)

  work_dir: str = ""
  output_dir: str = ""

  plan_id: str | None = None
  report_path: str | None = None

  messages: list[JobMessage] = Field(default_factory=list)
  errors: list[JobError] = Field(default_factory=list)
  artifacts: list[ArtifactRecord] = Field(default_factory=list)
  metrics: dict[str, Any] = Field(default_factory=dict)
  report: EditReport | None = None

  # ------------------------------------------------------------- mutations --

  def mTouch(self) -> None:
    self.updated_at = dt.datetime.now(dt.UTC).isoformat()

  def mSetState(self, pState: JobState, pMessage: str | None = None) -> None:
    self.state = pState
    self.progress = max(self.progress, cStateProgress.get(pState, self.progress))
    if pState in cTerminalStates:
      self.finished_at = dt.datetime.now(dt.UTC).isoformat()
      self.progress = 1.0
    if pMessage:
      self.mAddMessage(pMessage)
    self.mTouch()

  def mSetProgress(self, pFraction: float) -> None:
    self.progress = min(1.0, max(self.progress, float(pFraction)))
    self.mTouch()

  def mAddMessage(self, pMessage: str, pLevel: str = "info", pDetails: dict[str, Any] | None = None) -> None:
    self.messages.append(JobMessage(level=pLevel, message=pMessage, details=pDetails or {}))
    # Keep the record bounded: a runaway loop must not grow the file forever.
    if len(self.messages) > 500:
      self.messages = self.messages[-500:]
    self.mTouch()

  def mAddError(self, pCode: str, pMessage: str, pDetails: dict[str, Any] | None = None) -> None:
    self.errors.append(JobError(code=pCode, message=pMessage, state=str(self.state), details=pDetails or {}))
    self.mTouch()

  def mAddArtifact(self, pArtifact: ArtifactRecord) -> None:
    self.artifacts = [vItem for vItem in self.artifacts if vItem.path != pArtifact.path]
    self.artifacts.append(pArtifact)
    self.mTouch()

  # -------------------------------------------------------------- queries --

  def mGetIsTerminal(self) -> bool:
    return self.state in cTerminalStates

  is_terminal = property(mGetIsTerminal)

  def mSummary(self) -> dict[str, Any]:
    return {
      "id": self.id,
      "state": str(self.state),
      "progress": round(self.progress, 3),
      "source": self.source_label,
      "profile": self.profile,
      "created_at": self.created_at,
      "updated_at": self.updated_at,
      "finished_at": self.finished_at,
      "duration_seconds": round(self.source_duration_seconds, 2),
      "artifact_count": len(self.artifacts),
      "error_count": len(self.errors),
      "last_message": self.messages[-1].message if self.messages else None,
    }
