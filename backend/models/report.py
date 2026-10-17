"""The edit report: what actually happened, with enough provenance to redo it."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from podcast_editor_mcp_server.version import cProductName, cVersion


class ReportModel(BaseModel):
  model_config = ConfigDict(extra="forbid")


class ArtifactRecord(ReportModel):
  """One produced file."""

  kind: str
  path: str
  label: str
  size_bytes: int = Field(default=0, ge=0)
  sha256: str = ""
  duration_seconds: float | None = None
  media_type: str | None = None


class PhaseTiming(ReportModel):
  """Wall-clock cost of one pipeline phase."""

  phase: str
  seconds: float = Field(ge=0.0)


class LoudnessBeforeAfter(ReportModel):
  """Loudness figures on both sides of the edit."""

  integrated_lufs_before: float | None = None
  integrated_lufs_after: float | None = None
  true_peak_dbfs_before: float | None = None
  true_peak_dbfs_after: float | None = None
  sample_peak_dbfs_before: float | None = None
  loudness_range_before: float | None = None
  loudness_range_after: float | None = None
  target_lufs: float | None = None
  target_true_peak_dbtp: float | None = None


class EditCounts(ReportModel):
  """How many edits of each category were applied."""

  total_operations: int = 0
  structural_operations: int = 0
  mastering_operations: int = 0
  cuts: int = 0
  shortened_silences: int = 0
  removed_fillers: int = 0
  removed_repetitions: int = 0
  removed_false_starts: int = 0
  inserted_silences: int = 0
  by_type: dict[str, int] = Field(default_factory=dict)
  by_reason: dict[str, int] = Field(default_factory=dict)


class EditReport(ReportModel):
  """Final, self-contained record of one edit."""

  schema_version: int = 1
  generator: str = f"{cProductName}/{cVersion}"
  generated_at: str
  job_id: str | None = None
  plan_id: str
  profile: str

  source_path: str
  source_label: str
  source_sha256: str
  source_duration_seconds: float = Field(ge=0.0)

  output_duration_seconds: float = Field(ge=0.0)
  removed_seconds: float = Field(default=0.0, ge=0.0)
  inserted_seconds: float = Field(default=0.0, ge=0.0)
  removed_fraction: float = Field(default=0.0, ge=0.0, le=1.0)

  counts: EditCounts = Field(default_factory=EditCounts)
  loudness: LoudnessBeforeAfter = Field(default_factory=LoudnessBeforeAfter)

  transcription_backend: str | None = None
  transcription_model: str | None = None
  transcription_device: str | None = None
  transcription_language: str | None = None
  vad_backend: str | None = None
  edit_backend: str = "ffmpeg"
  audacity_project: str | None = None
  audacity_status: str = "not requested"

  phases: list[PhaseTiming] = Field(default_factory=list)
  total_seconds: float = Field(default=0.0, ge=0.0)
  artifacts: list[ArtifactRecord] = Field(default_factory=list)
  tool_versions: dict[str, str] = Field(default_factory=dict)
  warnings: list[str] = Field(default_factory=list)
  source_unchanged: bool = True

  def mSummary(self) -> dict[str, Any]:
    return {
      "plan_id": self.plan_id,
      "profile": self.profile,
      "source": self.source_label,
      "source_duration_seconds": round(self.source_duration_seconds, 3),
      "output_duration_seconds": round(self.output_duration_seconds, 3),
      "removed_seconds": round(self.removed_seconds, 3),
      "removed_percent": round(self.removed_fraction * 100.0, 2),
      "cuts": self.counts.cuts,
      "shortened_silences": self.counts.shortened_silences,
      "integrated_lufs_before": self.loudness.integrated_lufs_before,
      "integrated_lufs_after": self.loudness.integrated_lufs_after,
      "true_peak_dbfs_after": self.loudness.true_peak_dbfs_after,
      "artifacts": [{"kind": vArtifact.kind, "path": vArtifact.label} for vArtifact in self.artifacts],
      "audacity_status": self.audacity_status,
      "warnings": self.warnings,
      "total_seconds": round(self.total_seconds, 2),
    }


class SimulationResult(ReportModel):
  """What ``simulate_edit_plan`` returns: no audio is touched."""

  plan_id: str
  source: str
  source_sha256: str = ""
  source_duration_seconds: float = Field(ge=0.0)
  estimated_output_seconds: float = Field(ge=0.0)
  removed_seconds: float = Field(default=0.0, ge=0.0)
  inserted_seconds: float = Field(default=0.0, ge=0.0)
  removed_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
  operation_count: int = 0
  structural_count: int = 0
  mastering_count: int = 0
  segment_count: int = 0
  removed_regions: list[dict[str, Any]] = Field(default_factory=list)
  by_type: dict[str, int] = Field(default_factory=dict)
  by_reason: dict[str, int] = Field(default_factory=dict)
  valid: bool = True
  errors: list[dict[str, Any]] = Field(default_factory=list)
  warnings: list[dict[str, Any]] = Field(default_factory=list)
  conflicts: list[dict[str, Any]] = Field(default_factory=list)
