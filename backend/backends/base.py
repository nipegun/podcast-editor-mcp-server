"""Types shared by the backends.

:class:`~podcast_editor_mcp_server.backends.ffmpeg.FfmpegBackend` renders a compiled
:class:`~podcast_editor_mcp_server.util.timeline.Timeline` from a :class:`RenderRequest`;
every backend, FFmpeg, Audacity, transcription and VAD, reports its availability as a
:class:`BackendStatus`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from podcast_editor_mcp_server.models.edit_plan import EditPlan
from podcast_editor_mcp_server.util.timeline import Timeline


@dataclass(slots=True)
class RenderRequest:
  """Everything a backend needs to render one plan."""

  source: Path
  plan: EditPlan
  timeline: Timeline
  work_dir: Path
  target: Path
  sample_rate: int
  channels: int


@dataclass(slots=True)
class RenderResult:
  """What a backend produced."""

  path: Path
  applied_operations: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BackendStatus:
  """Availability report for one backend."""

  name: str
  available: bool
  reason: str = ""
  version: str | None = None
  details: dict[str, Any] = field(default_factory=dict)

  def mAsDict(self) -> dict[str, Any]:
    dResult: dict[str, Any] = {"name": self.name, "available": self.available}
    if self.version:
      dResult["version"] = self.version
    if self.reason:
      dResult["reason"] = self.reason
    if self.details:
      dResult["details"] = self.details
    return dResult
