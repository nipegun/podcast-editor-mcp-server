"""Exception hierarchy for podcast-editor-mcp-server.

Every error carries a short, user-facing ``message`` and an optional ``details``
mapping with structured context. MCP tools surface ``message`` (plus ``details``
when it is safe); tracebacks only ever reach the log files.
"""

from __future__ import annotations

from typing import Any


class PodcastEditorMcpServerError(Exception):
  """Base class for every error raised by this product."""

  cCode: str = "podcast_editor_mcp_server_error"

  def __init__(self, pMessage: str, pDetails: dict[str, Any] | None = None) -> None:
    super().__init__(pMessage)
    self.message = pMessage
    self.details: dict[str, Any] = dict(pDetails) if pDetails else {}

  def __str__(self) -> str:
    if self.details:
      return f"{self.message} ({self.details})"
    return self.message


class ConfigurationError(PodcastEditorMcpServerError):
  """A profile, configuration file or environment value is invalid."""

  cCode = "configuration_error"


class SecurityError(PodcastEditorMcpServerError):
  """A request tried to escape the configured workspace or broke a safety rule."""

  cCode = "security_error"


class AudioProbeError(PodcastEditorMcpServerError):
  """ffprobe could not describe the file, or the file is not usable audio."""

  cCode = "audio_probe_error"


class AnalysisError(PodcastEditorMcpServerError):
  """A DSP analysis step failed."""

  cCode = "analysis_error"


class TranscriptionError(PodcastEditorMcpServerError):
  """The transcription backend is unavailable or failed."""

  cCode = "transcription_error"


class VadError(PodcastEditorMcpServerError):
  """The voice-activity-detection backend is unavailable or failed."""

  cCode = "vad_error"


class EditPlanError(PodcastEditorMcpServerError):
  """An EditPlan could not be built, loaded or serialised."""

  cCode = "edit_plan_error"


class EditPlanValidationError(PodcastEditorMcpServerError):
  """An EditPlan was rejected by the validator."""

  cCode = "edit_plan_validation_error"


class ExecutionError(PodcastEditorMcpServerError):
  """The execution engine failed while rendering a plan."""

  cCode = "execution_error"


class FfmpegError(PodcastEditorMcpServerError):
  """An ffmpeg/ffprobe invocation failed, timed out or produced no output."""

  cCode = "ffmpeg_error"


class ExportError(PodcastEditorMcpServerError):
  """A final artifact could not be written."""

  cCode = "export_error"


class AudacityUnavailableError(PodcastEditorMcpServerError):
  """Audacity or mod-script-pipe is not usable right now."""

  cCode = "audacity_unavailable"


class AudacityCommandError(PodcastEditorMcpServerError):
  """Audacity accepted the connection but rejected or failed a command."""

  cCode = "audacity_command_error"


class JobError(PodcastEditorMcpServerError):
  """A job could not be created, loaded, updated or resumed."""

  cCode = "job_error"
