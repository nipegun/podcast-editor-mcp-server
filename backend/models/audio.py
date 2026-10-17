"""Audio description and analysis models.

These are the deterministic facts about a file. Anything a planner or an LLM
decides must be justified by numbers that live here.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AudioModel(BaseModel):
  """Base model for analysis payloads (unknown keys rejected)."""

  model_config = ConfigDict(extra="forbid")


class AudioProbe(AudioModel):
  """Container and stream facts reported by ffprobe."""

  path: str
  label: str
  size_bytes: int = Field(ge=0)
  sha256: str
  duration_seconds: float = Field(ge=0.0)
  sample_rate: int = Field(gt=0)
  channels: int = Field(gt=0)
  channel_layout: str | None = None
  codec_name: str
  codec_long_name: str | None = None
  sample_format: str | None = None
  bits_per_sample: int | None = None
  bit_rate: int | None = None
  format_name: str
  format_long_name: str | None = None
  stream_count: int = Field(ge=1)
  tags: dict[str, str] = Field(default_factory=dict)


class SilenceRegion(AudioModel):
  """A stretch of the source that sits below the silence threshold."""

  start: float = Field(ge=0.0)
  end: float = Field(ge=0.0)
  duration: float = Field(ge=0.0)


class SpeechRegion(AudioModel):
  """A stretch classified as speech by the VAD backend."""

  start: float = Field(ge=0.0)
  end: float = Field(ge=0.0)
  duration: float = Field(ge=0.0)


class ClippingRegion(AudioModel):
  """A run of consecutive samples at or beyond full scale."""

  start: float = Field(ge=0.0)
  end: float = Field(ge=0.0)
  duration: float = Field(ge=0.0)
  sample_count: int = Field(ge=1)


class LoudnessMeasurement(AudioModel):
  """EBU R128 measurement of a file or region."""

  integrated_lufs: float | None = None
  loudness_range: float | None = None
  threshold_lufs: float | None = None
  lra_low_lufs: float | None = None
  lra_high_lufs: float | None = None
  true_peak_dbfs: float | None = None
  sample_peak_dbfs: float | None = None
  source: Literal["ebur128", "loudnorm", "astats"] = "ebur128"


class DynamicsStatistics(AudioModel):
  """Per-file statistics from FFmpeg's ``astats`` filter."""

  peak_dbfs: float | None = None
  rms_dbfs: float | None = None
  rms_peak_dbfs: float | None = None
  rms_trough_dbfs: float | None = None
  dc_offset: float | None = None
  noise_floor_dbfs: float | None = None
  noise_floor_count: int | None = None
  dynamic_range_db: float | None = None
  flat_factor: float | None = None
  peak_count: int | None = None
  crest_factor: float | None = None
  entropy: float | None = None
  bit_depth: float | None = None
  number_of_samples: int | None = None
  zero_crossings_rate: float | None = None


class AnalysisReport(AudioModel):
  """Everything measured about one source file."""

  schema_version: int = 1
  generated_at: str
  tool_versions: dict[str, str] = Field(default_factory=dict)
  probe: AudioProbe
  loudness: LoudnessMeasurement
  dynamics: DynamicsStatistics
  silence_threshold_db: float
  silences: list[SilenceRegion] = Field(default_factory=list)
  speech_regions: list[SpeechRegion] = Field(default_factory=list)
  clipping: list[ClippingRegion] = Field(default_factory=list)
  clipping_sample_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
  speech_seconds: float = Field(default=0.0, ge=0.0)
  silence_seconds: float = Field(default=0.0, ge=0.0)
  speech_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
  longest_silence_seconds: float = Field(default=0.0, ge=0.0)
  vad_backend: str = "none"
  warnings: list[str] = Field(default_factory=list)

  def mSummary(self) -> dict[str, Any]:
    """Compact form used by MCP tool responses and job metrics."""
    return {
      "path": self.probe.label,
      "duration_seconds": round(self.probe.duration_seconds, 3),
      "sample_rate": self.probe.sample_rate,
      "channels": self.probe.channels,
      "codec": self.probe.codec_name,
      "integrated_lufs": self.loudness.integrated_lufs,
      "true_peak_dbfs": self.loudness.true_peak_dbfs,
      "loudness_range": self.loudness.loudness_range,
      "peak_dbfs": self.dynamics.peak_dbfs,
      "rms_dbfs": self.dynamics.rms_dbfs,
      "noise_floor_dbfs": self.dynamics.noise_floor_dbfs,
      "dc_offset": self.dynamics.dc_offset,
      "silence_count": len(self.silences),
      "silence_seconds": round(self.silence_seconds, 3),
      "speech_ratio": round(self.speech_ratio, 4),
      "longest_silence_seconds": round(self.longest_silence_seconds, 3),
      "clipping_regions": len(self.clipping),
      "warnings": self.warnings,
    }
