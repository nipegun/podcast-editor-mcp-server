"""The EditPlan: a strict, versioned description of an edit.

Design rules that the rest of the engine depends on:

* **Every structural timestamp is expressed in coordinates of the original
  source file.** Nothing in a plan ever refers to a partially edited timeline,
  which is what makes plans reviewable, cacheable and re-runnable.
* **Operations are a closed set.** A plan carrying an unknown ``type`` fails to
  parse; it never reaches the execution engine.
* **Numbers are finite.** ``NaN`` and ``Infinity`` are rejected at the model
  level, not later.
* **A plan is bound to one exact file** through ``source_sha256``.
"""

from __future__ import annotations

import datetime as dt
import uuid
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from podcast_editor_mcp_server.version import cEditPlanSchemaVersion, cProductName, cVersion

# Finite numeric aliases -------------------------------------------------------

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
TimeSeconds = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
PositiveSeconds = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
NonNegativeSeconds = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]

cMaxOperationIdLength: int = 64
cMaxReasonTextLength: int = 500


class EditReason(StrEnum):
  """Why an operation exists. Used for reporting and for per-category counts."""

  MANUAL = "manual"
  LONG_PAUSE = "long_pause"
  SILENCE = "silence"
  HEAD_SILENCE = "head_silence"
  TAIL_SILENCE = "tail_silence"
  PADDING = "padding"
  FILLER_WORD = "filler_word"
  REPETITION = "repetition"
  FALSE_START = "false_start"
  MISTAKE = "mistake"
  NOISE = "noise"
  CLEANUP = "cleanup"
  MASTERING = "mastering"
  OTHER = "other"


class BaseOperation(BaseModel):
  """Fields shared by every operation."""

  model_config = ConfigDict(extra="forbid")

  id: str = Field(default="", max_length=cMaxOperationIdLength)
  reason: EditReason = EditReason.MANUAL
  confidence: FiniteFloat = Field(default=1.0, ge=0.0, le=1.0)
  note: str = Field(default="", max_length=cMaxReasonTextLength)
  source: str = Field(default="planner", max_length=64)


class RangeOperation(BaseOperation):
  """An operation bounded by ``[start, end]`` in source coordinates."""

  start: TimeSeconds
  end: TimeSeconds

  @model_validator(mode="after")
  def mCheckRange(self) -> RangeOperation:
    if self.end <= self.start:
      raise ValueError("end must be strictly greater than start")
    return self

  def mGetDuration(self) -> float:
    return self.end - self.start

  duration = property(mGetDuration)


# --------------------------------------------------------------------------- #
# Structural operations (rewrite the timeline)
# --------------------------------------------------------------------------- #


class DeleteRange(RangeOperation):
  """Remove ``[start, end]`` from the source entirely."""

  type: Literal["delete_range"] = "delete_range"

  def mDescribe(self) -> str:
    return f"delete {self.start:.3f}-{self.end:.3f}s ({self.duration:.3f}s)"


class ShortenSilence(RangeOperation):
  """Shrink a silent region to ``target_duration``, trimmed from the middle.

  Trimming from the centre keeps the natural decay of the preceding phrase and
  the natural onset of the following one, which a one-sided cut destroys.
  """

  type: Literal["shorten_silence"] = "shorten_silence"
  target_duration: NonNegativeSeconds = 0.6

  @model_validator(mode="after")
  def mCheckTarget(self) -> ShortenSilence:
    if self.target_duration >= self.duration:
      raise ValueError("target_duration must be shorter than the silence itself")
    return self

  def mDescribe(self) -> str:
    return f"shorten silence {self.start:.3f}-{self.end:.3f}s to {self.target_duration:.3f}s"


class ReplaceRangeWithSilence(RangeOperation):
  """Replace ``[start, end]`` with digital silence.

  ``duration`` defaults to the length of the removed region, which keeps the
  total running time unchanged (useful for redactions).
  """

  type: Literal["replace_range_with_silence"] = "replace_range_with_silence"
  # Named explicitly so it cannot collide with RangeOperation.duration, which
  # is the length of the *removed* region rather than of the inserted silence.
  silence_duration: NonNegativeSeconds | None = None

  def mGetEffectiveSilenceDuration(self) -> float:
    if self.silence_duration is not None:
      return self.silence_duration
    return self.end - self.start

  effective_silence_duration = property(mGetEffectiveSilenceDuration)

  def mDescribe(self) -> str:
    return f"silence {self.start:.3f}-{self.end:.3f}s ({self.effective_silence_duration:.3f}s of silence)"


class TrimStart(BaseOperation):
  """Drop the first ``duration`` seconds of the source."""

  type: Literal["trim_start"] = "trim_start"
  duration: PositiveSeconds

  def mDescribe(self) -> str:
    return f"trim {self.duration:.3f}s from the start"


class TrimEnd(BaseOperation):
  """Drop the last ``duration`` seconds of the source."""

  type: Literal["trim_end"] = "trim_end"
  duration: PositiveSeconds

  def mDescribe(self) -> str:
    return f"trim {self.duration:.3f}s from the end"


class InsertSilence(BaseOperation):
  """Insert ``duration`` seconds of silence at source position ``at``.

  ``at == 0`` pads the head, ``at == source_duration`` pads the tail.
  """

  type: Literal["insert_silence"] = "insert_silence"
  at: TimeSeconds
  duration: PositiveSeconds

  def mDescribe(self) -> str:
    return f"insert {self.duration:.3f}s of silence at {self.at:.3f}s"


# --------------------------------------------------------------------------- #
# Mastering operations (filter the rendered audio)
# --------------------------------------------------------------------------- #


class FadeIn(BaseOperation):
  """Fade the rendered audio in over ``duration`` seconds."""

  type: Literal["fade_in"] = "fade_in"
  duration: PositiveSeconds
  curve: Literal["tri", "qsin", "hsin", "esin", "log", "exp", "cub", "squ", "cbr", "par", "ipar"] = "tri"

  def mDescribe(self) -> str:
    return f"fade in over {self.duration:.3f}s"


class FadeOut(BaseOperation):
  """Fade the rendered audio out over ``duration`` seconds."""

  type: Literal["fade_out"] = "fade_out"
  duration: PositiveSeconds
  curve: Literal["tri", "qsin", "hsin", "esin", "log", "exp", "cub", "squ", "cbr", "par", "ipar"] = "tri"

  def mDescribe(self) -> str:
    return f"fade out over {self.duration:.3f}s"


class NormalizeLoudness(BaseOperation):
  """EBU R128 loudness normalisation to an integrated target."""

  type: Literal["normalize_loudness"] = "normalize_loudness"
  integrated_lufs: FiniteFloat = Field(default=-16.0, ge=-40.0, le=-5.0)
  true_peak_dbtp: FiniteFloat = Field(default=-1.0, ge=-12.0, le=0.0)
  loudness_range: FiniteFloat = Field(default=11.0, gt=0.0, le=30.0)
  dual_pass: bool = True

  def mDescribe(self) -> str:
    return f"normalise to {self.integrated_lufs:g} LUFS / {self.true_peak_dbtp:g} dBTP"


class LimitTruePeak(BaseOperation):
  """Brick-wall limiter placed after normalisation."""

  type: Literal["limit_true_peak"] = "limit_true_peak"
  true_peak_dbtp: FiniteFloat = Field(default=-1.0, ge=-12.0, le=0.0)
  attack_ms: FiniteFloat = Field(default=5.0, gt=0.0, le=100.0)
  release_ms: FiniteFloat = Field(default=50.0, gt=0.0, le=5_000.0)

  def mDescribe(self) -> str:
    return f"limit true peak to {self.true_peak_dbtp:g} dBTP"


class HighPassFilter(BaseOperation):
  """Remove rumble below ``frequency``."""

  type: Literal["high_pass_filter"] = "high_pass_filter"
  frequency: FiniteFloat = Field(default=80.0, ge=10.0, le=1_000.0)
  poles: int = Field(default=2, ge=1, le=2)

  def mDescribe(self) -> str:
    return f"high-pass at {self.frequency:g} Hz"


class LowPassFilter(BaseOperation):
  """Remove hiss above ``frequency``."""

  type: Literal["low_pass_filter"] = "low_pass_filter"
  frequency: FiniteFloat = Field(default=16_000.0, ge=1_000.0, le=24_000.0)
  poles: int = Field(default=2, ge=1, le=2)

  def mDescribe(self) -> str:
    return f"low-pass at {self.frequency:g} Hz"


class ApplyGain(BaseOperation):
  """Static gain change in dB."""

  type: Literal["apply_gain"] = "apply_gain"
  gain_db: FiniteFloat = Field(default=0.0, ge=-60.0, le=60.0)

  def mDescribe(self) -> str:
    return f"apply {self.gain_db:+g} dB"


class Denoise(BaseOperation):
  """Broadband noise reduction via FFmpeg's ``afftdn``."""

  type: Literal["denoise"] = "denoise"
  reduction_db: FiniteFloat = Field(default=12.0, gt=0.0, le=97.0)
  noise_floor_db: FiniteFloat | None = Field(default=None, ge=-80.0, le=-20.0)

  def mDescribe(self) -> str:
    return f"denoise by {self.reduction_db:g} dB"


class Compress(BaseOperation):
  """Dynamic range compression via FFmpeg's ``acompressor``."""

  type: Literal["compress"] = "compress"
  threshold_db: FiniteFloat = Field(default=-18.0, ge=-60.0, le=0.0)
  ratio: FiniteFloat = Field(default=3.0, ge=1.0, le=20.0)
  attack_ms: FiniteFloat = Field(default=20.0, ge=0.01, le=2_000.0)
  release_ms: FiniteFloat = Field(default=250.0, ge=0.01, le=9_000.0)
  makeup_db: FiniteFloat = Field(default=0.0, ge=0.0, le=36.0)
  knee_db: FiniteFloat = Field(default=9.0, ge=0.0, le=18.0)

  def mDescribe(self) -> str:
    return f"compress {self.ratio:g}:1 above {self.threshold_db:g} dB"


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #

EditOperation = Annotated[
  DeleteRange
  | ShortenSilence
  | ReplaceRangeWithSilence
  | TrimStart
  | TrimEnd
  | InsertSilence
  | FadeIn
  | FadeOut
  | NormalizeLoudness
  | LimitTruePeak
  | HighPassFilter
  | LowPassFilter
  | ApplyGain
  | Denoise
  | Compress,
  Field(discriminator="type"),
]

cStructuralTypes: frozenset[str] = frozenset(
  {"delete_range", "shorten_silence", "replace_range_with_silence", "trim_start", "trim_end", "insert_silence"}
)

cMasteringTypes: frozenset[str] = frozenset(
  {
    "fade_in", "fade_out", "normalize_loudness", "limit_true_peak",
    "high_pass_filter", "low_pass_filter", "apply_gain", "denoise", "compress",
  }
)

cSupportedTypes: frozenset[str] = cStructuralTypes | cMasteringTypes

# Mastering is applied in this fixed order regardless of the order operations
# appear in the plan: filtering before dynamics, dynamics before loudness,
# loudness before the limiter, fades last. Rendering must be deterministic.
cMasteringOrder: dict[str, int] = {
  "high_pass_filter": 10,
  "low_pass_filter": 20,
  "denoise": 30,
  "compress": 40,
  "apply_gain": 50,
  "normalize_loudness": 60,
  "limit_true_peak": 70,
  "fade_in": 80,
  "fade_out": 90,
}

# The most operations a plan document may hold: the ceiling of
# safety.max_operations. A longer list is refused while it is parsed, before
# any check walks it.
cMaxPlanOperations: int = 200_000

# Types that may legitimately appear more than once in a single plan.
cRepeatableMasteringTypes: frozenset[str] = frozenset({"high_pass_filter", "low_pass_filter"})


class PlanProvenance(BaseModel):
  """Who and what produced this plan."""

  model_config = ConfigDict(extra="forbid")

  generator: str = f"{cProductName}/{cVersion}"
  profile: str = ""
  profile_sha256: str = ""
  transcription_backend: str | None = None
  transcription_model: str | None = None
  vad_backend: str | None = None
  tool_versions: dict[str, str] = Field(default_factory=dict)


class EditPlan(BaseModel):
  """A validated, self-describing edit description."""

  model_config = ConfigDict(extra="forbid")

  version: int = cEditPlanSchemaVersion
  id: str = Field(default_factory=lambda: f"plan-{uuid.uuid4().hex[:12]}", max_length=64)
  created_at: str = Field(default_factory=lambda: dt.datetime.now(dt.UTC).isoformat())
  source: str
  source_path: str = ""
  source_sha256: str = Field(default="", max_length=64)
  source_duration: TimeSeconds
  sample_rate: int = Field(default=0, ge=0)
  channels: int = Field(default=0, ge=0)
  provenance: PlanProvenance = Field(default_factory=PlanProvenance)
  operations: list[EditOperation] = Field(default_factory=list, max_length=cMaxPlanOperations)
  notes: str = Field(default="", max_length=4_000)
  metadata: dict[str, Any] = Field(default_factory=dict)

  @model_validator(mode="after")
  def mAssignOperationIds(self) -> EditPlan:
    """Give every operation a stable id and reject duplicates."""
    sSeen: set[str] = set()
    for vIndex, vOperation in enumerate(self.operations, start=1):
      if not vOperation.id:
        vOperation.id = f"op-{vIndex:03d}"
      if vOperation.id in sSeen:
        raise ValueError(f"duplicate operation id: {vOperation.id}")
      sSeen.add(vOperation.id)
    return self

  # -------------------------------------------------------------- accessors --

  def mStructuralOperations(self) -> list[EditOperation]:
    """Timeline-rewriting operations, in source order."""
    return [vOperation for vOperation in self.operations if vOperation.type in cStructuralTypes]

  def mMasteringOperations(self) -> list[EditOperation]:
    """Filter operations, sorted into the fixed mastering order."""
    lMastering = [vOperation for vOperation in self.operations if vOperation.type in cMasteringTypes]
    return sorted(lMastering, key=lambda pOperation: (cMasteringOrder.get(pOperation.type, 999), pOperation.id))

  def mWithoutMastering(self) -> EditPlan:
    """The same plan with only what is cut, shortened or inserted: no filter, loudness or fade."""
    lOperations = [vOperation for vOperation in self.operations if vOperation.type not in cMasteringTypes]
    return self.model_copy(update={"operations": lOperations})

  def mCountByReason(self) -> dict[str, int]:
    dCounts: dict[str, int] = {}
    for vOperation in self.operations:
      vKey = str(vOperation.reason)
      dCounts[vKey] = dCounts.get(vKey, 0) + 1
    return dCounts

  def mCountByType(self) -> dict[str, int]:
    dCounts: dict[str, int] = {}
    for vOperation in self.operations:
      dCounts[vOperation.type] = dCounts.get(vOperation.type, 0) + 1
    return dCounts

  def mSummary(self) -> dict[str, Any]:
    return {
      "id": self.id,
      "version": self.version,
      "source": self.source,
      "source_sha256": self.source_sha256,
      "source_duration": round(self.source_duration, 3),
      "operation_count": len(self.operations),
      "structural_count": len(self.mStructuralOperations()),
      "mastering_count": len(self.mMasteringOperations()),
      "by_type": self.mCountByType(),
      "by_reason": self.mCountByReason(),
      "profile": self.provenance.profile,
    }


def fNewOperationId(pIndex: int) -> str:
  """Build the canonical operation id for position ``pIndex`` (1-based)."""
  return f"op-{pIndex:03d}"
