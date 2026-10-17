"""EditPlan validation.

**A plan produced by a language model is never executed directly.** Everything
goes through this validator first, whether it came from the built-in planner,
from a file on disk, or straight out of an MCP tool call.

Pydantic already guarantees the *shape*: known operation types, finite numbers,
ranges the right way round. This module checks what a schema cannot: whether
the plan makes sense against this particular file, whether it stays inside the
profile's safety budget, and whether two operations contradict each other.

Safety limits can be relaxed, but only through :class:`SafetyOverride`, whose
fields are explicit, typed and individually bounded. There is no "just run it
anyway" flag that takes a free-form value from a caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from podcast_editor_mcp_server.config import Profile
from podcast_editor_mcp_server.errors import EditPlanValidationError
from podcast_editor_mcp_server.logging_setup import fGetLogger
from podcast_editor_mcp_server.models.audio import AudioProbe
from podcast_editor_mcp_server.models.edit_plan import (
  ApplyGain,
  DeleteRange,
  EditPlan,
  HighPassFilter,
  InsertSilence,
  LimitTruePeak,
  LowPassFilter,
  NormalizeLoudness,
  RangeOperation,
  ReplaceRangeWithSilence,
  ShortenSilence,
  TrimEnd,
  TrimStart,
  cMasteringTypes,
  cRepeatableMasteringTypes,
)
from podcast_editor_mcp_server.util.timeline import Timeline, fBuildTimeline
from podcast_editor_mcp_server.version import cEditPlanSchemaVersion

vLog = fGetLogger("validation")

cTimeTolerance: float = 1e-6
# A plan whose timestamps sit slightly beyond the end of the file is a rounding
# artefact up to this margin, and a real error beyond it.
cDurationGrace: float = 0.05
# Diagnostics returned to a caller are capped at this many per list; the
# counts still give the full numbers.
cMaxReportedIssues: int = 100


@dataclass(slots=True)
class ValidationIssue:
  """One problem found in a plan."""

  code: str
  message: str
  severity: str = "error"
  operation_id: str | None = None
  details: dict[str, Any] = field(default_factory=dict)

  def mAsDict(self) -> dict[str, Any]:
    dResult: dict[str, Any] = {"code": self.code, "message": self.message, "severity": self.severity}
    if self.operation_id:
      dResult["operation_id"] = self.operation_id
    if self.details:
      dResult["details"] = self.details
    return dResult


def fIssueDicts(pIssues: list[ValidationIssue]) -> list[dict[str, Any]]:
  """The first issues of a list as dictionaries, never more than the reporting cap."""
  return [vIssue.mAsDict() for vIssue in pIssues[:cMaxReportedIssues]]


@dataclass(slots=True)
class SafetyOverride:
  """Explicit, bounded relaxations of the profile's safety limits.

  Every field is a typed value with its own ceiling. A caller can raise the
  deletion budget, but cannot switch validation off.
  """

  max_deleted_fraction: float | None = None
  max_single_delete_seconds: float | None = None
  allow_overlapping_operations: bool = False

  def mResolve(self, pProfile: Profile) -> dict[str, Any]:
    """Merge the override onto the profile, clamping every value."""
    vSafety = pProfile.safety
    vFraction = vSafety.max_deleted_fraction
    if self.max_deleted_fraction is not None:
      vFraction = min(0.99, max(0.0, float(self.max_deleted_fraction)))
    vSingle = vSafety.max_single_delete_seconds
    if self.max_single_delete_seconds is not None:
      vSingle = min(36_000.0, max(0.1, float(self.max_single_delete_seconds)))
    return {
      "max_deleted_fraction": vFraction,
      "max_single_delete_seconds": vSingle,
      "max_single_retake_seconds": max(vSingle, vSafety.max_single_retake_seconds),
      "reject_overlapping_operations": vSafety.reject_overlapping_operations and not self.allow_overlapping_operations,
      "min_output_duration_seconds": vSafety.min_output_duration_seconds,
      "max_operations": vSafety.max_operations,
      "max_inserted_seconds": vSafety.max_inserted_seconds,
      "max_gain_db": vSafety.max_gain_db,
    }


@dataclass(slots=True)
class ValidationResult:
  """The outcome of validating one plan."""

  valid: bool
  errors: list[ValidationIssue] = field(default_factory=list)
  warnings: list[ValidationIssue] = field(default_factory=list)
  timeline: Timeline | None = None
  estimated_output_seconds: float = 0.0
  removed_seconds: float = 0.0
  inserted_seconds: float = 0.0
  removed_fraction: float = 0.0
  effective_limits: dict[str, Any] = field(default_factory=dict)

  def mAsDict(self) -> dict[str, Any]:
    return {
      "valid": self.valid,
      "errors": fIssueDicts(self.errors),
      "warnings": fIssueDicts(self.warnings),
      "error_count": len(self.errors),
      "warning_count": len(self.warnings),
      "estimated_output_seconds": round(self.estimated_output_seconds, 3),
      "removed_seconds": round(self.removed_seconds, 3),
      "inserted_seconds": round(self.inserted_seconds, 3),
      "removed_fraction": round(self.removed_fraction, 4),
      "effective_limits": self.effective_limits,
    }

  def mRaiseIfInvalid(self) -> None:
    if self.valid:
      return
    raise EditPlanValidationError(
      "The edit plan was rejected by the validator",
      {
        "errors": fIssueDicts(self.errors),
        "error_count": len(self.errors),
        "warning_count": len(self.warnings),
      },
    )


def fValidateEditPlan(
  pPlan: EditPlan,
  pProfile: Profile,
  pProbe: AudioProbe | None = None,
  pOverride: SafetyOverride | None = None,
) -> ValidationResult:
  """Validate ``pPlan`` against ``pProfile`` and, when given, a probed file."""
  lErrors: list[ValidationIssue] = []
  lWarnings: list[ValidationIssue] = []
  vOverride = pOverride or SafetyOverride()
  dLimits = vOverride.mResolve(pProfile)

  _fCheckDocument(pPlan, lErrors, lWarnings)
  _fCheckAgainstProbe(pPlan, pProbe, lErrors, lWarnings)
  if not _fCheckOperationCount(pPlan, dLimits, lErrors):
    # Every other check walks the operations, so an oversized plan is turned
    # away before any of them runs.
    vLog.info("Edit plan rejected", extra={"plan_id": pPlan.id, "errors": len(lErrors), "warnings": len(lWarnings)})
    return ValidationResult(valid=False, errors=lErrors, warnings=lWarnings, effective_limits=dLimits)
  _fCheckRanges(pPlan, lErrors, lWarnings)
  _fCheckOverlaps(pPlan, dLimits, lErrors, lWarnings)
  _fCheckMasteringConsistency(pPlan, dLimits, lErrors, lWarnings)

  vTimeline: Timeline | None = None
  vOutputSeconds = 0.0
  vRemoved = 0.0
  vInserted = 0.0
  vFraction = 0.0

  if not lErrors:
    try:
      vTimeline = fBuildTimeline(pPlan, pProfile.cuts.crossfade_ms / 1000.0)
      vOutputSeconds = vTimeline.output_duration
      vRemoved = vTimeline.removed_seconds
      vInserted = vTimeline.inserted_seconds
      vFraction = vTimeline.removed_fraction
      _fCheckBudget(vTimeline, dLimits, lErrors, lWarnings)
      _fCheckFades(pPlan, vOutputSeconds, lErrors, lWarnings)
    except Exception as vError:  # noqa: BLE001 - surface any compilation failure as an error
      lErrors.append(
        ValidationIssue(
          code="timeline_compilation_failed",
          message=f"The plan could not be compiled into a timeline: {vError}",
        )
      )

  vValid = not lErrors
  if not vValid:
    vLog.info(
      "Edit plan rejected",
      extra={"plan_id": pPlan.id, "errors": len(lErrors), "warnings": len(lWarnings)},
    )
  return ValidationResult(
    valid=vValid,
    errors=lErrors,
    warnings=lWarnings,
    timeline=vTimeline,
    estimated_output_seconds=vOutputSeconds,
    removed_seconds=vRemoved,
    inserted_seconds=vInserted,
    removed_fraction=vFraction,
    effective_limits=dLimits,
  )


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #


def _fCheckDocument(pPlan: EditPlan, pErrors: list[ValidationIssue], pWarnings: list[ValidationIssue]) -> None:
  if pPlan.version != cEditPlanSchemaVersion:
    pErrors.append(
      ValidationIssue(
        code="unsupported_schema_version",
        message=f"Plan schema version {pPlan.version} is not the one this build supports ({cEditPlanSchemaVersion})",
      )
    )
  if pPlan.source_duration <= 0.0:
    pErrors.append(
      ValidationIssue(code="invalid_source_duration", message="source_duration must be greater than zero")
    )
  if not pPlan.source_sha256:
    pWarnings.append(
      ValidationIssue(
        code="missing_source_hash",
        message="The plan carries no source hash, so it cannot be bound to a specific file",
        severity="warning",
      )
    )


def _fCheckAgainstProbe(
  pPlan: EditPlan,
  pProbe: AudioProbe | None,
  pErrors: list[ValidationIssue],
  pWarnings: list[ValidationIssue],
) -> None:
  if pProbe is None:
    return
  if pPlan.source_sha256 and pProbe.sha256 and pPlan.source_sha256 != pProbe.sha256:
    pErrors.append(
      ValidationIssue(
        code="source_hash_mismatch",
        message="This plan was generated for a different file",
        details={"plan_sha256": pPlan.source_sha256[:16], "file_sha256": pProbe.sha256[:16]},
      )
    )
  vDelta = abs(pPlan.source_duration - pProbe.duration_seconds)
  if vDelta > cDurationGrace:
    pErrors.append(
      ValidationIssue(
        code="source_duration_mismatch",
        message="The plan's source duration does not match the file",
        details={
          "plan_duration": round(pPlan.source_duration, 3),
          "file_duration": round(pProbe.duration_seconds, 3),
        },
      )
    )
  elif vDelta > cTimeTolerance:
    pWarnings.append(
      ValidationIssue(
        code="source_duration_drift",
        message="The plan's source duration differs slightly from the file",
        severity="warning",
        details={"delta_seconds": round(vDelta, 6)},
      )
    )


def _fCheckOperationCount(pPlan: EditPlan, pLimits: dict[str, Any], pErrors: list[ValidationIssue]) -> bool:
  """Whether the plan is within the operation limit; records the error when not."""
  vMaximum = int(pLimits["max_operations"])
  if len(pPlan.operations) > vMaximum:
    pErrors.append(
      ValidationIssue(
        code="too_many_operations",
        message=f"The plan has {len(pPlan.operations)} operations, above the limit of {vMaximum}",
      )
    )
    return False
  return True


def _fCheckRanges(pPlan: EditPlan, pErrors: list[ValidationIssue], pWarnings: list[ValidationIssue]) -> None:
  vDuration = pPlan.source_duration
  for vOperation in pPlan.operations:
    if isinstance(vOperation, DeleteRange | ShortenSilence | ReplaceRangeWithSilence):
      if vOperation.start < -cTimeTolerance:
        pErrors.append(
          ValidationIssue(
            code="negative_timestamp",
            message="start must not be negative",
            operation_id=vOperation.id,
          )
        )
      if vOperation.end > vDuration + cDurationGrace:
        pErrors.append(
          ValidationIssue(
            code="range_beyond_source",
            message="end lies beyond the end of the source",
            operation_id=vOperation.id,
            details={"end": round(vOperation.end, 3), "source_duration": round(vDuration, 3)},
          )
        )
      elif vOperation.end > vDuration + cTimeTolerance:
        pWarnings.append(
          ValidationIssue(
            code="range_rounding",
            message="end sits marginally past the source duration and will be clamped",
            severity="warning",
            operation_id=vOperation.id,
          )
        )
    elif isinstance(vOperation, TrimStart | TrimEnd):
      if vOperation.duration >= vDuration:
        pErrors.append(
          ValidationIssue(
            code="trim_removes_everything",
            message=f"{vOperation.type} would remove the whole recording",
            operation_id=vOperation.id,
          )
        )
    elif isinstance(vOperation, InsertSilence):
      if vOperation.at > vDuration + cDurationGrace:
        pErrors.append(
          ValidationIssue(
            code="insert_beyond_source",
            message="insert_silence position lies beyond the end of the source",
            operation_id=vOperation.id,
          )
        )


def _fCheckOverlaps(
  pPlan: EditPlan,
  pLimits: dict[str, Any],
  pErrors: list[ValidationIssue],
  pWarnings: list[ValidationIssue],
) -> None:
  lRanges: list[tuple[float, float, str, str]] = []
  for vOperation in pPlan.operations:
    if isinstance(vOperation, RangeOperation):
      lRanges.append((vOperation.start, vOperation.end, vOperation.id, vOperation.type))
  lRanges.sort(key=lambda pItem: (pItem[0], pItem[1]))

  vReject = bool(pLimits["reject_overlapping_operations"])
  for vIndex in range(1, len(lRanges)):
    vPreviousStart, vPreviousEnd, vPreviousId, vPreviousType = lRanges[vIndex - 1]
    vStart, vEnd, vId, vType = lRanges[vIndex]
    if vStart < vPreviousEnd - cTimeTolerance:
      dDetails = {
        "first": {"id": vPreviousId, "type": vPreviousType, "start": round(vPreviousStart, 3), "end": round(vPreviousEnd, 3)},
        "second": {"id": vId, "type": vType, "start": round(vStart, 3), "end": round(vEnd, 3)},
        "overlap_seconds": round(min(vPreviousEnd, vEnd) - vStart, 3),
      }
      if vReject:
        pErrors.append(
          ValidationIssue(
            code="overlapping_operations",
            message=f"Operations {vPreviousId} and {vId} overlap",
            operation_id=vId,
            details=dDetails,
          )
        )
      else:
        pWarnings.append(
          ValidationIssue(
            code="overlapping_operations",
            message=f"Operations {vPreviousId} and {vId} overlap and will be merged",
            severity="warning",
            operation_id=vId,
            details=dDetails,
          )
        )


def _fCheckMasteringConsistency(
  pPlan: EditPlan,
  pLimits: dict[str, Any],
  pErrors: list[ValidationIssue],
  pWarnings: list[ValidationIssue],
) -> None:
  dCounts: dict[str, int] = {}
  for vOperation in pPlan.operations:
    if vOperation.type in cMasteringTypes:
      dCounts[vOperation.type] = dCounts.get(vOperation.type, 0) + 1
  for vType, vCount in dCounts.items():
    if vCount > 1 and vType not in cRepeatableMasteringTypes:
      pErrors.append(
        ValidationIssue(
          code="duplicate_mastering_operation",
          message=f"'{vType}' appears {vCount} times; it may only appear once",
          details={"type": vType, "count": vCount},
        )
      )

  vMaxGain = float(pLimits["max_gain_db"])
  vTotalGain = 0.0
  for vOperation in pPlan.operations:
    if isinstance(vOperation, ApplyGain):
      vTotalGain += abs(vOperation.gain_db)
  if vTotalGain > vMaxGain:
    pErrors.append(
      ValidationIssue(
        code="excessive_gain",
        message=f"Total gain change of {vTotalGain:.1f} dB exceeds the limit of {vMaxGain:.1f} dB",
      )
    )

  lHighPass = [vItem for vItem in pPlan.operations if isinstance(vItem, HighPassFilter)]
  lLowPass = [vItem for vItem in pPlan.operations if isinstance(vItem, LowPassFilter)]
  if lHighPass and lLowPass:
    # Only the highest high-pass and the lowest low-pass decide whether any
    # band survives, so one comparison answers for every pair.
    vHigh = max(lHighPass, key=lambda pItem: pItem.frequency)
    vLow = min(lLowPass, key=lambda pItem: pItem.frequency)
    if vHigh.frequency >= vLow.frequency:
      pErrors.append(
        ValidationIssue(
          code="contradictory_filters",
          message="The high-pass cutoff is at or above the low-pass cutoff, which removes all audio",
          operation_id=vHigh.id,
          details={"high_pass_hz": vHigh.frequency, "low_pass_hz": vLow.frequency, "low_pass_id": vLow.id},
        )
      )

  lNormalize = [vItem for vItem in pPlan.operations if isinstance(vItem, NormalizeLoudness)]
  lLimit = [vItem for vItem in pPlan.operations if isinstance(vItem, LimitTruePeak)]
  if lNormalize and lLimit and lLimit[0].true_peak_dbtp > lNormalize[0].true_peak_dbtp + 0.01:
    pWarnings.append(
      ValidationIssue(
        code="limiter_above_normalisation_target",
        message="The limiter ceiling is above the normalisation true-peak target, so it will never engage",
        severity="warning",
        operation_id=lLimit[0].id,
      )
    )
  if lNormalize and any(isinstance(vItem, ApplyGain) for vItem in pPlan.operations):
    pWarnings.append(
      ValidationIssue(
        code="gain_before_normalisation",
        message="apply_gain runs before normalize_loudness, so its effect on the final level is cancelled out",
        severity="warning",
      )
    )


def _fCheckBudget(
  pTimeline: Timeline,
  pLimits: dict[str, Any],
  pErrors: list[ValidationIssue],
  pWarnings: list[ValidationIssue],
) -> None:
  vMaxFraction = float(pLimits["max_deleted_fraction"])
  if pTimeline.removed_fraction > vMaxFraction + 1e-9:
    pErrors.append(
      ValidationIssue(
        code="deleted_fraction_exceeded",
        message=(
          f"The plan removes {pTimeline.removed_fraction * 100:.1f}% of the recording, "
          f"above the limit of {vMaxFraction * 100:.1f}%"
        ),
        details={
          "removed_seconds": round(pTimeline.removed_seconds, 2),
          "source_seconds": round(pTimeline.source_duration, 2),
        },
      )
    )

  vMaxSingle = float(pLimits["max_single_delete_seconds"])
  vMaxRetake = float(pLimits["max_single_retake_seconds"])
  for vRemoval in pTimeline.removals:
    # A removal that includes a retake ("mistake") may be a whole paragraph
    # recorded again, so it is held to the retake limit instead.
    vLimit = vMaxRetake if "mistake" in vRemoval.reasons else vMaxSingle
    if vRemoval.duration > vLimit:
      pErrors.append(
        ValidationIssue(
          code="single_delete_too_long",
          message=f"A single removal of {vRemoval.duration:.1f}s exceeds the limit of {vLimit:.1f}s",
          operation_id=vRemoval.operation_ids[0] if vRemoval.operation_ids else None,
          details={"start": round(vRemoval.start, 3), "end": round(vRemoval.end, 3)},
        )
      )

  vMinimumOutput = float(pLimits["min_output_duration_seconds"])
  if pTimeline.output_duration < vMinimumOutput:
    pErrors.append(
      ValidationIssue(
        code="output_too_short",
        message=(
          f"The resulting audio would be {pTimeline.output_duration:.2f}s, "
          f"below the minimum of {vMinimumOutput:.2f}s"
        ),
      )
    )

  vMaxInserted = float(pLimits["max_inserted_seconds"])
  if pTimeline.inserted_seconds > vMaxInserted:
    pErrors.append(
      ValidationIssue(
        code="inserted_silence_exceeded",
        message=(
          f"The plan inserts {pTimeline.inserted_seconds:.1f}s of silence, "
          f"above the limit of {vMaxInserted:.1f}s"
        ),
      )
    )

  if pTimeline.removed_fraction > 0.25:
    pWarnings.append(
      ValidationIssue(
        code="large_removal",
        message=f"The plan removes {pTimeline.removed_fraction * 100:.1f}% of the recording",
        severity="warning",
      )
    )
  if len(pTimeline.keep_segments) > 500:
    pWarnings.append(
      ValidationIssue(
        code="many_segments",
        message=f"The timeline has {len(pTimeline.keep_segments)} segments; rendering will take several passes",
        severity="warning",
      )
    )


def _fCheckFades(
  pPlan: EditPlan,
  pOutputSeconds: float,
  pErrors: list[ValidationIssue],
  pWarnings: list[ValidationIssue],
) -> None:
  from podcast_editor_mcp_server.models.edit_plan import FadeIn, FadeOut

  vTotalFade = 0.0
  for vOperation in pPlan.operations:
    if not isinstance(vOperation, FadeIn | FadeOut):
      continue
    vTotalFade += vOperation.duration
    if vOperation.duration > pOutputSeconds:
      pErrors.append(
        ValidationIssue(
          code="fade_longer_than_output",
          message=f"The {vOperation.type} lasts longer than the edited audio ({pOutputSeconds:.2f}s)",
          operation_id=vOperation.id,
        )
      )
  if 0.0 < pOutputSeconds < vTotalFade:
    pWarnings.append(
      ValidationIssue(
        code="fades_overlap",
        message="The fade-in and fade-out together cover the whole file",
        severity="warning",
      )
    )

