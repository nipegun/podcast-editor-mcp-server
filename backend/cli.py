"""Command line interface.

Exists so the engine can be exercised, debugged and scripted without a language
model anywhere in the loop. It calls exactly the same
:class:`~podcast_editor_mcp_server.pipeline.PodcastPipeline` the MCP tools call; there is no
second implementation of anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from podcast_editor_mcp_server.config import AppContext, fListProfiles, fLoadProfile
from podcast_editor_mcp_server.errors import PodcastEditorMcpServerError
from podcast_editor_mcp_server.jobs.models import JobState
from podcast_editor_mcp_server.logging_setup import fGetLogger, fSetupLogging
from podcast_editor_mcp_server.pipeline import PodcastPipeline, ProcessOptions
from podcast_editor_mcp_server.version import cProductName, cVersion

vLog = fGetLogger("cli")

cExitOk: int = 0
cExitError: int = 1
cExitUsage: int = 2
cExitInvalidPlan: int = 3


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #


def fPrint(pText: str = "") -> None:
  sys.stdout.write(pText + "\n")


def fPrintJson(pPayload: Any) -> None:
  sys.stdout.write(json.dumps(pPayload, indent=2, ensure_ascii=False, default=str) + "\n")


def fPrintError(pText: str) -> None:
  sys.stderr.write(f"{cProductName}: {pText}\n")


def fFormatDuration(pSeconds: float) -> str:
  """Render seconds as ``H:MM:SS.mmm`` or ``M:SS.mmm``."""
  vTotal = max(0.0, pSeconds)
  vHours = int(vTotal // 3600)
  vMinutes = int((vTotal % 3600) // 60)
  vRest = vTotal % 60
  if vHours:
    return f"{vHours}:{vMinutes:02d}:{vRest:06.3f}"
  return f"{vMinutes}:{vRest:06.3f}"


def fFormatTable(pRows: list[tuple[str, str]], pIndent: str = "  ") -> str:
  if not pRows:
    return ""
  vWidth = max(len(vLabel) for vLabel, _ in pRows)
  return "\n".join(f"{pIndent}{vLabel:<{vWidth}}  {vValue}" for vLabel, vValue in pRows)


def _fOptional(pValue: Any, pSuffix: str = "", pDigits: int = 2) -> str:
  if pValue is None:
    return "n/a"
  if isinstance(pValue, float):
    return f"{pValue:.{pDigits}f}{pSuffix}"
  return f"{pValue}{pSuffix}"


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def fCommandProbe(pPipeline: PodcastPipeline, pArguments: argparse.Namespace) -> int:
  vSource = pPipeline.mResolveSource(pArguments.file)
  vProbe = pPipeline.mProbe(vSource)
  if pArguments.json:
    fPrintJson(vProbe.model_dump(mode="json"))
    return cExitOk
  fPrint(f"{vProbe.label}")
  fPrint(
    fFormatTable(
      [
        ("Duration", f"{fFormatDuration(vProbe.duration_seconds)} ({vProbe.duration_seconds:.3f} s)"),
        ("Sample rate", f"{vProbe.sample_rate} Hz"),
        ("Channels", str(vProbe.channels)),
        ("Codec", f"{vProbe.codec_name} ({vProbe.sample_format or 'n/a'})"),
        ("Container", vProbe.format_name),
        ("Bit rate", _fOptional(vProbe.bit_rate, " bit/s", 0)),
        ("Size", f"{vProbe.size_bytes / (1024 * 1024):.2f} MiB"),
        ("SHA-256", vProbe.sha256),
      ]
    )
  )
  return cExitOk


def fCommandAnalyze(pPipeline: PodcastPipeline, pArguments: argparse.Namespace) -> int:
  vSource = pPipeline.mResolveSource(pArguments.file)
  vProfile = pPipeline.mResolveProfile(pArguments.profile)
  vWorkDir = pPipeline.context.workspace.work / "cli-analysis"
  vWorkDir.mkdir(parents=True, exist_ok=True)
  vReport = pPipeline.mAnalyze(vSource, vProfile, vWorkDir)

  if pArguments.json:
    fPrintJson(vReport.model_dump(mode="json"))
    return cExitOk

  dSummary = vReport.mSummary()
  fPrint(f"{vReport.probe.label} (profile: {vProfile.name})")
  fPrint(
    fFormatTable(
      [
        ("Duration", fFormatDuration(vReport.probe.duration_seconds)),
        ("Integrated loudness", _fOptional(dSummary["integrated_lufs"], " LUFS")),
        ("Loudness range", _fOptional(dSummary["loudness_range"], " LU")),
        ("True peak", _fOptional(dSummary["true_peak_dbfs"], " dBFS")),
        ("Sample peak", _fOptional(dSummary["peak_dbfs"], " dBFS")),
        ("RMS", _fOptional(dSummary["rms_dbfs"], " dBFS")),
        ("Noise floor", _fOptional(dSummary["noise_floor_dbfs"], " dBFS")),
        ("DC offset", _fOptional(dSummary["dc_offset"], "", 5)),
        ("Silence threshold", f"{vReport.silence_threshold_db:.2f} dBFS"),
        ("Silences", f"{dSummary['silence_count']} covering {dSummary['silence_seconds']:.2f} s"),
        ("Longest silence", f"{dSummary['longest_silence_seconds']:.2f} s"),
        ("Speech ratio", f"{dSummary['speech_ratio'] * 100:.1f}% ({vReport.vad_backend} VAD)"),
        ("Clipping", f"{dSummary['clipping_regions']} region(s)"),
      ]
    )
  )
  if pArguments.show_silences and vReport.silences:
    fPrint("\nSilences:")
    for vRegion in vReport.silences[:50]:
      fPrint(f"  {vRegion.start:9.3f} - {vRegion.end:9.3f}  ({vRegion.duration:6.3f} s)")
    if len(vReport.silences) > 50:
      fPrint(f"  ... and {len(vReport.silences) - 50} more")
  for vWarning in vReport.warnings:
    fPrintError(f"warning: {vWarning}")
  return cExitOk


def fCommandTranscribe(pPipeline: PodcastPipeline, pArguments: argparse.Namespace) -> int:
  vSource = pPipeline.mResolveSource(pArguments.file)
  vProfile = pPipeline.mResolveProfile(pArguments.profile)
  if pArguments.model or pArguments.language:
    dUpdates: dict[str, Any] = {}
    if pArguments.model:
      dUpdates["model"] = pArguments.model
    if pArguments.language:
      dUpdates["language"] = pArguments.language
    vProfile = vProfile.model_copy(update={"transcription": vProfile.transcription.model_copy(update=dUpdates)})

  vProbe = pPipeline.mProbe(vSource)
  fPrintError(f"Transcribing {vProbe.label} with model '{vProfile.transcription.model}'...")
  vTranscript = pPipeline.mTranscribe(vSource, vProfile, vProbe.sha256, pUseCache=not pArguments.force)

  if pArguments.output:
    from podcast_editor_mcp_server.transcription.formats import fRender
    from podcast_editor_mcp_server.util.atomic import fWriteJsonAtomic, fWriteTextAtomic

    vTarget = pPipeline.guard.mResolveOutput(
      pArguments.output, "output", pAllowedExtensions=frozenset({".json", ".txt", ".srt", ".vtt"}), pOverwrite=True
    )
    if vTarget.suffix == ".json":
      fWriteJsonAtomic(vTarget, vTranscript.model_dump(mode="json"))
    else:
      fWriteTextAtomic(vTarget, fRender(vTranscript, vTarget.suffix.lstrip(".")))
    fPrintError(f"Wrote {pPipeline.guard.mLabel(vTarget)}")

  if pArguments.json:
    fPrintJson(vTranscript.model_dump(mode="json"))
    return cExitOk
  if pArguments.format != "summary":
    from podcast_editor_mcp_server.transcription.formats import fRender

    fPrint(fRender(vTranscript, pArguments.format))
    return cExitOk

  fPrint(
    fFormatTable(
      [
        ("Language", f"{vTranscript.language} ({_fOptional(vTranscript.language_probability, '', 3)})"),
        ("Backend", f"{vTranscript.backend} / {vTranscript.model} on {vTranscript.device}"),
        ("Segments", str(len(vTranscript.segments))),
        ("Words", str(vTranscript.word_count)),
        ("Word timestamps", "yes" if vTranscript.has_word_timestamps else "no"),
        ("Duration", fFormatDuration(vTranscript.duration)),
      ]
    )
  )
  return cExitOk


def fCommandCandidates(pPipeline: PodcastPipeline, pArguments: argparse.Namespace) -> int:
  vSource = pPipeline.mResolveSource(pArguments.file)
  vProfile = pPipeline.mResolveProfile(pArguments.profile)
  vWorkDir = pPipeline.context.workspace.work / "cli-analysis"
  vWorkDir.mkdir(parents=True, exist_ok=True)
  vProbe = pPipeline.mProbe(vSource)
  vAnalysis = pPipeline.mAnalyze(vSource, vProfile, vWorkDir, vProbe)
  vTranscript = pPipeline.mLoadCachedTranscript(vProbe.sha256, vProfile)
  vCandidates = pPipeline.mFindCandidates(vAnalysis, vProfile, vTranscript)

  if pArguments.json:
    fPrintJson(vCandidates.model_dump(mode="json"))
    return cExitOk

  fPrint(f"{vCandidates.source}: {len(vCandidates.candidates)} candidate(s)")
  for vType, vCount in sorted(vCandidates.mCountByType().items()):
    fPrint(f"  {vType:<24} {vCount}")
  if vCandidates.candidates:
    fPrint("")
    for vCandidate in vCandidates.candidates[: pArguments.limit]:
      vText = vCandidate.text[:60] + ("..." if len(vCandidate.text) > 60 else "")
      fPrint(
        f"  {vCandidate.id:<22} {vCandidate.start:9.3f}-{vCandidate.end:9.3f} "
        f"({vCandidate.duration:6.3f}s) conf={vCandidate.confidence:.2f} {vText}"
      )
    if len(vCandidates.candidates) > pArguments.limit:
      fPrint(f"  ... and {len(vCandidates.candidates) - pArguments.limit} more")
  for vWarning in vCandidates.warnings:
    fPrintError(f"warning: {vWarning}")
  return cExitOk


def fCommandPlan(pPipeline: PodcastPipeline, pArguments: argparse.Namespace) -> int:
  vSource = pPipeline.mResolveSource(pArguments.file)
  vProfile = pPipeline.mResolveProfile(pArguments.profile)
  vWorkDir = pPipeline.context.workspace.work / "cli-analysis"
  vWorkDir.mkdir(parents=True, exist_ok=True)

  vProbe = pPipeline.mProbe(vSource)
  vAnalysis = pPipeline.mAnalyze(vSource, vProfile, vWorkDir, vProbe)
  vTranscript = pPipeline.mLoadCachedTranscript(vProbe.sha256, vProfile)
  vCandidates = pPipeline.mFindCandidates(vAnalysis, vProfile, vTranscript)
  vEnvelope = pPipeline.mEnvelope(vSource, vWorkDir)
  vPlan = pPipeline.mCreatePlan(vAnalysis, vProfile, vCandidates, vTranscript, vEnvelope)
  vSimulation = pPipeline.mSimulate(vPlan, vProfile, vProbe)

  vTarget: Path | None = None
  if pArguments.output:
    vTarget = pPipeline.guard.mResolveOutput(
      pArguments.output, "output", pAllowedExtensions=frozenset({".json"}), pOverwrite=True
    )
    pPipeline.mSavePlan(vPlan, vTarget)
  elif pArguments.save:
    vTarget = pPipeline.mSavePlan(vPlan)

  if pArguments.json:
    dPayload = vPlan.model_dump(mode="json")
    dPayload["_simulation"] = vSimulation.model_dump(mode="json")
    fPrintJson(dPayload)
    return cExitOk if vSimulation.valid else cExitInvalidPlan

  fPrint(f"Plan {vPlan.id} for {vPlan.source} (profile: {vProfile.name})")
  fPrint(
    fFormatTable(
      [
        ("Operations", str(len(vPlan.operations))),
        ("Original duration", fFormatDuration(vPlan.source_duration)),
        ("Estimated result", fFormatDuration(vSimulation.estimated_output_seconds)),
        ("Removed", f"{vSimulation.removed_seconds:.2f} s ({vSimulation.removed_fraction * 100:.1f}%)"),
        ("Inserted", f"{vSimulation.inserted_seconds:.2f} s"),
        ("Valid", "yes" if vSimulation.valid else "NO"),
      ]
    )
  )
  if vTarget is not None:
    fPrint(f"  Saved to {pPipeline.guard.mLabel(vTarget)}")
  fPrint("\nOperations:")
  for vOperation in vPlan.operations[: pArguments.limit]:
    fPrint(f"  {vOperation.id:<8} {str(vOperation.reason):<16} {vOperation.mDescribe()}")
  if len(vPlan.operations) > pArguments.limit:
    fPrint(f"  ... and {len(vPlan.operations) - pArguments.limit} more")
  _fPrintIssues(vSimulation.errors, vSimulation.warnings)
  return cExitOk if vSimulation.valid else cExitInvalidPlan


def fCommandValidate(pPipeline: PodcastPipeline, pArguments: argparse.Namespace) -> int:
  vPlan = pPipeline.mLoadPlan(pArguments.plan)
  vProfile = pPipeline.mResolveProfile(pArguments.profile)
  vProbe = None
  if pArguments.file:
    vProbe = pPipeline.mProbe(pPipeline.mResolveSource(pArguments.file))
  vValidation = pPipeline.mValidate(vPlan, vProfile, vProbe)

  if pArguments.json:
    fPrintJson(vValidation.mAsDict())
    return cExitOk if vValidation.valid else cExitInvalidPlan

  fPrint(f"Plan {vPlan.id}: {'VALID' if vValidation.valid else 'INVALID'}")
  fPrint(
    fFormatTable(
      [
        ("Estimated result", fFormatDuration(vValidation.estimated_output_seconds)),
        ("Removed", f"{vValidation.removed_seconds:.2f} s ({vValidation.removed_fraction * 100:.1f}%)"),
        ("Errors", str(len(vValidation.errors))),
        ("Warnings", str(len(vValidation.warnings))),
      ]
    )
  )
  _fPrintIssues(
    [vIssue.mAsDict() for vIssue in vValidation.errors],
    [vIssue.mAsDict() for vIssue in vValidation.warnings],
  )
  return cExitOk if vValidation.valid else cExitInvalidPlan


def fCommandSimulate(pPipeline: PodcastPipeline, pArguments: argparse.Namespace) -> int:
  vPlan = pPipeline.mLoadPlan(pArguments.plan)
  vProfile = pPipeline.mResolveProfile(pArguments.profile)
  vProbe = None
  if pArguments.file:
    vProbe = pPipeline.mProbe(pPipeline.mResolveSource(pArguments.file))
  vSimulation = pPipeline.mSimulate(vPlan, vProfile, vProbe)

  if pArguments.json:
    fPrintJson(vSimulation.model_dump(mode="json"))
    return cExitOk if vSimulation.valid else cExitInvalidPlan

  fPrint(f"Simulation of plan {vSimulation.plan_id}")
  fPrint(
    fFormatTable(
      [
        ("Original duration", fFormatDuration(vSimulation.source_duration_seconds)),
        ("Estimated result", fFormatDuration(vSimulation.estimated_output_seconds)),
        ("Removed", f"{vSimulation.removed_seconds:.2f} s ({vSimulation.removed_fraction * 100:.1f}%)"),
        ("Inserted", f"{vSimulation.inserted_seconds:.2f} s"),
        ("Operations", f"{vSimulation.operation_count} ({vSimulation.structural_count} structural)"),
        ("Segments kept", str(vSimulation.segment_count)),
        ("Valid", "yes" if vSimulation.valid else "NO"),
      ]
    )
  )
  if vSimulation.removed_regions:
    fPrint("\nRemoved regions:")
    for dRegion in vSimulation.removed_regions[: pArguments.limit]:
      fPrint(
        f"  {dRegion['start']:9.3f} - {dRegion['end']:9.3f}  ({dRegion['duration']:6.3f} s)  "
        f"{', '.join(dRegion.get('reasons', []))}"
      )
    if len(vSimulation.removed_regions) > pArguments.limit:
      fPrint(f"  ... and {len(vSimulation.removed_regions) - pArguments.limit} more")
  _fPrintIssues(vSimulation.errors, vSimulation.warnings)
  return cExitOk if vSimulation.valid else cExitInvalidPlan


def fCommandExecute(pPipeline: PodcastPipeline, pArguments: argparse.Namespace) -> int:
  vPlan = pPipeline.mLoadPlan(pArguments.plan)
  vOptions = ProcessOptions(
    profile=pArguments.profile,
    transcribe=False,
    overwrite=pArguments.overwrite,
    audacity_project=_fAudacityChoice(pArguments),
    formats=pArguments.format or None,
    plan=vPlan,
    output_name=pArguments.name,
    keep_work_dir=pArguments.keep_work,
  )
  vResult = pPipeline.mProcess(pArguments.file, vOptions, _fProgressPrinter(pArguments))
  return _fReportProcessResult(pPipeline, vResult, pArguments)


def fParseDecisions(pItems: list[str]) -> dict[str, str]:
  """Turn ``ID=cut`` / ``ID=keep`` arguments into a decisions mapping."""
  dDecisions: dict[str, str] = {}
  for vItem in pItems:
    vKey, vSeparator, vValue = vItem.partition("=")
    vValue = vValue.strip().lower()
    if not vSeparator or not vKey.strip() or vValue not in {"cut", "keep"}:
      raise ValueError(f"invalid decision '{vItem}': use ID=cut or ID=keep")
    dDecisions[vKey.strip()] = vValue
  return dDecisions


def fParseCuts(pItems: list[str]) -> list[dict[str, object]]:
  """Turn ``START-END`` (seconds of the source) arguments into requested cuts."""
  ldCuts: list[dict[str, object]] = []
  for vItem in pItems:
    vRange, _, vWhy = vItem.partition(":")
    vStart, vSeparator, vEnd = vRange.partition("-")
    try:
      vFrom = float(vStart)
      vTo = float(vEnd)
    except ValueError as vError:
      raise ValueError(f"invalid cut '{vItem}': use START-END in seconds, e.g. 1159.27-1163.33") from vError
    if not vSeparator or vTo <= vFrom or vFrom < 0.0:
      raise ValueError(f"invalid cut '{vItem}': use START-END in seconds, e.g. 1159.27-1163.33")
    ldCuts.append({"start": vFrom, "end": vTo, "why": vWhy.strip() or "requested on the command line"})
  return ldCuts


def fCommandProcess(pPipeline: PodcastPipeline, pArguments: argparse.Namespace) -> int:
  try:
    dDecisions = fParseDecisions(pArguments.decide)
    ldCuts = fParseCuts(pArguments.cut)
  except ValueError as vError:
    fPrintError(str(vError))
    return cExitUsage
  vOptions = ProcessOptions(
    profile=pArguments.profile,
    dry_run=pArguments.dry_run,
    overwrite=pArguments.overwrite,
    transcribe=not pArguments.no_transcribe,
    audacity_project=_fAudacityChoice(pArguments),
    formats=pArguments.format or None,
    output_name=pArguments.name,
    keep_work_dir=pArguments.keep_work,
    decisions=dDecisions,
    apply_proposals=True,
    extra_cuts=ldCuts,
    mastering=not pArguments.no_mastering,
  )
  vResult = pPipeline.mProcess(pArguments.file, vOptions, _fProgressPrinter(pArguments))
  return _fReportProcessResult(pPipeline, vResult, pArguments)


def fCommandPrepare(pPipeline: PodcastPipeline, pArguments: argparse.Namespace) -> int:
  vPrepared = pPipeline.mPrepareEdit(pArguments.file, pArguments.profile)
  vPlan = vPrepared.plan
  ldReview = vPlan.metadata.get("review_items", [])
  if pArguments.json:
    fPrintJson(
      {
        "plan_path": pPipeline.guard.mLabel(vPrepared.plan_path) if vPrepared.plan_path else None,
        "simulation": vPrepared.simulation.model_dump(mode="json"),
        "review_items": ldReview,
        "joins": vPlan.metadata.get("joins"),
        "warnings": vPrepared.warnings,
      }
    )
    return cExitOk if vPrepared.validation.valid else cExitInvalidPlan

  vSimulation = vPrepared.simulation
  fPrint(
    f"{vPrepared.probe.label}: {fFormatDuration(vSimulation.source_duration_seconds)} -> "
    f"{fFormatDuration(vSimulation.estimated_output_seconds)} ({vSimulation.removed_fraction * 100:.1f}% removed)"
  )
  for vOperation in vPlan.mStructuralOperations():
    if vOperation.type != "delete_range":
      continue
    fPrint(
      f"  {vOperation.id:<12} {vOperation.start:9.3f}-{vOperation.end:9.3f} "  # type: ignore[attr-defined]
      f"{str(vOperation.reason):<12} {vOperation.note}"
    )
  if ldReview:
    fPrint("")
    fPrint(f"{len(ldReview)} item(s) need a decision (pass --decide ID=cut|keep to 'process'):")
    for dItem in ldReview:
      fPrint(f"  {dItem['id']:<22} {dItem['start']:9.3f}-{dItem['end']:9.3f} proposed={dItem['proposed_action']:<5} {dItem['text'][:70]}")
  dJoins = vPlan.metadata.get("joins")
  if dJoins:
    fPrint("")
    fPrint(f"joins: {dJoins['checked']} checked, {dJoins['adjusted']} adjusted, {len(dJoins['unverified'])} unverified")
  for vWarning in vPrepared.warnings:
    fPrintError(f"warning: {vWarning}")
  if vPrepared.plan_path:
    fPrint(f"plan: {pPipeline.guard.mLabel(vPrepared.plan_path)}")
  return cExitOk if vPrepared.validation.valid else cExitInvalidPlan


def fCommandNormalize(pPipeline: PodcastPipeline, pArguments: argparse.Namespace) -> int:
  from podcast_editor_mcp_server.models.edit_plan import EditPlan, EditReason, LimitTruePeak, NormalizeLoudness, PlanProvenance

  vSource = pPipeline.mResolveSource(pArguments.file)
  vProfile = pPipeline.mResolveProfile(pArguments.profile)
  vProbe = pPipeline.mProbe(vSource)
  vTargetLufs = pArguments.lufs if pArguments.lufs is not None else vProfile.loudness.integrated_lufs
  vTargetPeak = pArguments.true_peak if pArguments.true_peak is not None else vProfile.loudness.true_peak_dbtp

  vPlan = EditPlan(
    source=vProbe.label,
    source_path=vProbe.path,
    source_sha256=vProbe.sha256,
    source_duration=vProbe.duration_seconds,
    sample_rate=vProbe.sample_rate,
    channels=vProbe.channels,
    provenance=PlanProvenance(profile=vProfile.name),
    operations=[
      NormalizeLoudness(
        integrated_lufs=vTargetLufs,
        true_peak_dbtp=vTargetPeak,
        loudness_range=vProfile.loudness.loudness_range,
        reason=EditReason.MASTERING,
      ),
      LimitTruePeak(true_peak_dbtp=vTargetPeak, reason=EditReason.MASTERING),
    ],
    notes="Mastering only; no content was removed.",
  )
  vOptions = ProcessOptions(
    profile=pArguments.profile,
    transcribe=False,
    audacity_project=False,
    overwrite=pArguments.overwrite,
    formats=pArguments.format or None,
    plan=vPlan,
    output_name=pArguments.name,
  )
  vResult = pPipeline.mProcess(vSource, vOptions, _fProgressPrinter(pArguments))
  return _fReportProcessResult(pPipeline, vResult, pArguments)


def fCommandMaster(pPipeline: PodcastPipeline, pArguments: argparse.Namespace) -> int:
  vSource = pPipeline.guard.mResolveInput(pArguments.file, "output")
  vResult = pPipeline.mMasterFile(
    vSource, pArguments.profile, pArguments.name, pArguments.format or None, pArguments.overwrite,
  )
  vOutcome = vResult.outcome
  if pArguments.json:
    fPrintJson(
      {
        "source": str(vSource),
        "artifacts": {vFormat: str(vPath) for vFormat, vPath in vOutcome.exports.items()},
        "applied_operations": vOutcome.applied_operations,
        "loudness_before": vOutcome.loudness_before.model_dump(mode="json") if vOutcome.loudness_before else None,
        "loudness_after": vOutcome.loudness_after.model_dump(mode="json") if vOutcome.loudness_after else None,
        "warnings": vOutcome.warnings,
      }
    )
    return cExitOk
  vBefore = vOutcome.loudness_before
  vAfter = vOutcome.loudness_after
  fPrint(
    fFormatTable(
      [
        ("Source", str(vSource)),
        ("Duration", fFormatDuration(vOutcome.output_duration)),
        ("Applied", ", ".join(vOperation.type for vOperation in vResult.plan.mMasteringOperations())),
        ("Loudness before", _fOptional(vBefore.integrated_lufs if vBefore else None, " LUFS")),
        ("Loudness after", _fOptional(vAfter.integrated_lufs if vAfter else None, " LUFS")),
        ("True peak after", _fOptional(vAfter.true_peak_dbfs if vAfter else None, " dBFS")),
      ]
      + [(f"Output ({vFormat})", str(vPath)) for vFormat, vPath in vOutcome.exports.items()]
    )
  )
  for vWarning in vOutcome.warnings:
    fPrint(f"warning: {vWarning}")
  return cExitOk


def fCommandJobs(pPipeline: PodcastPipeline, pArguments: argparse.Namespace) -> int:
  if pArguments.purge:
    vDays = pPipeline.context.config.jobs.retention_days
    vRemoved = pPipeline.jobs.mPurgeOlderThan(vDays)
    fPrint(f"Removed {vRemoved} finished job record(s) older than {vDays} days.")
    fPrint("Artifacts on disk were left alone.")
    return cExitOk

  if pArguments.job_id:
    vJob = pPipeline.jobs.mLoad(pArguments.job_id)
    if pArguments.json:
      fPrintJson(vJob.model_dump(mode="json"))
      return cExitOk
    fPrint(f"Job {vJob.id}")
    fPrint(
      fFormatTable(
        [
          ("State", str(vJob.state)),
          ("Progress", f"{vJob.progress * 100:.0f}%"),
          ("Source", vJob.source_label),
          ("Profile", vJob.profile),
          ("Created", vJob.created_at),
          ("Finished", vJob.finished_at or "-"),
          ("Artifacts", str(len(vJob.artifacts))),
        ]
      )
    )
    if vJob.artifacts:
      fPrint("\nArtifacts:")
      for vArtifact in vJob.artifacts:
        fPrint(f"  {vArtifact.kind:<18} {vArtifact.label}")
    if vJob.errors:
      fPrint("\nErrors:")
      for vError in vJob.errors:
        fPrint(f"  [{vError.code}] {vError.message}")
    if pArguments.messages and vJob.messages:
      fPrint("\nMessages:")
      for vMessage in vJob.messages[-30:]:
        fPrint(f"  {vMessage.at}  {vMessage.level:<7} {vMessage.message}")
    return cExitOk

  vState = JobState(pArguments.state) if pArguments.state else None
  lJobs = pPipeline.jobs.mList(vState, pArguments.limit)
  if pArguments.json:
    fPrintJson([vJob.mSummary() for vJob in lJobs])
    return cExitOk
  if not lJobs:
    fPrint("No jobs recorded yet.")
    return cExitOk
  fPrint(f"{'JOB ID':<38} {'STATE':<13} {'SOURCE':<30} {'CREATED'}")
  for vJob in lJobs:
    fPrint(f"{vJob.id:<38} {str(vJob.state):<13} {vJob.source_label[:30]:<30} {vJob.created_at}")
  return cExitOk


def fCommandProfiles(pPipeline: PodcastPipeline, pArguments: argparse.Namespace) -> int:
  if pArguments.name:
    vProfile = fLoadProfile(pArguments.name, pPipeline.context.config)
    if pArguments.json:
      fPrintJson(vProfile.model_dump(mode="json"))
      return cExitOk
    import yaml

    fPrint(yaml.safe_dump(vProfile.model_dump(mode="json"), sort_keys=False, allow_unicode=True))
    return cExitOk

  ldProfiles = fListProfiles(pPipeline.context.config)
  if pArguments.json:
    fPrintJson(ldProfiles)
    return cExitOk
  fPrint(f"Active profile: {pPipeline.context.profile.name}\n")
  for dProfile in ldProfiles:
    fPrint(f"  {dProfile['name']}")
    if dProfile["description"]:
      vDescription = " ".join(dProfile["description"].split())
      fPrint(f"      {vDescription[:110]}")
  return cExitOk


def fCommandDoctor(pArguments: argparse.Namespace) -> int:
  from podcast_editor_mcp_server.doctor import fRunDoctor

  vReport = fRunDoctor(pCheckModel=not pArguments.no_model_check, pCheckMcp=not pArguments.no_mcp_check)
  if pArguments.json:
    fPrintJson(vReport.mAsDict())
  else:
    fPrint(vReport.mRender())
  return cExitError if vReport.failed else cExitOk


def fCommandAudacityStatus(pPipeline: PodcastPipeline, pArguments: argparse.Namespace) -> int:
  from podcast_editor_mcp_server.backends.audacity import (
    AudacityBackend,
    AudacityLauncher,
    cModuleStatusNames,
    fAudacityConfigPath,
    fAudacityIsRunning,
    fEnableScriptPipeModule,
    fReadModuleStatus,
  )

  vSettings = pPipeline.context.config.audacity

  if pArguments.enable_module:
    if fAudacityIsRunning():
      fPrintError("Audacity is running. Close it first: it rewrites audacity.cfg when it exits.")
      return cExitError
    dResult = fEnableScriptPipeModule()
    if "error" in dResult:
      fPrintError(str(dResult["error"]))
      return cExitError
    fPrint(f"mod-script-pipe enabled in {dResult['config']}")
    if dResult.get("backup"):
      fPrint(f"  Previous file kept at {dResult['backup']}")
    fPrint("  Start Audacity again for the change to take effect.")
    return cExitOk

  if pArguments.start:
    dUpdates: dict[str, Any] = {}
    if pArguments.xvfb:
      dUpdates["use_xvfb"] = "always"
    if pArguments.display:
      dUpdates["display"] = pArguments.display
      dUpdates["use_xvfb"] = "never"
    if dUpdates:
      vSettings = vSettings.model_copy(update=dUpdates)
    vLauncher = AudacityLauncher(vSettings)
    try:
      dStarted = vLauncher.mStart()
    except PodcastEditorMcpServerError as vError:
      fPrintError(str(vError))
      return cExitError
    fPrint("Audacity started and answering on its script pipe.")
    fPrint(fFormatTable([(vKey, str(vValue)) for vKey, vValue in dStarted.items()]))
    fPrint("\nThe process keeps running after this command exits.")
    return cExitOk

  vBackend = AudacityBackend(vSettings)
  vStatus = vBackend.mStatus()
  vModule = fReadModuleStatus()

  if pArguments.json:
    fPrintJson(
      {
        "available": vStatus.available,
        "reason": vStatus.reason,
        "version": vStatus.version,
        "running": fAudacityIsRunning(),
        "module_status": cModuleStatusNames.get(vModule) if vModule is not None else None,
        "config_file": str(fAudacityConfigPath()),
        "details": vStatus.details,
      }
    )
    return cExitOk if vStatus.available else cExitError

  fPrint(f"Audacity: {'available' if vStatus.available else 'not available'}")
  if vStatus.reason:
    fPrint(f"  Reason: {vStatus.reason}")
  fPrint(
    fFormatTable(
      [
        ("Version", vStatus.version or "unknown"),
        ("Running", "yes" if fAudacityIsRunning() else "no"),
        ("Module status", cModuleStatusNames.get(vModule, "unknown") if vModule is not None else "not configured"),
        ("Config file", str(fAudacityConfigPath())),
        ("Command pipe", str(vStatus.details.get("to_pipe", "n/a"))),
        ("Reply pipe", str(vStatus.details.get("from_pipe", "n/a"))),
      ]
    )
  )
  if not vStatus.available:
    fPrint("\nTo fix:")
    fPrint("  1. Close Audacity.")
    fPrint("  2. podcast-editor-mcp-server-cli audacity-status --enable-module")
    fPrint("  3. Start Audacity, or: podcast-editor-mcp-server-cli audacity-status --start")
  return cExitOk if vStatus.available else cExitError


def fCommandDemo(pPipeline: PodcastPipeline, pArguments: argparse.Namespace) -> int:
  """Write a synthetic test recording into the input directory."""
  from podcast_editor_mcp_server.audio.synth import fWriteSpeechLikeWav

  vTarget = pPipeline.context.workspace.input / pArguments.name
  if vTarget.exists() and not pArguments.overwrite:
    fPrintError(f"{vTarget} already exists; pass --overwrite to replace it")
    return cExitError
  vTarget.parent.mkdir(parents=True, exist_ok=True)
  fWriteSpeechLikeWav(vTarget, pSampleRate=pArguments.sample_rate, pChannels=pArguments.channels)
  vProbe = pPipeline.mProbe(vTarget)
  fPrint(f"Wrote {pPipeline.guard.mLabel(vTarget)} ({vProbe.duration_seconds:.2f} s)")
  fPrint(f"Try: podcast-editor-mcp-server-cli process {pArguments.name} --profile voice-cleanup")
  return cExitOk


def fCommandWorkspace(pPipeline: PodcastPipeline, pArguments: argparse.Namespace) -> int:
  dDescription = pPipeline.context.workspace.mDescribe()
  if pArguments.json:
    fPrintJson(dDescription)
    return cExitOk
  fPrint("Workspace:")
  fPrint(fFormatTable([(vKey, str(vValue)) for vKey, vValue in dDescription.items() if vKey != "extra_roots"]))
  if dDescription["extra_roots"]:
    fPrint("\nExtra roots:")
    for dRoot in dDescription["extra_roots"]:
      fPrint(f"  {dRoot['role']:<12} {dRoot['path']} ({'rw' if dRoot['writable'] else 'ro'})")
  return cExitOk


# --------------------------------------------------------------------------- #
# Shared command helpers
# --------------------------------------------------------------------------- #


def _fAudacityChoice(pArguments: argparse.Namespace) -> bool | None:
  if getattr(pArguments, "no_audacity", False):
    return False
  if getattr(pArguments, "audacity", False):
    return True
  return None


def _fProgressPrinter(pArguments: argparse.Namespace):  # noqa: ANN202 - a closure type adds nothing
  if pArguments.quiet or pArguments.json:
    return None

  def fProgress(pPhase: str, pFraction: float, pJobId: str) -> None:
    sys.stderr.write(f"\r[{pFraction * 100:3.0f}%] {pPhase:<14}")
    sys.stderr.flush()
    if pPhase == "done":
      sys.stderr.write("\n")

  return fProgress


def _fPrintIssues(pErrors: list[dict[str, Any]], pWarnings: list[dict[str, Any]]) -> None:
  if pErrors:
    fPrint("\nErrors:")
    for dIssue in pErrors:
      vOperation = f" [{dIssue['operation_id']}]" if dIssue.get("operation_id") else ""
      fPrint(f"  {dIssue['code']}{vOperation}: {dIssue['message']}")
  if pWarnings:
    fPrint("\nWarnings:")
    for dIssue in pWarnings:
      vOperation = f" [{dIssue['operation_id']}]" if dIssue.get("operation_id") else ""
      fPrint(f"  {dIssue['code']}{vOperation}: {dIssue['message']}")


def _fReportProcessResult(pPipeline: PodcastPipeline, pResult: Any, pArguments: argparse.Namespace) -> int:
  if pArguments.json:
    dPayload: dict[str, Any] = {
      "job_id": pResult.job.id,
      "state": str(pResult.job.state),
      "artifacts": {vKind: str(vPath) for vKind, vPath in pResult.artifacts.items()},
      "warnings": pResult.warnings,
    }
    if pResult.report is not None:
      dPayload["report"] = pResult.report.model_dump(mode="json")
    if pResult.simulation is not None:
      dPayload["simulation"] = pResult.simulation.model_dump(mode="json")
    fPrintJson(dPayload)
    return cExitOk

  if pResult.report is not None:
    vReport = pResult.report
    fPrint(f"\nJob {pResult.job.id} - {pResult.job.state}")
    fPrint(
      fFormatTable(
        [
          ("Source", vReport.source_label),
          ("Original duration", fFormatDuration(vReport.source_duration_seconds)),
          ("Final duration", fFormatDuration(vReport.output_duration_seconds)),
          ("Removed", f"{vReport.removed_seconds:.2f} s ({vReport.removed_fraction * 100:.1f}%)"),
          ("Cuts", str(vReport.counts.cuts)),
          ("Shortened silences", str(vReport.counts.shortened_silences)),
          ("Loudness before", _fOptional(vReport.loudness.integrated_lufs_before, " LUFS")),
          ("Loudness after", _fOptional(vReport.loudness.integrated_lufs_after, " LUFS")),
          ("True peak after", _fOptional(vReport.loudness.true_peak_dbfs_after, " dBFS")),
          ("Audacity project", vReport.audacity_status),
          ("Total time", f"{vReport.total_seconds:.1f} s"),
          ("Source unchanged", "yes" if vReport.source_unchanged else "NO - INVESTIGATE"),
        ]
      )
    )
  elif pResult.simulation is not None:
    vSimulation = pResult.simulation
    fPrint(f"\nDry run - job {pResult.job.id}; no audio was written")
    fPrint(
      fFormatTable(
        [
          ("Original duration", fFormatDuration(vSimulation.source_duration_seconds)),
          ("Estimated result", fFormatDuration(vSimulation.estimated_output_seconds)),
          ("Removed", f"{vSimulation.removed_seconds:.2f} s ({vSimulation.removed_fraction * 100:.1f}%)"),
          ("Operations", str(vSimulation.operation_count)),
          ("Valid", "yes" if vSimulation.valid else "NO"),
        ]
      )
    )
    _fPrintIssues(vSimulation.errors, vSimulation.warnings)

  if pResult.artifacts:
    fPrint("\nArtifacts:")
    for vKind, vPath in sorted(pResult.artifacts.items()):
      fPrint(f"  {vKind:<18} {pPipeline.guard.mLabel(vPath)}")
  for vWarning in pResult.warnings:
    fPrintError(f"warning: {vWarning}")

  if pResult.simulation is not None and not pResult.simulation.valid:
    return cExitInvalidPlan
  return cExitOk


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def fBuildCommonParser() -> argparse.ArgumentParser:
  """Options accepted both before and after the subcommand.

  ``SUPPRESS`` as the default is the point: without it, a subparser that did
  not see the flag would write its own False over the value the top-level
  parser already stored, so "podcast-editor-mcp-server-cli --json probe x.wav" would silently
  lose the flag.
  """
  vCommon = argparse.ArgumentParser(add_help=False)
  vCommon.add_argument(
    "--json", action="store_true", default=argparse.SUPPRESS, help="print machine-readable JSON"
  )
  vCommon.add_argument(
    "--quiet", "-q", action="store_true", default=argparse.SUPPRESS, help="suppress progress output"
  )
  vCommon.add_argument(
    "--verbose", "-v", action="store_true", default=argparse.SUPPRESS, help="log at DEBUG level"
  )
  return vCommon


def fBuildParser() -> argparse.ArgumentParser:
  vParser = argparse.ArgumentParser(
    prog="podcast-editor-mcp-server-cli",
    description="Edit and master podcasts from the command line. Uses the same engine as the MCP server.",
  )
  vParser.add_argument("--version", action="version", version=f"{cProductName} {cVersion}")
  vParser.add_argument("--config", metavar="PATH", help="runtime configuration file to use")
  vParser.add_argument("--workspace", metavar="PATH", help="workspace root (default: /opt/podcast-editor-mcp-server/var/workspace/<user>)")
  vParser.add_argument("--json", action="store_true", help="print machine-readable JSON")
  vParser.add_argument("--quiet", "-q", action="store_true", help="suppress progress output")
  vParser.add_argument("--verbose", "-v", action="store_true", help="log at DEBUG level")

  vCommon = fBuildCommonParser()
  vSubparsers = vParser.add_subparsers(dest="command", metavar="COMMAND")

  vProbe = vSubparsers.add_parser("probe", help="describe an audio file", parents=[vCommon])
  vProbe.add_argument("file", help="audio file in the input directory")

  vAnalyze = vSubparsers.add_parser("analyze", help="measure loudness, silence and speech", parents=[vCommon])
  vAnalyze.add_argument("file")
  vAnalyze.add_argument("--profile", "-p", help="profile to use")
  vAnalyze.add_argument("--show-silences", action="store_true", help="list every detected silence")

  vTranscribe = vSubparsers.add_parser("transcribe", help="transcribe an audio file locally", parents=[vCommon])
  vTranscribe.add_argument("file")
  vTranscribe.add_argument("--profile", "-p")
  vTranscribe.add_argument("--model", "-m", help="Whisper model, e.g. tiny, base, small, medium, large-v3")
  vTranscribe.add_argument("--language", "-l", help="language code, or 'auto'")
  vTranscribe.add_argument("--force", action="store_true", help="ignore the cache")
  vTranscribe.add_argument("--output", "-o", help="write the transcript to this file in the output directory")
  vTranscribe.add_argument(
    "--format", "-f", choices=["summary", "txt", "srt", "vtt", "labels"], default="summary"
  )

  vCandidates = vSubparsers.add_parser("candidates", help="list detected edit candidates", parents=[vCommon])
  vCandidates.add_argument("file")
  vCandidates.add_argument("--profile", "-p")
  vCandidates.add_argument("--limit", type=int, default=40)

  vPlan = vSubparsers.add_parser("plan", help="build an edit plan", parents=[vCommon])
  vPlan.add_argument("file")
  vPlan.add_argument("--profile", "-p")
  vPlan.add_argument("--output", "-o", help="write the plan to this file in the output directory")
  vPlan.add_argument("--save", action="store_true", help="save the plan into the workspace")
  vPlan.add_argument("--limit", type=int, default=40)

  vValidate = vSubparsers.add_parser("validate", help="validate a saved edit plan", parents=[vCommon])
  vValidate.add_argument("plan", help="edit plan JSON file inside the workspace")
  vValidate.add_argument("--file", help="audio file to check the plan against")
  vValidate.add_argument("--profile", "-p")

  vSimulate = vSubparsers.add_parser("simulate", help="report what a plan would do, without touching audio", parents=[vCommon])
  vSimulate.add_argument("plan")
  vSimulate.add_argument("--file")
  vSimulate.add_argument("--profile", "-p")
  vSimulate.add_argument("--limit", type=int, default=40)

  vExecute = vSubparsers.add_parser("execute", help="render a saved edit plan", parents=[vCommon])
  vExecute.add_argument("plan")
  vExecute.add_argument("file", help="audio file the plan was built for")
  vExecute.add_argument("--profile", "-p")
  _fAddRenderArguments(vExecute)

  vProcess = vSubparsers.add_parser("process", help="run the whole pipeline on a file", parents=[vCommon])
  vProcess.add_argument("file")
  vProcess.add_argument("--profile", "-p")
  vProcess.add_argument("--dry-run", action="store_true", help="plan and simulate only; write no audio")
  vProcess.add_argument("--no-transcribe", action="store_true", help="skip transcription")
  vProcess.add_argument(
    "--decide", action="append", default=[], metavar="ID=cut|keep",
    help="decide a review candidate listed by 'prepare' (repeatable); undecided ones follow their proposal",
  )
  vProcess.add_argument(
    "--cut", action="append", default=[], metavar="START-END[:WHY]",
    help="also remove the words between these seconds of the source (repeatable)",
  )
  vProcess.add_argument(
    "--no-mastering", action="store_true",
    help="render only the cuts, as <name>.wav, and leave the clean-up and mastering to 'master'",
  )
  _fAddRenderArguments(vProcess)

  vPrepare = vSubparsers.add_parser(
    "prepare", help="plan an automatic edit and list what needs a decision, without rendering", parents=[vCommon],
  )
  vPrepare.add_argument("file")
  vPrepare.add_argument("--profile", "-p")

  vNormalize = vSubparsers.add_parser("normalize", help="master a file to a loudness target", parents=[vCommon])
  vNormalize.add_argument("file")
  vNormalize.add_argument("--profile", "-p")
  vNormalize.add_argument("--lufs", type=float, help="integrated loudness target, e.g. -16")
  vNormalize.add_argument("--true-peak", type=float, help="true-peak ceiling in dBTP, e.g. -1")
  vNormalize.add_argument("--overwrite", action="store_true", help="replace artifacts that already exist")
  vNormalize.add_argument(
    "--format", "-f", action="append", choices=["wav", "flac", "mp3"], help="delivery format (repeatable)"
  )
  vNormalize.add_argument("--name", help="base name for the artifacts")

  vMaster = vSubparsers.add_parser(
    "master", help="clean up and master an edited file, without cutting anything", parents=[vCommon],
  )
  vMaster.add_argument("file", help="edited audio file, relative to the output directory")
  vMaster.add_argument("--profile", "-p")
  vMaster.add_argument("--overwrite", action="store_true", help="replace artifacts that already exist")
  vMaster.add_argument(
    "--format", "-f", action="append", choices=["wav", "flac", "mp3"], help="delivery format (repeatable)"
  )
  vMaster.add_argument("--name", help="base name for the artifacts")

  vJobs = vSubparsers.add_parser("jobs", help="list or inspect jobs", parents=[vCommon])
  vJobs.add_argument("job_id", nargs="?", help="job to show; omit to list")
  vJobs.add_argument("--state", choices=[vItem.value for vItem in JobState])
  vJobs.add_argument("--limit", type=int, default=20)
  vJobs.add_argument("--messages", action="store_true", help="include the message log")
  vJobs.add_argument(
    "--purge",
    action="store_true",
    help="delete finished job records older than jobs.retention_days (artifacts are kept)",
  )

  vProfiles = vSubparsers.add_parser("profiles", help="list or show editing profiles", parents=[vCommon])
  vProfiles.add_argument("name", nargs="?", help="profile to show; omit to list")

  vDoctor = vSubparsers.add_parser("doctor", help="diagnose the installation", parents=[vCommon])
  vDoctor.add_argument("--no-model-check", action="store_true")
  vDoctor.add_argument("--no-mcp-check", action="store_true")

  vAudacity = vSubparsers.add_parser("audacity-status", help="check or set up the Audacity integration", parents=[vCommon])
  vAudacity.add_argument("--enable-module", action="store_true", help="set mod-script-pipe=1 in audacity.cfg")
  vAudacity.add_argument("--start", action="store_true", help="start Audacity (with Xvfb if needed) and wait for its pipes")
  vAudacity.add_argument("--xvfb", action="store_true", help="always start a virtual display, even on a desktop session")
  vAudacity.add_argument("--display", help="use this X display instead of starting one")

  vDemo = vSubparsers.add_parser("demo", help="write a synthetic test recording into the input directory", parents=[vCommon])
  vDemo.add_argument("--name", default="demo.wav")
  vDemo.add_argument("--sample-rate", type=int, default=48000)
  vDemo.add_argument("--channels", type=int, default=1, choices=[1, 2])
  vDemo.add_argument("--overwrite", action="store_true")

  vSubparsers.add_parser("workspace", help="show the workspace layout", parents=[vCommon])

  return vParser


def _fAddRenderArguments(pParser: argparse.ArgumentParser) -> None:
  pParser.add_argument("--overwrite", action="store_true", help="replace artifacts that already exist")
  pParser.add_argument(
    "--format", "-f", action="append", choices=["wav", "flac", "mp3"], help="delivery format (repeatable)"
  )
  pParser.add_argument("--name", help="base name for the artifacts")
  pParser.add_argument("--audacity", action="store_true", help="force creating the .aup3 project")
  pParser.add_argument("--no-audacity", action="store_true", help="skip the .aup3 project")
  pParser.add_argument("--keep-work", action="store_true", help="keep the per-job work directory")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def fMain(pArgv: list[str] | None = None) -> int:
  """Entry point for ``podcast-editor-mcp-server-cli``."""
  import os

  vParser = fBuildParser()
  vArguments = vParser.parse_args(pArgv)

  if not vArguments.command:
    vParser.print_help()
    return cExitUsage

  if vArguments.config:
    os.environ["PODCAST_EDITOR_MCP_SERVER_CONFIG"] = vArguments.config
  if vArguments.workspace:
    os.environ["PODCAST_EDITOR_MCP_SERVER_WORKSPACE"] = vArguments.workspace

  fSetupLogging(pLevel="DEBUG" if vArguments.verbose else "WARNING", pForce=True)

  # The doctor builds its own context so that it can report configuration
  # problems instead of failing to start.
  if vArguments.command == "doctor":
    return fCommandDoctor(vArguments)

  try:
    vPipeline = PodcastPipeline(AppContext())
  except PodcastEditorMcpServerError as vError:
    fPrintError(str(vError))
    return cExitError

  dHandlers = {
    "probe": fCommandProbe,
    "analyze": fCommandAnalyze,
    "transcribe": fCommandTranscribe,
    "candidates": fCommandCandidates,
    "plan": fCommandPlan,
    "validate": fCommandValidate,
    "simulate": fCommandSimulate,
    "execute": fCommandExecute,
    "process": fCommandProcess,
    "prepare": fCommandPrepare,
    "normalize": fCommandNormalize,
    "master": fCommandMaster,
    "jobs": fCommandJobs,
    "profiles": fCommandProfiles,
    "audacity-status": fCommandAudacityStatus,
    "demo": fCommandDemo,
    "workspace": fCommandWorkspace,
  }
  vHandler = dHandlers[vArguments.command]

  try:
    return vHandler(vPipeline, vArguments)
  except PodcastEditorMcpServerError as vError:
    fPrintError(str(vError))
    if vArguments.verbose:
      vLog.exception("Command failed", extra={"command": vArguments.command})
    return cExitError
  except KeyboardInterrupt:  # pragma: no cover - interactive interruption
    fPrintError("interrupted")
    return cExitError


if __name__ == "__main__":  # pragma: no cover - module entry point
  raise SystemExit(fMain())
