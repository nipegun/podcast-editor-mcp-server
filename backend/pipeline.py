"""The product engine.

Every entry point - the MCP server, the CLI, the doctor - goes through this
module. There is exactly one implementation of "process a podcast", and both
front ends call it.

Pipeline
--------

    probe -> analyse -> transcribe -> candidates -> plan -> check joins
          -> validate -> execute -> export -> transcribe the master
          -> Audacity project -> report

Each step is also callable on its own, which is what makes the MCP surface
fine-grained: a client can transcribe, look at the candidates, build its own
plan, validate it, simulate it and only then execute.

The automatic edit is two calls: :meth:`PodcastPipeline.mPrepareEdit` builds
and checks the plan and lists what needs a decision; :meth:`mProcess` with
``decisions`` renders it. A run with no decisions applies the proposals.

Editing and mastering can also be two steps: :meth:`mProcess` with
``mastering=False`` renders only the cuts (``<name>.wav``), and
:meth:`mMasterFile` later applies the profile's clean-up and mastering chain to
that file (``<name>.master.wav`` and the other delivery formats).
"""

from __future__ import annotations

import datetime as dt
import shutil
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from podcast_editor_mcp_server.analysis.candidates import fCollectCandidates
from podcast_editor_mcp_server.analysis.retakes import fRequestedCutCandidates
from podcast_editor_mcp_server.audio.analysis import fAnalyzeAudio, fEnsureReadableCopy
from podcast_editor_mcp_server.audio.probe import fProbeAudio
from podcast_editor_mcp_server.audio.samples import Envelope, fComputeEnvelope, fDecodeMono, fReadSamples
from podcast_editor_mcp_server.backends.audacity import AudacityBackend, fInvertKeepSegments
from podcast_editor_mcp_server.backends.ffmpeg import FfmpegBackend
from podcast_editor_mcp_server.config import AppContext, Profile
from podcast_editor_mcp_server.errors import (
  EditPlanError,
  ExportError,
  PodcastEditorMcpServerError,
  TranscriptionError,
)
from podcast_editor_mcp_server.errors import (
  JobError as JobFailure,
)
from podcast_editor_mcp_server.execution.engine import ExecutionEngine, ExecutionOutcome, ExecutionRequest
from podcast_editor_mcp_server.jobs.models import Job, JobState
from podcast_editor_mcp_server.jobs.store import JobStore, fValidateJobId
from podcast_editor_mcp_server.logging_setup import fGetLogger, fJobLogHandler
from podcast_editor_mcp_server.models.audio import AnalysisReport, AudioProbe
from podcast_editor_mcp_server.models.candidates import CandidateSet
from podcast_editor_mcp_server.models.edit_plan import EditPlan, PlanProvenance
from podcast_editor_mcp_server.models.report import (
  ArtifactRecord,
  EditCounts,
  EditReport,
  LoudnessBeforeAfter,
  PhaseTiming,
  SimulationResult,
)
from podcast_editor_mcp_server.models.transcript import Transcript
from podcast_editor_mcp_server.planning.joins import JoinCheck, fVerifyJoins
from podcast_editor_mcp_server.planning.pacing import SpeechEdges, fListenToGaps
from podcast_editor_mcp_server.planning.planner import fBuildAutomaticPlan, fBuildMasteringOperations
from podcast_editor_mcp_server.security.paths import cAudioWriteExtensions, cDataExtensions, fSanitiseName
from podcast_editor_mcp_server.transcription.base import TranscriptionBackend, fCreateTranscriptionBackend
from podcast_editor_mcp_server.transcription.cache import TranscriptCache
from podcast_editor_mcp_server.transcription.cleanup import fPostProcessTranscript
from podcast_editor_mcp_server.transcription.formats import fRender
from podcast_editor_mcp_server.transcription.remap import fRemapTranscript
from podcast_editor_mcp_server.util.atomic import fAcquireLock, fReleaseLock, fWriteJsonAtomic, fWriteTextAtomic
from podcast_editor_mcp_server.util.hashing import fHashFile, fHashMapping
from podcast_editor_mcp_server.util.timeline import fBuildTimeline
from podcast_editor_mcp_server.validation.validator import SafetyOverride, ValidationResult, fIssueDicts, fValidateEditPlan
from podcast_editor_mcp_server.version import cProductName, cVersion

vLog = fGetLogger("pipeline")

ProgressCallback = Callable[[str, float, str], None]

# Secondary suffixes stripped from a source stem when naming artifacts, so
# "Podcast-042.raw.wav" produces "Podcast-042.master.wav".
cStrippedStemSuffixes: tuple[str, ...] = (".raw", ".original", ".source", ".unedited")


@dataclass(slots=True)
class ProcessOptions:
  """Options for a full ``process`` run."""

  profile: str | None = None
  dry_run: bool = False
  overwrite: bool = False
  transcribe: bool = True
  audacity_project: bool | None = None
  formats: list[str] | None = None
  keep_work_dir: bool | None = None
  plan: EditPlan | None = None
  output_name: str | None = None
  # Called between phases; returning True aborts the run cleanly.
  cancel_check: Callable[[], bool] | None = None
  # Pre-assigned job id, so a caller that starts a run in the background can
  # hand the id back before the work begins.
  job_id: str | None = None
  # Candidate id -> "cut" or "keep", for the candidates that needed a decision.
  decisions: dict[str, str] | None = None
  # Apply the proposed action of every undecided review candidate. True is a
  # fully automatic run; False leaves undecided candidates out.
  apply_proposals: bool = True
  # Deliver a transcript of the edited result, even when the source was not
  # transcribed in this run (a supplied plan).
  export_transcript: bool = True
  # Cuts asked for by time, ``{"start", "end", "why"}`` in seconds of the
  # source: removed word by word and placed, checked and reported like a take.
  extra_cuts: list[dict[str, Any]] | None = None
  # Apply the profile's clean-up and mastering chain (filters, noise
  # reduction, compressor, loudness, limiter, fades) while rendering. False
  # renders only the cuts, as "<name>.wav" in lossless formats only, for
  # :meth:`PodcastPipeline.mMasterFile` to master later.
  mastering: bool = True


@dataclass(slots=True)
class PreparedEdit:
  """An automatic edit, planned and checked but not rendered."""

  probe: AudioProbe
  transcript: Transcript | None
  plan: EditPlan
  validation: ValidationResult
  simulation: SimulationResult
  plan_path: Path | None = None
  warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MasterResult:
  """What mastering an already edited file produced."""

  plan: EditPlan
  outcome: ExecutionOutcome


@dataclass(slots=True)
class ProcessResult:
  """What a full run produced."""

  job: Job
  transcript: Transcript | None = None
  plan: EditPlan | None = None
  simulation: SimulationResult | None = None
  report: EditReport | None = None
  artifacts: dict[str, Path] = field(default_factory=dict)
  warnings: list[str] = field(default_factory=list)


class PodcastPipeline:
  """The single implementation of every product operation."""

  def __init__(self, pContext: AppContext | None = None) -> None:
    self.context = pContext if pContext is not None else AppContext()
    self.ffmpeg = FfmpegBackend()
    self.engine = ExecutionEngine(self.ffmpeg)
    self.jobs = JobStore(self.context.workspace.jobs)
    self.cache = TranscriptCache(
      self.context.workspace.cache,
      self.context.config.jobs.transcription_cache,
    )
    # One backend per engine for the pipeline's lifetime, so the Whisper model
    # loaded to transcribe is the one reused to check the joins.
    self._dTranscribers: dict[str, TranscriptionBackend] = {}

  # ------------------------------------------------------------ resolution --

  def mGetGuard(self):  # noqa: ANN201 - the PathGuard type is internal plumbing
    return self.context.workspace.guard

  guard = property(mGetGuard)

  def mResolveProfile(self, pName: str | None) -> Profile:
    return self.context.mWithProfile(pName)

  def mResolveSource(self, pPath: str | Path) -> Path:
    """Resolve a user-supplied source path inside the workspace."""
    return self.guard.mResolveInput(pPath)

  def mArtifactBaseName(self, pSource: Path, pOverride: str | None = None) -> str:
    """Derive the artifact stem from a source file name."""
    if pOverride:
      return fSanitiseName(pOverride)
    vStem = pSource.stem
    for vSuffix in cStrippedStemSuffixes:
      if vStem.lower().endswith(vSuffix):
        vStem = vStem[: -len(vSuffix)]
        break
    return fSanitiseName(vStem)

  def mOutputDirectory(self, pBaseName: str) -> Path:
    return self.guard.mResolve(
      self.context.workspace.output / pBaseName,
      pRequireExists=False, pForWrite=True, pMustBeFile=False,
    )

  @contextmanager
  def mReserveOutput(self, pOutput: Path) -> Iterator[None]:
    """Hold an artifact directory (or file) for the whole of one run.

    Two runs aimed at the same place would both pass the existence checks and
    then write over each other, whatever ``overwrite`` said. The second one is
    refused at once instead. The lock lives under ``<work>/locks`` and the
    kernel drops it if the process dies.
    """
    vKey = fHashMapping({"output": str(pOutput)})[:16]
    vLockPath = self.context.workspace.work / "locks" / f"output-{vKey}.lock"
    vDescriptor = fAcquireLock(vLockPath)
    if vDescriptor is None:
      raise ExportError(
        "Another run is writing artifacts with this name right now",
        {"path": pOutput.name, "hint": "wait for it to finish, or pass another output name"},
      )
    try:
      yield
    finally:
      fReleaseLock(vLockPath, vDescriptor)

  def _mResolveArtifact(self, pPath: Path, pSource: Path, pOverwrite: bool) -> Path:
    """Check an artifact without creating directories or replacing the source."""
    vPath = self.guard.mResolveOutput(
      pPath, pOverwrite=True, pSource=pSource, pCreateParent=False,
    )
    if vPath.exists() and not pOverwrite:
      raise ExportError(
        "An artifact with this name already exists",
        {"path": vPath.name, "hint": "pass overwrite to replace it"},
      )
    return vPath

  def _mFormats(self, pProfile: Profile, pOptions: ProcessOptions) -> list[str]:
    """The delivery formats of a run: the profile's, lossless only without mastering."""
    if pOptions.formats:
      return list(pOptions.formats)
    lFormats = [str(vFormat) for vFormat in pProfile.output.formats]
    if pOptions.mastering:
      return lFormats
    # Audio left unmastered is the working file of the mastering step.
    return [vFormat for vFormat in lFormats if vFormat in {"wav", "flac"}] or ["wav"]

  def _mCheckOutputTargets(
    self, pSource: Path, pOutputDir: Path, pBaseName: str, pProfile: Profile, pOptions: ProcessOptions,
  ) -> None:
    """Check every potential artifact before any output from this run is written."""
    lSuffixes = ["analysis.json", "edit-plan.json", "candidates.json"]
    vWantsTranscript = pOptions.transcribe or (pOptions.export_transcript and not pOptions.dry_run)
    if vWantsTranscript and pProfile.transcription.backend != "none":
      lSuffixes.extend(
        f"transcript.{vFormat}" if vFormat in {"json", "txt"} else str(vFormat)
        for vFormat in pProfile.output.transcript_formats
      )
      lSuffixes.append("labels.txt")
    if pOptions.dry_run:
      lSuffixes.append("simulation.json")
    else:
      lSuffixes.extend(["edit-report.json", "job.json"])
      vWavSuffix = "master.wav" if pOptions.mastering else "wav"
      lSuffixes.extend(
        vWavSuffix if vFormat == "wav" else str(vFormat)
        for vFormat in self._mFormats(pProfile, pOptions)
      )
      vWantProject = pOptions.audacity_project
      if vWantProject is None:
        vWantProject = pProfile.output.audacity_project
      if vWantProject:
        lSuffixes.append("aup3")
      if self.context.config.logging.per_job_log:
        lSuffixes.append("job.log")
    for vSuffix in lSuffixes:
      self._mResolveArtifact(pOutputDir / f"{pBaseName}.{vSuffix}", pSource, pOptions.overwrite)

  # ----------------------------------------------------------------- steps --

  def mProbe(self, pSource: Path) -> AudioProbe:
    """Describe a source file."""
    return fProbeAudio(pSource, self.ffmpeg, self.guard)

  def mAnalyze(
    self,
    pSource: Path,
    pProfile: Profile,
    pWorkDir: Path,
    pProbe: AudioProbe | None = None,
  ) -> AnalysisReport:
    """Measure a source file."""
    return fAnalyzeAudio(pSource, self.ffmpeg, pProfile, pWorkDir, self.guard, pProbe=pProbe)

  def mEnvelope(self, pSource: Path, pWorkDir: Path) -> Envelope | None:
    """Compute the RMS envelope used to place cuts, tolerating failure."""
    try:
      vReadable = fEnsureReadableCopy(pSource, self.ffmpeg, pWorkDir)
      return fComputeEnvelope(fReadSamples(vReadable))
    except PodcastEditorMcpServerError as vError:
      vLog.warning("Could not compute the energy envelope", extra={"reason": str(vError)})
      return None

  def mTranscriber(self, pProfile: Profile) -> TranscriptionBackend:
    """The (shared) transcription backend a profile uses, checked for availability."""
    vName = pProfile.transcription.backend
    vBackend = self._dTranscribers.get(vName)
    if vBackend is None:
      vBackend = fCreateTranscriptionBackend(vName)
      if vBackend is None:
        raise TranscriptionError(
          "Transcription is disabled in this profile",
          {"profile": pProfile.name},
        )
      self._dTranscribers[vName] = vBackend
    vStatus = vBackend.mStatus()
    if not vStatus.available:
      raise TranscriptionError(vStatus.reason or "The transcription backend is unavailable", vStatus.details)
    return vBackend

  def mTranscribe(
    self,
    pSource: Path,
    pProfile: Profile,
    pSourceSha256: str = "",
    pUseCache: bool = True,
  ) -> Transcript:
    """Transcribe a file, reusing a cached result when one matches exactly.

    The cache holds the recogniser's raw output; what is returned has the
    hallucinations removed and the profile's corrections applied.
    """
    vBackend = self.mTranscriber(pProfile)

    if pUseCache:
      vCached = self.cache.mLoad(pSourceSha256, pProfile.transcription)
      if vCached is not None:
        return fPostProcessTranscript(vCached, pProfile.transcription)

    vTranscript = vBackend.mTranscribe(pSource, pProfile.transcription)
    vTranscript = vTranscript.model_copy(update={"source_sha256": pSourceSha256})
    self.cache.mStore(pSourceSha256, pProfile.transcription, vTranscript)
    return fPostProcessTranscript(vTranscript, pProfile.transcription)

  def mLoadCachedTranscript(self, pSourceSha256: str, pProfile: Profile) -> Transcript | None:
    """The cached transcript for these bytes and settings, post-processed, or None."""
    vCached = self.cache.mLoad(pSourceSha256, pProfile.transcription)
    if vCached is None:
      return None
    return fPostProcessTranscript(vCached, pProfile.transcription)

  def mVerifyJoins(
    self,
    pPlan: EditPlan,
    pSource: Path,
    pTranscript: Transcript,
    pProfile: Profile,
  ) -> tuple[EditPlan, list[JoinCheck], list[str]]:
    """Listen to every cut through speech and fix the seams that bite.

    Never fatal: without a working recogniser the plan is returned unchanged
    with a warning.
    """
    try:
      vBackend = self.mTranscriber(pProfile)
      aAudio = fDecodeMono(pSource, 16_000)
    except PodcastEditorMcpServerError as vError:
      return (pPlan, [], [f"Joins not checked: {vError.message}"])

    vSettings = pProfile.transcription
    vModel = pProfile.cuts.verify_model

    def fRecognize(pSamples: Any) -> str:
      return vBackend.mRecognizeSamples(pSamples, vSettings, vModel)

    try:
      vPlan, lChecks = fVerifyJoins(pPlan, aAudio, pTranscript, pProfile, fRecognize)
    except PodcastEditorMcpServerError as vError:
      return (pPlan, [], [f"Joins not checked: {vError.message}"])
    lWarnings: list[str] = []
    vUnverified = sum(1 for vCheck in lChecks if not vCheck.verified)
    if vUnverified:
      lWarnings.append(f"{vUnverified} cut(s) could not be verified by listening to the join")
    return (vPlan, lChecks, lWarnings)

  def mListenToGaps(
    self,
    pPlan: EditPlan,
    pSource: Path,
    pProfile: Profile,
    pEnvelope: Envelope | None,
    pTranscript: Transcript | None = None,
  ) -> tuple[EditPlan, list[str]]:
    """With tight pacing, keep the gaps whose long sounds are words the transcript lacks.

    Never fatal: without a recogniser or an envelope the plan is returned
    unchanged.
    """
    if not pProfile.silence.tighten_word_gaps or pEnvelope is None:
      return (pPlan, [])
    if not any(str(getattr(vOperation, "note", "") or "").startswith("gap-") for vOperation in pPlan.operations):
      return (pPlan, [])
    try:
      vBackend = self.mTranscriber(pProfile)
      aAudio = fDecodeMono(pSource, 16_000)
    except PodcastEditorMcpServerError as vError:
      return (pPlan, [f"Gaps not listened to: {vError.message}"])
    vSettings = pProfile.transcription

    def fRecognize(pSamples: Any) -> str:
      return vBackend.mRecognizeSamples(pSamples, vSettings, pProfile.cuts.verify_model)

    vPlan, ldKept = fListenToGaps(
      pPlan, aAudio, SpeechEdges(pEnvelope), fRecognize, vSettings.hallucination_phrases, pProfile.silence, pTranscript,
    )
    dMetadata = dict(vPlan.metadata)
    dPacing = dict(dMetadata.get("pacing") or {})
    dPacing["kept_for_speech"] = ldKept
    dMetadata["pacing"] = dPacing
    vPlan = vPlan.model_copy(update={"metadata": dMetadata})
    lWarnings = [f"{len(ldKept)} gap(s) kept because the recogniser heard words there"] if ldKept else []
    return (vPlan, lWarnings)

  def mPrepareEdit(
    self,
    pSource: str | Path,
    pProfile: str | None = None,
    pDecisions: dict[str, str] | None = None,
    pApplyProposals: bool = False,
    pSave: bool = True,
    pExtraCuts: list[dict[str, Any]] | None = None,
  ) -> PreparedEdit:
    """Analyse, transcribe, detect, plan and check the joins, without rendering.

    The result lists the candidates that need a decision. Rendering is
    :meth:`mProcess` with the same profile and the decisions. ``pExtraCuts``
    are cuts asked for by time (see :class:`ProcessOptions`).
    """
    vSource = self.mResolveSource(pSource)
    vProfile = self.mResolveProfile(pProfile)
    vWorkDir = self.guard.mResolve(
      self.context.workspace.work / f"prepare-{fHashMapping({'path': str(vSource)})[:16]}",
      pRequireExists=False, pForWrite=True, pMustBeFile=False,
    )
    vWorkDir.mkdir(parents=True, exist_ok=True)
    lWarnings: list[str] = []
    try:
      vProbe = self.mProbe(vSource)
      vAnalysis = self.mAnalyze(vSource, vProfile, vWorkDir, vProbe)
      lWarnings.extend(vAnalysis.warnings)
      vEnvelope = self.mEnvelope(vSource, vWorkDir)
      vTranscript: Transcript | None = None
      if vProfile.transcription.backend != "none":
        try:
          vTranscript = self.mTranscribe(vSource, vProfile, vProbe.sha256)
        except TranscriptionError as vError:
          lWarnings.append(f"Transcription unavailable: {vError.message}")
      vCandidates = self.mFindCandidates(vAnalysis, vProfile, vTranscript)
      lWarnings.extend(vCandidates.warnings)
      vCandidates, dDecisions, lRequestWarnings = self.mAddRequestedCuts(vCandidates, vTranscript, pExtraCuts, pDecisions)
      lWarnings.extend(lRequestWarnings)
      vPlan = self.mCreatePlan(vAnalysis, vProfile, vCandidates, vTranscript, vEnvelope, dDecisions, pApplyProposals)
      vPlan, lGapWarnings = self.mListenToGaps(vPlan, vSource, vProfile, vEnvelope, vTranscript)
      lWarnings.extend(lGapWarnings)
      if vProfile.cuts.verify_joins and vTranscript is not None:
        vPlan, _, lJoinWarnings = self.mVerifyJoins(vPlan, vSource, vTranscript, vProfile)
        lWarnings.extend(lJoinWarnings)
      vValidation = self.mValidate(vPlan, vProfile, vProbe)
      vSimulation = self.mSimulate(vPlan, vProfile, vProbe)
    finally:
      shutil.rmtree(vWorkDir, ignore_errors=True)
    vPlanPath = self.mSavePlan(vPlan) if pSave else None
    return PreparedEdit(
      probe=vProbe,
      transcript=vTranscript,
      plan=vPlan,
      validation=vValidation,
      simulation=vSimulation,
      plan_path=vPlanPath,
      warnings=lWarnings,
    )

  def mFindCandidates(
    self,
    pAnalysis: AnalysisReport,
    pProfile: Profile,
    pTranscript: Transcript | None,
  ) -> CandidateSet:
    """Run every candidate detector."""
    return fCollectCandidates(pAnalysis, pProfile, pTranscript)

  def mAddRequestedCuts(
    self,
    pCandidates: CandidateSet,
    pTranscript: Transcript | None,
    pCuts: list[dict[str, Any]] | None,
    pDecisions: dict[str, str] | None,
  ) -> tuple[CandidateSet, dict[str, str], list[str]]:
    """Add the cuts asked for by time as take candidates already decided as ``cut``."""
    dDecisions = dict(pDecisions or {})
    if not pCuts:
      return (pCandidates, dDecisions, [])
    if pTranscript is None:
      return (pCandidates, dDecisions, ["Requested cuts ignored: there is no transcript to place them on"])
    lRequested, lWarnings = fRequestedCutCandidates(pTranscript, pCuts)
    for vCandidate in lRequested:
      dDecisions.setdefault(vCandidate.id, "cut")
    lAll = sorted(list(pCandidates.candidates) + lRequested, key=lambda pCandidate: (pCandidate.start, pCandidate.end))
    return (pCandidates.model_copy(update={"candidates": lAll}), dDecisions, lWarnings)

  def mCreatePlan(
    self,
    pAnalysis: AnalysisReport,
    pProfile: Profile,
    pCandidates: CandidateSet,
    pTranscript: Transcript | None = None,
    pEnvelope: Envelope | None = None,
    pDecisions: dict[str, str] | None = None,
    pApplyProposals: bool = False,
  ) -> EditPlan:
    """Build an EditPlan with the deterministic planner."""
    return fBuildAutomaticPlan(
      pAnalysis, pProfile, pCandidates, pTranscript, pEnvelope, pDecisions, pApplyProposals,
    )

  def mValidate(
    self,
    pPlan: EditPlan,
    pProfile: Profile,
    pProbe: AudioProbe | None = None,
    pOverride: SafetyOverride | None = None,
  ) -> ValidationResult:
    """Validate a plan against a profile and, when given, the actual file."""
    return fValidateEditPlan(pPlan, pProfile, pProbe, pOverride)

  def mSimulate(
    self,
    pPlan: EditPlan,
    pProfile: Profile,
    pProbe: AudioProbe | None = None,
  ) -> SimulationResult:
    """Report what a plan would do without touching any audio."""
    vValidation = self.mValidate(pPlan, pProfile, pProbe)
    vTimeline = vValidation.timeline
    ldRemoved: list[dict[str, Any]] = []
    if vTimeline is not None:
      ldRemoved = [
        {
          "start": round(vRemoval.start, 3),
          "end": round(vRemoval.end, 3),
          "duration": round(vRemoval.duration, 3),
          "operations": vRemoval.operation_ids,
          "reasons": sorted(set(vRemoval.reasons)),
        }
        for vRemoval in vTimeline.removals
      ]
    return SimulationResult(
      plan_id=pPlan.id,
      source=pPlan.source,
      source_sha256=pPlan.source_sha256,
      source_duration_seconds=pPlan.source_duration,
      estimated_output_seconds=vValidation.estimated_output_seconds,
      removed_seconds=vValidation.removed_seconds,
      inserted_seconds=vValidation.inserted_seconds,
      removed_fraction=vValidation.removed_fraction,
      operation_count=len(pPlan.operations),
      structural_count=len(pPlan.mStructuralOperations()),
      mastering_count=len(pPlan.mMasteringOperations()),
      segment_count=len(vTimeline.keep_segments) if vTimeline else 0,
      removed_regions=ldRemoved,
      by_type=pPlan.mCountByType(),
      by_reason=pPlan.mCountByReason(),
      valid=vValidation.valid,
      errors=fIssueDicts(vValidation.errors),
      warnings=fIssueDicts(vValidation.warnings),
      conflicts=fIssueDicts([vIssue for vIssue in vValidation.errors if vIssue.code == "overlapping_operations"]),
    )

  # --------------------------------------------------------------- process --

  def mProcess(
    self,
    pSource: str | Path,
    pOptions: ProcessOptions | None = None,
    pProgress: ProgressCallback | None = None,
  ) -> ProcessResult:
    """Run the whole pipeline and produce every artifact."""
    vOptions = pOptions if pOptions is not None else ProcessOptions()
    vSource = self.mResolveSource(pSource)
    vProfile = self.mResolveProfile(vOptions.profile)

    vBaseName = self.mArtifactBaseName(vSource, vOptions.output_name)
    vOutputDir = self.mOutputDirectory(vBaseName)
    vSourceHashBefore = fHashFile(vSource)
    vSourceMtimeBefore = vSource.stat().st_mtime_ns

    dJobFields: dict[str, Any] = {}
    if vOptions.job_id:
      dJobFields["id"] = fValidateJobId(vOptions.job_id)
      if self.jobs.mExists(vOptions.job_id):
        vQueued = self.jobs.mLoad(vOptions.job_id)
        if vQueued.is_terminal:
          raise JobFailure("The job has already finished", {"job_id": vQueued.id})
        dJobFields["created_at"] = vQueued.created_at
        dJobFields["messages"] = vQueued.messages
    vJob = Job(
      **dJobFields,
      source_path=str(vSource),
      source_label=self.guard.mLabel(vSource),
      source_sha256=vSourceHashBefore,
      profile=vProfile.name,
      profile_sha256=fHashMapping(vProfile.mFingerprint()),
      options={
        "dry_run": vOptions.dry_run,
        "transcribe": vOptions.transcribe,
        "formats": self._mFormats(vProfile, vOptions),
        "audacity_project": vOptions.audacity_project,
        "mastering": vOptions.mastering,
        "overwrite": vOptions.overwrite,
      },
      output_dir=str(vOutputDir),
    )
    vWorkDir = self.guard.mResolve(
      self.context.workspace.mJobWorkDir(vJob.id),
      pRequireExists=False, pForWrite=True, pMustBeFile=False,
    )
    vWorkDir.mkdir(parents=True, exist_ok=True)
    vJob.work_dir = str(vWorkDir)
    self.jobs.mSave(vJob)

    vKeepWork = vOptions.keep_work_dir
    if vKeepWork is None:
      vKeepWork = self.context.config.jobs.keep_work_dir

    vLogFile = vWorkDir / "job.log"
    vResult = ProcessResult(job=vJob)
    dPhaseSeconds: dict[str, float] = {}
    vOutputsChecked = False
    vSourceChecked = False
    vSourceIntact = False
    vReservation = ExitStack()

    try:
      self._mCheckCancelled(vOptions, vJob)
      vReservation.enter_context(self.mReserveOutput(vOutputDir))
      self._mCheckOutputTargets(vSource, vOutputDir, vBaseName, vProfile, vOptions)
      vOutputsChecked = True
      if self.context.config.logging.per_job_log:
        with fJobLogHandler(vLogFile):
          vResult = self._mRunPipeline(
            vSource, vProfile, vJob, vOptions, vWorkDir, vOutputDir, vBaseName, dPhaseSeconds, pProgress
          )
      else:
        vResult = self._mRunPipeline(
          vSource, vProfile, vJob, vOptions, vWorkDir, vOutputDir, vBaseName, dPhaseSeconds, pProgress
        )
      vSourceIntact = self._mVerifySourceUnchanged(vSource, vSourceHashBefore, vSourceMtimeBefore, vJob, vResult)
      vSourceChecked = True
      self._mFinalizeResult(vResult, vSource, vOutputDir, vBaseName, vOptions, vSourceIntact)
    except PodcastEditorMcpServerError as vError:
      # A cancelled job has already recorded its outcome; do not relabel it as
      # a failure on the way out.
      if vJob.state != JobState.CANCELLED:
        vJob.mAddError(vError.cCode, vError.message, vError.details)
        vJob.mSetState(JobState.FAILED, f"Failed: {vError.message}")
        vLog.error("Job failed", extra={"job_id": vJob.id, "error": vError.cCode, "reason": vError.message})
      self.jobs.mSave(vJob)
      raise
    except Exception as vError:  # noqa: BLE001 - record any crash on the job before re-raising
      if vJob.state != JobState.CANCELLED:
        vJob.mAddError("internal_error", str(vError))
        vJob.mSetState(JobState.FAILED, "Failed with an internal error")
      self.jobs.mSave(vJob)
      vLog.exception("Job crashed", extra={"job_id": vJob.id})
      raise
    finally:
      try:
        if not vSourceChecked:
          self._mVerifySourceUnchanged(vSource, vSourceHashBefore, vSourceMtimeBefore, vJob, vResult)
        if not vKeepWork:
          # A dry run delivers no job log, so its log is not copied out.
          self._mCleanWorkDir(
            vWorkDir, vLogFile, vOutputDir, vBaseName, vSource, vOptions.overwrite,
            vOutputsChecked and not vOptions.dry_run,
          )
        self.jobs.mSave(vJob)
      finally:
        # Held until the job log is copied out, the last write of the run.
        vReservation.close()

    if not vSourceIntact:
      raise JobFailure("Source integrity verification failed", {"job_id": vJob.id, "path": vJob.source_label})
    return vResult

  def mMasterFile(
    self,
    pSource: Path,
    pProfileName: str | None = None,
    pOutputName: str | None = None,
    pFormats: list[str] | None = None,
    pOverwrite: bool = False,
  ) -> MasterResult:
    """Clean up and master a file that is already edited, without cutting anything.

    The plan holds only the profile's mastering chain, rendered in the
    engine's fixed order. The delivery formats go to ``<output>/<name>/`` as
    ``<name>.master.wav`` and ``<name>.<format>``; nothing else is written, so
    the plan, report and transcript of the edit stay as they were.
    """
    vProfile = self.mResolveProfile(pProfileName)
    vBaseName = self.mArtifactBaseName(pSource, pOutputName)
    vOutputDir = self.mOutputDirectory(vBaseName)
    lFormats = list(pFormats) if pFormats else [str(vFormat) for vFormat in vProfile.output.formats]
    with self.mReserveOutput(vOutputDir):
      for vFormat in lFormats:
        vSuffix = "master.wav" if vFormat == "wav" else vFormat
        self._mResolveArtifact(vOutputDir / f"{vBaseName}.{vSuffix}", pSource, pOverwrite)

      vWorkDir = self.guard.mResolve(
        self.context.workspace.mJobWorkDir(str(uuid.uuid4())),
        pRequireExists=False, pForWrite=True, pMustBeFile=False,
      )
      vWorkDir.mkdir(parents=True, exist_ok=True)
      try:
        vProbe = self.mProbe(pSource)
        vAnalysis = self.mAnalyze(pSource, vProfile, vWorkDir, vProbe)
        lOperations = fBuildMasteringOperations(vAnalysis, vProfile)
        if not lOperations:
          raise EditPlanError(
            "The profile applies no clean-up or mastering",
            {"profile": vProfile.name, "hint": "set cleanup or loudness in the profile"},
          )
        vPlan = EditPlan(
          source=vProbe.label,
          source_path=vProbe.path,
          source_sha256=vProbe.sha256,
          source_duration=vProbe.duration_seconds,
          sample_rate=vProbe.sample_rate,
          channels=vProbe.channels,
          provenance=PlanProvenance(profile=vProfile.name),
          operations=lOperations,
          notes="Mastering only; nothing was cut.",
        )
        self.mValidate(vPlan, vProfile, vProbe).mRaiseIfInvalid()
        vRequest = ExecutionRequest(
          source=pSource,
          plan=vPlan,
          profile=vProfile,
          work_dir=vWorkDir,
          output_dir=vOutputDir,
          base_name=vBaseName,
          overwrite=pOverwrite,
          noise_floor_db=vAnalysis.dynamics.noise_floor_dbfs,
          measure_before=vAnalysis.loudness,
          formats=lFormats,
          metadata=self._mBuildMetadata(vProbe, vProfile),
          guard=self.guard,
        )
        vOutcome = self.engine.mExecute(vRequest)
      finally:
        shutil.rmtree(vWorkDir, ignore_errors=True)
    vLog.info(
      "Mastered an edited file",
      extra={"source": vProbe.label, "profile": vProfile.name, "formats": lFormats},
    )
    return MasterResult(plan=vPlan, outcome=vOutcome)

  def _mRunPipeline(
    self,
    pSource: Path,
    pProfile: Profile,
    pJob: Job,
    pOptions: ProcessOptions,
    pWorkDir: Path,
    pOutputDir: Path,
    pBaseName: str,
    pPhaseSeconds: dict[str, float],
    pProgress: ProgressCallback | None,
  ) -> ProcessResult:
    vStartedAt = time.monotonic()
    lWarnings: list[str] = []
    dArtifacts: dict[str, Path] = {}

    # ------------------------------------------------------------- probing --
    pJob.mSetState(JobState.PROBING, "Probing the source file")
    self._mReport(pProgress, pJob, "probing")
    self.jobs.mSave(pJob)
    vStart = time.monotonic()
    vProbe = self.mProbe(pSource)
    pPhaseSeconds["probing"] = time.monotonic() - vStart
    pJob.source_duration_seconds = vProbe.duration_seconds
    pJob.mAddMessage(
      f"{vProbe.duration_seconds:.1f}s, {vProbe.sample_rate} Hz, {vProbe.channels} ch, {vProbe.codec_name}"
    )

    # ----------------------------------------------------------- analysing --
    self._mCheckCancelled(pOptions, pJob)
    pJob.mSetState(JobState.ANALYZING, "Analysing the audio")
    self._mReport(pProgress, pJob, "analyzing")
    self.jobs.mSave(pJob)
    vStart = time.monotonic()
    vAnalysis = self.mAnalyze(pSource, pProfile, pWorkDir, vProbe)
    pPhaseSeconds["analyzing"] = time.monotonic() - vStart
    lWarnings.extend(vAnalysis.warnings)
    vEnvelope = self.mEnvelope(pSource, pWorkDir)

    # -------------------------------------------------------- transcribing --
    vTranscript: Transcript | None = None
    if pOptions.transcribe and pProfile.transcription.backend != "none":
      self._mCheckCancelled(pOptions, pJob)
      pJob.mSetState(JobState.TRANSCRIBING, "Transcribing")
      self._mReport(pProgress, pJob, "transcribing")
      self.jobs.mSave(pJob)
      vStart = time.monotonic()
      try:
        vTranscript = self.mTranscribe(pSource, pProfile, vProbe.sha256)
        pJob.mAddMessage(
          f"Transcribed {len(vTranscript.segments)} segments in {vTranscript.language}"
        )
      except TranscriptionError as vError:
        lWarnings.append(f"Transcription unavailable: {vError.message}")
        pJob.mAddMessage(f"Transcription skipped: {vError.message}", "warning", vError.details)
      pPhaseSeconds["transcribing"] = time.monotonic() - vStart

    # ------------------------------------------------------------ planning --
    self._mCheckCancelled(pOptions, pJob)
    pJob.mSetState(JobState.PLANNING, "Building the edit plan")
    self._mReport(pProgress, pJob, "planning")
    self.jobs.mSave(pJob)
    vStart = time.monotonic()
    vCandidates = self.mFindCandidates(vAnalysis, pProfile, vTranscript)
    lWarnings.extend(vCandidates.warnings)
    vCandidates, dDecisions, lRequestWarnings = self.mAddRequestedCuts(
      vCandidates, vTranscript, pOptions.extra_cuts, pOptions.decisions,
    )
    lWarnings.extend(lRequestWarnings)
    vPlan = pOptions.plan
    if vPlan is None:
      vPlan = self.mCreatePlan(
        vAnalysis, pProfile, vCandidates, vTranscript, vEnvelope, dDecisions, pOptions.apply_proposals,
      )
      vPlan, lGapWarnings = self.mListenToGaps(vPlan, pSource, pProfile, vEnvelope, vTranscript)
      lWarnings.extend(lGapWarnings)
      if pProfile.cuts.verify_joins and vTranscript is not None:
        self._mCheckCancelled(pOptions, pJob)
        pJob.mAddMessage("Checking the joins")
        vPlan, lChecks, lJoinWarnings = self.mVerifyJoins(vPlan, pSource, vTranscript, pProfile)
        lWarnings.extend(lJoinWarnings)
        pJob.mAddMessage(
          f"Checked {len(lChecks)} joins, {sum(1 for vCheck in lChecks if vCheck.adjusted)} adjusted"
        )
    elif vPlan.source_sha256 and vPlan.source_sha256 != vProbe.sha256:
      raise EditPlanError(
        "The supplied plan was generated for a different file",
        {"plan_sha256": vPlan.source_sha256[:16], "file_sha256": vProbe.sha256[:16]},
      )
    if not pOptions.mastering:
      vPlan = vPlan.mWithoutMastering()
    pPhaseSeconds["planning"] = time.monotonic() - vStart
    pJob.plan_id = vPlan.id
    pJob.mAddMessage(f"Plan has {len(vPlan.operations)} operations")

    # ---------------------------------------------------------- validating --
    self._mCheckCancelled(pOptions, pJob)
    pJob.mSetState(JobState.VALIDATING, "Validating the edit plan")
    self._mReport(pProgress, pJob, "validating")
    self.jobs.mSave(pJob)
    vStart = time.monotonic()
    vValidation = self.mValidate(vPlan, pProfile, vProbe)
    pPhaseSeconds["validating"] = time.monotonic() - vStart
    for vIssue in vValidation.warnings:
      lWarnings.append(f"{vIssue.code}: {vIssue.message}")
    vSimulation = self.mSimulate(vPlan, pProfile, vProbe)

    if pOptions.dry_run:
      pOutputDir.mkdir(parents=True, exist_ok=True)
      dArtifacts.update(
        self._mWriteAnalysisArtifacts(
          pOutputDir, pBaseName, vAnalysis, vPlan, vTranscript, vCandidates, pProfile, pSource, pOptions.overwrite,
        )
      )
      vSimulationPath = self._mResolveArtifact(
        pOutputDir / f"{pBaseName}.simulation.json", pSource, pOptions.overwrite,
      )
      fWriteJsonAtomic(vSimulationPath, vSimulation.model_dump(mode="json"), pReplace=pOptions.overwrite)
      dArtifacts["simulation"] = vSimulationPath
      self._mRecordArtifacts(pJob, dArtifacts)
      return ProcessResult(
        job=pJob,
        transcript=vTranscript,
        plan=vPlan,
        simulation=vSimulation,
        artifacts=dArtifacts,
        warnings=lWarnings,
      )

    vValidation.mRaiseIfInvalid()

    # ------------------------------------------------------------- editing --
    self._mCheckCancelled(pOptions, pJob)
    pJob.mSetState(JobState.EDITING, "Rendering the edit")
    self._mReport(pProgress, pJob, "editing")
    self.jobs.mSave(pJob)
    pOutputDir.mkdir(parents=True, exist_ok=True)

    def fEngineProgress(pPhase: str, pFraction: float) -> None:
      vBase = 0.80
      pJob.mSetProgress(vBase + pFraction * 0.15)
      if pPhase == "mastering":
        pJob.state = JobState.MASTERING
      elif pPhase == "exporting":
        pJob.state = JobState.EXPORTING
      self._mReport(pProgress, pJob, pPhase)

    vRequest = ExecutionRequest(
      source=pSource,
      plan=vPlan,
      profile=pProfile,
      work_dir=pWorkDir,
      output_dir=pOutputDir,
      base_name=pBaseName,
      overwrite=pOptions.overwrite,
      keep_intermediates=True,
      noise_floor_db=vAnalysis.dynamics.noise_floor_dbfs,
      measure_before=vAnalysis.loudness,
      formats=self._mFormats(pProfile, pOptions),
      metadata=self._mBuildMetadata(vProbe, pProfile),
      guard=self.guard,
      master_suffix=pOptions.mastering,
    )
    vStart = time.monotonic()
    vOutcome = self.engine.mExecute(vRequest, fEngineProgress)
    pPhaseSeconds["editing"] = time.monotonic() - vStart
    lWarnings.extend(vOutcome.warnings)
    for vFormat, vPath in vOutcome.exports.items():
      dArtifacts[vFormat] = vPath

    # ------------------------------------------------------------ artifacts --
    pJob.mSetState(JobState.EXPORTING, "Writing artifacts")
    self._mReport(pProgress, pJob, "exporting")
    vDelivered: Transcript | None = None if pOptions.export_transcript else vTranscript
    if pOptions.export_transcript and pProfile.transcription.backend != "none" and pProfile.output.transcript_formats:
      self._mCheckCancelled(pOptions, pJob)
      vStart = time.monotonic()
      vDelivered, lTranscriptWarnings = self._mDeliverableTranscript(vOutcome, pProfile, vTranscript)
      lWarnings.extend(lTranscriptWarnings)
      pPhaseSeconds["transcribing_master"] = time.monotonic() - vStart
    dArtifacts.update(
      self._mWriteAnalysisArtifacts(
        pOutputDir, pBaseName, vAnalysis, vPlan, vDelivered, vCandidates, pProfile, pSource, pOptions.overwrite,
      )
    )

    # ------------------------------------------------------------- Audacity --
    vWantProject = pOptions.audacity_project
    if vWantProject is None:
      vWantProject = pProfile.output.audacity_project
    vAudacityStatus = "not requested"
    vProjectPath: Path | None = None
    if vWantProject:
      vProjectPath, vAudacityStatus, lProjectWarnings = self._mCreateAudacityProject(
        vOutcome, pSource, pOutputDir, pBaseName, vPlan, pProfile, pOptions.overwrite,
      )
      lWarnings.extend(lProjectWarnings)
      if vProjectPath is not None:
        dArtifacts["aup3"] = vProjectPath

    # --------------------------------------------------------------- report --
    vTotalSeconds = time.monotonic() - vStartedAt
    vReport = self._mBuildReport(
      pJob, vProbe, vPlan, pProfile, vOutcome, vTranscript, vAnalysis,
      dArtifacts, lWarnings, pPhaseSeconds, vTotalSeconds, vAudacityStatus, vProjectPath,
    )
    pJob.report = vReport
    self._mRecordArtifacts(pJob, dArtifacts)

    return ProcessResult(
      job=pJob,
      transcript=vTranscript,
      plan=vPlan,
      simulation=vSimulation,
      report=vReport,
      artifacts=dArtifacts,
      warnings=lWarnings,
    )

  # ------------------------------------------------------------- artifacts --

  def _mDeliverableTranscript(
    self,
    pOutcome: ExecutionOutcome,
    pProfile: Profile,
    pSourceTranscript: Transcript | None,
  ) -> tuple[Transcript | None, list[str]]:
    """The transcript delivered with the episode, timed to the rendered audio.

    Preferably a fresh transcription of the edited master, by stretch of
    speech: its timestamps are the listener's, and every stretch is heard on
    its own. A whole-file pass reads a little better but skips text - on a 41
    minute episode it dropped ten seconds of plain speech when it jumped from
    one 30 s window to the next. Otherwise the source transcript mapped
    through the edit.
    """
    lWarnings: list[str] = []
    if pProfile.output.transcript_source == "master":
      vMaster = pOutcome.exports.get("wav") or pOutcome.master
      try:
        vSettings = pProfile.transcription.model_copy(update={"chunking": "speech"})
        vProfile = pProfile.model_copy(update={"transcription": vSettings})
        vTranscript = self.mTranscribe(vMaster, vProfile, fHashFile(vMaster))
        return (vTranscript, lWarnings)
      except PodcastEditorMcpServerError as vError:
        lWarnings.append(f"Could not transcribe the edited master ({vError.message}); the source transcript was mapped instead")
    if pSourceTranscript is None:
      return (None, lWarnings)
    return (fRemapTranscript(pSourceTranscript, pOutcome.timeline), lWarnings)

  def _mFinalizeResult(
    self, pResult: ProcessResult, pSource: Path, pOutputDir: Path, pBaseName: str,
    pOptions: ProcessOptions, pSourceIntact: bool,
  ) -> None:
    """Publish the report and terminal state only after source verification."""
    vJob = pResult.job
    if pResult.report is not None:
      vReportPath = self._mResolveArtifact(
        pOutputDir / f"{pBaseName}.edit-report.json", pSource, pOptions.overwrite,
      )
      fWriteJsonAtomic(vReportPath, pResult.report.model_dump(mode="json"), pReplace=pOptions.overwrite)
      pResult.artifacts["report"] = vReportPath
      vJob.report_path = str(vReportPath)
      vJob.metrics = pResult.report.mSummary()
      self._mRecordArtifacts(vJob, {"report": vReportPath})
    if pSourceIntact:
      vMessage = "Dry run finished; no audio was written" if pOptions.dry_run else "Finished"
      vJob.mSetState(JobState.COMPLETED, vMessage)
    if not pOptions.dry_run:
      vJobCopy = self._mResolveArtifact(pOutputDir / f"{pBaseName}.job.json", pSource, pOptions.overwrite)
      fWriteJsonAtomic(vJobCopy, vJob.model_dump(mode="json"), pReplace=pOptions.overwrite)
      pResult.artifacts["job"] = vJobCopy

  def _mWriteAnalysisArtifacts(
    self,
    pOutputDir: Path,
    pBaseName: str,
    pAnalysis: AnalysisReport,
    pPlan: EditPlan,
    pTranscript: Transcript | None,
    pCandidates: CandidateSet,
    pProfile: Profile,
    pSource: Path,
    pOverwrite: bool,
  ) -> dict[str, Path]:
    """Write the JSON/TXT artifacts that describe the edit."""
    dArtifacts: dict[str, Path] = {}

    def fTarget(pSuffix: str) -> Path:
      return self._mResolveArtifact(pOutputDir / f"{pBaseName}.{pSuffix}", pSource, pOverwrite)

    vAnalysisPath = fTarget("analysis.json")
    fWriteJsonAtomic(vAnalysisPath, pAnalysis.model_dump(mode="json"), pReplace=pOverwrite)
    dArtifacts["analysis"] = vAnalysisPath

    vPlanPath = fTarget("edit-plan.json")
    fWriteJsonAtomic(vPlanPath, pPlan.model_dump(mode="json"), pReplace=pOverwrite)
    dArtifacts["plan"] = vPlanPath

    vCandidatePath = fTarget("candidates.json")
    fWriteJsonAtomic(vCandidatePath, pCandidates.model_dump(mode="json"), pReplace=pOverwrite)
    dArtifacts["candidates"] = vCandidatePath

    if pTranscript is not None:
      for vFormat in pProfile.output.transcript_formats:
        if vFormat == "json":
          vPath = fTarget("transcript.json")
          fWriteJsonAtomic(vPath, pTranscript.model_dump(mode="json"), pReplace=pOverwrite)
        elif vFormat == "txt":
          vPath = fTarget("transcript.txt")
          fWriteTextAtomic(vPath, fRender(pTranscript, "txt"), pReplace=pOverwrite)
        elif vFormat == "srt":
          vPath = fTarget("srt")
          fWriteTextAtomic(vPath, fRender(pTranscript, "srt"), pReplace=pOverwrite)
        elif vFormat == "vtt":
          vPath = fTarget("vtt")
          fWriteTextAtomic(vPath, fRender(pTranscript, "vtt"), pReplace=pOverwrite)
        else:  # pragma: no cover - the schema forbids anything else
          continue
        dArtifacts[f"transcript_{vFormat}"] = vPath
      vLabels = fTarget("labels.txt")
      fWriteTextAtomic(vLabels, fRender(pTranscript, "labels"), pReplace=pOverwrite)
      dArtifacts["labels"] = vLabels

    return dArtifacts

  def _mRecordArtifacts(self, pJob: Job, pArtifacts: dict[str, Path]) -> None:
    for vKind, vPath in pArtifacts.items():
      if not vPath.exists():
        continue
      vDuration: float | None = None
      if vPath.suffix.lower() in cAudioWriteExtensions:
        try:
          vDuration = self.ffmpeg.mDuration(vPath)
        except PodcastEditorMcpServerError:
          vDuration = None
      pJob.mAddArtifact(
        ArtifactRecord(
          kind=vKind,
          path=str(vPath),
          label=self.guard.mLabel(vPath),
          size_bytes=vPath.stat().st_size,
          sha256=fHashFile(vPath),
          duration_seconds=vDuration,
          media_type=_fMediaType(vPath),
        )
      )

  def _mBuildMetadata(self, pProbe: AudioProbe, pProfile: Profile) -> dict[str, str]:
    dMetadata: dict[str, str] = {
      "encoded_by": f"{cProductName} {cVersion}",
      "comment": f"Edited with {cProductName} using profile '{pProfile.name}'",
    }
    for vKey in ("title", "artist", "album", "date"):
      if vKey in pProbe.tags:
        dMetadata[vKey] = pProbe.tags[vKey]
    return dMetadata

  # -------------------------------------------------------------- Audacity --

  def _mCreateAudacityProject(
    self,
    pOutcome: ExecutionOutcome,
    pSource: Path,
    pOutputDir: Path,
    pBaseName: str,
    pPlan: EditPlan,
    pProfile: Profile,
    pOverwrite: bool = False,
  ) -> tuple[Path | None, str, list[str]]:
    """Build the ``.aup3`` project when Audacity can be reached."""
    lWarnings: list[str] = []
    vBackend = AudacityBackend(self.context.config.audacity)
    vStatus = vBackend.mStatus()
    if not vStatus.available:
      vMessage = vStatus.reason or "Audacity is unavailable"
      lWarnings.append(f"Audacity project not created: {vMessage}")
      return (None, f"unavailable: {vMessage}", lWarnings)

    vProject = self._mResolveArtifact(pOutputDir / f"{pBaseName}.aup3", pSource, pOverwrite)
    vMode = self.context.config.audacity.project_mode
    lRemovals: list[tuple[float, float]] | None = None
    if vMode == "replay":
      vTimeline = fBuildTimeline(pPlan, pProfile.cuts.crossfade_ms / 1000.0)
      if vTimeline.silence_segments:
        lWarnings.append("Replay mode cannot reproduce inserted silence; the project uses the mastered audio")
        vMode = "master"
      else:
        lRemovals = fInvertKeepSegments(vTimeline.keep_segments, vTimeline.source_duration)

    try:
      vResult = vBackend.mCreateProject(
        pAudio=pOutcome.master,
        pProject=vProject,
        pMode=vMode,
        pSourceAudio=pSource,
        pRemovals=lRemovals,
      )
    except PodcastEditorMcpServerError as vError:
      lWarnings.append(f"Audacity project not created: {vError.message}")
      return (None, f"failed: {vError.message}", lWarnings)

    lWarnings.extend(vResult.warnings)
    return (vResult.project, f"created ({vResult.mode} mode)", lWarnings)

  # --------------------------------------------------------------- reports --

  def _mBuildReport(
    self,
    pJob: Job,
    pProbe: AudioProbe,
    pPlan: EditPlan,
    pProfile: Profile,
    pOutcome: ExecutionOutcome,
    pTranscript: Transcript | None,
    pAnalysis: AnalysisReport,
    pArtifacts: dict[str, Path],
    pWarnings: list[str],
    pPhaseSeconds: dict[str, float],
    pTotalSeconds: float,
    pAudacityStatus: str,
    pProject: Path | None,
  ) -> EditReport:
    dByReason = pPlan.mCountByReason()
    dByType = pPlan.mCountByType()
    vCounts = EditCounts(
      total_operations=len(pPlan.operations),
      structural_operations=len(pPlan.mStructuralOperations()),
      mastering_operations=len(pPlan.mMasteringOperations()),
      cuts=dByType.get("delete_range", 0),
      shortened_silences=dByType.get("shorten_silence", 0),
      removed_fillers=dByReason.get("filler_word", 0),
      removed_repetitions=dByReason.get("repetition", 0),
      removed_false_starts=dByReason.get("false_start", 0),
      inserted_silences=dByType.get("insert_silence", 0),
      by_type=dByType,
      by_reason=dByReason,
    )

    vLoudness = LoudnessBeforeAfter(
      integrated_lufs_before=pOutcome.loudness_before.integrated_lufs if pOutcome.loudness_before else None,
      integrated_lufs_after=pOutcome.loudness_after.integrated_lufs if pOutcome.loudness_after else None,
      true_peak_dbfs_before=pOutcome.loudness_before.true_peak_dbfs if pOutcome.loudness_before else None,
      true_peak_dbfs_after=pOutcome.loudness_after.true_peak_dbfs if pOutcome.loudness_after else None,
      sample_peak_dbfs_before=pAnalysis.dynamics.peak_dbfs,
      loudness_range_before=pOutcome.loudness_before.loudness_range if pOutcome.loudness_before else None,
      loudness_range_after=pOutcome.loudness_after.loudness_range if pOutcome.loudness_after else None,
      target_lufs=pProfile.loudness.integrated_lufs,
      target_true_peak_dbtp=pProfile.loudness.true_peak_dbtp,
    )

    lArtifacts: list[ArtifactRecord] = []
    for vKind, vPath in pArtifacts.items():
      if not vPath.exists():
        continue
      lArtifacts.append(
        ArtifactRecord(
          kind=vKind,
          path=str(vPath),
          label=self.guard.mLabel(vPath),
          size_bytes=vPath.stat().st_size,
          sha256=fHashFile(vPath),
          media_type=_fMediaType(vPath),
        )
      )

    dTools = dict(pAnalysis.tool_versions)
    if pProject is not None:
      vAudacityVersion = AudacityBackend(self.context.config.audacity).mVersion()
      if vAudacityVersion:
        dTools["audacity"] = vAudacityVersion

    vRemoved = pOutcome.timeline.removed_seconds
    vFraction = pOutcome.timeline.removed_fraction

    return EditReport(
      generated_at=dt.datetime.now(dt.UTC).isoformat(),
      job_id=pJob.id,
      plan_id=pPlan.id,
      profile=pProfile.name,
      source_path=str(pProbe.path),
      source_label=pProbe.label,
      source_sha256=pProbe.sha256,
      source_duration_seconds=pProbe.duration_seconds,
      output_duration_seconds=pOutcome.output_duration,
      removed_seconds=vRemoved,
      inserted_seconds=pOutcome.timeline.inserted_seconds,
      removed_fraction=vFraction,
      counts=vCounts,
      loudness=vLoudness,
      transcription_backend=pTranscript.backend if pTranscript else None,
      transcription_model=pTranscript.model if pTranscript else None,
      transcription_device=pTranscript.device if pTranscript else None,
      transcription_language=pTranscript.language if pTranscript else None,
      vad_backend=pAnalysis.vad_backend,
      edit_backend=pOutcome.backend,
      audacity_project=str(pProject) if pProject else None,
      audacity_status=pAudacityStatus,
      phases=[PhaseTiming(phase=vName, seconds=vSeconds) for vName, vSeconds in pPhaseSeconds.items()]
      + [PhaseTiming(phase=vName, seconds=vSeconds) for vName, vSeconds in pOutcome.phase_seconds.items()],
      total_seconds=pTotalSeconds,
      artifacts=lArtifacts,
      tool_versions=dTools,
      warnings=pWarnings,
      source_unchanged=True,
    )

  # ------------------------------------------------------------ safeguards --

  def _mVerifySourceUnchanged(
    self,
    pSource: Path,
    pHashBefore: str,
    pMtimeBefore: int,
    pJob: Job,
    pResult: ProcessResult,
  ) -> bool:
    """Prove the original file was not touched. This is a hard guarantee."""
    try:
      vHashAfter = fHashFile(pSource)
      vMtimeAfter = pSource.stat().st_mtime_ns
    except OSError as vError:
      vMessage = f"Could not re-verify the source file: {vError}"
      vCode = "source_unverifiable"
    else:
      if vHashAfter == pHashBefore and vMtimeAfter == pMtimeBefore:
        return True
      vMessage = "The source file changed during processing"
      vCode = "source_modified"
    pJob.mAddError(vCode, vMessage, {"path": pJob.source_label})
    pJob.mSetState(JobState.FAILED, vMessage)
    if pResult.report is not None:
      pResult.report.source_unchanged = False
      pResult.report.warnings.append(vMessage)
    vLog.error(vMessage, extra={"path": pJob.source_label})
    return False

  def _mCleanWorkDir(
    self, pWorkDir: Path, pLogFile: Path, pOutputDir: Path, pBaseName: str,
    pSource: Path, pOverwrite: bool, pSaveLog: bool,
  ) -> None:
    """Remove the per-job scratch directory, preserving its log."""
    try:
      if pSaveLog and pLogFile.is_file() and pOutputDir.is_dir():
        vTarget = self._mResolveArtifact(pOutputDir / f"{pBaseName}.job.log", pSource, pOverwrite)
        shutil.copy2(pLogFile, vTarget)
    except (OSError, PodcastEditorMcpServerError) as vError:
      vLog.warning("Could not clean the work directory", extra={"path": str(pWorkDir), "reason": str(vError)})
    finally:
      shutil.rmtree(pWorkDir, ignore_errors=True)

  def _mReport(self, pProgress: ProgressCallback | None, pJob: Job, pPhase: str) -> None:
    if pProgress is not None:
      pProgress(pPhase, pJob.progress, pJob.id)

  def _mCheckCancelled(self, pOptions: ProcessOptions, pJob: Job) -> None:
    """Abort between phases when the caller asked for cancellation."""
    if pOptions.cancel_check is None or not pOptions.cancel_check():
      return
    pJob.mSetState(JobState.CANCELLED, "Cancelled by request")
    self.jobs.mSave(pJob)
    raise JobFailure("The job was cancelled", {"job_id": pJob.id, "state": str(pJob.state)})

  # ---------------------------------------------------------- plan storage --

  def mSavePlan(self, pPlan: EditPlan, pPath: Path | None = None) -> Path:
    """Persist a plan inside the workspace."""
    vTarget = pPath
    if vTarget is None:
      vTarget = self.context.workspace.work / "plans" / f"{pPlan.id}.edit-plan.json"
    vTarget = self.guard.mResolveOutput(vTarget, "work", cDataExtensions, pOverwrite=True)
    fWriteJsonAtomic(vTarget, pPlan.model_dump(mode="json"))
    return vTarget

  def mLoadPlan(self, pPath: str | Path) -> EditPlan:
    """Load a plan from inside the workspace."""
    vPath = self.guard.mResolve(pPath, pDefaultRole="output", pRequireExists=True, pAllowedExtensions=frozenset({".json"}))
    try:
      return EditPlan.model_validate_json(vPath.read_text(encoding="utf-8"))
    except (ValueError, OSError) as vError:
      raise EditPlanError(
        "The file is not a valid edit plan",
        {"path": self.guard.mLabel(vPath), "reason": str(vError)[:400]},
      ) from vError

  def mFindPlanForJob(self, pJobId: str) -> EditPlan | None:
    """Return the plan a completed job used, if its artifact still exists."""
    vJob = self.jobs.mLoad(pJobId)
    for vArtifact in vJob.artifacts:
      if vArtifact.kind == "plan":
        vPath = Path(vArtifact.path)
        if vPath.is_file():
          try:
            return EditPlan.model_validate_json(vPath.read_text(encoding="utf-8"))
          except (ValueError, OSError):
            return None
    return None


cMediaTypes: dict[str, str] = {
  ".wav": "audio/wav",
  ".mp3": "audio/mpeg",
  ".flac": "audio/flac",
  ".ogg": "audio/ogg",
  ".opus": "audio/opus",
  ".json": "application/json",
  ".txt": "text/plain",
  ".srt": "application/x-subrip",
  ".vtt": "text/vtt",
  ".log": "text/plain",
  ".aup3": "application/x-audacity-project",
}


def _fMediaType(pPath: Path) -> str | None:
  return cMediaTypes.get(pPath.suffix.lower())
