"""Execution engine: a validated plan becomes audio.

The engine owns the *order* of operations, which the plan deliberately does
not. Rendering is split into stages so each one is independently verifiable and
so the two-pass loudness measurement happens on exactly the signal it will be
applied to:

  prepare -> structural -> clean-up/dynamics -> loudness + limiter + fades -> export

Every intermediate lives in the per-job work directory. The source file is
opened read-only and never written to.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from podcast_editor_mcp_server.backends.base import RenderRequest, RenderResult
from podcast_editor_mcp_server.backends.ffmpeg import FfmpegBackend
from podcast_editor_mcp_server.config import Profile
from podcast_editor_mcp_server.errors import ExecutionError, ExportError
from podcast_editor_mcp_server.logging_setup import fGetLogger
from podcast_editor_mcp_server.models.audio import LoudnessMeasurement
from podcast_editor_mcp_server.models.edit_plan import (
  ApplyGain,
  Compress,
  Denoise,
  EditPlan,
  FadeIn,
  FadeOut,
  HighPassFilter,
  LimitTruePeak,
  LowPassFilter,
  NormalizeLoudness,
)
from podcast_editor_mcp_server.security.paths import PathGuard, cAudioWriteExtensions
from podcast_editor_mcp_server.util.timeline import Timeline, fBuildTimeline

vLog = fGetLogger("execution")

# A measured true peak is allowed to sit this far above the requested ceiling
# before a corrective gain pass runs.
cTruePeakToleranceDb: float = 0.05
# A master this close to the loudness target is left as it is. Further from it,
# up to two corrective renders run; a gap wider than the cap is not a loudnorm
# shortfall but something else (silence, a broken measurement), and is left.
cLoudnessToleranceLu: float = 0.2
cLoudnessCorrectionPasses: int = 2
cMaxLoudnessCorrectionDb: float = 6.0
cStageStructural: str = "structural.wav"
cStagePrepared: str = "prepared.wav"
cStageCleaned: str = "cleaned.wav"
cStageMastered: str = "mastered.wav"
cStagePeakCorrected: str = "peak-corrected.wav"

ProgressCallback = Callable[[str, float], None]


@dataclass(slots=True)
class ExecutionRequest:
  """Inputs for one render."""

  source: Path
  plan: EditPlan
  profile: Profile
  work_dir: Path
  output_dir: Path
  base_name: str
  overwrite: bool = False
  keep_intermediates: bool = False
  noise_floor_db: float | None = None
  measure_before: LoudnessMeasurement | None = None
  formats: list[str] | None = None
  metadata: dict[str, str] = field(default_factory=dict)
  guard: PathGuard | None = None
  # The WAV is named "<base>.master.wav"; False names it "<base>.wav", for an
  # edit rendered without its mastering chain.
  master_suffix: bool = True


@dataclass(slots=True)
class ExecutionOutcome:
  """What a render produced."""

  master: Path
  exports: dict[str, Path]
  timeline: Timeline
  output_duration: float
  applied_operations: list[str]
  loudness_before: LoudnessMeasurement | None
  loudness_after: LoudnessMeasurement | None
  backend: str
  warnings: list[str] = field(default_factory=list)
  phase_seconds: dict[str, float] = field(default_factory=dict)


class ExecutionEngine:
  """Renders validated EditPlans."""

  def __init__(self, pFfmpeg: FfmpegBackend | None = None) -> None:
    self.ffmpeg = pFfmpeg if pFfmpeg is not None else FfmpegBackend()

  def mExecute(self, pRequest: ExecutionRequest, pProgress: ProgressCallback | None = None) -> ExecutionOutcome:
    """Render ``pRequest.plan`` into the output directory."""
    vProfile = pRequest.profile
    pRequest.work_dir.mkdir(parents=True, exist_ok=True)
    pRequest.output_dir.mkdir(parents=True, exist_ok=True)

    lWarnings: list[str] = []
    dPhases: dict[str, float] = {}

    vSampleRate = vProfile.output.sample_rate
    vChannels = vProfile.output.channels or pRequest.plan.channels or 1

    vTimeline = fBuildTimeline(pRequest.plan, vProfile.cuts.crossfade_ms / 1000.0)
    if vTimeline.is_empty:
      raise ExecutionError("The plan leaves no audio at all", {"plan_id": pRequest.plan.id})

    # ------------------------------------------------------------- prepare --
    _fReport(pProgress, "prepare", 0.05)
    vStart = time.monotonic()
    vPrepared = self.ffmpeg.mPrepare(
      pRequest.source,
      pRequest.work_dir / cStagePrepared,
      vSampleRate,
      vChannels,
    )
    dPhases["prepare"] = time.monotonic() - vStart

    vDelta = abs(vPrepared.duration_seconds - pRequest.plan.source_duration)
    if vDelta > 0.05:
      lWarnings.append(
        f"The decoded working copy is {vPrepared.duration_seconds:.3f}s but the plan expects "
        f"{pRequest.plan.source_duration:.3f}s"
      )


    # ---------------------------------------------------------- structural --
    _fReport(pProgress, "editing", 0.25)
    vStart = time.monotonic()
    vRenderRequest = RenderRequest(
      source=vPrepared.path,
      plan=pRequest.plan,
      timeline=vTimeline,
      work_dir=pRequest.work_dir,
      target=pRequest.work_dir / cStageStructural,
      sample_rate=vSampleRate,
      channels=vChannels,
    )
    vRender: RenderResult = self.ffmpeg.mRenderStructural(vRenderRequest)
    dPhases["editing"] = time.monotonic() - vStart

    # ------------------------------------------------------------ mastering --
    _fReport(pProgress, "mastering", 0.55)
    vStart = time.monotonic()
    vMastered, lApplied, lMasterWarnings = self._mApplyMastering(
      vRender.path,
      pRequest,
      vSampleRate,
      vChannels,
    )
    dPhases["mastering"] = time.monotonic() - vStart
    lWarnings.extend(lMasterWarnings)

    vMastered, lPeakWarnings = self._mEnforceTruePeak(
      vMastered, pRequest, vSampleRate, vChannels
    )
    lWarnings.extend(lPeakWarnings)

    # --------------------------------------------------------------- export --
    _fReport(pProgress, "exporting", 0.85)
    vStart = time.monotonic()
    dExports = self._mExportAll(vMastered, pRequest, vSampleRate, vChannels)
    dPhases["exporting"] = time.monotonic() - vStart

    vStart = time.monotonic()
    vLoudnessAfter = self.ffmpeg.mMeasureEbur128(vMastered)
    dPhases["measure_after"] = time.monotonic() - vStart

    vOutputDuration = self.ffmpeg.mDuration(vMastered)
    vExpected = vTimeline.output_duration
    if abs(vOutputDuration - vExpected) > 0.25:
      lWarnings.append(
        f"The rendered duration ({vOutputDuration:.3f}s) differs from the simulated duration ({vExpected:.3f}s)"
      )

    # The master is always a delivered artifact, never a work-directory stage
    # file, because those are deleted a few lines below.
    vMaster = dExports.get("wav") or dExports.get("flac")
    if vMaster is None:
      vMaster = next(iter(dExports.values()))
    if not pRequest.keep_intermediates:
      self._mCleanIntermediates(pRequest.work_dir)

    _fReport(pProgress, "done", 1.0)
    return ExecutionOutcome(
      master=vMaster,
      exports=dExports,
      timeline=vTimeline,
      output_duration=vOutputDuration,
      applied_operations=list(vRender.applied_operations) + lApplied,
      loudness_before=pRequest.measure_before,
      loudness_after=vLoudnessAfter,
      backend=self.ffmpeg.name,
      warnings=lWarnings,
      phase_seconds=dPhases,
    )

  # ------------------------------------------------------------- mastering --

  def _mApplyMastering(
    self,
    pSource: Path,
    pRequest: ExecutionRequest,
    pSampleRate: int,
    pChannels: int,
  ) -> tuple[Path, list[str], list[str]]:
    # isinstance dispatch, not a comparison on ``type``: the type checker then
    # proves every branch reads fields the operation really has.
    """Apply the mastering chain in the engine's fixed order."""
    lApplied: list[str] = []
    lWarnings: list[str] = []
    lMastering = pRequest.plan.mMasteringOperations()
    if not lMastering:
      return (pSource, lApplied, lWarnings)

    # Stage A: spectral clean-up and dynamics, everything that changes the
    # signal before its loudness is measured.
    lStageA: list[str] = []
    lNormalise: list[NormalizeLoudness] = []
    lLimit: list[LimitTruePeak] = []
    lFades: list[FadeIn | FadeOut] = []

    for vOperation in lMastering:
      if isinstance(vOperation, HighPassFilter):
        lStageA.append(self.ffmpeg.mBuildHighPass(vOperation.frequency, vOperation.poles))
        lApplied.append(vOperation.id)
      elif isinstance(vOperation, LowPassFilter):
        lStageA.append(self.ffmpeg.mBuildLowPass(vOperation.frequency, vOperation.poles))
        lApplied.append(vOperation.id)
      elif isinstance(vOperation, Denoise):
        vFloor = vOperation.noise_floor_db if vOperation.noise_floor_db is not None else pRequest.noise_floor_db
        lStageA.append(self.ffmpeg.mBuildDenoise(vOperation.reduction_db, vFloor))
        lApplied.append(vOperation.id)
      elif isinstance(vOperation, Compress):
        lStageA.append(
          self.ffmpeg.mBuildCompressor(
            vOperation.threshold_db,
            vOperation.ratio,
            vOperation.attack_ms,
            vOperation.release_ms,
            vOperation.makeup_db,
            vOperation.knee_db,
          )
        )
        lApplied.append(vOperation.id)
      elif isinstance(vOperation, ApplyGain):
        if abs(vOperation.gain_db) >= 1e-6:
          lStageA.append(self.ffmpeg.mBuildGain(vOperation.gain_db))
          lApplied.append(vOperation.id)
      elif isinstance(vOperation, NormalizeLoudness):
        lNormalise.append(vOperation)
      elif isinstance(vOperation, LimitTruePeak):
        lLimit.append(vOperation)
      elif isinstance(vOperation, FadeIn | FadeOut):
        lFades.append(vOperation)

    vCurrent = pSource
    if lStageA:
      vCleaned = pRequest.work_dir / cStageCleaned
      self.ffmpeg.mApplyFilters(vCurrent, vCleaned, lStageA, pSampleRate, pChannels, "cleaning up the audio")
      vCurrent = vCleaned

    lStageB: list[str] = []
    if lNormalise:
      vNormalise = lNormalise[0]
      dMeasured: dict[str, float] | None = None
      if vNormalise.dual_pass:
        try:
          dMeasured = self.ffmpeg.mMeasureLoudnorm(
            vCurrent,
            vNormalise.integrated_lufs,
            vNormalise.true_peak_dbtp,
            vNormalise.loudness_range,
          )
        except Exception as vError:  # noqa: BLE001 - fall back to single pass
          lWarnings.append(f"Two-pass loudness measurement failed, using a single pass: {vError}")
      lStageB.append(
        self.ffmpeg.mBuildLoudnorm(
          vNormalise.integrated_lufs,
          vNormalise.true_peak_dbtp,
          vNormalise.loudness_range,
          dMeasured,
        )
      )
      lApplied.append(vNormalise.id)
    # A loudness correction goes right after loudnorm, in front of the limiter.
    vGainIndex = len(lStageB)

    if lLimit:
      vLimiter = lLimit[0]
      lStageB.append(
        self.ffmpeg.mBuildLimiter(vLimiter.true_peak_dbtp, vLimiter.attack_ms, vLimiter.release_ms)
      )
      lApplied.append(vLimiter.id)

    if lFades:
      vDuration = self.ffmpeg.mDuration(vCurrent)
      for vFade in lFades:
        if isinstance(vFade, FadeIn):
          lStageB.append(self.ffmpeg.mBuildFade("in", 0.0, vFade.duration, vFade.curve))
        else:
          vStartTime = max(0.0, vDuration - vFade.duration)
          lStageB.append(self.ffmpeg.mBuildFade("out", vStartTime, vFade.duration, vFade.curve))
        lApplied.append(vFade.id)

    if lStageB:
      vMastered = pRequest.work_dir / cStageMastered
      self.ffmpeg.mApplyFilters(vCurrent, vMastered, lStageB, pSampleRate, pChannels, "mastering")
      if lNormalise:
        self._mCorrectLoudness(
          vCurrent, vMastered, lStageB, vGainIndex, lNormalise[0], bool(lLimit), pSampleRate, pChannels,
        )
      vCurrent = vMastered

    return (vCurrent, lApplied, lWarnings)

  def _mCorrectLoudness(
    self,
    pSource: Path,
    pTarget: Path,
    pStage: list[str],
    pGainIndex: int,
    pNormalise: NormalizeLoudness,
    pLimited: bool,
    pSampleRate: int,
    pChannels: int,
  ) -> None:
    """Close the gap ``loudnorm`` leaves when it falls back to its dynamic mode.

    When the true-peak ceiling forbids the linear gain, ``loudnorm`` switches to
    dynamic normalisation, which can land a full LU short of the target. The
    gap is measured on the render and added as gain in front of the limiter,
    which keeps the peaks under the ceiling. Without a limiter only a master
    that came out too loud is corrected, because raising it would break the
    ceiling and the true-peak check would take the gain back.
    """
    vMakeupDb = 0.0
    for _ in range(cLoudnessCorrectionPasses):
      vMeasured = self.ffmpeg.mMeasureEbur128(pTarget).integrated_lufs
      if vMeasured is None:
        return
      vGap = pNormalise.integrated_lufs - vMeasured
      if abs(vGap) <= cLoudnessToleranceLu or (vGap > 0.0 and not pLimited):
        return
      if abs(vMakeupDb + vGap) > cMaxLoudnessCorrectionDb:
        return
      vMakeupDb += vGap
      vLog.info(
        "Correcting the integrated loudness",
        extra={"measured_lufs": round(vMeasured, 2), "target_lufs": pNormalise.integrated_lufs, "gain_db": round(vMakeupDb, 2)},
      )
      lStage = list(pStage)
      lStage.insert(pGainIndex, self.ffmpeg.mBuildGain(vMakeupDb))
      self.ffmpeg.mApplyFilters(pSource, pTarget, lStage, pSampleRate, pChannels, "correcting the loudness")

  def _mEnforceTruePeak(
    self,
    pSource: Path,
    pRequest: ExecutionRequest,
    pSampleRate: int,
    pChannels: int,
  ) -> tuple[Path, list[str]]:
    """Verify the rendered true peak and pull it down if it overshot.

    The limiter works on sample peaks, so an inter-sample overshoot is
    possible. Rather than hope, the engine measures the result and applies an
    exact corrective gain when the ceiling was missed. Meeting the true-peak
    ceiling takes priority over hitting the loudness target to the last tenth
    of a dB, because the ceiling is what distribution platforms enforce.
    """
    lLimiters = [vItem for vItem in pRequest.plan.operations if isinstance(vItem, LimitTruePeak)]
    lNormalisers = [vItem for vItem in pRequest.plan.operations if isinstance(vItem, NormalizeLoudness)]
    if not lLimiters and not lNormalisers:
      return (pSource, [])
    vCeiling = lLimiters[0].true_peak_dbtp if lLimiters else lNormalisers[0].true_peak_dbtp

    try:
      vMeasured = self.ffmpeg.mMeasureEbur128(pSource)
    except Exception as vError:  # noqa: BLE001 - verification must not fail the job
      return (pSource, [f"Could not verify the true peak: {vError}"])

    vPeak = vMeasured.true_peak_dbfs
    if vPeak is None or vPeak <= vCeiling + cTruePeakToleranceDb:
      return (pSource, [])

    vCorrection = vCeiling - vPeak
    vLog.info(
      "Correcting a true-peak overshoot",
      extra={"measured_dbfs": round(vPeak, 3), "ceiling_dbtp": vCeiling, "gain_db": round(vCorrection, 3)},
    )
    vTarget = pRequest.work_dir / cStagePeakCorrected
    self.ffmpeg.mApplyFilters(
      pSource,
      vTarget,
      [self.ffmpeg.mBuildGain(vCorrection)],
      pSampleRate,
      pChannels,
      "correcting the true peak",
    )
    return (
      vTarget,
      [
        f"The true peak measured {vPeak:.2f} dBFS against a {vCeiling:.2f} dBTP ceiling; "
        f"a corrective gain of {vCorrection:.2f} dB was applied"
      ],
    )

  # ---------------------------------------------------------------- export --

  def _mExportAll(
    self,
    pSource: Path,
    pRequest: ExecutionRequest,
    pSampleRate: int,
    pChannels: int,
  ) -> dict[str, Path]:
    """Encode the mastered audio into every requested delivery format."""
    lFormats = pRequest.formats if pRequest.formats is not None else list(pRequest.profile.output.formats)
    dOutputs: dict[str, Path] = {}
    vOutput = pRequest.profile.output

    for vFormat in lFormats:
      vSuffix = ".wav" if vFormat == "wav" else f".{vFormat}"
      vMasterTag = ".master" if vFormat == "wav" and pRequest.master_suffix else ""
      vName = f"{pRequest.base_name}{vMasterTag}{vSuffix}"
      vTarget = pRequest.output_dir / vName
      if pRequest.guard is not None:
        vTarget = pRequest.guard.mResolveOutput(
          vTarget, pAllowedExtensions=cAudioWriteExtensions, pOverwrite=True, pSource=pRequest.source,
        )
      if vTarget.exists() and not pRequest.overwrite:
        raise ExportError(
          "An artifact with this name already exists",
          {"path": vTarget.name, "hint": "pass overwrite to replace it"},
        )
      self.ffmpeg.mExport(
        pSource,
        vTarget,
        vFormat,
        pSampleRate,
        pChannels,
        pWavCodec=vOutput.wav_codec,
        pMp3BitrateKbps=vOutput.mp3_bitrate_kbps,
        pMp3Mode=vOutput.mp3_mode,
        pMp3VbrQuality=vOutput.mp3_vbr_quality,
        pFlacCompression=vOutput.flac_compression,
        pMetadata=pRequest.metadata or None,
        pOverwrite=pRequest.overwrite,
      )
      dOutputs[vFormat] = vTarget
      vLog.info("Exported artifact", extra={"format": vFormat, "path": vTarget.name})

    return dOutputs

  def _mCleanIntermediates(self, pWorkDir: Path) -> None:
    """Delete stage files, keeping the directory itself for the job record."""
    for vName in (
      cStagePrepared, cStageStructural, cStageCleaned, cStageMastered,
      cStagePeakCorrected,
    ):
      (pWorkDir / vName).unlink(missing_ok=True)
    for vPartial in pWorkDir.glob("structural-part-*.wav"):
      vPartial.unlink(missing_ok=True)
    for vDecoded in pWorkDir.glob("analysis-source*.wav"):
      vDecoded.unlink(missing_ok=True)


def _fReport(pProgress: ProgressCallback | None, pPhase: str, pFraction: float) -> None:
  if pProgress is not None:
    pProgress(pPhase, pFraction)
