"""Deterministic audio analysis.

The point of this module is to leave the language model as little to guess as
possible: loudness, peaks, noise floor, silence and speech boundaries are all
*measured*, not estimated from a transcript.

Division of labour:

* **FFmpeg** does loudness (``ebur128``), overall statistics (``astats``) and
  threshold-based silence detection (``silencedetect``). These are mature,
  fast, and better than anything worth re-implementing here.
* **NumPy** does the things FFmpeg does not expose: the short-term energy
  envelope used to place cuts, and sample-accurate clipping runs.
* **A VAD backend** answers "is anybody speaking", which is a different
  question from "is the signal below -45 dB".
"""

from __future__ import annotations

import datetime as dt
import shutil
import uuid
from pathlib import Path

from podcast_editor_mcp_server.audio.probe import fProbeAudio, fStreamShape
from podcast_editor_mcp_server.audio.samples import (
  Envelope,
  SampleBlock,
  cMaxDecodedBytes,
  fCanReadDirectly,
  fClippedSampleFraction,
  fComputeEnvelope,
  fEstimatePeakNoiseFloorDb,
  fFindClippingRuns,
  fReadSamples,
)
from podcast_editor_mcp_server.backends.ffmpeg import FfmpegBackend
from podcast_editor_mcp_server.config import Profile
from podcast_editor_mcp_server.errors import AnalysisError
from podcast_editor_mcp_server.logging_setup import fGetLogger
from podcast_editor_mcp_server.models.audio import (
  AnalysisReport,
  AudioProbe,
  ClippingRegion,
  DynamicsStatistics,
  LoudnessMeasurement,
  SilenceRegion,
  SpeechRegion,
)
from podcast_editor_mcp_server.security.paths import PathGuard
from podcast_editor_mcp_server.util.hashing import fHashText, fShortHash
from podcast_editor_mcp_server.vad.silero import fCreateVadBackend

vLog = fGetLogger("audio.analysis")

cMinimumThresholdDb: float = -80.0
cMaximumThresholdDb: float = -20.0
cDefaultNoiseFloorDb: float = -60.0
# Background noise peaks this far above its own RMS. Used only when the
# sample-level pass did not run and the peak floor has to be inferred from
# astats' RMS-style figure.
cCrestFactorAllowanceDb: float = 12.0
# Above this duration the sample-level pass is skipped unless explicitly asked
# for, because it needs the whole file in memory as float32.
cMaxSampleAnalysisSeconds: float = 4 * 3600.0
# A decoded copy is only made when it leaves at least this much free space on
# the work disk.
cScratchFreeReserveBytes: int = 2 * 1024**3
# Room for the WAV header and for a duration that ffprobe rounds down.
cDecodeSizeMargin: float = 1.05
cDecodeHeaderBytes: int = 1_048_576


def fResolveSilenceThresholdDb(pProfile: Profile, pPeakNoiseFloorDb: float | None) -> float:
  """Pick the silence threshold, either fixed or relative to the noise floor.

  ``pPeakNoiseFloorDb`` must be a *peak* figure, because that is what
  ``silencedetect`` compares. :func:`fPeakNoiseFloorDb` produces one.
  """
  if pProfile.silence.threshold_mode == "fixed":
    return float(pProfile.silence.threshold_db)
  vFloor = pPeakNoiseFloorDb if pPeakNoiseFloorDb is not None else cDefaultNoiseFloorDb
  vThreshold = vFloor + pProfile.silence.auto_margin_db
  return float(min(max(vThreshold, cMinimumThresholdDb), cMaximumThresholdDb))


def fPeakNoiseFloorDb(pEnvelope: Envelope | None, pRmsNoiseFloorDb: float | None) -> float | None:
  """Best available estimate of the peak noise floor.

  Prefers the measured peak envelope. Falls back to the RMS-style figure plus a
  crest-factor allowance when the sample-level pass did not run.
  """
  if pEnvelope is not None:
    return fEstimatePeakNoiseFloorDb(pEnvelope, 10.0)
  if pRmsNoiseFloorDb is not None:
    return pRmsNoiseFloorDb + cCrestFactorAllowanceDb
  return None


def fSourceIdentity(pPath: Path) -> str:
  """A short key that changes whenever the file at ``pPath`` is rewritten or replaced.

  The change time is part of it because, unlike the modification time, no
  program can set it back.
  """
  vStat = pPath.stat()
  vText = f"{vStat.st_dev}:{vStat.st_ino}:{vStat.st_size}:{vStat.st_mtime_ns}:{vStat.st_ctime_ns}"
  return fShortHash(fHashText(vText), 16)


def fEnsureReadableCopy(pPath: Path, pBackend: FfmpegBackend, pWorkDir: Path) -> Path:
  """Return a path libsndfile can open, decoding once if necessary.

  The decoded copy is named after the identity of the source, so a file
  replaced at the same path is decoded again instead of answering with the
  samples of the old one. It is decoded under a temporary name and renamed
  when complete, so an interrupted decode never passes for a finished one.
  """
  if fCanReadDirectly(pPath):
    return pPath
  pWorkDir.mkdir(parents=True, exist_ok=True)
  vTarget = pWorkDir / f"analysis-source-{fSourceIdentity(pPath)}.wav"
  if vTarget.is_file() and vTarget.stat().st_size > 0:
    return vTarget
  # Copies of earlier versions of the file are of no further use.
  for vStale in pWorkDir.glob("analysis-source-*.wav"):
    vStale.unlink(missing_ok=True)
  vLimit = _fDecodedSizeLimit(pPath, pBackend, pWorkDir)
  vPartial = pWorkDir / f"analysis-source.partial-{uuid.uuid4().hex}.wav"
  try:
    pBackend.mDecode(pPath, vPartial, vLimit)
    if vPartial.stat().st_size >= vLimit:
      raise AnalysisError(
        "The decoded recording is larger than its duration allows",
        {"path": pPath.name, "limit_bytes": vLimit},
      )
    vPartial.replace(vTarget)
  finally:
    vPartial.unlink(missing_ok=True)
  return vTarget


def _fDecodedSizeLimit(pPath: Path, pBackend: FfmpegBackend, pWorkDir: Path) -> int:
  """The most bytes a decoded copy of ``pPath`` may take, refusing when it cannot fit.

  The size is estimated from the stream before decoding: its mono samples
  must fit the sample budget, and the copy must leave
  ``cScratchFreeReserveBytes`` free on the disk. The estimate is also the
  limit FFmpeg writes up to, so a stream that lies about its duration cannot
  fill the disk either.
  """
  vDuration, vSampleRate, vChannels = fStreamShape(pPath, pBackend)
  vChannels = max(1, vChannels or 1)
  if vDuration is None or not vSampleRate or vSampleRate <= 0:
    vLimit = cMaxDecodedBytes * vChannels
  else:
    vMonoBytes = vDuration * vSampleRate * 4
    if vMonoBytes > cMaxDecodedBytes:
      raise AnalysisError(
        "The recording is too long to hold in memory as samples",
        {"path": pPath.name, "decoded_bytes": int(vMonoBytes), "limit_bytes": cMaxDecodedBytes},
      )
    vLimit = int(vMonoBytes * vChannels * cDecodeSizeMargin) + cDecodeHeaderBytes
  vFree = shutil.disk_usage(pWorkDir).free
  if vFree < vLimit + cScratchFreeReserveBytes:
    raise AnalysisError(
      "There is not enough free disk space to decode the recording",
      {"path": pPath.name, "needed_bytes": vLimit, "free_bytes": vFree, "reserve_bytes": cScratchFreeReserveBytes},
    )
  return vLimit


def fAnalyzeAudio(
  pPath: Path,
  pBackend: FfmpegBackend,
  pProfile: Profile,
  pWorkDir: Path,
  pGuard: PathGuard | None = None,
  pProbe: AudioProbe | None = None,
) -> AnalysisReport:
  """Measure everything the planner needs to know about a file."""
  vProbe = pProbe if pProbe is not None else fProbeAudio(pPath, pBackend, pGuard)
  lWarnings: list[str] = []

  vLoudness = _fMeasureLoudness(pPath, pBackend, lWarnings)
  vDynamics = _fMeasureDynamics(pPath, pBackend, lWarnings)

  vBlock: SampleBlock | None = None
  vEnvelope: Envelope | None = None
  lClipping: list[ClippingRegion] = []
  vClippedFraction = 0.0

  if vProbe.duration_seconds > cMaxSampleAnalysisSeconds:
    lWarnings.append(
      f"Sample-level analysis was skipped: the file is longer than {cMaxSampleAnalysisSeconds / 3600:.0f} h"
    )
  else:
    try:
      vReadable = fEnsureReadableCopy(pPath, pBackend, pWorkDir)
      vBlock = fReadSamples(vReadable)
      vEnvelope = fComputeEnvelope(vBlock)
      vClippedFraction = fClippedSampleFraction(vBlock)
      for vStart, vEnd, vCount in fFindClippingRuns(vBlock):
        lClipping.append(
          ClippingRegion(start=vStart, end=vEnd, duration=vEnd - vStart, sample_count=vCount)
        )
    except AnalysisError as vError:
      lWarnings.append(f"Sample-level analysis failed: {vError.message}")

  vNoiseFloor = vDynamics.noise_floor_dbfs
  if vNoiseFloor is None and vEnvelope is not None:
    vNoiseFloor = vEnvelope.mPercentile(10.0)
    vDynamics = vDynamics.model_copy(update={"noise_floor_dbfs": vNoiseFloor})

  vThresholdDb = fResolveSilenceThresholdDb(pProfile, fPeakNoiseFloorDb(vEnvelope, vNoiseFloor))
  lSilences = _fDetectSilences(pPath, pBackend, pProfile, vThresholdDb, lWarnings)

  lSpeech, vVadName = _fDetectSpeech(pPath, pProfile, vBlock, vEnvelope, lWarnings)

  vSpeechSeconds = sum(vRegion.duration for vRegion in lSpeech)
  vSilenceSeconds = sum(vRegion.duration for vRegion in lSilences)
  vLongestSilence = max((vRegion.duration for vRegion in lSilences), default=0.0)
  vSpeechRatio = 0.0
  if vProbe.duration_seconds > 0.0:
    vSpeechRatio = min(1.0, vSpeechSeconds / vProbe.duration_seconds)

  if vLoudness.sample_peak_dbfs is None and vDynamics.peak_dbfs is not None:
    vLoudness = vLoudness.model_copy(update={"sample_peak_dbfs": vDynamics.peak_dbfs})

  if vDynamics.dc_offset is not None and abs(vDynamics.dc_offset) > 0.01:
    lWarnings.append(f"Significant DC offset detected ({vDynamics.dc_offset:+.4f})")
  if lClipping:
    lWarnings.append(f"{len(lClipping)} clipped region(s) detected in the source")

  return AnalysisReport(
    generated_at=dt.datetime.now(dt.UTC).isoformat(),
    tool_versions=pBackend.mToolVersions(),
    probe=vProbe,
    loudness=vLoudness,
    dynamics=vDynamics,
    silence_threshold_db=vThresholdDb,
    silences=lSilences,
    speech_regions=lSpeech,
    clipping=lClipping,
    clipping_sample_fraction=vClippedFraction,
    speech_seconds=vSpeechSeconds,
    silence_seconds=vSilenceSeconds,
    speech_ratio=vSpeechRatio,
    longest_silence_seconds=vLongestSilence,
    vad_backend=vVadName,
    warnings=lWarnings,
  )


def _fMeasureLoudness(pPath: Path, pBackend: FfmpegBackend, pWarnings: list[str]) -> LoudnessMeasurement:
  try:
    return pBackend.mMeasureEbur128(pPath)
  except Exception as vError:  # noqa: BLE001 - measurement must never abort analysis
    vLog.warning("Loudness measurement failed", extra={"path": pPath.name, "reason": str(vError)})
    pWarnings.append(f"Loudness measurement failed: {vError}")
    return LoudnessMeasurement()


def _fMeasureDynamics(pPath: Path, pBackend: FfmpegBackend, pWarnings: list[str]) -> DynamicsStatistics:
  try:
    return pBackend.mMeasureAstats(pPath)
  except Exception as vError:  # noqa: BLE001 - measurement must never abort analysis
    vLog.warning("Statistics measurement failed", extra={"path": pPath.name, "reason": str(vError)})
    pWarnings.append(f"Statistics measurement failed: {vError}")
    return DynamicsStatistics()


def _fDetectSilences(
  pPath: Path,
  pBackend: FfmpegBackend,
  pProfile: Profile,
  pThresholdDb: float,
  pWarnings: list[str],
) -> list[SilenceRegion]:
  vMinimumSeconds = pProfile.silence.minimum_detectable_ms / 1000.0
  try:
    return pBackend.mDetectSilence(pPath, pThresholdDb, vMinimumSeconds)
  except Exception as vError:  # noqa: BLE001 - detection must never abort analysis
    vLog.warning("Silence detection failed", extra={"path": pPath.name, "reason": str(vError)})
    pWarnings.append(f"Silence detection failed: {vError}")
    return []


def _fDetectSpeech(
  pPath: Path,
  pProfile: Profile,
  pBlock: SampleBlock | None,
  pEnvelope: Envelope | None,
  pWarnings: list[str],
) -> tuple[list[SpeechRegion], str]:
  try:
    vBackend = fCreateVadBackend(pProfile.vad.backend)
  except Exception as vError:  # noqa: BLE001 - a VAD problem must not abort analysis
    pWarnings.append(f"Voice activity detection unavailable: {vError}")
    return ([], "none")
  if vBackend is None:
    return ([], "none")

  # The energy backend can reuse samples we already have in memory.
  if vBackend.name == "energy" and pBlock is not None and pEnvelope is not None:
    from podcast_editor_mcp_server.vad.energy import EnergyVadBackend

    if isinstance(vBackend, EnergyVadBackend):
      return (vBackend.mDetectFromEnvelope(pBlock, pEnvelope, pProfile.vad), vBackend.name)

  try:
    return (vBackend.mDetectSpeech(pPath, pProfile.vad), vBackend.name)
  except Exception as vError:  # noqa: BLE001 - fall back rather than fail the job
    vLog.warning("VAD failed", extra={"backend": vBackend.name, "reason": str(vError)})
    pWarnings.append(f"Voice activity detection failed ({vBackend.name}): {vError}")
    if pBlock is not None and pEnvelope is not None:
      from podcast_editor_mcp_server.vad.energy import EnergyVadBackend

      try:
        vFallback = EnergyVadBackend()
        return (vFallback.mDetectFromEnvelope(pBlock, pEnvelope, pProfile.vad), vFallback.name)
      except Exception as vFallbackError:  # noqa: BLE001
        pWarnings.append(f"Energy VAD fallback failed: {vFallbackError}")
    return ([], "none")
