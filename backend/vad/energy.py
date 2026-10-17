"""Energy-based voice activity detection.

Always available: it needs nothing beyond NumPy and libsndfile. The threshold
adapts to the recording's own noise floor, so it works on a quiet studio file
and on a noisy field recording without retuning.

It is less accurate than Silero on breathy or overlapping speech, which is why
it is the fallback rather than the default.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from podcast_editor_mcp_server.audio.samples import Envelope, SampleBlock, fComputeEnvelope, fReadSamples
from podcast_editor_mcp_server.backends.base import BackendStatus
from podcast_editor_mcp_server.config import VadSettings
from podcast_editor_mcp_server.logging_setup import fGetLogger
from podcast_editor_mcp_server.models.audio import SpeechRegion
from podcast_editor_mcp_server.vad.base import VadBackend, fMergeSpeechRegions

vLog = fGetLogger("vad.energy")

# The speech threshold sits this far above the estimated noise floor, but never
# below an absolute gate: a recording that is silent throughout must not have
# its own hiss promoted to "speech".
cNoiseFloorMarginDb: float = 10.0
cAbsoluteGateDb: float = -55.0
cFloorPercentile: float = 15.0
cPeakPercentile: float = 95.0
cMinimumSeparationDb: float = 6.0


class EnergyVadBackend(VadBackend):
  """RMS-envelope voice activity detection."""

  name = "energy"

  def mStatus(self) -> BackendStatus:
    return BackendStatus(name=self.name, available=True, version="builtin")

  def mDetectSpeech(self, pPath: Path, pSettings: VadSettings) -> list[SpeechRegion]:
    vBlock = fReadSamples(pPath)
    vEnvelope = fComputeEnvelope(vBlock)
    return self.mDetectFromEnvelope(vBlock, vEnvelope, pSettings)

  def mDetectFromEnvelope(
    self,
    pBlock: SampleBlock,
    pEnvelope: Envelope,
    pSettings: VadSettings,
  ) -> list[SpeechRegion]:
    """Detect speech from an already-computed envelope (saves a re-read)."""
    if pEnvelope.rms_db.size == 0:
      return []

    vThreshold = fEnergyThresholdDb(pEnvelope)
    aActive = pEnvelope.rms_db >= vThreshold
    if not bool(aActive.any()):
      vLog.debug("Energy VAD found no speech", extra={"threshold_db": round(vThreshold, 2)})
      return []

    lRegions = fRunsToRegions(aActive, pEnvelope, pBlock.duration_seconds)

    vPadSeconds = pSettings.speech_pad_ms / 1000.0
    lPadded: list[SpeechRegion] = []
    for vRegion in lRegions:
      vStart = max(0.0, vRegion.start - vPadSeconds)
      vEnd = min(pBlock.duration_seconds, vRegion.end + vPadSeconds)
      lPadded.append(SpeechRegion(start=vStart, end=vEnd, duration=vEnd - vStart))

    lMerged = fMergeSpeechRegions(lPadded, pSettings.min_silence_duration_ms / 1000.0)
    vMinimumSpeech = pSettings.min_speech_duration_ms / 1000.0
    return [vRegion for vRegion in lMerged if vRegion.duration >= vMinimumSpeech]


def fEnergyThresholdDb(pEnvelope: Envelope) -> float:
  """Choose a speech/non-speech threshold from the envelope's own statistics."""
  vFloor = pEnvelope.mPercentile(cFloorPercentile)
  vPeak = pEnvelope.mPercentile(cPeakPercentile)
  if vPeak - vFloor < cMinimumSeparationDb:
    # Flat material: either constant tone or constant noise. Put the threshold
    # just under the peak so a constant tone still reads as "active".
    return max(cAbsoluteGateDb, vPeak - 3.0)
  vThreshold = vFloor + cNoiseFloorMarginDb
  vCeiling = vPeak - 3.0
  return float(min(max(vThreshold, cAbsoluteGateDb), vCeiling))


def fRunsToRegions(pActive: np.ndarray, pEnvelope: Envelope, pDuration: float) -> list[SpeechRegion]:
  """Convert a boolean activity mask into time regions."""
  aPadded = np.concatenate(([False], pActive, [False]))
  aDiff = np.diff(aPadded.astype(np.int8))
  aStarts = np.flatnonzero(aDiff == 1)
  aEnds = np.flatnonzero(aDiff == -1)
  vHalfFrame = pEnvelope.frame_seconds / 2.0
  lRegions: list[SpeechRegion] = []
  for vStartIndex, vEndIndex in zip(aStarts, aEnds, strict=True):
    vStart = max(0.0, float(pEnvelope.times[vStartIndex]) - vHalfFrame)
    vLastIndex = min(int(vEndIndex) - 1, pEnvelope.times.size - 1)
    vEnd = min(pDuration, float(pEnvelope.times[vLastIndex]) + vHalfFrame)
    if vEnd > vStart:
      lRegions.append(SpeechRegion(start=vStart, end=vEnd, duration=vEnd - vStart))
  return lRegions
