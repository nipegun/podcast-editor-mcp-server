"""Sample-level access to audio.

FFmpeg answers most questions faster and better than hand-written DSP, so this
module exists only for the things that genuinely need the samples themselves:
the short-term energy envelope (used to place cuts in quiet spots), clipping
runs, and the energy VAD fallback.

Files that libsndfile can read are read directly; anything else is decoded once
into a temporary WAV.

Whisper and Silero get their audio from ``fDecodeMono`` instead, already mixed
to mono and resampled to the rate they expect.

Decoded samples are the one thing that grows with the recording, so no whole
file is held in memory beyond ``cMaxDecodedBytes`` of mono float32, and every
computation over a long signal runs a bounded chunk at a time.
"""

from __future__ import annotations

import math
import os
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from podcast_editor_mcp_server.backends.ffmpeg import cEnvFfmpeg, fInputArguments
from podcast_editor_mcp_server.errors import AnalysisError
from podcast_editor_mcp_server.logging_setup import fGetLogger

vLog = fGetLogger("audio.samples")

# Extensions libsndfile handles natively in Debian 13; everything else is
# decoded through FFmpeg first.
cDirectReadExtensions: frozenset[str] = frozenset({".wav", ".wave", ".flac", ".aiff", ".aif", ".caf", ".ogg", ".oga"})

# Decoding is far faster than real time; this only stops a hung FFmpeg.
cDecodeTimeout: float = 3_600.0

# The most decoded audio held in memory at once, counted as mono float32
# samples: more than 4 h at 48 kHz, and more than 14 h at the 16 kHz that
# transcription and Silero use.
cMaxDecodedBytes: int = 3 * 1024**3
# Long signals are reduced this many samples at a time, so the temporary
# float64 copies stay small whatever the length of the file.
cChunkSamples: int = 1 << 20
# FFmpeg's raw output is read in pieces of this size, and this much of its
# diagnostics is kept for the error message.
cReadChunkBytes: int = 1 << 20
cStderrTailBytes: int = 65_536

cDefaultFrameMs: float = 20.0
cDefaultHopMs: float = 10.0
cClipThreshold: float = 0.9995
cMinClipRun: int = 3


@dataclass(slots=True)
class SampleBlock:
  """Mono-mixed samples plus the sample rate they were read at."""

  samples: np.ndarray
  sample_rate: int
  duration_seconds: float

  def mGetFrameCount(self) -> int:
    return int(self.samples.shape[0])

  frame_count = property(mGetFrameCount)


@dataclass(slots=True)
class Envelope:
  """Short-term envelope in dBFS, both RMS and peak.

  Both are needed, and they answer different questions. RMS is what a listener
  perceives, so it drives cut placement. Peak is what FFmpeg's
  ``silencedetect`` actually compares against its threshold, so it is what a
  silence threshold has to be derived from. For background noise the two differ
  by the crest factor - typically 12 dB or more - which is enough to make a
  threshold computed from RMS never fire.
  """

  times: np.ndarray
  rms_db: np.ndarray
  peak_db: np.ndarray
  frame_seconds: float
  hop_seconds: float

  def mPercentile(self, pPercentile: float) -> float:
    """Percentile of the RMS envelope, in dBFS."""
    if self.rms_db.size == 0:
      return -120.0
    return float(np.percentile(self.rms_db, pPercentile))

  def mPeakPercentile(self, pPercentile: float) -> float:
    """Percentile of the peak envelope, in dBFS."""
    if self.peak_db.size == 0:
      return -120.0
    return float(np.percentile(self.peak_db, pPercentile))


def fCanReadDirectly(pPath: Path) -> bool:
  """Whether libsndfile is expected to open this file without help."""
  return pPath.suffix.lower() in cDirectReadExtensions


def fCheckDecodedBudget(pSampleCount: int, pPath: Path) -> None:
  """Refuse to hold more decoded samples in memory than ``cMaxDecodedBytes``."""
  vBytes = int(pSampleCount) * 4
  if vBytes > cMaxDecodedBytes:
    raise AnalysisError(
      "The recording is too long to hold in memory as samples",
      {"path": pPath.name, "decoded_bytes": vBytes, "limit_bytes": cMaxDecodedBytes},
    )


def fReadSamples(pPath: Path) -> SampleBlock:
  """Read a file and return its mono mix as float32 samples.

  The size of the mono mix is checked against the budget before anything is
  read, and the file is mixed down one block at a time, so the channels are
  never all in memory together.
  """
  try:
    with sf.SoundFile(str(pPath)) as vFile:
      vSampleRate = int(vFile.samplerate)
      vFrames = int(vFile.frames)
      fCheckDecodedBudget(vFrames, pPath)
      aMono = np.empty(vFrames, dtype=np.float32)
      vFilled = 0
      for aData in vFile.blocks(blocksize=cChunkSamples, frames=vFrames, dtype="float32", always_2d=True):
        vCount = min(int(aData.shape[0]), vFrames - vFilled)
        if vCount <= 0:
          break
        aData = aData[:vCount]
        aMono[vFilled : vFilled + vCount] = aData.mean(axis=1) if aData.shape[1] > 1 else aData[:, 0]
        vFilled += vCount
  except (sf.LibsndfileError, RuntimeError, OSError) as vError:
    raise AnalysisError(
      "Could not read audio samples",
      {"path": pPath.name, "reason": str(vError)},
    ) from vError

  aMono = aMono[:vFilled]
  if aMono.size == 0:
    raise AnalysisError("Audio file contains no samples", {"path": pPath.name})

  vDuration = float(aMono.shape[0]) / float(vSampleRate)
  return SampleBlock(samples=aMono, sample_rate=vSampleRate, duration_seconds=vDuration)


def fDecodeMono(pPath: Path, pSampleRate: int) -> np.ndarray:
  """Decode any file FFmpeg can read into mono float32 samples at ``pSampleRate``.

  This replaces ``faster_whisper.audio.decode_audio`` for transcription and for
  Silero. That one goes through PyAV, whose API keeps moving under it: PyAV 15
  dropped the ``metadata_errors`` argument faster-whisper 1.2 still passes, so
  with a current PyAV every decode fails. FFmpeg is the decoder the rest of the
  product already relies on, and it is not a Python package that pip can
  upgrade behind our back.
  """
  lCommand = [
    os.environ.get(cEnvFfmpeg) or "ffmpeg",
    "-nostdin",
    "-hide_banner",
    "-loglevel", "error",
    *fInputArguments(pPath),
    "-map", "0:a:0",
    "-vn", "-sn", "-dn",
    "-ac", "1",
    "-ar", str(pSampleRate),
    "-f", "f32le",
    "-",
  ]
  try:
    # Not fRunProcess: that one captures text, and these are raw samples. The
    # output is read a piece at a time, so the budget stops FFmpeg as soon as
    # it is exceeded instead of after the whole file is in memory.
    vProcess = subprocess.Popen(  # noqa: S603 - argument list, shell=False
      lCommand,
      stdin=subprocess.DEVNULL,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
    )
  except OSError as vError:
    raise AnalysisError(
      "Could not decode audio",
      {"path": pPath.name, "reason": str(vError)},
    ) from vError

  lStderr: list[bytes] = []
  vStderrThread = threading.Thread(target=_fDrainStderr, args=(vProcess.stderr, lStderr), daemon=True)
  vStderrThread.start()
  vTimedOut = threading.Event()

  def fOnTimeout() -> None:
    vTimedOut.set()
    vProcess.kill()

  vTimer = threading.Timer(cDecodeTimeout, fOnTimeout)
  vTimer.daemon = True
  vTimer.start()
  vBuffer = bytearray()
  vTooLarge = False
  try:
    while True:
      vChunk = vProcess.stdout.read(cReadChunkBytes)  # type: ignore[union-attr]
      if not vChunk:
        break
      if len(vBuffer) + len(vChunk) > cMaxDecodedBytes:
        vTooLarge = True
        vProcess.kill()
        break
      vBuffer += vChunk
  finally:
    vProcess.stdout.close()  # type: ignore[union-attr]
    vReturnCode = vProcess.wait()
    vTimer.cancel()
    vStderrThread.join(timeout=5.0)

  if vTooLarge:
    raise AnalysisError(
      "The recording is too long to hold in memory as samples",
      {"path": pPath.name, "sample_rate": pSampleRate, "limit_bytes": cMaxDecodedBytes},
    )
  if vTimedOut.is_set():
    raise AnalysisError(
      "Could not decode audio",
      {"path": pPath.name, "reason": f"FFmpeg did not finish within {cDecodeTimeout:.0f} s"},
    )
  if vReturnCode != 0:
    vReason = b"".join(lStderr).decode("utf-8", errors="replace").strip()[-500:]
    raise AnalysisError(
      "Could not decode audio",
      {"path": pPath.name, "returncode": vReturnCode, "reason": vReason},
    )

  # A view over the bytearray, which is writable, so no second copy is made.
  aSamples = np.frombuffer(vBuffer, dtype="<f4", count=len(vBuffer) // 4).astype(np.float32, copy=False)
  if aSamples.size == 0:
    raise AnalysisError("Audio file contains no samples", {"path": pPath.name})
  vLog.debug(
    "Decoded audio",
    extra={"path": pPath.name, "sample_rate": pSampleRate, "seconds": round(aSamples.size / pSampleRate, 3)},
  )
  return aSamples


def _fDrainStderr(pStream: Any, pChunks: list[bytes]) -> None:
  """Read FFmpeg's diagnostics so it never blocks on a full pipe, keeping the tail."""
  vKept = 0
  try:
    for vChunk in iter(lambda: pStream.read(4096), b""):
      pChunks.append(vChunk)
      vKept += len(vChunk)
      while vKept > cStderrTailBytes and len(pChunks) > 1:
        vKept -= len(pChunks.pop(0))
  finally:
    pStream.close()


def fComputeEnvelope(
  pBlock: SampleBlock,
  pFrameMs: float = cDefaultFrameMs,
  pHopMs: float = cDefaultHopMs,
) -> Envelope:
  """Compute a short-term RMS envelope in dBFS.

  Every frame is a whole number of hops, so the signal is first reduced to one
  sum of squares and one peak per hop, a bounded chunk at a time, and each
  frame then adds up the hops it spans through a cumulative sum. The cost is
  linear in the number of samples, and the memory in the number of hops.
  """
  vHopLength = max(1, int(round(pHopMs * pBlock.sample_rate / 1000.0)))
  # Force the frame to be a whole number of hops, which lets the peak envelope
  # be computed from per-hop block maxima instead of a quadratic scan.
  vBlocksPerFrame = max(1, int(round(pFrameMs / max(pHopMs, 0.001))))
  vFrameLength = vBlocksPerFrame * vHopLength
  vTotal = pBlock.frame_count

  if vTotal < vFrameLength:
    vRms = float(np.sqrt(np.mean(np.square(pBlock.samples, dtype=np.float64))))
    vPeak = float(np.max(np.abs(pBlock.samples))) if pBlock.samples.size else 0.0
    return Envelope(
      times=np.array([0.0]),
      rms_db=np.array([fAmplitudeToDb(vRms)]),
      peak_db=np.array([fAmplitudeToDb(vPeak)]),
      frame_seconds=vFrameLength / pBlock.sample_rate,
      hop_seconds=vHopLength / pBlock.sample_rate,
    )

  vFrameCount = (vTotal - vFrameLength) // vHopLength + 1
  # The frames span exactly these hops, all of them complete.
  vHopCount = vFrameCount + vBlocksPerFrame - 1
  aHopSquares, aHopPeaks = _fHopReductions(pBlock.samples, vHopLength, vHopCount)
  aCumulative = np.concatenate(([0.0], np.cumsum(aHopSquares)))
  aFrames = np.arange(vFrameCount, dtype=np.int64)
  aStarts = aFrames * vHopLength
  aMeanSquare = (aCumulative[aFrames + vBlocksPerFrame] - aCumulative[aFrames]) / float(vFrameLength)
  aRms = np.sqrt(np.maximum(aMeanSquare, 0.0))
  aRmsDb = fAmplitudeArrayToDb(aRms)
  aPeakDb = fAmplitudeArrayToDb(_fWindowPeaks(aHopPeaks, vFrameCount, vBlocksPerFrame))
  aTimes = (aStarts + vFrameLength / 2.0) / float(pBlock.sample_rate)
  return Envelope(
    times=aTimes,
    rms_db=aRmsDb,
    peak_db=aPeakDb,
    frame_seconds=vFrameLength / pBlock.sample_rate,
    hop_seconds=vHopLength / pBlock.sample_rate,
  )


def _fHopReductions(pSamples: np.ndarray, pHopLength: int, pHopCount: int) -> tuple[np.ndarray, np.ndarray]:
  """Sum of squares and peak amplitude of each of the first ``pHopCount`` hops.

  Computed ``cChunkSamples`` at a time, so the float64 copy never spans more
  than one chunk.
  """
  aSquares = np.empty(pHopCount, dtype=np.float64)
  aPeaks = np.empty(pHopCount, dtype=np.float32)
  vHopsPerChunk = max(1, cChunkSamples // pHopLength)
  for vFirst in range(0, pHopCount, vHopsPerChunk):
    vLast = min(pHopCount, vFirst + vHopsPerChunk)
    aChunk = pSamples[vFirst * pHopLength : vLast * pHopLength].reshape(vLast - vFirst, pHopLength)
    aWide = aChunk.astype(np.float64)
    aSquares[vFirst:vLast] = np.einsum("ij,ij->i", aWide, aWide)
    aPeaks[vFirst:vLast] = np.abs(aChunk).max(axis=1)
  return (aSquares, aPeaks)


def _fWindowPeaks(pHopPeaks: np.ndarray, pFrameCount: int, pBlocksPerFrame: int) -> np.ndarray:
  """Peak amplitude of every analysis window: the maximum of the hops it spans."""
  aWindowMax = pHopPeaks[:pFrameCount].copy()
  for vOffset in range(1, pBlocksPerFrame):
    aWindowMax = np.maximum(aWindowMax, pHopPeaks[vOffset : vOffset + pFrameCount])
  return aWindowMax


def fAmplitudeToDb(pAmplitude: float, pFloorDb: float = -120.0) -> float:
  """Convert a linear amplitude into dBFS with a floor."""
  if pAmplitude <= 0.0:
    return pFloorDb
  return max(pFloorDb, float(20.0 * math.log10(pAmplitude)))


def fAmplitudeArrayToDb(pAmplitudes: np.ndarray, pFloorDb: float = -120.0) -> np.ndarray:
  """Vectorised :func:`fAmplitudeToDb`."""
  aSafe = np.maximum(pAmplitudes, 10.0 ** (pFloorDb / 20.0))
  return np.maximum(20.0 * np.log10(aSafe), pFloorDb).astype(np.float64, copy=False)


def fFindClippingRuns(
  pBlock: SampleBlock,
  pThreshold: float = cClipThreshold,
  pMinimumRun: int = cMinClipRun,
) -> list[tuple[float, float, int]]:
  """Locate runs of consecutive samples at or beyond full scale.

  The signal is scanned ``cChunkSamples`` at a time. A run still going at the
  end of a chunk is carried into the next one, so the runs are the same as a
  scan of the whole signal would find.
  """
  lRuns: list[tuple[float, float, int]] = []
  vTotal = pBlock.frame_count
  vOpenStart: int | None = None
  for vOffset in range(0, vTotal, cChunkSamples):
    aOver = np.abs(pBlock.samples[vOffset : vOffset + cChunkSamples]) >= pThreshold
    if vOpenStart is None and not bool(aOver.any()):
      continue
    # Locate the boundaries of every True run of the chunk in one pass.
    aPadded = np.concatenate(([vOpenStart is not None], aOver, [False]))
    aDiff = np.diff(aPadded.astype(np.int8))
    lStarts = (np.flatnonzero(aDiff == 1) + vOffset).tolist()
    lEnds = (np.flatnonzero(aDiff == -1) + vOffset).tolist()
    if vOpenStart is not None:
      lStarts.insert(0, vOpenStart)
    vOpenStart = None
    if bool(aOver[-1]) and vOffset + aOver.size < vTotal:
      vOpenStart = lStarts.pop()
      lEnds.pop()
    for vStart, vEnd in zip(lStarts, lEnds, strict=True):
      vLength = int(vEnd - vStart)
      if vLength < pMinimumRun:
        continue
      lRuns.append(
        (
          float(vStart) / pBlock.sample_rate,
          float(vEnd) / pBlock.sample_rate,
          vLength,
        )
      )
  return lRuns


def fClippedSampleFraction(pBlock: SampleBlock, pThreshold: float = cClipThreshold) -> float:
  """Fraction of samples sitting at or beyond full scale."""
  if pBlock.frame_count == 0:
    return 0.0
  vClipped = 0
  for vOffset in range(0, pBlock.frame_count, cChunkSamples):
    vClipped += int(np.count_nonzero(np.abs(pBlock.samples[vOffset : vOffset + cChunkSamples]) >= pThreshold))
  return float(vClipped) / float(pBlock.frame_count)

def fEstimatePeakNoiseFloorDb(pEnvelope: Envelope, pPercentile: float = 10.0) -> float:
  """Estimate the *peak* noise floor, which is what a silence threshold needs.

  ``silencedetect`` compares sample peaks, so a threshold derived from the RMS
  floor sits a full crest factor too low and never fires.
  """
  return pEnvelope.mPeakPercentile(pPercentile)


def fFindQuietestPoint(
  pEnvelope: Envelope,
  pCentre: float,
  pWindowSeconds: float,
) -> float:
  """Return the instant of lowest energy within a window around ``pCentre``.

  Used to nudge a cut boundary off a phoneme and into a gap, which is what
  keeps splices from sounding chopped.
  """
  if pWindowSeconds <= 0.0 or pEnvelope.times.size == 0:
    return pCentre
  vLow = pCentre - pWindowSeconds
  vHigh = pCentre + pWindowSeconds
  aMask = (pEnvelope.times >= vLow) & (pEnvelope.times <= vHigh)
  if not bool(aMask.any()):
    return pCentre
  aIndices = np.flatnonzero(aMask)
  vBest = aIndices[int(np.argmin(pEnvelope.rms_db[aIndices]))]
  return float(pEnvelope.times[vBest])


# --------------------------------------------------------------------------- #
# Stretches of sound
# --------------------------------------------------------------------------- #

# The speech threshold sits this share of the way from the noise floor to the
# usual speech level, and at least this many dB above the floor.
cSpeechThresholdShare: float = 0.3
cSpeechThresholdMinimumDb: float = 6.0
cNoisePercentile: float = 10.0
cSpeechPercentile: float = 90.0


def fSpeechThresholdDb(pLevelsDb: np.ndarray) -> float:
  """Level above which a frame holds sound worth keeping, from the recording itself."""
  if pLevelsDb.size == 0:
    return -120.0
  vNoise = float(np.percentile(pLevelsDb, cNoisePercentile))
  vSpeech = float(np.percentile(pLevelsDb, cSpeechPercentile))
  return vNoise + max(cSpeechThresholdMinimumDb, cSpeechThresholdShare * (vSpeech - vNoise))


def fFrameLevelsDb(pSamples: np.ndarray, pSampleRate: int, pFrameSeconds: float = 0.01) -> np.ndarray:
  """RMS level of consecutive, non-overlapping frames, in dBFS, a chunk at a time."""
  vFrame = max(1, int(round(pFrameSeconds * pSampleRate)))
  vCount = pSamples.size // vFrame
  if vCount == 0:
    return np.zeros(0)
  aLevels = np.empty(vCount, dtype=np.float64)
  vFramesPerChunk = max(1, cChunkSamples // vFrame)
  for vFirst in range(0, vCount, vFramesPerChunk):
    vLast = min(vCount, vFirst + vFramesPerChunk)
    aFrames = pSamples[vFirst * vFrame : vLast * vFrame].reshape(vLast - vFirst, vFrame).astype(np.float64)
    aLevels[vFirst:vLast] = np.sqrt(np.mean(aFrames * aFrames, axis=1))
  return fAmplitudeArrayToDb(aLevels)


def fLoudRuns(pLoud: np.ndarray, pStarts: np.ndarray, pEnds: np.ndarray, pBridgeFrames: int) -> list[tuple[float, float]]:
  """Runs of loud frames as ``(start, end)`` in seconds.

  Quiet dips shorter than ``pBridgeFrames`` frames (a stop consonant) do not
  split a run. ``pStarts`` and ``pEnds`` give each frame's time span.
  """
  lRuns: list[tuple[float, float]] = []
  vCount = int(pLoud.size)
  vIndex = 0
  while vIndex < vCount:
    if not pLoud[vIndex]:
      vIndex += 1
      continue
    vFirst = vIndex
    vLast = vIndex
    vQuiet = 0
    vIndex += 1
    while vIndex < vCount:
      if pLoud[vIndex]:
        vLast = vIndex
        vQuiet = 0
      else:
        vQuiet += 1
        if vQuiet >= pBridgeFrames:
          break
      vIndex += 1
    lRuns.append((float(pStarts[vFirst]), float(pEnds[vLast])))
    vIndex = vLast + 1
  return lRuns

