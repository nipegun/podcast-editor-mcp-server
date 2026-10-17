"""Synthetic test audio.

Used by the test suite and by ``podcast-editor-mcp-server-cli demo`` so that the pipeline can
be exercised end to end without shipping binary fixtures in the repository.

``fWriteSpeechLikeWav`` produces something that behaves like speech for the
purposes of silence detection and loudness measurement: bursts of band-limited,
amplitude-modulated noise plus harmonics, separated by real silence. It is not
speech and will not transcribe into words - that is not what it is for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

cDefaultSampleRate: int = 48_000
cDefaultFundamentalHz: float = 130.0


@dataclass(slots=True)
class SynthSegment:
  """One stretch of the generated file."""

  duration: float
  kind: str = "speech"  # "speech" or "silence"
  level_db: float = -20.0
  fundamental_hz: float = cDefaultFundamentalHz


def fGenerateSpeechLike(
  pDuration: float,
  pSampleRate: int,
  pLevelDb: float,
  pFundamentalHz: float,
  pGenerator: np.random.Generator,
) -> np.ndarray:
  """Generate an amplitude-modulated harmonic burst that reads as speech."""
  vFrames = max(1, int(pDuration * pSampleRate))
  aTime = np.arange(vFrames, dtype=np.float64) / pSampleRate

  # A few harmonics of a voice-like fundamental, falling off with order.
  aSignal = np.zeros(vFrames, dtype=np.float64)
  for vHarmonic in range(1, 9):
    aSignal += (1.0 / vHarmonic) * np.sin(2.0 * math.pi * pFundamentalHz * vHarmonic * aTime)
  aSignal /= np.max(np.abs(aSignal)) or 1.0

  # Syllable-rate amplitude modulation, roughly 4 Hz, plus a little jitter.
  aSyllables = 0.5 + 0.5 * np.sin(2.0 * math.pi * 4.0 * aTime - math.pi / 2.0)
  aJitter = 1.0 + 0.08 * pGenerator.standard_normal(vFrames)
  aSignal = aSignal * aSyllables * aJitter

  # A touch of breath noise so the spectrum is not purely harmonic.
  aSignal += 0.05 * pGenerator.standard_normal(vFrames)

  # Short fades so each burst starts and ends without a click.
  vFade = min(vFrames // 4, int(0.01 * pSampleRate))
  if vFade > 0:
    aSignal[:vFade] *= np.linspace(0.0, 1.0, vFade)
    aSignal[-vFade:] *= np.linspace(1.0, 0.0, vFade)

  vPeak = float(np.max(np.abs(aSignal))) or 1.0
  vTarget = 10.0 ** (pLevelDb / 20.0)
  return (aSignal / vPeak * vTarget).astype(np.float32)


def fBuildSegments(
  pSegments: list[SynthSegment],
  pSampleRate: int = cDefaultSampleRate,
  pSeed: int = 20260921,
  pNoiseFloorDb: float | None = -85.0,
) -> np.ndarray:
  """Concatenate segments into one mono float32 signal."""
  vGenerator = np.random.default_rng(pSeed)
  lParts: list[np.ndarray] = []
  for vSegment in pSegments:
    if vSegment.kind == "speech":
      lParts.append(
        fGenerateSpeechLike(vSegment.duration, pSampleRate, vSegment.level_db, vSegment.fundamental_hz, vGenerator)
      )
    else:
      vFrames = max(1, int(vSegment.duration * pSampleRate))
      lParts.append(np.zeros(vFrames, dtype=np.float32))
  aSignal = np.concatenate(lParts) if lParts else np.zeros(1, dtype=np.float32)

  if pNoiseFloorDb is not None:
    # A tiny dither floor keeps "silence" realistic without defeating
    # silencedetect at the thresholds this product uses.
    vLevel = 10.0 ** (pNoiseFloorDb / 20.0)
    aSignal = aSignal + (vGenerator.standard_normal(aSignal.shape[0]).astype(np.float32) * vLevel)
  return np.clip(aSignal, -1.0, 1.0).astype(np.float32)


def fWriteWav(
  pPath: Path,
  pSignal: np.ndarray,
  pSampleRate: int = cDefaultSampleRate,
  pChannels: int = 1,
  pSubtype: str = "PCM_16",
) -> Path:
  """Write a mono or stereo WAV from a mono signal."""
  pPath.parent.mkdir(parents=True, exist_ok=True)
  aData = pSignal if pChannels == 1 else np.column_stack([pSignal] * pChannels)
  sf.write(str(pPath), aData, pSampleRate, subtype=pSubtype)
  return pPath


def fWriteSpeechLikeWav(
  pPath: Path,
  pSampleRate: int = cDefaultSampleRate,
  pChannels: int = 1,
  pSeed: int = 20260921,
) -> Path:
  """Write the demo recording.

  The layout is leading silence, speech, a long pause, more speech and
  trailing silence.
  """
  lSegments = [
    SynthSegment(duration=2.0, kind="silence"),
    SynthSegment(duration=3.0, kind="speech", level_db=-14.0),
    SynthSegment(duration=4.0, kind="silence"),
    SynthSegment(duration=3.0, kind="speech", level_db=-16.0, fundamental_hz=155.0),
    SynthSegment(duration=2.5, kind="silence"),
  ]
  aSignal = fBuildSegments(lSegments, pSampleRate, pSeed)
  return fWriteWav(pPath, aSignal, pSampleRate, pChannels)

