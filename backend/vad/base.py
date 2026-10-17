"""Voice activity detection abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from podcast_editor_mcp_server.backends.base import BackendStatus
from podcast_editor_mcp_server.config import VadSettings
from podcast_editor_mcp_server.models.audio import SpeechRegion


class VadBackend(ABC):
  """Splits a recording into speech and non-speech regions."""

  name: str = "base"

  @abstractmethod
  def mStatus(self) -> BackendStatus:
    """Report whether this backend can run."""

  @abstractmethod
  def mDetectSpeech(self, pPath: Path, pSettings: VadSettings) -> list[SpeechRegion]:
    """Return the speech regions of a file, in source coordinates."""

  def mIsAvailable(self) -> bool:
    return self.mStatus().available


def fMergeSpeechRegions(pRegions: list[SpeechRegion], pMaximumGapSeconds: float) -> list[SpeechRegion]:
  """Merge speech regions separated by less than ``pMaximumGapSeconds``."""
  lSorted = sorted(pRegions, key=lambda pRegion: (pRegion.start, pRegion.end))
  lMerged: list[SpeechRegion] = []
  for vRegion in lSorted:
    if lMerged and vRegion.start - lMerged[-1].end <= pMaximumGapSeconds:
      vLast = lMerged[-1]
      vEnd = max(vLast.end, vRegion.end)
      lMerged[-1] = SpeechRegion(
        start=vLast.start,
        end=vEnd,
        duration=vEnd - vLast.start,
      )
    else:
      lMerged.append(vRegion)
  return lMerged
