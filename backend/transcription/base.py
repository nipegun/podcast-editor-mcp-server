"""Transcription backend abstraction.

Whatever engine runs underneath, the product only ever consumes a
:class:`~podcast_editor_mcp_server.models.transcript.Transcript`. Adding a new engine means
implementing one class and registering it in :func:`fCreateTranscriptionBackend`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from podcast_editor_mcp_server.backends.base import BackendStatus
from podcast_editor_mcp_server.config import TranscriptionSettings
from podcast_editor_mcp_server.models.transcript import Transcript


class TranscriptionBackend(ABC):
  """Turns audio into a timestamped transcript."""

  name: str = "base"

  @abstractmethod
  def mStatus(self) -> BackendStatus:
    """Report whether this backend can run."""

  @abstractmethod
  def mTranscribe(self, pPath: Path, pSettings: TranscriptionSettings) -> Transcript:
    """Transcribe a file and return the normalised result."""

  @abstractmethod
  def mRecognizeSamples(self, pSamples: np.ndarray, pSettings: TranscriptionSettings, pModel: str | None = None) -> str:
    """Recognise a short stretch of 16 kHz mono samples and return its text."""


def fCreateTranscriptionBackend(pName: str) -> TranscriptionBackend | None:
  """Instantiate the transcription backend a profile names, or ``None`` for \"none\"."""
  from podcast_editor_mcp_server.transcription.faster_whisper_backend import FasterWhisperBackend

  if pName == "none":
    return None
  return FasterWhisperBackend()
