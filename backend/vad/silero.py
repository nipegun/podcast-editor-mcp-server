"""Silero VAD backend.

Silero v5 ships as an ONNX model inside ``faster-whisper``'s assets directory,
so this backend needs no download and no network access: if faster-whisper is
installed, the model is already on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from podcast_editor_mcp_server.audio.samples import fDecodeMono
from podcast_editor_mcp_server.backends.base import BackendStatus
from podcast_editor_mcp_server.config import VadSettings
from podcast_editor_mcp_server.errors import VadError
from podcast_editor_mcp_server.logging_setup import fGetLogger
from podcast_editor_mcp_server.models.audio import SpeechRegion
from podcast_editor_mcp_server.vad.base import VadBackend

vLog = fGetLogger("vad.silero")

# Silero expects 16 kHz mono.
cSileroSampleRate: int = 16_000


class SileroVadBackend(VadBackend):
  """Neural voice activity detection through faster-whisper's bundled model."""

  name = "silero"

  def __init__(self) -> None:
    self._vImportError: str | None = None
    self._dModules: dict[str, Any] | None = None

  def _mLoadModules(self) -> dict[str, Any] | None:
    if self._dModules is not None:
      return self._dModules
    if self._vImportError is not None:
      return None
    try:
      from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ImportError as vError:
      self._vImportError = str(vError)
      return None
    # The audio is decoded with FFmpeg (fDecodeMono), not with faster-whisper's
    # PyAV-based decode_audio, which a current PyAV breaks.
    self._dModules = {
      "VadOptions": VadOptions,
      "get_speech_timestamps": get_speech_timestamps,
    }
    return self._dModules

  def mStatus(self) -> BackendStatus:
    dModules = self._mLoadModules()
    if dModules is None:
      return BackendStatus(
        name=self.name,
        available=False,
        reason="faster-whisper is not installed (pip install 'podcast-editor-mcp-server[whisper]')",
        details={"import_error": self._vImportError or ""},
      )
    try:
      import faster_whisper

      vVersion = getattr(faster_whisper.version, "__version__", None) or "unknown"
    except (ImportError, AttributeError):
      vVersion = "unknown"
    return BackendStatus(name=self.name, available=True, version=f"silero-v5 via faster-whisper {vVersion}")

  def mDetectSpeech(self, pPath: Path, pSettings: VadSettings) -> list[SpeechRegion]:
    dModules = self._mLoadModules()
    if dModules is None:
      raise VadError(
        "Silero VAD is unavailable because faster-whisper is not installed",
        {"install": "pip install 'podcast-editor-mcp-server[whisper]'"},
      )
    try:
      aAudio = fDecodeMono(pPath, cSileroSampleRate)
    except Exception as vError:  # noqa: BLE001 - a failed decode must become a VadError
      raise VadError(
        "Could not decode audio for voice activity detection",
        {"path": pPath.name, "reason": str(vError)},
      ) from vError

    vOptions = dModules["VadOptions"](
      threshold=pSettings.threshold,
      min_speech_duration_ms=pSettings.min_speech_duration_ms,
      max_speech_duration_s=pSettings.max_speech_duration_s,
      min_silence_duration_ms=pSettings.min_silence_duration_ms,
      speech_pad_ms=pSettings.speech_pad_ms,
    )
    try:
      ldChunks = dModules["get_speech_timestamps"](aAudio, vOptions, sampling_rate=cSileroSampleRate)
    except Exception as vError:  # noqa: BLE001 - onnxruntime raises opaque errors
      raise VadError(
        "Silero VAD failed",
        {"path": pPath.name, "reason": str(vError)},
      ) from vError

    lRegions: list[SpeechRegion] = []
    for dChunk in ldChunks:
      vStart = float(dChunk["start"]) / cSileroSampleRate
      vEnd = float(dChunk["end"]) / cSileroSampleRate
      if vEnd > vStart:
        lRegions.append(SpeechRegion(start=vStart, end=vEnd, duration=vEnd - vStart))
    vLog.debug("Silero VAD finished", extra={"regions": len(lRegions), "path": pPath.name})
    return lRegions


def fCreateVadBackend(pName: str) -> VadBackend | None:
  """Instantiate a VAD backend by name, falling back to energy when needed."""
  from podcast_editor_mcp_server.vad.energy import EnergyVadBackend

  if pName == "none":
    return None
  if pName == "energy":
    return EnergyVadBackend()
  vSilero = SileroVadBackend()
  if vSilero.mIsAvailable():
    return vSilero
  vLog.info("Silero VAD unavailable; falling back to the energy backend")
  return EnergyVadBackend()
