"""Transcript cache.

Transcribing an hour of audio is expensive; doing it twice for the same bytes
and the same settings is waste. The cache key combines the audio's SHA-256 with
a digest of everything that could change the result (backend, model, language,
word timestamps, VAD filter, beam size, prompt, hotwords, chunking, product
schema version), so a cache hit is only ever returned when it is genuinely the
same computation.

The cache stores the recogniser's raw output. Hallucination removal and the
corrections table are applied when a transcript is read
(:mod:`podcast_editor_mcp_server.transcription.cleanup`), so neither is part of the key.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from podcast_editor_mcp_server.config import TranscriptionSettings
from podcast_editor_mcp_server.logging_setup import fGetLogger
from podcast_editor_mcp_server.models.transcript import Transcript
from podcast_editor_mcp_server.util.atomic import fWriteJsonAtomic
from podcast_editor_mcp_server.util.hashing import fHashMapping, fShortHash

vLog = fGetLogger("transcription.cache")

cCacheSchemaVersion: int = 1
# Version of the transcription by stretch of speech; part of the key of every
# transcript made that way.
cChunkingVersion: int = 6


def fCacheKey(pSourceSha256: str, pSettings: TranscriptionSettings) -> str:
  """Build the cache key for one (audio, settings) pair."""
  dRelevant = {
    "schema": cCacheSchemaVersion,
    "backend": pSettings.backend,
    "model": pSettings.model,
    "language": pSettings.language,
    "word_timestamps": pSettings.word_timestamps,
    "vad_filter": pSettings.vad_filter,
    "beam_size": pSettings.beam_size,
    "condition_on_previous_text": pSettings.condition_on_previous_text,
    "initial_prompt": pSettings.initial_prompt or "",
    "batch_size": pSettings.batch_size,
    "hotwords": pSettings.hotwords or "",
    "chunking": pSettings.chunking,
  }
  # The stretch settings only matter when the audio is transcribed by stretch.
  if pSettings.chunking != "none":
    # Bumped whenever the way stretches are transcribed changes.
    dRelevant["chunking_version"] = cChunkingVersion
    dRelevant["chunk_max_seconds"] = pSettings.chunk_max_seconds
    dRelevant["chunk_merge_gap_ms"] = pSettings.chunk_merge_gap_ms
    dRelevant["chunk_min_silence_ms"] = pSettings.chunk_min_silence_ms
    dRelevant["chunk_vad_threshold"] = pSettings.chunk_vad_threshold
  return f"{pSourceSha256}-{fShortHash(fHashMapping(dRelevant), 16)}"


class TranscriptCache:
  """A directory of cached transcripts, keyed by audio hash plus settings."""

  def __init__(self, pDirectory: Path, pEnabled: bool = True) -> None:
    self.directory = pDirectory
    self.enabled = pEnabled

  def mPathFor(self, pSourceSha256: str, pSettings: TranscriptionSettings) -> Path:
    return self.directory / "transcripts" / f"{fCacheKey(pSourceSha256, pSettings)}.json"

  def mLoad(self, pSourceSha256: str, pSettings: TranscriptionSettings) -> Transcript | None:
    """Return a cached transcript, or ``None`` when there is no valid hit."""
    if not self.enabled or not pSourceSha256:
      return None
    vPath = self.mPathFor(pSourceSha256, pSettings)
    if not vPath.is_file():
      return None
    try:
      vTranscript = Transcript.model_validate_json(vPath.read_text(encoding="utf-8"))
    except (ValidationError, ValueError, OSError) as vError:
      vLog.warning("Discarding unreadable cached transcript", extra={"path": vPath.name, "reason": str(vError)})
      vPath.unlink(missing_ok=True)
      return None
    if vTranscript.source_sha256 != pSourceSha256:
      vLog.warning("Cached transcript hash mismatch; ignoring", extra={"path": vPath.name})
      return None
    vLog.info("Reusing cached transcript", extra={"key": vPath.stem, "segments": len(vTranscript.segments)})
    return vTranscript

  def mStore(self, pSourceSha256: str, pSettings: TranscriptionSettings, pTranscript: Transcript) -> Path | None:
    """Persist a transcript, returning the file it was written to."""
    if not self.enabled or not pSourceSha256:
      return None
    vPath = self.mPathFor(pSourceSha256, pSettings)
    vTranscript = pTranscript.model_copy(update={"source_sha256": pSourceSha256})
    try:
      fWriteJsonAtomic(vPath, vTranscript.model_dump(mode="json"))
    except OSError as vError:
      vLog.warning("Could not write the transcript cache", extra={"path": str(vPath), "reason": str(vError)})
      return None
    return vPath
