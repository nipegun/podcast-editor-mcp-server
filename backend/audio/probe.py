"""Container and stream description of a source file."""

from __future__ import annotations

from pathlib import Path

from podcast_editor_mcp_server.backends.ffmpeg import FfmpegBackend
from podcast_editor_mcp_server.errors import AudioProbeError
from podcast_editor_mcp_server.logging_setup import fGetLogger
from podcast_editor_mcp_server.models.audio import AudioProbe
from podcast_editor_mcp_server.security.paths import PathGuard
from podcast_editor_mcp_server.util.hashing import fHashFile

vLog = fGetLogger("audio.probe")

# Tags worth keeping; everything else is dropped so a crafted file cannot push
# megabytes of metadata into our JSON artifacts.
cKeptTags: frozenset[str] = frozenset(
  {"title", "artist", "album", "date", "comment", "genre", "encoder", "track", "album_artist"}
)
cMaxTagLength: int = 500


def fProbeAudio(pPath: Path, pBackend: FfmpegBackend, pGuard: PathGuard | None = None) -> AudioProbe:
  """Describe a file with ffprobe and hash it for provenance."""
  if not pPath.is_file():
    raise AudioProbeError("File does not exist", {"path": pPath.name})

  dRaw = pBackend.mProbeRaw(pPath)
  dFormat = dRaw.get("format") or {}
  ldStreams = dRaw.get("streams") or []
  ldAudioStreams = [dStream for dStream in ldStreams if dStream.get("codec_type") == "audio"]
  if not ldAudioStreams:
    raise AudioProbeError("File contains no audio stream", {"path": pPath.name})
  dStream = ldAudioStreams[0]

  vDuration = _fFloat(dStream.get("duration")) or _fFloat(dFormat.get("duration"))
  if vDuration is None or vDuration <= 0.0:
    raise AudioProbeError("File has no usable duration", {"path": pPath.name})

  vSampleRate = _fInt(dStream.get("sample_rate"))
  if not vSampleRate or vSampleRate <= 0:
    raise AudioProbeError("Audio stream has no sample rate", {"path": pPath.name})

  vChannels = _fInt(dStream.get("channels")) or 0
  if vChannels <= 0:
    raise AudioProbeError("Audio stream has no channels", {"path": pPath.name})

  dTags: dict[str, str] = {}
  for vKey, vValue in (dFormat.get("tags") or {}).items():
    vLowerKey = str(vKey).lower()
    if vLowerKey in cKeptTags:
      dTags[vLowerKey] = str(vValue)[:cMaxTagLength]

  vLabel = pGuard.mLabel(pPath) if pGuard is not None else pPath.name

  return AudioProbe(
    path=str(pPath),
    label=vLabel,
    size_bytes=pPath.stat().st_size,
    sha256=fHashFile(pPath),
    duration_seconds=vDuration,
    sample_rate=vSampleRate,
    channels=vChannels,
    channel_layout=dStream.get("channel_layout"),
    codec_name=str(dStream.get("codec_name") or "unknown"),
    codec_long_name=dStream.get("codec_long_name"),
    sample_format=dStream.get("sample_fmt"),
    bits_per_sample=_fInt(dStream.get("bits_per_raw_sample")) or _fInt(dStream.get("bits_per_sample")) or None,
    bit_rate=_fInt(dStream.get("bit_rate")) or _fInt(dFormat.get("bit_rate")) or None,
    format_name=str(dFormat.get("format_name") or "unknown"),
    format_long_name=dFormat.get("format_long_name"),
    stream_count=max(1, len(ldStreams)),
    tags=dTags,
  )


def fStreamShape(pPath: Path, pBackend: FfmpegBackend) -> tuple[float | None, int | None, int | None]:
  """Duration, sample rate and channel count of the first audio stream.

  The cheap half of :func:`fProbeAudio`: ffprobe only, without hashing the
  file. Any of the three is ``None`` when ffprobe does not report it.
  """
  dRaw = pBackend.mProbeRaw(pPath)
  dFormat = dRaw.get("format") or {}
  ldAudioStreams = [dStream for dStream in dRaw.get("streams") or [] if dStream.get("codec_type") == "audio"]
  dStream = ldAudioStreams[0] if ldAudioStreams else {}
  return (
    _fFloat(dStream.get("duration")) or _fFloat(dFormat.get("duration")),
    _fInt(dStream.get("sample_rate")),
    _fInt(dStream.get("channels")),
  )


def _fFloat(pValue: object) -> float | None:
  if pValue is None:
    return None
  try:
    vNumber = float(pValue)  # type: ignore[arg-type]
  except (TypeError, ValueError):
    return None
  if vNumber != vNumber or vNumber in (float("inf"), float("-inf")):
    return None
  return vNumber


def _fInt(pValue: object) -> int | None:
  vNumber = _fFloat(pValue)
  if vNumber is None:
    return None
  return int(vNumber)
