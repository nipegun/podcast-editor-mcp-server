"""Transcript serialisation: plain text, SubRip and WebVTT."""

from __future__ import annotations

from podcast_editor_mcp_server.models.transcript import Transcript


def fFormatTimestampSrt(pSeconds: float) -> str:
  """``HH:MM:SS,mmm`` as SubRip expects."""
  vTotal = max(0.0, pSeconds)
  vHours = int(vTotal // 3600)
  vMinutes = int((vTotal % 3600) // 60)
  vWholeSeconds = int(vTotal % 60)
  vMilliseconds = int(round((vTotal - int(vTotal)) * 1000.0))
  if vMilliseconds == 1000:
    vMilliseconds = 0
    vWholeSeconds += 1
  return f"{vHours:02d}:{vMinutes:02d}:{vWholeSeconds:02d},{vMilliseconds:03d}"


def fFormatTimestampVtt(pSeconds: float) -> str:
  """``HH:MM:SS.mmm`` as WebVTT expects."""
  return fFormatTimestampSrt(pSeconds).replace(",", ".")


def fToPlainText(pTranscript: Transcript) -> str:
  """Render the transcript as readable text."""
  lLines: list[str] = []
  for vSegment in pTranscript.segments:
    vText = vSegment.text.strip()
    if not vText:
      continue
    lLines.append(vText)
  return "\n".join(lLines) + ("\n" if lLines else "")


def fToSrt(pTranscript: Transcript) -> str:
  """Render the transcript as SubRip subtitles."""
  lBlocks: list[str] = []
  vIndex = 0
  for vSegment in pTranscript.segments:
    vText = vSegment.text.strip()
    if not vText:
      continue
    vIndex += 1
    lBlocks.append(
      f"{vIndex}\n"
      f"{fFormatTimestampSrt(vSegment.start)} --> {fFormatTimestampSrt(vSegment.end)}\n"
      f"{vText}\n"
    )
  return "\n".join(lBlocks)


def fToVtt(pTranscript: Transcript) -> str:
  """Render the transcript as WebVTT."""
  lBlocks: list[str] = ["WEBVTT\n"]
  for vSegment in pTranscript.segments:
    vText = vSegment.text.strip()
    if not vText:
      continue
    lBlocks.append(
      f"{fFormatTimestampVtt(vSegment.start)} --> {fFormatTimestampVtt(vSegment.end)}\n"
      f"{vText}\n"
    )
  return "\n".join(lBlocks)


def fToLabelTrack(pTranscript: Transcript) -> str:
  """Render the transcript as an Audacity label track (TSV).

  Audacity imports this format directly through File > Import > Labels, which
  gives the user a navigable transcript alongside the audio.
  """
  lLines: list[str] = []
  for vSegment in pTranscript.segments:
    vText = vSegment.text.strip().replace("\t", " ").replace("\n", " ")
    if not vText:
      continue
    lLines.append(f"{vSegment.start:.6f}\t{vSegment.end:.6f}\t{vText}")
  return "\n".join(lLines) + ("\n" if lLines else "")


def fRender(pTranscript: Transcript, pFormat: str) -> str:
  """Render the transcript in the requested format."""
  vFormat = pFormat.strip().lower()
  if vFormat == "txt":
    return fToPlainText(pTranscript)
  if vFormat == "srt":
    return fToSrt(pTranscript)
  if vFormat == "vtt":
    return fToVtt(pTranscript)
  if vFormat == "labels":
    return fToLabelTrack(pTranscript)
  raise ValueError(f"Unsupported transcript format: {pFormat}")
