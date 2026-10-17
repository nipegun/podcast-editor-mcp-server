"""Map a source transcript onto the edited timeline.

A transcript's timestamps are coordinates of the source file. After an edit,
the delivered subtitles must follow the rendered audio instead, so every word
is carried through :meth:`Timeline.mMapSourceToOutput`. Words whose middle was
removed disappear with the audio they described; segments that lose all their
words disappear too. This is the instant fallback; transcribing the master
itself gives better text, and is what the pipeline prefers.
"""

from __future__ import annotations

from podcast_editor_mcp_server.models.transcript import Segment, Transcript, Word
from podcast_editor_mcp_server.util.timeline import Timeline


def _fMap(pTimeline: Timeline, pTime: float) -> float | None:
  return pTimeline.mMapSourceToOutput(pTime)


def fRemapTranscript(pTranscript: Transcript, pTimeline: Timeline) -> Transcript:
  """Return ``pTranscript`` re-timed onto the output of ``pTimeline``."""
  lSegments: list[Segment] = []
  vNextId = 1
  for vSegment in pTranscript.segments:
    if vSegment.words:
      lWords: list[Word] = []
      for vWord in vSegment.words:
        vMiddle = _fMap(pTimeline, (vWord.start + vWord.end) / 2.0)
        if vMiddle is None:
          continue
        vStart = _fMap(pTimeline, vWord.start)
        vEnd = _fMap(pTimeline, vWord.end)
        vStart = vMiddle if vStart is None else min(vStart, vMiddle)
        vEnd = vMiddle if vEnd is None else max(vEnd, vMiddle)
        lWords.append(vWord.model_copy(update={"start": max(0.0, vStart), "end": max(0.0, vEnd)}))
      if not lWords:
        continue
      vText = "".join(vWord.word for vWord in lWords).strip()
      lSegments.append(
        vSegment.model_copy(
          update={
            "id": vNextId,
            "start": lWords[0].start,
            "end": max(lWords[-1].end, lWords[0].start),
            "text": vText,
            "words": lWords,
          }
        )
      )
      vNextId += 1
      continue

    vStart = _fMap(pTimeline, vSegment.start)
    vEnd = _fMap(pTimeline, vSegment.end)
    if vStart is None or vEnd is None or vEnd < vStart:
      continue
    lSegments.append(vSegment.model_copy(update={"id": vNextId, "start": vStart, "end": vEnd}))
    vNextId += 1

  return pTranscript.model_copy(
    update={
      "segments": lSegments,
      "duration": pTimeline.output_duration,
      "source_sha256": "",
    }
  )
