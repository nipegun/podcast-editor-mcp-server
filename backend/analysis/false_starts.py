"""False-start candidate detection.

The reliable signal for a false start is a self-correction marker:

    "vamos a instalar... perdón... vamos a configurar el paquete"

When a marker appears, the abandoned attempt is whatever precedes it back to
the nearest breath - a pause long enough to be a phrase boundary - bounded by
``lookback_seconds`` and ``max_removed_seconds``. The marker itself is included
in the candidate, since "perdón" is part of what the listener does not need.
"""

from __future__ import annotations

from podcast_editor_mcp_server.config import FalseStartSettings
from podcast_editor_mcp_server.models.candidates import CandidateType, EditCandidate
from podcast_editor_mcp_server.models.transcript import Transcript, Word

# A gap this long is treated as a phrase boundary when walking backwards.
cPhraseBoundarySeconds: float = 0.32
cMinimumAttemptSeconds: float = 0.25


def fFindFalseStartCandidates(pTranscript: Transcript, pSettings: FalseStartSettings) -> list[EditCandidate]:
  """Return false-start candidates found in ``pTranscript``."""
  if not pSettings.enabled or not pSettings.markers:
    return []
  if not pTranscript.has_word_timestamps:
    return []

  lWords = pTranscript.mAllWords()
  if not lWords:
    return []
  lTokens = [vWord.normalised for vWord in lWords]
  lPhrases = [tuple(vMarker.split()) for vMarker in pSettings.markers if vMarker.strip()]
  lPhrases.sort(key=len, reverse=True)

  lCandidates: list[EditCandidate] = []
  vCounter = 0
  vIndex = 0
  vLastEnd = -1.0

  while vIndex < len(lWords):
    vMatchLength = _fMatchMarker(lTokens, vIndex, lPhrases)
    if vMatchLength == 0:
      vIndex += 1
      continue

    vMarkerStart = lWords[vIndex].start
    vMarkerEnd = lWords[vIndex + vMatchLength - 1].end
    # The hard limit is measured from the *end* of the marker, because that is
    # where the removal ends: max_removed_seconds bounds the whole cut.
    vAttemptStart = _fWalkBackToPhraseStart(
      lWords,
      vIndex,
      vMarkerStart - pSettings.lookback_seconds,
      vMarkerEnd - pSettings.max_removed_seconds,
    )

    if vAttemptStart is None or vMarkerEnd - vAttemptStart < cMinimumAttemptSeconds:
      vIndex += vMatchLength
      continue
    if vAttemptStart < vLastEnd:
      vIndex += vMatchLength
      continue

    vCounter += 1
    vBefore, vAfter = pTranscript.mContextText(vAttemptStart, vMarkerEnd)
    lAbandoned = [vWord for vWord in lWords if vWord.start >= vAttemptStart and vWord.end <= vMarkerEnd]
    lCandidates.append(
      EditCandidate(
        id=f"false-start-{vCounter:04d}",
        type=CandidateType.POSSIBLE_FALSE_START,
        start=vAttemptStart,
        end=vMarkerEnd,
        text=" ".join(vWord.word.strip() for vWord in lAbandoned),
        context_before=vBefore[-240:],
        context_after=vAfter[:240],
        confidence=_fScore(vMarkerEnd - vAttemptStart, vMatchLength),
        details={
          "marker": " ".join(lTokens[vIndex : vIndex + vMatchLength]),
          "marker_start": round(vMarkerStart, 3),
          "marker_end": round(vMarkerEnd, 3),
          "abandoned_seconds": round(vMarkerEnd - vAttemptStart, 3),
          "word_count": len(lAbandoned),
          "suggested_action": "delete the abandoned attempt together with the marker",
        },
      )
    )
    vLastEnd = vMarkerEnd
    vIndex += vMatchLength

  return lCandidates


def _fMatchMarker(pTokens: list[str], pIndex: int, pPhrases: list[tuple[str, ...]]) -> int:
  for lPhrase in pPhrases:
    vLength = len(lPhrase)
    if pIndex + vLength > len(pTokens):
      continue
    if tuple(pTokens[pIndex : pIndex + vLength]) == lPhrase:
      return vLength
  return 0


def _fWalkBackToPhraseStart(
  pWords: list[Word],
  pMarkerIndex: int,
  pEarliestPreferred: float,
  pHardLimit: float,
) -> float | None:
  """Walk backwards from the marker to the start of the abandoned attempt."""
  if pMarkerIndex == 0:
    return None
  vIndex = pMarkerIndex - 1
  vStart = pWords[vIndex].start
  while vIndex > 0:
    vGap = pWords[vIndex].start - pWords[vIndex - 1].end
    if vGap >= cPhraseBoundarySeconds:
      break
    if pWords[vIndex - 1].start < pEarliestPreferred:
      break
    vIndex -= 1
    vStart = pWords[vIndex].start
  if vStart < pHardLimit:
    vStart = pHardLimit
  return max(0.0, vStart)


def _fScore(pDuration: float, pMarkerWords: int) -> float:
  """A multi-word marker ("mejor dicho") is stronger evidence than a single one."""
  vMarkerScore = min(1.0, 0.6 + 0.2 * pMarkerWords)
  vDurationScore = 1.0 if pDuration <= 6.0 else max(0.3, 6.0 / pDuration)
  return float(min(1.0, max(0.0, 0.6 * vMarkerScore + 0.4 * vDurationScore)))
