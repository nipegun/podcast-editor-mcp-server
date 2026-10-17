"""Accidental repetition candidate detection.

The pattern this looks for is a speaker restarting a phrase:

    "lo hacemos con apt... lo hacemos con apt install"

Both halves are compared as normalised word sequences with
``difflib.SequenceMatcher``, so "vamos a instalar" and "vamos a instalarlo"
still register as the same attempt. What is emitted is a candidate naming the
*first* occurrence, with both spans in ``details``, because deciding which
attempt to keep is a judgement call that belongs to the planner or the model -
not to a string comparison.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from podcast_editor_mcp_server.config import RepetitionSettings
from podcast_editor_mcp_server.models.candidates import CandidateType, EditCandidate
from podcast_editor_mcp_server.models.transcript import Transcript, Word

# A repetition made after a long pause is more likely to be deliberate
# rhetoric than a stumble, so the score decays with the gap.
cGapDecaySeconds: float = 4.0
cMinimumTokenLength: int = 2


def fNormaliseWords(pWords: list[Word]) -> list[str]:
  """Normalised token list, dropping punctuation-only entries."""
  lTokens: list[str] = []
  for vWord in pWords:
    vToken = vWord.normalised
    lTokens.append(vToken)
  return lTokens


def fFindRepetitionCandidates(pTranscript: Transcript, pSettings: RepetitionSettings) -> list[EditCandidate]:
  """Return repetition candidates found in ``pTranscript``."""
  if not pSettings.enabled:
    return []
  if not pTranscript.has_word_timestamps:
    return []

  lWords = pTranscript.mAllWords()
  lTokens = fNormaliseWords(lWords)
  vCount = len(lWords)
  if vCount < pSettings.min_words * 2:
    return []

  lCandidates: list[EditCandidate] = []
  lClaimed: list[tuple[float, float]] = []
  vCounter = 0
  vIndex = 0

  while vIndex < vCount:
    vBest: tuple[float, int, int] | None = None  # (similarity, window, second_start)
    for vWindow in range(pSettings.max_words, pSettings.min_words - 1, -1):
      vFirstEnd = vIndex + vWindow
      if vFirstEnd + vWindow > vCount:
        continue
      lFirst = lTokens[vIndex:vFirstEnd]
      if not _fIsMeaningful(lFirst):
        continue
      # The second attempt may start immediately or after a few stray tokens.
      for vOffset in range(0, 4):
        vSecondStart = vFirstEnd + vOffset
        vSecondEnd = vSecondStart + vWindow
        if vSecondEnd > vCount:
          break
        # Both attempts must begin with the same word. Without this anchor a
        # sliding window happily matches "eh lo hacemos con" against
        # "apt lo hacemos con" at 0.86 similarity, which is an artefact of the
        # offset rather than anything the speaker repeated.
        if lTokens[vIndex] != lTokens[vSecondStart]:
          continue
        vGap = lWords[vSecondStart].start - lWords[vFirstEnd - 1].end
        if vGap < 0.0 or vGap > pSettings.max_gap_seconds:
          continue
        lSecond = lTokens[vSecondStart:vSecondEnd]
        vSimilarity = _fSimilarity(lFirst, lSecond)
        if vSimilarity >= pSettings.similarity and (vBest is None or vSimilarity > vBest[0]):
          vBest = (vSimilarity, vWindow, vSecondStart)
      if vBest is not None:
        break

    if vBest is None:
      vIndex += 1
      continue

    vSimilarity, vWindow, vSecondStart = vBest
    vStart = lWords[vIndex].start
    vEnd = lWords[vIndex + vWindow - 1].end
    vSecondEndTime = lWords[min(vSecondStart + vWindow - 1, vCount - 1)].end
    vGap = lWords[vSecondStart].start - vEnd

    if _fOverlapsClaimed(vStart, vEnd, lClaimed):
      vIndex += 1
      continue

    vCounter += 1
    vBefore, vAfter = pTranscript.mContextText(vStart, vSecondEndTime)
    lCandidates.append(
      EditCandidate(
        id=f"repetition-{vCounter:04d}",
        type=CandidateType.POSSIBLE_REPETITION,
        start=vStart,
        end=vEnd,
        text=" ".join(vWord.word.strip() for vWord in lWords[vIndex : vIndex + vWindow]),
        context_before=vBefore[-240:],
        context_after=vAfter[:240],
        confidence=_fScore(vSimilarity, vGap, vWindow),
        details={
          "similarity": round(vSimilarity, 3),
          "gap_seconds": round(max(0.0, vGap), 3),
          "window_words": vWindow,
          "first_occurrence": {"start": round(vStart, 3), "end": round(vEnd, 3)},
          "second_occurrence": {
            "start": round(lWords[vSecondStart].start, 3),
            "end": round(vSecondEndTime, 3),
            "text": " ".join(
              vWord.word.strip() for vWord in lWords[vSecondStart : vSecondStart + vWindow]
            ),
          },
          "suggested_action": "delete the first occurrence and keep the second",
        },
      )
    )
    lClaimed.append((vStart, vSecondEndTime))
    vIndex = vSecondStart + vWindow

  return lCandidates


def _fIsMeaningful(pTokens: list[str]) -> bool:
  """Reject windows made only of punctuation or single letters."""
  lUseful = [vToken for vToken in pTokens if len(vToken) >= cMinimumTokenLength]
  return len(lUseful) >= max(1, len(pTokens) // 2)


def _fSimilarity(pFirst: list[str], pSecond: list[str]) -> float:
  """Similarity of two token sequences, 0..1."""
  if not pFirst or not pSecond:
    return 0.0
  if pFirst == pSecond:
    return 1.0
  return float(SequenceMatcher(None, " ".join(pFirst), " ".join(pSecond)).ratio())


def _fOverlapsClaimed(pStart: float, pEnd: float, pClaimed: list[tuple[float, float]]) -> bool:
  return any(pStart < vEnd and pEnd > vStart for vStart, vEnd in pClaimed)


def _fScore(pSimilarity: float, pGap: float, pWindow: int) -> float:
  """Higher for near-identical, closely spaced, longer repetitions."""
  vGapScore = max(0.2, 1.0 - (max(0.0, pGap) / cGapDecaySeconds))
  vLengthScore = min(1.0, 0.4 + 0.1 * pWindow)
  vScore = 0.55 * pSimilarity + 0.25 * vGapScore + 0.20 * vLengthScore
  return float(min(1.0, max(0.0, vScore)))
