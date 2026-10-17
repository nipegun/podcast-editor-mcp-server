"""Filler-word candidate detection.

A textual match is never enough on its own: "esto" is a filler in
"y esto... vamos a instalar" and an ordinary pronoun in "esto es un problema".
So every hit is scored against measurable evidence - how long the word lasted,
how confident the recogniser was, and whether it stands alone between two
pauses - and the result is a *candidate*, not a deletion.
"""

from __future__ import annotations

import re

from podcast_editor_mcp_server.config import FillerSettings
from podcast_editor_mcp_server.models.candidates import CandidateType, EditCandidate
from podcast_editor_mcp_server.models.transcript import Transcript, Word

# A filler that is part of a fluent run of speech is far more likely to be a
# real word, so isolation carries most of the weight in the score.
cIsolationWeight: float = 0.45
cDurationWeight: float = 0.25
cProbabilityWeight: float = 0.30
cShortFillerSeconds: float = 0.35


def fNormaliseToken(pText: str) -> str:
  """Lower-case a token and strip punctuation for comparison."""
  return pText.strip().strip(".,;:!?¡¿\"'()[]…-").lower()


def fFindFillerCandidates(pTranscript: Transcript, pSettings: FillerSettings) -> list[EditCandidate]:
  """Return filler-word candidates found in ``pTranscript``."""
  if not pSettings.enabled or not (pSettings.words or pSettings.patterns):
    return []
  if not pTranscript.has_word_timestamps:
    return _fSegmentLevelCandidates(pTranscript, pSettings)

  lPhrases = [tuple(vPhrase.split()) for vPhrase in pSettings.words if vPhrase.strip()]
  lPhrases.sort(key=len, reverse=True)
  lPatterns = [re.compile(vPattern) for vPattern in pSettings.patterns]
  sKeepBefore = set(pSettings.keep_when_followed_by)
  vMaxDuration = pSettings.max_duration_ms / 1000.0
  vIsolationGap = pSettings.isolation_gap_ms / 1000.0

  lCandidates: list[EditCandidate] = []
  vCounter = 0

  for vSegment in pTranscript.segments:
    lWords = vSegment.words
    if not lWords:
      continue
    lNormalised = [fNormaliseToken(vWord.word) for vWord in lWords]
    vIndex = 0
    while vIndex < len(lWords):
      vMatchLength = _fMatchPhrase(lNormalised, vIndex, lPhrases)
      if vMatchLength == 0 and any(vPattern.fullmatch(lNormalised[vIndex]) for vPattern in lPatterns):
        vMatchLength = 1
      if vMatchLength == 0:
        vIndex += 1
        continue
      # "o sea que", "a ver que": the filler is part of the sentence.
      vNextIndex = vIndex + vMatchLength
      if vNextIndex < len(lNormalised) and lNormalised[vNextIndex] in sKeepBefore:
        vIndex += vMatchLength
        continue

      lMatched = lWords[vIndex : vIndex + vMatchLength]
      vStart = lMatched[0].start
      vEnd = lMatched[-1].end
      vDuration = max(0.0, vEnd - vStart)
      if vDuration > vMaxDuration:
        vIndex += vMatchLength
        continue

      vProbability = _fAverageProbability(lMatched)
      if vProbability is not None and vProbability < pSettings.min_probability:
        vIndex += vMatchLength
        continue

      vGapBefore = _fGapBefore(lWords, vIndex)
      vGapAfter = _fGapAfter(lWords, vIndex + vMatchLength - 1)
      vIsolated = (vGapBefore >= vIsolationGap) or (vGapAfter >= vIsolationGap)
      if pSettings.require_isolation and not vIsolated:
        vIndex += vMatchLength
        continue

      vCounter += 1
      vBefore, vAfter = pTranscript.mContextText(vStart, vEnd)
      lCandidates.append(
        EditCandidate(
          id=f"filler-{vCounter:04d}",
          type=CandidateType.FILLER_WORD,
          start=vStart,
          end=vEnd,
          text=" ".join(vWord.word.strip() for vWord in lMatched),
          context_before=vBefore[-240:],
          context_after=vAfter[:240],
          confidence=_fScore(vDuration, vProbability, vIsolated),
          details={
            "segment_id": vSegment.id,
            "duration": round(vDuration, 3),
            "gap_before": round(vGapBefore, 3),
            "gap_after": round(vGapAfter, 3),
            "isolated": vIsolated,
            "word_probability": round(vProbability, 3) if vProbability is not None else None,
          },
        )
      )
      vIndex += vMatchLength

  return _fApplyRateLimit(lCandidates, pTranscript.duration, pSettings.max_per_minute)


def _fMatchPhrase(pNormalised: list[str], pIndex: int, pPhrases: list[tuple[str, ...]]) -> int:
  """Return the length of the longest configured phrase starting at ``pIndex``."""
  for lPhrase in pPhrases:
    vLength = len(lPhrase)
    if pIndex + vLength > len(pNormalised):
      continue
    if tuple(pNormalised[pIndex : pIndex + vLength]) == lPhrase:
      return vLength
  return 0


def _fAverageProbability(pWords: list[Word]) -> float | None:
  lValues = [vWord.probability for vWord in pWords if vWord.probability is not None]
  if not lValues:
    return None
  return sum(lValues) / len(lValues)


def _fGapBefore(pWords: list[Word], pIndex: int) -> float:
  if pIndex == 0:
    return 999.0
  return max(0.0, pWords[pIndex].start - pWords[pIndex - 1].end)


def _fGapAfter(pWords: list[Word], pIndex: int) -> float:
  if pIndex >= len(pWords) - 1:
    return 999.0
  return max(0.0, pWords[pIndex + 1].start - pWords[pIndex].end)


def _fScore(pDuration: float, pProbability: float | None, pIsolated: bool) -> float:
  """Blend the evidence into a 0..1 confidence."""
  vIsolation = 1.0 if pIsolated else 0.25
  vDurationScore = 1.0 if pDuration <= cShortFillerSeconds else max(0.2, cShortFillerSeconds / max(pDuration, 1e-6))
  vProbabilityScore = pProbability if pProbability is not None else 0.6
  vScore = (
    cIsolationWeight * vIsolation
    + cDurationWeight * vDurationScore
    + cProbabilityWeight * vProbabilityScore
  )
  return float(min(1.0, max(0.0, vScore)))


def _fSegmentLevelCandidates(pTranscript: Transcript, pSettings: FillerSettings) -> list[EditCandidate]:
  """Fallback when the backend produced no word timestamps.

  Without word boundaries a filler cannot be cut safely, so these candidates
  are reported with a low confidence and flagged as not directly actionable.
  """
  lCandidates: list[EditCandidate] = []
  vCounter = 0
  sWords = {vWord for vWord in pSettings.words if " " not in vWord}
  for vSegment in pTranscript.segments:
    lTokens = [fNormaliseToken(vToken) for vToken in vSegment.text.split()]
    lHits = [vToken for vToken in lTokens if vToken in sWords]
    if not lHits:
      continue
    vCounter += 1
    lCandidates.append(
      EditCandidate(
        id=f"filler-seg-{vCounter:04d}",
        type=CandidateType.FILLER_WORD,
        start=vSegment.start,
        end=vSegment.end,
        text=", ".join(sorted(set(lHits))),
        context_before="",
        context_after=vSegment.text[:240],
        confidence=0.15,
        details={
          "segment_id": vSegment.id,
          "actionable": False,
          "why": "no word timestamps; enable transcription.word_timestamps to cut fillers",
        },
      )
    )
  return lCandidates


def _fApplyRateLimit(
  pCandidates: list[EditCandidate],
  pDuration: float,
  pMaxPerMinute: float,
) -> list[EditCandidate]:
  """Drop the least confident candidates when a recording looks over-detected.

  A run-away detector is a sign the word list is too aggressive for this
  speaker, not that the speaker said "eh" four hundred times.
  """
  if pMaxPerMinute <= 0.0 or pDuration <= 0.0:
    return pCandidates
  vAllowed = int(pMaxPerMinute * (pDuration / 60.0)) + 1
  if len(pCandidates) <= vAllowed:
    return pCandidates
  lRanked = sorted(pCandidates, key=lambda pCandidate: pCandidate.confidence, reverse=True)[:vAllowed]
  return sorted(lRanked, key=lambda pCandidate: pCandidate.start)

