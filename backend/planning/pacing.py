"""Tight pacing: every gap between two words is shortened to a breath.

An editor working by hand does not hunt for silences. Listening through the
episode, they cut whatever stands between two words when it lasts longer than
a breath - a pause, a breath, a throat clearing, a chair, a click, a sound of
the mouth - and leave about a tenth of a second. That is what makes an
episode sound tight rather than slow, and it leaves no low noise in the pauses
for anybody to clean up afterwards. Measured on an episode edited by hand, the
pause left at a join was 0.12 s at the median and under 0.25 s nine times out
of ten, after a full stop or not.

The gap is measured from the energy, not from the word timestamps: Whisper's
timestamps run early or late by a few tenths of a second, and a cut placed on
them clips the tail of a word ("cos-" for "cosas"). The end of a word is where
its energy decays below the speech threshold and stays there; the start of the
next one is where it rises above it.

A phrase made only of mouth noises ("¿Eh?", "¡Pssst!", "Mmm") standing alone
between two gaps goes with the gap around it. Long sounds without words are
listened to before they are cut (:func:`fListenToGaps`): if the recogniser
hears words the transcript lacks, the gap is kept.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from podcast_editor_mcp_server.analysis.retakes import fFold
from podcast_editor_mcp_server.audio.samples import Envelope, fLoudRuns, fSpeechThresholdDb
from podcast_editor_mcp_server.config import SilenceSettings
from podcast_editor_mcp_server.logging_setup import fGetLogger
from podcast_editor_mcp_server.models.edit_plan import EditPlan
from podcast_editor_mcp_server.models.transcript import Transcript, Word
from podcast_editor_mcp_server.transcription.cleanup import fContainsHallucination, fFoldText, fIsHallucinationText

vLog = fGetLogger("planning.pacing")

# The usual level of the voice: this percentile of the RMS envelope.
cSpeechLevelPercentile: float = 90.0
# Energy has to stay under the threshold this long to count as the end of a
# word (a stop consonant dips for less).
cQuietRunSeconds: float = 0.05
# How far the energy of a word may run past (or start before) its timestamp.
cEdgeReachSeconds: float = 0.35
# Shortest cut worth making.
cMinimumGapCutSeconds: float = 0.04
# Slack around a word's timestamp when checking that it holds any sound.
cFloatingMargin: float = 0.02
# How far from its timestamp a misplaced word's sound may be.
cFloatingReach: float = 0.5
# A stretch whose loudest instant stays this far under the usual level of the
# voice is a breath or a mouth noise (a hand edit removed 76-88 % of them),
# unless a word heard this clearly lies this much inside it.
cQuietRunDropDb: float = 10.0
cQuietRunWordProbability: float = 0.6
cQuietRunWordInside: float = 0.6
# A sound without words at least this long is listened to before it is cut.
cListenMinimumSeconds: float = 0.4
cListenMinimumWords: int = 2
# Words heard in a gap that are mostly the words just around it are a second
# copy of a phrase, not missing speech.
cListenContextSeconds: float = 4.0
cListenRepeatShare: float = 0.7
cSampleRate: int = 16_000

Recognizer = Callable[[np.ndarray], str]


@dataclass(slots=True)
class GapCut:
  """One gap between words, and the part of it that goes."""

  start: float
  end: float


class SpeechEdges:
  """Where words really start and end, from the RMS envelope."""

  def __init__(self, pEnvelope: Envelope) -> None:
    self.envelope = pEnvelope
    self.threshold_db = fSpeechThresholdDb(pEnvelope.rms_db)
    self.speech_level_db = pEnvelope.mPercentile(cSpeechLevelPercentile)
    self._aLoud = pEnvelope.rms_db > self.threshold_db
    self._vHop = max(pEnvelope.hop_seconds, 1e-3)
    self._vQuietFrames = max(1, int(round(cQuietRunSeconds / self._vHop)))
    self._vReachFrames = max(1, int(round(cEdgeReachSeconds / self._vHop)))
    self._aRuns: list[tuple[float, float]] | None = None

  def _mIndex(self, pTime: float) -> int:
    vTimes = self.envelope.times
    if vTimes.size == 0:
      return 0
    return int(np.clip(np.searchsorted(vTimes, pTime), 0, vTimes.size - 1))

  def _mTime(self, pIndex: int) -> float:
    return float(self.envelope.times[pIndex])

  def mRuns(self) -> list[tuple[float, float]]:
    """Stretches of sound, ``(start, end)`` in seconds; dips shorter than a stop consonant are bridged."""
    if self._aRuns is None:
      vHalf = self.envelope.frame_seconds / 2.0
      self._aRuns = fLoudRuns(self._aLoud, self.envelope.times - vHalf, self.envelope.times + vHalf, self._vQuietFrames)
    return self._aRuns

  def mPeakDb(self, pStart: float, pEnd: float) -> float:
    """Loudest RMS frame inside ``[pStart, pEnd]``, in dBFS."""
    if self.envelope.times.size == 0:
      return -120.0
    aSlice = self.envelope.rms_db[self._mIndex(pStart):self._mIndex(pEnd) + 1]
    return float(aSlice.max()) if aSlice.size else -120.0

  def mLoudSeconds(self, pStart: float, pEnd: float) -> float:
    """Longest run of loud frames inside ``[pStart, pEnd]``, in seconds."""
    if pEnd <= pStart or self.envelope.times.size == 0:
      return 0.0
    aSlice = self._aLoud[self._mIndex(pStart):self._mIndex(pEnd) + 1]
    vLongest = 0
    vRun = 0
    for vValue in aSlice:
      vRun = vRun + 1 if vValue else 0
      vLongest = max(vLongest, vRun)
    return vLongest * self._vHop

  def mEndOf(self, pWordStart: float, pWordEnd: float, pLimit: float) -> float:
    """Where the energy of the word timed ``[pWordStart, pWordEnd]`` dies away."""
    if self.envelope.times.size == 0:
      return min(pWordEnd, pLimit)
    vIndex = self._mIndex(pWordEnd)
    vLimitIndex = min(self._mIndex(pLimit), vIndex + self._vReachFrames)
    if self._aLoud[vIndex]:
      # Still sounding at the timestamp: follow the decay forward.
      vLast = vIndex
      vQuiet = 0
      vCursor = vIndex + 1
      while vCursor <= vLimitIndex and vQuiet < self._vQuietFrames:
        if self._aLoud[vCursor]:
          vLast = vCursor
          vQuiet = 0
        else:
          vQuiet += 1
        vCursor += 1
      return min(self._mTime(vLast) + self.envelope.frame_seconds / 2.0, pLimit)
    # Already quiet: the word ended earlier than its timestamp says - often
    # much earlier, when Whisper stretched it over the pause that follows.
    vFloor = self._mIndex(pWordStart)
    vCursor = vIndex - 1
    while vCursor >= vFloor:
      if self._aLoud[vCursor]:
        return min(self._mTime(vCursor) + self.envelope.frame_seconds / 2.0, pLimit)
      vCursor -= 1
    return min(pWordEnd, pLimit)

  def mStartOf(self, pWordStart: float, pWordEnd: float, pLimit: float) -> float:
    """Where the energy of the word timed ``[pWordStart, pWordEnd]`` rises."""
    if self.envelope.times.size == 0:
      return max(pWordStart, pLimit)
    vIndex = self._mIndex(pWordStart)
    vLimitIndex = max(self._mIndex(pLimit), vIndex - self._vReachFrames)
    if self._aLoud[vIndex]:
      # Already sounding at the timestamp: follow the onset backward.
      vFirst = vIndex
      vQuiet = 0
      vCursor = vIndex - 1
      while vCursor >= vLimitIndex and vQuiet < self._vQuietFrames:
        if self._aLoud[vCursor]:
          vFirst = vCursor
          vQuiet = 0
        else:
          vQuiet += 1
        vCursor -= 1
      return max(self._mTime(vFirst) - self.envelope.frame_seconds / 2.0, pLimit)
    # Still quiet: the word starts later than its timestamp says - often much
    # later, when Whisper stretched it over the pause in front of it.
    vCeiling = self._mIndex(pWordEnd)
    vCursor = vIndex + 1
    while vCursor <= vCeiling:
      if self._aLoud[vCursor]:
        return max(self._mTime(vCursor) - self.envelope.frame_seconds / 2.0, pLimit)
      vCursor += 1
    return max(pWordStart, pLimit)


def fSpokenWords(pTranscript: Transcript, pSettings: SilenceSettings) -> list[Word]:
  """The transcript's words, without phrases made only of mouth noises.

  A phrase is a run of words with no gap longer than ``word_gap_max_ms``
  inside; when every word of it is a vocal noise, the whole phrase is dropped,
  so it falls inside the gap around it.
  """
  lWords = sorted((vWord for vWord in pTranscript.mAllWords() if vWord.end > vWord.start), key=lambda pWord: pWord.start)
  if not lWords:
    return []
  sNoises = {fFold(vNoise) for vNoise in pSettings.vocal_noises if fFold(vNoise)}
  vMaxGap = pSettings.word_gap_max_ms / 1000.0
  lPhrases: list[list[Word]] = [[lWords[0]]]
  for vWord in lWords[1:]:
    if vWord.start - lPhrases[-1][-1].end > vMaxGap:
      lPhrases.append([vWord])
    else:
      lPhrases[-1].append(vWord)
  lSpoken: list[Word] = []
  for lPhrase in lPhrases:
    lFolded = [fFold(vWord.word) for vWord in lPhrase]
    if all(not vText or vText in sNoises for vText in lFolded):
      continue
    lSpoken.extend(lPhrase)
  return lSpoken


def fSpeechRuns(pTranscript: Transcript, pEdges: SpeechEdges, pSettings: SilenceSettings) -> list[tuple[float, float]]:
  """The stretches of sound that are speech, in time order.

  The recording is split into stretches of sound by its energy (dips shorter
  than a stop consonant do not split a stretch). A stretch is speech when a
  spoken word's timestamp overlaps it. A word whose timestamp holds no sound
  at all was misplaced by the recogniser into a pause; its sound is the
  nearest stretch, within ``cFloatingReach``. Every other stretch - a breath
  on its own, a throat clearing, a click, a chair - is not speech.
  """
  lRuns = pEdges.mRuns()
  if not lRuns:
    return []
  lStarts = [vRun[0] for vRun in lRuns]
  lSpeech = [False] * len(lRuns)
  # A stretch far quieter than the voice is a breath or a mouth noise even
  # when a word's timestamp touches it, unless a word the recogniser heard
  # clearly lies almost whole inside it.
  vQuietBelow = pEdges.speech_level_db - cQuietRunDropDb
  lQuiet = [pEdges.mPeakDb(vStart, vEnd) < vQuietBelow for vStart, vEnd in lRuns]
  for vWord in fSpokenWords(pTranscript, pSettings):
    vFrom = vWord.start - cFloatingMargin
    vTo = vWord.end + cFloatingMargin
    vIndex = max(0, int(np.searchsorted(lStarts, vFrom, side="right")) - 1)
    vFound = False
    while vIndex < len(lRuns) and lRuns[vIndex][0] < vTo:
      vRunStart, vRunEnd = lRuns[vIndex]
      if vRunEnd > vFrom:
        vFound = True
        if not lQuiet[vIndex]:
          lSpeech[vIndex] = True
        else:
          vInside = (min(vRunEnd, vWord.end) - max(vRunStart, vWord.start)) / max(vWord.end - vWord.start, 1e-3)
          vProbability = vWord.probability if vWord.probability is not None else 0.0
          if vInside >= cQuietRunWordInside and vProbability >= cQuietRunWordProbability:
            lSpeech[vIndex] = True
      vIndex += 1
    if vFound:
      continue
    # Misplaced word: the nearest stretch of sound is where it was said.
    vBest = -1
    vBestDistance = cFloatingReach
    for vCandidate in (vIndex - 1, vIndex):
      if 0 <= vCandidate < len(lRuns):
        vDistance = max(lRuns[vCandidate][0] - vTo, vFrom - lRuns[vCandidate][1], 0.0)
        if vDistance <= vBestDistance:
          vBest = vCandidate
          vBestDistance = vDistance
    if vBest >= 0 and not lQuiet[vBest]:
      lSpeech[vBest] = True
  return [vRun for vRun, vIsSpeech in zip(lRuns, lSpeech) if vIsSpeech]


def fWordGapCuts(
  pTranscript: Transcript,
  pEdges: SpeechEdges,
  pSettings: SilenceSettings,
) -> list[GapCut]:
  """Every gap between two stretches of speech that is longer than allowed.

  The gap is measured from the energy, so it is found even when a word's
  timestamp was stretched over it. Whatever sound the gap holds that is not
  speech goes with it. The lead-in before the first word and the tail after
  the last one are left to the head and tail rules.
  """
  lRuns = fSpeechRuns(pTranscript, pEdges, pSettings)
  vMaxGap = pSettings.word_gap_max_ms / 1000.0
  vKeepAfter = pSettings.word_gap_keep_after_ms / 1000.0
  vKeepBefore = pSettings.word_gap_keep_before_ms / 1000.0
  lCuts: list[GapCut] = []
  for (_, vEnd), (vStart, _) in zip(lRuns, lRuns[1:]):
    if vStart - vEnd <= vMaxGap:
      continue
    vCutStart = vEnd + vKeepAfter
    vCutEnd = vStart - vKeepBefore
    if vCutEnd - vCutStart < cMinimumGapCutSeconds:
      continue
    lCuts.append(GapCut(start=vCutStart, end=vCutEnd))
  return lCuts


def fListenToGaps(
  pPlan: EditPlan,
  pAudio: np.ndarray,
  pEdges: SpeechEdges,
  pRecognize: Recognizer,
  pHallucinationPhrases: list[str],
  pSettings: SilenceSettings,
  pTranscript: Transcript | None = None,
) -> tuple[EditPlan, list[dict[str, Any]]]:
  """Keep the gaps whose long sounds turn out to be words the transcript lacks.

  Only cuts of the pacing (notes ``gap-*``) holding a sound of at least
  ``cListenMinimumSeconds`` are listened to. When the recogniser hears at
  least two words there that are neither mouth noises nor a known
  hallucination, and that do not merely repeat the words around the gap (a
  phrase said twice in a row, of which the transcript kept one copy), the cut
  is dropped and reported.
  """
  sNoises = {fFold(vNoise) for vNoise in pSettings.vocal_noises if fFold(vNoise)}
  lAllWords = pTranscript.mAllWords() if pTranscript is not None else []
  ldKeptGaps: list[dict[str, Any]] = []
  sDropped: set[str] = set()
  for vOperation in pPlan.mStructuralOperations():
    vNote = str(getattr(vOperation, "note", "") or "")
    if vOperation.type != "delete_range" or not vNote.startswith("gap-"):
      continue
    vStart = float(vOperation.start)  # type: ignore[attr-defined]
    vEnd = float(vOperation.end)  # type: ignore[attr-defined]
    if pEdges.mLoudSeconds(vStart, vEnd) < cListenMinimumSeconds:
      continue
    aSamples = pAudio[int(vStart * cSampleRate):int(vEnd * cSampleRate)]
    if aSamples.size < int(0.2 * cSampleRate):
      continue
    try:
      vHeard = pRecognize(aSamples)
    except Exception as vError:  # noqa: BLE001 - never fatal
      vLog.warning("Could not listen to a gap", extra={"reason": str(vError)[:200]})
      continue
    if not vHeard or fIsHallucinationText(vHeard, pHallucinationPhrases) or fContainsHallucination(vHeard, pHallucinationPhrases):
      continue
    lWords = [vText for vText in (fFold(vRaw) for vRaw in fFoldText(vHeard).split()) if vText and vText not in sNoises]
    if len(lWords) < cListenMinimumWords:
      continue
    sAround = {
      fFold(vWord.word)
      for vWord in lAllWords
      if vStart - cListenContextSeconds <= vWord.start and vWord.end <= vEnd + cListenContextSeconds
    }
    if sum(1 for vText in lWords if vText in sAround) >= cListenRepeatShare * len(lWords):
      continue
    sDropped.add(str(vOperation.id))
    ldKeptGaps.append({"start": round(vStart, 3), "end": round(vEnd, 3), "heard": vHeard})
  if not sDropped:
    return (pPlan, ldKeptGaps)
  lOperations = [vOperation for vOperation in pPlan.operations if str(getattr(vOperation, "id", "")) not in sDropped]
  vLog.info("Gaps kept because they hold speech", extra={"count": len(sDropped)})
  return (pPlan.model_copy(update={"operations": lOperations}), ldKeptGaps)
