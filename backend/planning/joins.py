"""Join verification: listen to every seam before rendering.

Word timestamps are approximate, and they run early exactly where an editor
needs them most: where one word runs into the next. A cut placed by numbers
alone can swallow the tail of the last word that should stay ("con respec|")
or leave the head of the first word that should go ("po|"), and nothing in the
plan shows it. So every cut through speech is listened to:

1. The audio the listener will hear around the seam is assembled - a few
   seconds kept before the cut, crossfaded into a few seconds kept after it,
   skipping every other removal exactly as the renderer will.
2. Whisper transcribes that excerpt.
3. The words heard are aligned with the words the transcript says should be
   there. The seam is intact when the last word before it and the first word
   after it are both heard, and nothing extra is heard between them.

When a seam fails, the boundary on the failing side is moved to other quiet
instants nearby, one at a time, until a seam passes or the attempts run out. A
cut that never passes is kept as planned and reported as unverified.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import numpy as np

from podcast_editor_mcp_server.analysis.retakes import fFold, fSameToken
from podcast_editor_mcp_server.config import Profile
from podcast_editor_mcp_server.logging_setup import fGetLogger
from podcast_editor_mcp_server.models.edit_plan import EditPlan
from podcast_editor_mcp_server.models.transcript import Transcript

vLog = fGetLogger("planning.joins")

cSampleRate: int = 16_000
cCheckedReasons: frozenset[str] = frozenset({"mistake", "repetition", "false_start", "filler_word"})
cCrossfadeSeconds: float = 0.010
cExpectedWords: int = 4
cMinimumCutSeconds: float = 0.08
# Search windows for alternative boundaries, around the planned one.
cSearchEarlier: float = 0.30
cSearchLater: float = 0.45
cFrameSeconds: float = 0.010
cSmoothFrames: int = 3
cNeighbourMargin: float = 0.01
cHeardSimilarity: float = 0.75
cMinimumExcerptSeconds: float = 0.3

Recognizer = Callable[[np.ndarray], str]


@dataclass(slots=True)
class JoinCheck:
  """What the check found at one seam."""

  operation_id: str
  verified: bool
  adjusted: bool
  attempts: int
  heard: str
  expected: str
  start: float
  end: float
  original_start: float
  original_end: float

  def mAsDict(self) -> dict[str, Any]:
    return {
      "operation_id": self.operation_id,
      "verified": self.verified,
      "adjusted": self.adjusted,
      "attempts": self.attempts,
      "heard": self.heard,
      "expected": self.expected,
      "start": round(self.start, 3),
      "end": round(self.end, 3),
      "original_start": round(self.original_start, 3),
      "original_end": round(self.original_end, 3),
    }


# --------------------------------------------------------------------------- #
# Removals and kept audio
# --------------------------------------------------------------------------- #


def _fRemovals(pPlan: EditPlan) -> list[tuple[float, float, str]]:
  """Source spans the renderer removes, with the id of the operation behind each."""
  lRemovals: list[tuple[float, float, str]] = []
  for vOperation in pPlan.mStructuralOperations():
    vType = vOperation.type
    if vType == "delete_range":
      lRemovals.append((vOperation.start, vOperation.end, vOperation.id))  # type: ignore[attr-defined]
    elif vType == "shorten_silence":
      vHalf = vOperation.target_duration / 2.0  # type: ignore[attr-defined]
      vStart = vOperation.start + vHalf  # type: ignore[attr-defined]
      vEnd = vOperation.end - (vOperation.target_duration - vHalf)  # type: ignore[attr-defined]
      if vEnd > vStart:
        lRemovals.append((vStart, vEnd, vOperation.id))
    elif vType == "trim_start":
      lRemovals.append((0.0, vOperation.duration, vOperation.id))  # type: ignore[attr-defined]
    elif vType == "trim_end":
      lRemovals.append((pPlan.source_duration - vOperation.duration, pPlan.source_duration, vOperation.id))  # type: ignore[attr-defined]
  return sorted(lRemovals)


def _fKeptBefore(pTime: float, pSeconds: float, pRemovals: list[tuple[float, float, str]], pSkip: str) -> list[tuple[float, float]]:
  """Kept spans that end at ``pTime`` and add up to ``pSeconds``, oldest first."""
  lPieces: list[tuple[float, float]] = []
  vCursor = pTime
  vLeft = pSeconds
  lBlocking = sorted((vRemoval for vRemoval in pRemovals if vRemoval[2] != pSkip and vRemoval[0] < pTime), key=lambda pItem: -pItem[1])
  for vStart, vEnd, _ in lBlocking:
    if vLeft <= 0.0:
      break
    if vEnd >= vCursor:
      vCursor = min(vCursor, vStart)
      continue
    vTake = min(vLeft, vCursor - vEnd)
    if vTake > 0.0:
      lPieces.append((vCursor - vTake, vCursor))
      vLeft -= vTake
    vCursor = vStart
  if vLeft > 0.0 and vCursor > 0.0:
    vTake = min(vLeft, vCursor)
    lPieces.append((vCursor - vTake, vCursor))
  return sorted(lPieces)


def _fKeptAfter(
  pTime: float, pSeconds: float, pRemovals: list[tuple[float, float, str]], pSkip: str, pDuration: float,
) -> list[tuple[float, float]]:
  """Kept spans that start at ``pTime`` and add up to ``pSeconds``."""
  lPieces: list[tuple[float, float]] = []
  vCursor = pTime
  vLeft = pSeconds
  lBlocking = sorted((vRemoval for vRemoval in pRemovals if vRemoval[2] != pSkip and vRemoval[1] > pTime), key=lambda pItem: pItem[0])
  for vStart, vEnd, _ in lBlocking:
    if vLeft <= 0.0:
      break
    if vStart <= vCursor:
      vCursor = max(vCursor, vEnd)
      continue
    vTake = min(vLeft, vStart - vCursor)
    if vTake > 0.0:
      lPieces.append((vCursor, vCursor + vTake))
      vLeft -= vTake
    vCursor = vEnd
  if vLeft > 0.0 and vCursor < pDuration:
    vTake = min(vLeft, pDuration - vCursor)
    lPieces.append((vCursor, vCursor + vTake))
  return lPieces


def _fSamples(pAudio: np.ndarray, pPieces: list[tuple[float, float]]) -> np.ndarray:
  lParts = [pAudio[int(vStart * cSampleRate):int(vEnd * cSampleRate)] for vStart, vEnd in pPieces]
  lParts = [vPart for vPart in lParts if vPart.size]
  if not lParts:
    return np.zeros(0, dtype=np.float32)
  return np.concatenate(lParts).astype(np.float32, copy=False)


def _fJoin(pLeft: np.ndarray, pRight: np.ndarray) -> np.ndarray:
  vFade = int(cCrossfadeSeconds * cSampleRate)
  if pLeft.size < vFade or pRight.size < vFade:
    return np.concatenate([pLeft, pRight]).astype(np.float32, copy=False)
  aRamp = np.linspace(1.0, 0.0, vFade, dtype=np.float32)
  aMix = pLeft[-vFade:] * aRamp + pRight[:vFade] * (1.0 - aRamp)
  return np.concatenate([pLeft[:-vFade], aMix, pRight[vFade:]]).astype(np.float32, copy=False)


def _fWordsIn(pWords: list[tuple[float, float, str]], pPieces: list[tuple[float, float]]) -> list[str]:
  lTokens: list[str] = []
  for vStart, vEnd, vText in pWords:
    vMiddle = (vStart + vEnd) / 2.0
    if any(vPieceStart <= vMiddle <= vPieceEnd and vStart >= vPieceStart - 0.05 for vPieceStart, vPieceEnd in pPieces):
      lTokens.append(vText)
  return lTokens


# --------------------------------------------------------------------------- #
# Seam check
# --------------------------------------------------------------------------- #


def fSeamIntact(pLeft: list[str], pRight: list[str], pHeard: str) -> tuple[bool, bool]:
  """Whether the words on each side of a seam survive: (left intact, right intact)."""
  lExpected = pLeft + pRight
  if not lExpected:
    return (True, True)
  lHeard = [fFold(vPart) for vPart in pHeard.split()]
  lHeard = [vToken for vToken in lHeard if vToken]
  # Words heard slightly differently ("distros"/"distras") still count as heard.
  lMapped: list[str] = []
  for vToken in lHeard:
    vBest = max(lExpected, key=lambda pItem: SequenceMatcher(None, vToken, pItem).ratio())
    if vToken == vBest or fSameToken(vToken, vBest) or SequenceMatcher(None, vToken, vBest).ratio() >= cHeardSimilarity:
      lMapped.append(vBest)
    else:
      lMapped.append(vToken)

  vSeam = len(pLeft)
  vLeftOk = True
  vRightOk = True
  for vTag, vI1, vI2, vJ1, vJ2 in SequenceMatcher(None, lExpected, lMapped, autojunk=False).get_opcodes():
    if vTag == "equal":
      continue
    vMissing = vTag == "delete" or (vTag == "replace" and (vI2 - vI1) > (vJ2 - vJ1))
    vExtra = vTag == "insert" or (vTag == "replace" and (vJ2 - vJ1) > (vI2 - vI1))
    if vSeam > 0 and vI1 <= vSeam - 1 < vI2 and vMissing:
      # The last word before the seam is missing: tolerated only for a short
      # function word when the word before it was heard.
      if not (len(pLeft[-1]) <= 2 and vI1 == vSeam - 1 and vI2 == vSeam):
        vLeftOk = False
    if vSeam < len(lExpected) and vI1 <= vSeam < vI2 and vMissing:
      if not (len(pRight[0]) <= 2 and vI1 == vSeam and vI2 == vSeam + 1):
        vRightOk = False
    if vExtra and vI1 <= vSeam <= vI2 and (vI1 == vSeam or vI2 == vSeam or (vI1 < vSeam < vI2)):
      # Something extra at the seam: a fragment of a removed word leaked.
      vLeftOk = False
      vRightOk = False
  return (vLeftOk, vRightOk)


# --------------------------------------------------------------------------- #
# Alternatives
# --------------------------------------------------------------------------- #


def _fQuietInstants(pAudio: np.ndarray, pLow: float, pHigh: float, pAnchor: float) -> list[float]:
  """Local minima of the short-term RMS in ``[pLow, pHigh]``, nearest to ``pAnchor`` first."""
  vLow = max(0.0, pLow)
  vHigh = min(pAudio.size / cSampleRate, pHigh)
  if vHigh - vLow < 3 * cFrameSeconds:
    return []
  vFrame = int(cFrameSeconds * cSampleRate)
  aSegment = pAudio[int(vLow * cSampleRate):int(vHigh * cSampleRate)].astype(np.float64)
  vFrames = aSegment.size // vFrame
  if vFrames < 3:
    return []
  aRms = np.sqrt(np.mean(aSegment[: vFrames * vFrame].reshape(vFrames, vFrame) ** 2, axis=1) + 1e-12)
  aSmooth = np.convolve(aRms, np.ones(cSmoothFrames) / cSmoothFrames, mode="same")
  vMedian = float(np.median(aSmooth))
  lInstants: list[float] = []
  for vIndex in range(1, vFrames - 1):
    if aSmooth[vIndex] <= aSmooth[vIndex - 1] and aSmooth[vIndex] <= aSmooth[vIndex + 1] and aSmooth[vIndex] < 0.6 * vMedian:
      lInstants.append(vLow + (vIndex + 0.5) * cFrameSeconds)
  lInstants.sort(key=lambda pTime: abs(pTime - pAnchor))
  return lInstants


def _fNeighbourLimits(pPlan: EditPlan, pOperationId: str, pStart: float, pEnd: float) -> tuple[float, float]:
  """How far a cut may move before it touches another operation."""
  vLower = 0.0
  vUpper = pPlan.source_duration
  for vOperation in pPlan.mStructuralOperations():
    if vOperation.id == pOperationId or not hasattr(vOperation, "start"):
      continue
    if vOperation.end <= pStart + 1e-6:  # type: ignore[attr-defined]
      vLower = max(vLower, vOperation.end)  # type: ignore[attr-defined]
    elif vOperation.start >= pEnd - 1e-6:  # type: ignore[attr-defined]
      vUpper = min(vUpper, vOperation.start)  # type: ignore[attr-defined]
  return (vLower + cNeighbourMargin, vUpper - cNeighbourMargin)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def fVerifyJoins(
  pPlan: EditPlan,
  pAudio: np.ndarray,
  pTranscript: Transcript,
  pProfile: Profile,
  pRecognize: Recognizer,
) -> tuple[EditPlan, list[JoinCheck]]:
  """Listen to every cut through speech and move the boundaries that bite.

  ``pAudio`` is the source at 16 kHz mono; ``pRecognize`` turns a short buffer
  of it into text. Returns the adjusted plan and one check per seam.
  """
  vContext = pProfile.cuts.verify_context_seconds
  vAttemptsLimit = pProfile.cuts.verify_max_attempts
  lWords = [
    (vWord.start, max(vWord.end, vWord.start), fFold(vWord.word))
    for vWord in pTranscript.mAllWords()
    if fFold(vWord.word)
  ]
  dUpdates: dict[str, tuple[float, float]] = {}
  lChecks: list[JoinCheck] = []
  vPlan = pPlan

  for vOperation in pPlan.mStructuralOperations():
    if vOperation.type != "delete_range" or str(vOperation.reason) not in cCheckedReasons:
      continue
    vOriginalStart = float(vOperation.start)  # type: ignore[attr-defined]
    vOriginalEnd = float(vOperation.end)  # type: ignore[attr-defined]
    if vOriginalEnd - vOriginalStart < cMinimumCutSeconds:
      continue

    lRemovals = _fRemovals(vPlan)
    vLower, vUpper = _fNeighbourLimits(vPlan, vOperation.id, vOriginalStart, vOriginalEnd)

    def fTry(pStart: float, pEnd: float) -> tuple[bool, bool, str, str]:
      lLeftPieces = _fKeptBefore(pStart, vContext, lRemovals, vOperation.id)
      lRightPieces = _fKeptAfter(pEnd, vContext, lRemovals, vOperation.id, pPlan.source_duration)
      lLeftWords = _fWordsIn(lWords, lLeftPieces)[-cExpectedWords:]
      lRightWords = _fWordsIn(lWords, lRightPieces)[:cExpectedWords]
      aJoined = _fJoin(_fSamples(pAudio, lLeftPieces), _fSamples(pAudio, lRightPieces))
      if aJoined.size < cMinimumExcerptSeconds * cSampleRate or not (lLeftWords or lRightWords):
        # Nothing to listen to (a cut at the very start or end of the file).
        return (True, True, "", " ".join(lLeftWords) + " | " + " ".join(lRightWords))
      vHeard = pRecognize(aJoined)
      vLeftOk, vRightOk = fSeamIntact(lLeftWords, lRightWords, vHeard)
      return (vLeftOk, vRightOk, vHeard, " ".join(lLeftWords) + " | " + " ".join(lRightWords))

    vStart, vEnd = vOriginalStart, vOriginalEnd
    vLeftOk, vRightOk, vHeard, vExpected = fTry(vStart, vEnd)
    vAttempts = 1
    if not (vLeftOk and vRightOk):
      lStarts = [vStart]
      lEnds = [vEnd]
      if not vLeftOk:
        lStarts = [
          vTime for vTime in _fQuietInstants(pAudio, vStart - cSearchEarlier, vStart + cSearchLater, vStart)
          if vLower <= vTime < vEnd - cMinimumCutSeconds
        ] or [vStart]
      if not vRightOk:
        lEnds = [
          vTime for vTime in _fQuietInstants(pAudio, vEnd - cSearchLater, vEnd + cSearchEarlier, vEnd)
          if vStart + cMinimumCutSeconds < vTime <= vUpper
        ] or [vEnd]
      lCombinations = [(vCandidateStart, vCandidateEnd) for vCandidateStart in lStarts[:4] for vCandidateEnd in lEnds[:4]]
      lCombinations = [vPair for vPair in lCombinations if vPair != (vOriginalStart, vOriginalEnd)]
      for vCandidateStart, vCandidateEnd in lCombinations:
        if vAttempts >= vAttemptsLimit:
          break
        if vCandidateEnd - vCandidateStart < cMinimumCutSeconds:
          continue
        vAttempts += 1
        vLeftTry, vRightTry, vHeardTry, vExpectedTry = fTry(vCandidateStart, vCandidateEnd)
        if vLeftTry and vRightTry:
          vStart, vEnd = vCandidateStart, vCandidateEnd
          vLeftOk, vRightOk, vHeard, vExpected = vLeftTry, vRightTry, vHeardTry, vExpectedTry
          break

    vVerified = vLeftOk and vRightOk
    vAdjusted = vVerified and (abs(vStart - vOriginalStart) > 1e-6 or abs(vEnd - vOriginalEnd) > 1e-6)
    if vAdjusted:
      dUpdates[vOperation.id] = (vStart, vEnd)
      vPlan = _fWithBoundaries(vPlan, vOperation.id, vStart, vEnd)
    lChecks.append(
      JoinCheck(
        operation_id=vOperation.id,
        verified=vVerified,
        adjusted=vAdjusted,
        attempts=vAttempts,
        heard=vHeard[:300],
        expected=vExpected[:300],
        start=vStart if vVerified else vOriginalStart,
        end=vEnd if vVerified else vOriginalEnd,
        original_start=vOriginalStart,
        original_end=vOriginalEnd,
      )
    )

  vVerifiedCount = sum(1 for vCheck in lChecks if vCheck.verified)
  vLog.info(
    "Joins checked",
    extra={"checked": len(lChecks), "verified": vVerifiedCount, "adjusted": len(dUpdates)},
  )
  dMetadata = dict(vPlan.metadata)
  dMetadata["joins"] = {
    "checked": len(lChecks),
    "verified": vVerifiedCount,
    "adjusted": len(dUpdates),
    "unverified": [vCheck.mAsDict() for vCheck in lChecks if not vCheck.verified],
  }
  return (vPlan.model_copy(update={"metadata": dMetadata}), lChecks)


def _fWithBoundaries(pPlan: EditPlan, pOperationId: str, pStart: float, pEnd: float) -> EditPlan:
  lOperations = [
    vOperation.model_copy(update={"start": pStart, "end": pEnd}) if vOperation.id == pOperationId else vOperation
    for vOperation in pPlan.operations
  ]
  return pPlan.model_copy(update={"operations": lOperations})
