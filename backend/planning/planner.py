"""The built-in, deterministic planner.

Given measurements and candidates it emits an EditPlan. It is opinionated but
never creative: the same inputs always produce the same plan, which is what
makes the result reviewable and reproducible.

Three problems get solved here, and they are the ones that make or break the
audible result:

1. **Where exactly to cut.** Word timestamps are approximate. Boundaries are
   widened by the configured margins, clamped so they cannot eat into the
   neighbouring words, and then nudged to the quietest instant inside a small
   window. Cutting in a trough instead of mid-phoneme is the difference
   between a clean edit and a click.
2. **Overlapping intentions.** A filler can sit inside a long silence, a false
   start can swallow a repetition. Overlaps are reconciled *before* the plan is
   emitted, in a fixed priority order, so the validator never sees a conflict.
3. **Staying under the safety budget.** If the candidates add up to more than
   the profile allows, the least confident ones are dropped until the plan fits
   rather than the whole job failing.

Retakes, talk about the recording and abandoned phrases carry a decision:
``auto`` candidates follow ``retakes.auto_remove``; ``review`` candidates are
applied only when a decision for their id says ``cut`` - or, when the caller
asks for it, when their proposed action is ``cut``. A cut that removes a take
runs up to the next take and leaves a natural pause in front of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from podcast_editor_mcp_server.audio.samples import Envelope, fFindQuietestPoint
from podcast_editor_mcp_server.config import Profile
from podcast_editor_mcp_server.logging_setup import fGetLogger
from podcast_editor_mcp_server.models.audio import AnalysisReport
from podcast_editor_mcp_server.models.candidates import CandidateSet, CandidateType, EditCandidate
from podcast_editor_mcp_server.models.edit_plan import (
  Compress,
  DeleteRange,
  Denoise,
  EditPlan,
  EditReason,
  FadeIn,
  FadeOut,
  HighPassFilter,
  InsertSilence,
  LimitTruePeak,
  LowPassFilter,
  NormalizeLoudness,
  PlanProvenance,
  ShortenSilence,
  fNewOperationId,
)
from podcast_editor_mcp_server.models.transcript import Transcript
from podcast_editor_mcp_server.planning.pacing import SpeechEdges, fWordGapCuts
from podcast_editor_mcp_server.util.hashing import fHashMapping
from podcast_editor_mcp_server.version import cProductName, cVersion

vLog = fGetLogger("planning.planner")

# Higher wins when two structural intentions overlap.
cCandidatePriority: dict[CandidateType, int] = {
  CandidateType.HEAD_SILENCE: 100,
  CandidateType.TAIL_SILENCE: 100,
  CandidateType.META_TALK: 90,
  CandidateType.RETAKE: 85,
  CandidateType.POSSIBLE_FALSE_START: 80,
  CandidateType.ABANDONED_PHRASE: 75,
  CandidateType.POSSIBLE_REPETITION: 70,
  CandidateType.FILLER_WORD: 60,
  CandidateType.LONG_SILENCE: 40,
}

cReasonForCandidate: dict[CandidateType, EditReason] = {
  CandidateType.HEAD_SILENCE: EditReason.HEAD_SILENCE,
  CandidateType.TAIL_SILENCE: EditReason.TAIL_SILENCE,
  CandidateType.POSSIBLE_FALSE_START: EditReason.FALSE_START,
  CandidateType.POSSIBLE_REPETITION: EditReason.REPETITION,
  CandidateType.FILLER_WORD: EditReason.FILLER_WORD,
  CandidateType.LONG_SILENCE: EditReason.LONG_PAUSE,
  CandidateType.RETAKE: EditReason.MISTAKE,
  CandidateType.META_TALK: EditReason.MISTAKE,
  CandidateType.ABANDONED_PHRASE: EditReason.FALSE_START,
}

# Candidates that remove a whole take, placed with fPlaceTakeCut.
cTakeTypes: frozenset[CandidateType] = frozenset(
  {CandidateType.RETAKE, CandidateType.META_TALK, CandidateType.ABANDONED_PHRASE}
)

cMinimumViableSilenceSeconds: float = 0.15
# A gap at least this long in front of or behind a removed take is a pause, and
# the cut keeps part of it so the edit breathes.
cTakeGapSeconds: float = 0.7
# Pause kept after the last word before a removed take.
cKeptPauseSeconds: float = 0.3
# Where to look for a quiet instant when a take abuts the word kept next to it.
cTightSearchBefore: float = 0.04
cTightSearchAfter: float = 0.25
# Length assumed for the word at a take's edge when only one of its timestamps
# is known (tight pacing).
cTakeEdgeReach: float = 0.3


@dataclass(slots=True)
class PendingCut:
  """A structural intention before it becomes an operation."""

  start: float
  end: float
  candidate_type: CandidateType
  reason: EditReason
  confidence: float
  candidate_id: str
  target_duration: float | None = None  # set for shorten_silence

  def mGetDuration(self) -> float:
    return max(0.0, self.end - self.start)

  duration = property(mGetDuration)

  def mGetRemovedSeconds(self) -> float:
    if self.target_duration is None:
      return self.duration
    return max(0.0, self.duration - self.target_duration)

  removed_seconds = property(mGetRemovedSeconds)

  def mGetPriority(self) -> int:
    return cCandidatePriority.get(self.candidate_type, 0)

  priority = property(mGetPriority)


def fBuildAutomaticPlan(
  pAnalysis: AnalysisReport,
  pProfile: Profile,
  pCandidates: CandidateSet,
  pTranscript: Transcript | None = None,
  pEnvelope: Envelope | None = None,
  pDecisions: dict[str, str] | None = None,
  pApplyProposals: bool = False,
) -> EditPlan:
  """Build a complete EditPlan from measurements and candidates.

  ``pDecisions`` maps candidate ids to ``"cut"`` or ``"keep"`` and overrides
  everything else for those candidates. ``pApplyProposals`` applies the
  proposed action of every review candidate nobody decided on, which is what a
  fully automatic run does.
  """
  vDuration = pAnalysis.probe.duration_seconds
  dDecisions = {vKey: str(vValue).strip().lower() for vKey, vValue in (pDecisions or {}).items()}
  lPending: list[PendingCut] = []

  # Tight pacing measures every gap between words from the energy; it needs
  # the envelope and word timestamps, and replaces the long-pause rule.
  vEdges: SpeechEdges | None = None
  if (
    pProfile.silence.tighten_word_gaps
    and pEnvelope is not None
    and pTranscript is not None
    and pTranscript.has_word_timestamps
  ):
    vEdges = SpeechEdges(pEnvelope)

  for vCandidate in pCandidates.candidates:
    vPending = _fCandidateToPending(
      vCandidate, pProfile, pAnalysis, pTranscript, pEnvelope, vDuration, dDecisions, pApplyProposals, vEdges,
    )
    if vPending is not None:
      lPending.append(vPending)

  vGapCount = 0
  if vEdges is not None and pTranscript is not None:
    for vGap in fWordGapCuts(pTranscript, vEdges, pProfile.silence):
      vGapCount += 1
      lPending.append(
        PendingCut(
          start=vGap.start,
          end=vGap.end,
          candidate_type=CandidateType.LONG_SILENCE,
          reason=EditReason.LONG_PAUSE,
          confidence=0.9,
          candidate_id=f"gap-{vGapCount:04d}",
        )
      )

  lPending = fReconcile(lPending)
  lPending = fApplyRemovalBudget(lPending, vDuration, pProfile)

  lOperations: list[object] = []
  for vPending in sorted(lPending, key=lambda pItem: pItem.start):
    if vPending.target_duration is None:
      lOperations.append(
        DeleteRange(
          start=vPending.start,
          end=vPending.end,
          reason=vPending.reason,
          confidence=vPending.confidence,
          note=vPending.candidate_id,
          source="automatic-planner",
        )
      )
    else:
      lOperations.append(
        ShortenSilence(
          start=vPending.start,
          end=vPending.end,
          target_duration=vPending.target_duration,
          reason=vPending.reason,
          confidence=vPending.confidence,
          note=vPending.candidate_id,
          source="automatic-planner",
        )
      )

  lOperations.extend(fBuildPaddingOperations(pAnalysis, pProfile, lPending))
  lOperations.extend(fBuildMasteringOperations(pAnalysis, pProfile))

  for vIndex, vOperation in enumerate(lOperations, start=1):
    vOperation.id = fNewOperationId(vIndex)  # type: ignore[attr-defined]

  vProvenance = PlanProvenance(
    generator=f"{cProductName}/{cVersion}",
    profile=pProfile.name,
    profile_sha256=fHashMapping(pProfile.mFingerprint()),
    transcription_backend=pTranscript.backend if pTranscript else None,
    transcription_model=pTranscript.model if pTranscript else None,
    vad_backend=pAnalysis.vad_backend,
    tool_versions=dict(pAnalysis.tool_versions),
  )

  return EditPlan(
    source=pAnalysis.probe.label,
    source_path=pAnalysis.probe.path,
    source_sha256=pAnalysis.probe.sha256,
    source_duration=vDuration,
    sample_rate=pAnalysis.probe.sample_rate,
    channels=pAnalysis.probe.channels,
    provenance=vProvenance,
    operations=lOperations,  # type: ignore[arg-type]
    notes="Generated by the built-in automatic planner",
    metadata={
      "candidate_count": len(pCandidates.candidates),
      "candidates_by_type": pCandidates.mCountByType(),
      "silence_threshold_db": round(pAnalysis.silence_threshold_db, 2),
      "review_items": fReviewItems(pCandidates, dDecisions, pApplyProposals, lPending),
      "decisions": dDecisions,
      "pacing": {
        "tight": vEdges is not None,
        "gaps_found": vGapCount,
        "gaps_cut": sum(1 for vCut in lPending if vCut.candidate_id.startswith("gap-")),
        "speech_threshold_db": round(vEdges.threshold_db, 1) if vEdges is not None else None,
      },
    },
  )


def fReviewItems(
  pCandidates: CandidateSet,
  pDecisions: dict[str, str],
  pApplyProposals: bool,
  pPending: list[PendingCut],
) -> list[dict[str, Any]]:
  """Summarise the candidates that needed a decision, and what happened to them."""
  sApplied = {vCut.candidate_id for vCut in pPending}
  ldItems: list[dict[str, Any]] = []
  for vCandidate in pCandidates.mReviewItems():
    vDecided = pDecisions.get(vCandidate.id)
    if vDecided is None and pApplyProposals:
      vDecided = f"proposed:{vCandidate.details.get('proposed_action', 'keep')}"
    ldItems.append(
      {
        "id": vCandidate.id,
        "kind": vCandidate.details.get("review_kind", str(vCandidate.type)),
        "start": round(vCandidate.start, 3),
        "end": round(vCandidate.end, 3),
        "duration": round(vCandidate.duration, 3),
        "text": vCandidate.text,
        "context_before": vCandidate.context_before[-200:],
        "context_after": vCandidate.context_after[:200],
        "kept_take": vCandidate.details.get("kept_take"),
        "proposed_action": vCandidate.details.get("proposed_action", "keep"),
        "why": vCandidate.details.get("why", ""),
        "decision": vDecided,
        "applied": vCandidate.id in sApplied,
      }
    )
  return ldItems


# --------------------------------------------------------------------------- #
# Candidate conversion
# --------------------------------------------------------------------------- #


def _fCandidateToPending(
  pCandidate: EditCandidate,
  pProfile: Profile,
  pAnalysis: AnalysisReport,
  pTranscript: Transcript | None,
  pEnvelope: Envelope | None,
  pDuration: float,
  pDecisions: dict[str, str] | None = None,
  pApplyProposals: bool = False,
  pEdges: SpeechEdges | None = None,
) -> PendingCut | None:
  """Decide whether and how a candidate becomes a structural intention."""
  vReason = cReasonForCandidate.get(pCandidate.type, EditReason.OTHER)
  vDecision = (pDecisions or {}).get(pCandidate.id)
  if vDecision == "keep":
    return None

  if pCandidate.type in cTakeTypes:
    if vDecision != "cut":
      if pCandidate.details.get("decision", "auto") == "auto":
        if not pProfile.retakes.auto_remove:
          return None
      elif not (pApplyProposals and pCandidate.details.get("proposed_action") == "cut"):
        return None
    vStart, vEnd = fPlaceTakeCut(pCandidate, pAnalysis, pEnvelope, pDuration, pEdges, pProfile)
    if vEnd - vStart < pProfile.cuts.min_cut_duration_ms / 1000.0:
      return None
    return PendingCut(
      start=vStart,
      end=vEnd,
      candidate_type=pCandidate.type,
      reason=vReason,
      confidence=pCandidate.confidence,
      candidate_id=pCandidate.id,
    )

  if pCandidate.type == CandidateType.LONG_SILENCE:
    if pEdges is not None:
      # Tight pacing cuts every gap between words itself.
      return None
    vTarget = pProfile.silence.replacement_duration_ms / 1000.0
    if not pProfile.silence.preserve_natural_pauses:
      vTarget = 0.0
    if pCandidate.duration - vTarget < cMinimumViableSilenceSeconds:
      return None
    return PendingCut(
      start=pCandidate.start,
      end=pCandidate.end,
      candidate_type=pCandidate.type,
      reason=vReason,
      confidence=pCandidate.confidence,
      candidate_id=pCandidate.id,
      target_duration=vTarget,
    )

  if pCandidate.type == CandidateType.HEAD_SILENCE:
    vTarget = pProfile.silence.head_target_ms / 1000.0
    vCutEnd = pCandidate.end - vTarget
    if vCutEnd <= 0.0:
      return None
    return PendingCut(
      start=0.0,
      end=min(vCutEnd, pDuration),
      candidate_type=pCandidate.type,
      reason=vReason,
      confidence=pCandidate.confidence,
      candidate_id=pCandidate.id,
    )

  if pCandidate.type == CandidateType.TAIL_SILENCE:
    vTarget = pProfile.silence.tail_target_ms / 1000.0
    vCutStart = pCandidate.start + vTarget
    if vCutStart >= pDuration:
      return None
    return PendingCut(
      start=vCutStart,
      end=pDuration,
      candidate_type=pCandidate.type,
      reason=vReason,
      confidence=pCandidate.confidence,
      candidate_id=pCandidate.id,
    )

  vAutoRemove = {
    CandidateType.FILLER_WORD: pProfile.fillers.auto_remove,
    CandidateType.POSSIBLE_REPETITION: pProfile.repetitions.auto_remove,
    CandidateType.POSSIBLE_FALSE_START: pProfile.false_starts.auto_remove,
  }.get(pCandidate.type, False)
  if not vAutoRemove and vDecision != "cut":
    return None
  if pCandidate.details.get("actionable") is False:
    return None

  vStart, vEnd = fRefineCutBoundaries(
    pCandidate.start,
    pCandidate.end,
    pProfile,
    pTranscript,
    pEnvelope,
    pDuration,
  )
  if vEnd - vStart < pProfile.cuts.min_cut_duration_ms / 1000.0:
    return None
  return PendingCut(
    start=vStart,
    end=vEnd,
    candidate_type=pCandidate.type,
    reason=vReason,
    confidence=pCandidate.confidence,
    candidate_id=pCandidate.id,
  )


def fRefineCutBoundaries(
  pStart: float,
  pEnd: float,
  pProfile: Profile,
  pTranscript: Transcript | None,
  pEnvelope: Envelope | None,
  pDuration: float,
) -> tuple[float, float]:
  """Widen a speech-derived cut by the configured margins and snap it to a trough.

  Widening is clamped by the neighbouring words: the cut may grow into the
  silence around the region, but it may never start before the previous word
  ended or end after the next word began.
  """
  vLowerBound, vUpperBound = fNeighbourBounds(pStart, pEnd, pTranscript, pDuration)

  vPre = pProfile.cuts.pre_margin_ms / 1000.0
  vPost = pProfile.cuts.post_margin_ms / 1000.0
  vStart = max(vLowerBound, pStart - vPre)
  vEnd = min(vUpperBound, pEnd + vPost)

  if pProfile.cuts.snap_to_low_energy and pEnvelope is not None:
    vWindow = pProfile.cuts.snap_window_ms / 1000.0
    vSnappedStart = fFindQuietestPoint(pEnvelope, vStart, vWindow)
    vSnappedEnd = fFindQuietestPoint(pEnvelope, vEnd, vWindow)
    vStart = min(max(vSnappedStart, vLowerBound), pStart)
    vEnd = max(min(vSnappedEnd, vUpperBound), pEnd)

  vStart = max(0.0, min(vStart, pDuration))
  vEnd = max(0.0, min(vEnd, pDuration))
  if vEnd <= vStart:
    return (pStart, pEnd)
  return (vStart, vEnd)


def fPlaceTakeCut(
  pCandidate: EditCandidate,
  pAnalysis: AnalysisReport,
  pEnvelope: Envelope | None,
  pDuration: float,
  pEdges: SpeechEdges | None = None,
  pProfile: Profile | None = None,
) -> tuple[float, float]:
  """Place the cut that removes a take, from the word edges the detector recorded.

  When a pause separates the take from the word kept before it, the cut starts
  a short, natural pause after the end of that speech (measured by the VAD,
  because word timestamps run early). When a pause separates the take from the
  next one, the cut ends where that speech starts. When the words touch, the
  boundary goes to the quietest instant between them; the join check later
  listens to it and moves it if it bites a word.

  With tight pacing (``pEdges``) the edges come from the energy of the words
  themselves, and the pause kept on each side is the pacing's.
  """
  dDetails = pCandidate.details
  vRemovedStart = float(dDetails.get("removed_start", pCandidate.start))
  vRemovedEnd = float(dDetails.get("removed_end", pCandidate.end))
  vKeptBefore = dDetails.get("kept_before_end")
  vKeptAfter = dDetails.get("kept_after_start")

  if pEdges is not None and pProfile is not None:
    return _fPlaceTightTakeCut(
      pCandidate, pEdges, pProfile, pEnvelope, pDuration, vRemovedStart, vRemovedEnd, vKeptBefore, vKeptAfter,
    )

  if vKeptBefore is None:
    vStart = max(0.0, vRemovedStart - cTightSearchBefore)
  else:
    vKeptBefore = float(vKeptBefore)
    if vRemovedStart - vKeptBefore >= cTakeGapSeconds:
      vSpeechEnd = _fSpeechEdge(pAnalysis, vKeptBefore, pAfter=True)
      if vSpeechEnd is None or vSpeechEnd > vRemovedStart - 0.1:
        vSpeechEnd = vKeptBefore
      vStart = min(max(vSpeechEnd, vKeptBefore) + cKeptPauseSeconds, vRemovedStart - 0.02)
    else:
      vStart = _fQuietestBetween(
        pEnvelope, vKeptBefore - cTightSearchBefore, vRemovedStart + cTightSearchAfter, (vKeptBefore + vRemovedStart) / 2.0,
      )

  if vKeptAfter is None:
    vEnd = min(pDuration, vRemovedEnd + 0.1)
  else:
    vKeptAfter = float(vKeptAfter)
    if vKeptAfter - vRemovedEnd >= cTakeGapSeconds:
      vSpeechStart = _fSpeechEdge(pAnalysis, vKeptAfter, pAfter=False)
      if vSpeechStart is None or vSpeechStart < vRemovedEnd + 0.1:
        vSpeechStart = vKeptAfter
      vEnd = min(vSpeechStart, vKeptAfter)
    else:
      vEnd = _fQuietestBetween(
        pEnvelope, vRemovedEnd - cTightSearchAfter, vKeptAfter + cTightSearchBefore, (vRemovedEnd + vKeptAfter) / 2.0,
      )

  vStart = max(0.0, min(vStart, pDuration))
  vEnd = max(0.0, min(vEnd, pDuration))
  if vEnd <= vStart:
    return (pCandidate.start, max(pCandidate.end, pCandidate.start))
  return (vStart, vEnd)


def _fPlaceTightTakeCut(
  pCandidate: EditCandidate,
  pEdges: SpeechEdges,
  pProfile: Profile,
  pEnvelope: Envelope | None,
  pDuration: float,
  pRemovedStart: float,
  pRemovedEnd: float,
  pKeptBefore: Any,
  pKeptAfter: Any,
) -> tuple[float, float]:
  """:func:`fPlaceTakeCut` for tight pacing: energy edges, the pacing's pauses."""
  vKeepAfter = pProfile.silence.word_gap_keep_after_ms / 1000.0
  vKeepBefore = pProfile.silence.word_gap_keep_before_ms / 1000.0
  vReach = cTakeEdgeReach

  if pKeptBefore is None:
    vStart = max(0.0, pEdges.mStartOf(pRemovedStart, pRemovedStart + vReach, 0.0) - vKeepBefore)
  else:
    vKeptBeforeEnd = float(pKeptBefore)
    vSpeechEnd = pEdges.mEndOf(vKeptBeforeEnd - vReach, vKeptBeforeEnd, pRemovedStart)
    vTakeStart = pEdges.mStartOf(pRemovedStart, pRemovedStart + vReach, vSpeechEnd)
    if vTakeStart - vSpeechEnd > vKeepAfter:
      vStart = vSpeechEnd + vKeepAfter
    else:
      vStart = _fQuietestBetween(pEnvelope, vSpeechEnd - cTightSearchBefore, vTakeStart + cTightSearchBefore, (vSpeechEnd + vTakeStart) / 2.0)

  if pKeptAfter is None:
    vEnd = min(pDuration, pEdges.mEndOf(pRemovedEnd - vReach, pRemovedEnd, pDuration) + vKeepAfter)
  else:
    vKeptAfterStart = float(pKeptAfter)
    vSpeechStart = pEdges.mStartOf(vKeptAfterStart, vKeptAfterStart + vReach, pRemovedEnd)
    vTakeEnd = pEdges.mEndOf(pRemovedEnd - vReach, pRemovedEnd, vSpeechStart)
    if vSpeechStart - vTakeEnd > vKeepBefore:
      vEnd = vSpeechStart - vKeepBefore
    else:
      vEnd = _fQuietestBetween(pEnvelope, vTakeEnd - cTightSearchBefore, vSpeechStart + cTightSearchBefore, (vTakeEnd + vSpeechStart) / 2.0)

  vStart = max(0.0, min(vStart, pDuration))
  vEnd = max(0.0, min(vEnd, pDuration))
  if vEnd <= vStart:
    return (pCandidate.start, max(pCandidate.end, pCandidate.start))
  return (vStart, vEnd)


def _fSpeechEdge(pAnalysis: AnalysisReport, pTime: float, pAfter: bool) -> float | None:
  """End (``pAfter``) or start of the VAD speech region around ``pTime``."""
  for vRegion in pAnalysis.speech_regions:
    if vRegion.start - 0.2 <= pTime <= vRegion.end + 0.2:
      return vRegion.end if pAfter else vRegion.start
  return None


def _fQuietestBetween(pEnvelope: Envelope | None, pLow: float, pHigh: float, pFallback: float) -> float:
  """The quietest instant of the RMS envelope inside ``[pLow, pHigh]``."""
  if pEnvelope is None or pHigh <= pLow:
    return pFallback
  vCentre = (pLow + pHigh) / 2.0
  return fFindQuietestPoint(pEnvelope, vCentre, (pHigh - pLow) / 2.0)


def fNeighbourBounds(
  pStart: float,
  pEnd: float,
  pTranscript: Transcript | None,
  pDuration: float,
) -> tuple[float, float]:
  """Return how far a cut may grow before it touches a neighbouring word."""
  if pTranscript is None or not pTranscript.has_word_timestamps:
    return (max(0.0, pStart - 0.25), min(pDuration, pEnd + 0.25))
  vLower = 0.0
  vUpper = pDuration
  for vWord in pTranscript.mAllWords():
    if vWord.end <= pStart + 1e-6:
      vLower = max(vLower, vWord.end)
    elif vWord.start >= pEnd - 1e-6:
      vUpper = min(vUpper, vWord.start)
      break
  return (vLower, vUpper)


# --------------------------------------------------------------------------- #
# Reconciliation and budget
# --------------------------------------------------------------------------- #


def fReconcile(pPending: list[PendingCut]) -> list[PendingCut]:
  """Remove overlaps so the emitted plan can never contain a conflict.

  Higher-priority intentions win outright. A lower-priority one is trimmed to
  the parts that do not overlap, and dropped if nothing viable is left. A
  removal that straddles an accepted one keeps both free sides: what both
  remove is their union (a retake inside a long earlier take must not leave
  the rest of that take in). A shortened pause keeps only its longer side, as
  its target duration cannot be split.
  """
  lSorted = sorted(pPending, key=lambda pItem: (-pItem.priority, -pItem.confidence, pItem.start))
  lAccepted: list[PendingCut] = []

  for vCandidate in lSorted:
    lPieces = [vCandidate]
    for vAccepted in list(lAccepted):
      lNext: list[PendingCut] = []
      for vCurrent in lPieces:
        if vCurrent.end <= vAccepted.start or vCurrent.start >= vAccepted.end:
          lNext.append(vCurrent)
          continue
        # Fully swallowed by an accepted cut.
        if vCurrent.start >= vAccepted.start and vCurrent.end <= vAccepted.end:
          continue
        lSides: list[PendingCut] = []
        if vCurrent.start < vAccepted.start:
          lSides.append(_fResize(vCurrent, vCurrent.start, vAccepted.start))
        if vCurrent.end > vAccepted.end:
          lSides.append(_fResize(vCurrent, vAccepted.end, vCurrent.end))
        if len(lSides) == 2 and vCurrent.target_duration is not None:
          lSides = [max(lSides, key=lambda pSide: pSide.duration)]
        lNext.extend(vSide for vSide in lSides if vSide.duration > cMinimumViableSilenceSeconds)
      lPieces = lNext
    for vCurrent in lPieces:
      if vCurrent.target_duration is not None and vCurrent.duration - vCurrent.target_duration <= cMinimumViableSilenceSeconds:
        continue
      if vCurrent.duration <= 0.0:
        continue
      lAccepted.append(vCurrent)

  return sorted(lAccepted, key=lambda pItem: pItem.start)


def _fResize(pCut: PendingCut, pStart: float, pEnd: float) -> PendingCut:
  """Return a copy of ``pCut`` spanning ``[pStart, pEnd]``."""
  vTarget = pCut.target_duration
  if vTarget is not None:
    vTarget = min(vTarget, max(0.0, (pEnd - pStart) - cMinimumViableSilenceSeconds))
  return PendingCut(
    start=pStart,
    end=pEnd,
    candidate_type=pCut.candidate_type,
    reason=pCut.reason,
    confidence=pCut.confidence,
    candidate_id=pCut.candidate_id,
    target_duration=vTarget,
  )


def fApplyRemovalBudget(pPending: list[PendingCut], pDuration: float, pProfile: Profile) -> list[PendingCut]:
  """Drop the least confident cuts until the plan fits the safety budget."""
  if pDuration <= 0.0:
    return pPending
  vBudget = pProfile.safety.max_deleted_fraction * pDuration
  vMaxSingle = pProfile.safety.max_single_delete_seconds
  vMaxRetake = max(vMaxSingle, pProfile.safety.max_single_retake_seconds)

  lFiltered: list[PendingCut] = []
  for vCut in pPending:
    vLimit = vMaxRetake if vCut.candidate_type in cTakeTypes else vMaxSingle
    if vCut.target_duration is None and vCut.duration > vLimit:
      vLog.info(
        "Dropping a cut that exceeds max_single_delete_seconds",
        extra={"candidate": vCut.candidate_id, "duration": round(vCut.duration, 2)},
      )
      continue
    lFiltered.append(vCut)

  vTotal = sum(vCut.removed_seconds for vCut in lFiltered)
  if vTotal <= vBudget:
    return lFiltered

  # Keep the highest priority, then the most confident, until the budget fills.
  # A pause that does not fit whole is shortened less rather than dropped: half
  # a second saved is better than none, and it keeps the budget fully used.
  lRanked = sorted(lFiltered, key=lambda pItem: (-pItem.priority, -pItem.confidence, pItem.start))
  lKept: list[PendingCut] = []
  vAccumulated = 0.0
  for vCut in lRanked:
    vRemaining = vBudget - vAccumulated
    if vCut.removed_seconds <= vRemaining:
      lKept.append(vCut)
      vAccumulated += vCut.removed_seconds
      continue
    if vCut.target_duration is not None and vRemaining > cMinimumViableSilenceSeconds:
      vRelaxedTarget = vCut.duration - vRemaining
      if vCut.duration - vRelaxedTarget > cMinimumViableSilenceSeconds:
        lKept.append(
          PendingCut(
            start=vCut.start,
            end=vCut.end,
            candidate_type=vCut.candidate_type,
            reason=vCut.reason,
            confidence=vCut.confidence,
            candidate_id=vCut.candidate_id,
            target_duration=vRelaxedTarget,
          )
        )
        vAccumulated += vRemaining
  vLog.warning(
    "Removal budget reached; dropped low-priority cuts",
    extra={
      "budget_seconds": round(vBudget, 2),
      "requested_seconds": round(vTotal, 2),
      "kept": len(lKept),
      "dropped": len(lFiltered) - len(lKept),
    },
  )
  return sorted(lKept, key=lambda pItem: pItem.start)


# --------------------------------------------------------------------------- #
# Padding and mastering
# --------------------------------------------------------------------------- #


def fBuildPaddingOperations(
  pAnalysis: AnalysisReport,
  pProfile: Profile,
  pPending: list[PendingCut],
) -> list[object]:
  """Insert lead-in/lead-out silence when the recording is shorter than the target."""
  lOperations: list[object] = []
  vDuration = pAnalysis.probe.duration_seconds

  if pProfile.silence.trim_head:
    vTarget = pProfile.silence.head_target_ms / 1000.0
    vExisting = _fHeadSilenceSeconds(pAnalysis)
    vTrimmed = any(vCut.candidate_type == CandidateType.HEAD_SILENCE for vCut in pPending)
    if not vTrimmed and vTarget - vExisting > 0.01:
      lOperations.append(
        InsertSilence(
          at=0.0,
          duration=vTarget - vExisting,
          reason=EditReason.PADDING,
          confidence=1.0,
          note="pad the head to the configured lead-in",
          source="automatic-planner",
        )
      )

  if pProfile.silence.trim_tail:
    vTarget = pProfile.silence.tail_target_ms / 1000.0
    vExisting = _fTailSilenceSeconds(pAnalysis)
    vTrimmed = any(vCut.candidate_type == CandidateType.TAIL_SILENCE for vCut in pPending)
    if not vTrimmed and vTarget - vExisting > 0.01:
      lOperations.append(
        InsertSilence(
          at=vDuration,
          duration=vTarget - vExisting,
          reason=EditReason.PADDING,
          confidence=1.0,
          note="pad the tail to the configured lead-out",
          source="automatic-planner",
        )
      )

  return lOperations


def _fHeadSilenceSeconds(pAnalysis: AnalysisReport) -> float:
  for vRegion in pAnalysis.silences:
    if vRegion.start <= 0.05:
      return vRegion.duration
  return 0.0


def _fTailSilenceSeconds(pAnalysis: AnalysisReport) -> float:
  vDuration = pAnalysis.probe.duration_seconds
  for vRegion in pAnalysis.silences:
    if vRegion.end >= vDuration - 0.05:
      return vRegion.duration
  return 0.0


def fBuildMasteringOperations(pAnalysis: AnalysisReport, pProfile: Profile) -> list[object]:
  """Build the mastering chain implied by the profile and the measurements."""
  lOperations: list[object] = []
  vCleanup = pProfile.cleanup

  if vCleanup.high_pass_hz:
    lOperations.append(
      HighPassFilter(
        frequency=vCleanup.high_pass_hz,
        reason=EditReason.CLEANUP,
        note="remove rumble below the voice band",
        source="automatic-planner",
      )
    )
  if vCleanup.low_pass_hz:
    lOperations.append(
      LowPassFilter(
        frequency=vCleanup.low_pass_hz,
        reason=EditReason.CLEANUP,
        note="remove hiss above the voice band",
        source="automatic-planner",
      )
    )
  if vCleanup.denoise:
    vFloor = pAnalysis.dynamics.noise_floor_dbfs
    vNoiseFloor: float | None = None
    if vFloor is not None:
      vNoiseFloor = min(-20.0, max(-80.0, vFloor + vCleanup.denoise_floor_offset_db))
    lOperations.append(
      Denoise(
        reduction_db=vCleanup.denoise_reduction_db,
        noise_floor_db=vNoiseFloor,
        reason=EditReason.NOISE,
        note="broadband noise reduction based on the measured noise floor",
        source="automatic-planner",
      )
    )
  if vCleanup.compress:
    vCompressor = vCleanup.compressor
    lOperations.append(
      Compress(
        threshold_db=vCompressor.threshold_db,
        ratio=vCompressor.ratio,
        attack_ms=vCompressor.attack_ms,
        release_ms=vCompressor.release_ms,
        makeup_db=vCompressor.makeup_db,
        knee_db=vCompressor.knee_db,
        reason=EditReason.MASTERING,
        source="automatic-planner",
      )
    )

  if pProfile.loudness.normalize:
    lOperations.append(
      NormalizeLoudness(
        integrated_lufs=pProfile.loudness.integrated_lufs,
        true_peak_dbtp=pProfile.loudness.true_peak_dbtp,
        loudness_range=pProfile.loudness.loudness_range,
        dual_pass=pProfile.loudness.dual_pass,
        reason=EditReason.MASTERING,
        note="EBU R128 normalisation to the profile target",
        source="automatic-planner",
      )
    )
  if pProfile.loudness.apply_limiter:
    lOperations.append(
      LimitTruePeak(
        true_peak_dbtp=pProfile.loudness.true_peak_dbtp,
        attack_ms=pProfile.loudness.limiter_attack_ms,
        release_ms=pProfile.loudness.limiter_release_ms,
        reason=EditReason.MASTERING,
        source="automatic-planner",
      )
    )

  if pProfile.output.fade_in_ms > 0:
    lOperations.append(
      FadeIn(
        duration=pProfile.output.fade_in_ms / 1000.0,
        reason=EditReason.MASTERING,
        source="automatic-planner",
      )
    )
  if pProfile.output.fade_out_ms > 0:
    lOperations.append(
      FadeOut(
        duration=pProfile.output.fade_out_ms / 1000.0,
        reason=EditReason.MASTERING,
        source="automatic-planner",
      )
    )

  return lOperations

