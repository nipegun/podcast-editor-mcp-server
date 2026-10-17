"""Candidate collection: one call, every detector."""

from __future__ import annotations

from podcast_editor_mcp_server.analysis.false_starts import fFindFalseStartCandidates
from podcast_editor_mcp_server.analysis.fillers import fFindFillerCandidates
from podcast_editor_mcp_server.analysis.repetitions import fFindRepetitionCandidates
from podcast_editor_mcp_server.analysis.retakes import fFindRetakeCandidates
from podcast_editor_mcp_server.config import Profile
from podcast_editor_mcp_server.models.audio import AnalysisReport, SilenceRegion
from podcast_editor_mcp_server.models.candidates import CandidateSet, CandidateType, EditCandidate
from podcast_editor_mcp_server.models.transcript import Transcript

# Gaps between speech regions shorter than this are breaths, not pauses.
cMinimumVadGapSeconds: float = 0.25


def fVadGaps(pReport: AnalysisReport) -> list[SilenceRegion]:
  """The stretches between the VAD's speech regions, head and tail included."""
  vDuration = pReport.probe.duration_seconds
  lRegions = sorted(pReport.speech_regions, key=lambda pRegion: pRegion.start)
  if not lRegions:
    return []
  lGaps: list[SilenceRegion] = []
  vPrevious = 0.0
  for vRegion in lRegions:
    if vRegion.start - vPrevious >= cMinimumVadGapSeconds:
      lGaps.append(SilenceRegion(start=vPrevious, end=vRegion.start, duration=vRegion.start - vPrevious))
    vPrevious = max(vPrevious, vRegion.end)
  if vDuration - vPrevious >= cMinimumVadGapSeconds:
    lGaps.append(SilenceRegion(start=vPrevious, end=vDuration, duration=vDuration - vPrevious))
  return lGaps


def fPauseRegions(pReport: AnalysisReport, pProfile: Profile) -> list[SilenceRegion]:
  """Measured silences, joined with the VAD's gaps when the profile asks for them."""
  if not pProfile.silence.detect_from_vad or not pReport.speech_regions:
    return list(pReport.silences)
  lAll = sorted(list(pReport.silences) + fVadGaps(pReport), key=lambda pRegion: pRegion.start)
  lMerged: list[list[float]] = []
  for vRegion in lAll:
    if lMerged and vRegion.start <= lMerged[-1][1]:
      lMerged[-1][1] = max(lMerged[-1][1], vRegion.end)
      continue
    lMerged.append([vRegion.start, vRegion.end])
  return [SilenceRegion(start=vStart, end=vEnd, duration=vEnd - vStart) for vStart, vEnd in lMerged]


def fFindSilenceCandidates(pReport: AnalysisReport, pProfile: Profile) -> list[EditCandidate]:
  """Turn measured pauses into candidates, split by position in the file."""
  vDuration = pReport.probe.duration_seconds
  vRemoveAbove = pProfile.silence.remove_when_longer_than_ms / 1000.0
  vHeadTarget = pProfile.silence.head_target_ms / 1000.0
  vTailTarget = pProfile.silence.tail_target_ms / 1000.0
  lCandidates: list[EditCandidate] = []
  vCounter = 0
  vFromVad = pProfile.silence.detect_from_vad and bool(pReport.speech_regions)

  for vRegion in fPauseRegions(pReport, pProfile):
    vIsHead = vRegion.start <= 0.05
    vIsTail = vRegion.end >= vDuration - 0.05
    vCounter += 1

    if vIsHead and pProfile.silence.trim_head:
      if vRegion.duration <= vHeadTarget:
        continue
      lCandidates.append(
        EditCandidate(
          id=f"head-silence-{vCounter:04d}",
          type=CandidateType.HEAD_SILENCE,
          start=vRegion.start,
          end=vRegion.end,
          text="",
          confidence=0.95,
          details={
            "duration": round(vRegion.duration, 3),
            "target_duration": round(vHeadTarget, 3),
            "from_vad": vFromVad,
            "suggested_action": f"keep {vHeadTarget:.2f}s of lead-in",
          },
        )
      )
      continue

    if vIsTail and pProfile.silence.trim_tail:
      if vRegion.duration <= vTailTarget:
        continue
      lCandidates.append(
        EditCandidate(
          id=f"tail-silence-{vCounter:04d}",
          type=CandidateType.TAIL_SILENCE,
          start=vRegion.start,
          end=vRegion.end,
          text="",
          confidence=0.95,
          details={
            "duration": round(vRegion.duration, 3),
            "target_duration": round(vTailTarget, 3),
            "from_vad": vFromVad,
            "suggested_action": f"keep {vTailTarget:.2f}s of tail",
          },
        )
      )
      continue

    if vRegion.duration <= vRemoveAbove:
      continue

    lCandidates.append(
      EditCandidate(
        id=f"silence-{vCounter:04d}",
        type=CandidateType.LONG_SILENCE,
        start=vRegion.start,
        end=vRegion.end,
        text="",
        confidence=0.9,
        details={
          "duration": round(vRegion.duration, 3),
          "target_duration": round(pProfile.silence.replacement_duration_ms / 1000.0, 3),
          "threshold_db": round(pReport.silence_threshold_db, 2),
          "from_vad": vFromVad,
          "suggested_action": "shorten to the configured replacement duration",
        },
      )
    )

  return lCandidates


def fCollectCandidates(
  pReport: AnalysisReport,
  pProfile: Profile,
  pTranscript: Transcript | None = None,
) -> CandidateSet:
  """Run every detector and return one ordered candidate set."""
  lCandidates: list[EditCandidate] = list(fFindSilenceCandidates(pReport, pProfile))
  lWarnings: list[str] = []

  if pTranscript is not None:
    lCandidates.extend(fFindFillerCandidates(pTranscript, pProfile.fillers))
    lCandidates.extend(fFindRepetitionCandidates(pTranscript, pProfile.repetitions))
    lCandidates.extend(fFindFalseStartCandidates(pTranscript, pProfile.false_starts))
    lCandidates.extend(fFindRetakeCandidates(pTranscript, pProfile.retakes))
    if not pTranscript.has_word_timestamps:
      lWarnings.append(
        "The transcript has no word timestamps, so filler, repetition, false-start and "
        "retake detection are limited. Enable transcription.word_timestamps."
      )
  else:
    lWarnings.append("No transcript available: only silence candidates were produced")

  lCandidates.sort(key=lambda pCandidate: (pCandidate.start, pCandidate.end))
  return CandidateSet(
    source=pReport.probe.label,
    source_sha256=pReport.probe.sha256,
    source_duration=pReport.probe.duration_seconds,
    candidates=lCandidates,
    warnings=lWarnings,
  )
