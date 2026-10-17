"""Edit candidates.

A candidate is *evidence*, never a decision. The detectors in
:mod:`podcast_editor_mcp_server.analysis` emit candidates with enough context that either the
built-in planner or an MCP client's language model can decide what to do with
them. Nothing is removed just because a detector fired.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CandidateType(StrEnum):
  """The kinds of evidence the analysers produce."""

  FILLER_WORD = "filler_word"
  POSSIBLE_REPETITION = "possible_repetition"
  POSSIBLE_FALSE_START = "possible_false_start"
  LONG_SILENCE = "long_silence"
  HEAD_SILENCE = "head_silence"
  TAIL_SILENCE = "tail_silence"
  # A take the speaker recorded again later; the later take is the good one.
  RETAKE = "retake"
  # Talk about the recording itself: "lo voy a grabar otra vez".
  META_TALK = "meta_talk"
  # A phrase that trails off and is followed by a long pause. Whether it was
  # abandoned is a judgement call, so it always goes to review.
  ABANDONED_PHRASE = "abandoned_phrase"


class EditCandidate(BaseModel):
  """One region that *might* deserve an edit."""

  model_config = ConfigDict(extra="forbid")

  id: str = Field(max_length=64)
  type: CandidateType
  start: float = Field(ge=0.0, allow_inf_nan=False)
  end: float = Field(ge=0.0, allow_inf_nan=False)
  text: str = ""
  context_before: str = ""
  context_after: str = ""
  confidence: float = Field(default=0.0, ge=0.0, le=1.0)
  details: dict[str, Any] = Field(default_factory=dict)

  @model_validator(mode="after")
  def mCheckRange(self) -> EditCandidate:
    if self.end < self.start:
      raise ValueError("end must not be before start")
    return self

  def mGetDuration(self) -> float:
    return max(0.0, self.end - self.start)

  duration = property(mGetDuration)


class CandidateSet(BaseModel):
  """All candidates found for one source file."""

  model_config = ConfigDict(extra="forbid")

  schema_version: int = 1
  source: str
  source_sha256: str = ""
  source_duration: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
  candidates: list[EditCandidate] = Field(default_factory=list)
  warnings: list[str] = Field(default_factory=list)

  def mReviewItems(self) -> list[EditCandidate]:
    """Candidates that need a decision before they can be applied."""
    return [vCandidate for vCandidate in self.candidates if vCandidate.details.get("decision") == "review"]

  def mCountByType(self) -> dict[str, int]:
    dCounts: dict[str, int] = {}
    for vCandidate in self.candidates:
      vKey = str(vCandidate.type)
      dCounts[vKey] = dCounts.get(vKey, 0) + 1
    return dCounts
