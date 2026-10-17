"""Backend-independent transcript representation.

Whatever engine produced the text, the rest of the product only ever sees this
shape. Timestamps are always seconds from the start of the *source* file.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TranscriptModel(BaseModel):
  model_config = ConfigDict(extra="forbid")


class Word(TranscriptModel):
  """One recognised word with its time span."""

  word: str
  start: float = Field(ge=0.0)
  end: float = Field(ge=0.0)
  probability: float | None = Field(default=None, ge=0.0, le=1.0)

  def mGetNormalised(self) -> str:
    """Lower-cased text without surrounding punctuation or whitespace."""
    return self.word.strip().strip(".,;:!?¡¿\"'()[]…-").lower()

  normalised = property(mGetNormalised)


class Segment(TranscriptModel):
  """One transcript segment, typically a sentence or clause."""

  id: int = Field(ge=0)
  start: float = Field(ge=0.0)
  end: float = Field(ge=0.0)
  text: str
  words: list[Word] = Field(default_factory=list)
  avg_logprob: float | None = None
  no_speech_probability: float | None = Field(default=None, ge=0.0, le=1.0)
  compression_ratio: float | None = None
  temperature: float | None = None

  def mGetDuration(self) -> float:
    return max(0.0, self.end - self.start)

  duration = property(mGetDuration)


class Transcript(TranscriptModel):
  """A full transcript plus the provenance needed to trust it."""

  schema_version: int = 1
  language: str
  language_probability: float | None = Field(default=None, ge=0.0, le=1.0)
  duration: float = Field(ge=0.0)
  backend: str
  model: str
  device: str = "cpu"
  compute_type: str = "default"
  source_sha256: str = ""
  generated_at: str = ""
  segments: list[Segment] = Field(default_factory=list)
  has_word_timestamps: bool = False

  def mGetText(self) -> str:
    """The whole transcript as one whitespace-joined string."""
    return " ".join(vSegment.text.strip() for vSegment in self.segments if vSegment.text.strip())

  text = property(mGetText)

  def mGetWordCount(self) -> int:
    if self.has_word_timestamps:
      return sum(len(vSegment.words) for vSegment in self.segments)
    return len(self.text.split())

  word_count = property(mGetWordCount)

  def mAllWords(self) -> list[Word]:
    """Flatten every word of every segment, in chronological order."""
    lWords: list[Word] = []
    for vSegment in self.segments:
      lWords.extend(vSegment.words)
    lWords.sort(key=lambda pWord: (pWord.start, pWord.end))
    return lWords

  def mSegmentsBetween(self, pStart: float, pEnd: float) -> list[Segment]:
    """Every segment that overlaps ``[pStart, pEnd]``."""
    return [vSegment for vSegment in self.segments if vSegment.end > pStart and vSegment.start < pEnd]

  def mContextText(self, pStart: float, pEnd: float, pWindow: float = 6.0) -> tuple[str, str]:
    """Return the text just before and just after a region."""
    lBefore = [vS.text.strip() for vS in self.segments if vS.end <= pStart and vS.end >= pStart - pWindow]
    lAfter = [vS.text.strip() for vS in self.segments if vS.start >= pEnd and vS.start <= pEnd + pWindow]
    return (" ".join(lBefore).strip(), " ".join(lAfter).strip())
