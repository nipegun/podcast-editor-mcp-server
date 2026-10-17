"""Timeline algebra: the answer to "what happens to timestamps after a cut?".

Naively applying cuts one after another is wrong, because every cut shifts all
later timestamps. This module never does that. Instead it takes the *whole*
structural part of an EditPlan, all expressed in coordinates of the original
file, and compiles it once into an ordered list of items:

* :class:`KeepSegment` - a half-open span ``[source_start, source_end)`` of the
  original that survives;
* :class:`SilenceSegment` - generated silence inserted at a known source
  position.

The output duration, the source-to-output mapping and the rendering plan are
then all read off that single structure. Nothing mutates coordinates, so the
order in which operations were authored cannot change the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from podcast_editor_mcp_server.errors import ExecutionError
from podcast_editor_mcp_server.models.edit_plan import (
  DeleteRange,
  EditPlan,
  InsertSilence,
  ReplaceRangeWithSilence,
  ShortenSilence,
  TrimEnd,
  TrimStart,
)

# Anything shorter than this is numerical noise, not audio.
cEpsilon: float = 1e-6


@dataclass(slots=True)
class KeepSegment:
  """A surviving span of the original file."""

  source_start: float
  source_end: float
  fade_in: float = 0.0
  fade_out: float = 0.0

  def mGetDuration(self) -> float:
    return max(0.0, self.source_end - self.source_start)

  duration = property(mGetDuration)


@dataclass(slots=True)
class SilenceSegment:
  """Generated silence."""

  duration: float


TimelineItem = KeepSegment | SilenceSegment


@dataclass(slots=True)
class Removal:
  """A span of the original that the plan deletes, with its justification."""

  start: float
  end: float
  operation_ids: list[str] = field(default_factory=list)
  reasons: list[str] = field(default_factory=list)

  def mGetDuration(self) -> float:
    return max(0.0, self.end - self.start)

  duration = property(mGetDuration)


@dataclass(slots=True)
class Insertion:
  """A silence insertion request in source coordinates."""

  at: float
  duration: float
  order: int


class Timeline:
  """A compiled structural edit."""

  def __init__(
    self,
    pSourceDuration: float,
    pItems: list[TimelineItem],
    pRemovals: list[Removal],
  ) -> None:
    self.source_duration = float(pSourceDuration)
    self.items: list[TimelineItem] = list(pItems)
    self.removals: list[Removal] = list(pRemovals)

  # ----------------------------------------------------------- aggregation --

  def mGetKeepSegments(self) -> list[KeepSegment]:
    return [vItem for vItem in self.items if isinstance(vItem, KeepSegment)]

  keep_segments = property(mGetKeepSegments)

  def mGetSilenceSegments(self) -> list[SilenceSegment]:
    return [vItem for vItem in self.items if isinstance(vItem, SilenceSegment)]

  silence_segments = property(mGetSilenceSegments)

  def mGetKeptSeconds(self) -> float:
    return sum(vItem.duration for vItem in self.keep_segments)

  kept_seconds = property(mGetKeptSeconds)

  def mGetInsertedSeconds(self) -> float:
    return sum(vItem.duration for vItem in self.silence_segments)

  inserted_seconds = property(mGetInsertedSeconds)

  def mGetRemovedSeconds(self) -> float:
    return sum(vRemoval.duration for vRemoval in self.removals)

  removed_seconds = property(mGetRemovedSeconds)

  def mGetOutputDuration(self) -> float:
    return self.kept_seconds + self.inserted_seconds

  output_duration = property(mGetOutputDuration)

  def mGetRemovedFraction(self) -> float:
    if self.source_duration <= 0.0:
      return 0.0
    return min(1.0, self.removed_seconds / self.source_duration)

  removed_fraction = property(mGetRemovedFraction)

  def mGetIsEmpty(self) -> bool:
    return self.output_duration <= cEpsilon

  is_empty = property(mGetIsEmpty)

  def mGetIsIdentity(self) -> bool:
    """True when the timeline reproduces the source unchanged."""
    if self.silence_segments:
      return False
    lKeep = self.keep_segments
    if len(lKeep) != 1:
      return False
    return abs(lKeep[0].source_start) <= cEpsilon and abs(lKeep[0].source_end - self.source_duration) <= cEpsilon

  is_identity = property(mGetIsIdentity)

  # -------------------------------------------------------------- mappings --

  def mMapSourceToOutput(self, pTime: float) -> float | None:
    """Where a source instant ends up in the rendered file.

    Returns ``None`` when the instant was deleted.
    """
    vElapsed = 0.0
    for vItem in self.items:
      if isinstance(vItem, SilenceSegment):
        vElapsed += vItem.duration
        continue
      if vItem.source_start - cEpsilon <= pTime <= vItem.source_end + cEpsilon:
        return vElapsed + max(0.0, pTime - vItem.source_start)
      vElapsed += vItem.duration
    return None


# --------------------------------------------------------------------------- #
# Compilation
# --------------------------------------------------------------------------- #


def fMergeRanges(pRemovals: list[Removal], pTolerance: float = cEpsilon) -> list[Removal]:
  """Sort and merge overlapping or touching removals.

  Merging is what makes overlapping operations harmless at render time: the
  validator may still reject them as a policy decision, but the engine never
  double-counts a second of audio.
  """
  lSorted = sorted((vRemoval for vRemoval in pRemovals if vRemoval.duration > pTolerance), key=lambda pR: (pR.start, pR.end))
  lMerged: list[Removal] = []
  for vRemoval in lSorted:
    if lMerged and vRemoval.start <= lMerged[-1].end + pTolerance:
      vLast = lMerged[-1]
      vLast.end = max(vLast.end, vRemoval.end)
      vLast.operation_ids.extend(vRemoval.operation_ids)
      vLast.reasons.extend(vRemoval.reasons)
    else:
      lMerged.append(Removal(start=vRemoval.start, end=vRemoval.end, operation_ids=list(vRemoval.operation_ids), reasons=list(vRemoval.reasons)))
  return lMerged


def fCollectRemovals(pPlan: EditPlan) -> list[Removal]:
  """Translate every structural operation into source-coordinate removals."""
  vDuration = pPlan.source_duration
  lRemovals: list[Removal] = []

  # isinstance rather than a comparison on ``type``: the union is closed, so
  # the type checker verifies that every branch really has the fields it uses,
  # and an operation added later without a branch here fails loudly.
  for vOperation in pPlan.mStructuralOperations():
    vReason = str(vOperation.reason)
    if isinstance(vOperation, DeleteRange | ReplaceRangeWithSilence):
      lRemovals.append(Removal(vOperation.start, vOperation.end, [vOperation.id], [vReason]))
    elif isinstance(vOperation, ShortenSilence):
      # Keep half of the target duration on each side so the decay of the
      # previous phrase and the onset of the next one both survive.
      vKeepLeft = vOperation.target_duration / 2.0
      vKeepRight = vOperation.target_duration - vKeepLeft
      lRemovals.append(
        Removal(vOperation.start + vKeepLeft, vOperation.end - vKeepRight, [vOperation.id], [vReason])
      )
    elif isinstance(vOperation, TrimStart):
      lRemovals.append(Removal(0.0, min(vOperation.duration, vDuration), [vOperation.id], [vReason]))
    elif isinstance(vOperation, TrimEnd):
      lRemovals.append(Removal(max(0.0, vDuration - vOperation.duration), vDuration, [vOperation.id], [vReason]))
    elif isinstance(vOperation, InsertSilence):
      continue
    else:  # pragma: no cover - the discriminated union prevents this
      raise ExecutionError(
        f"Unsupported structural operation: {vOperation.type}", {"operation_id": vOperation.id}
      )

  # Clamp to the file and drop anything degenerate.
  lClamped: list[Removal] = []
  for vRemoval in lRemovals:
    vStart = max(0.0, min(vRemoval.start, vDuration))
    vEnd = max(0.0, min(vRemoval.end, vDuration))
    if vEnd - vStart > cEpsilon:
      lClamped.append(Removal(vStart, vEnd, vRemoval.operation_ids, vRemoval.reasons))
  return lClamped


def fCollectInsertions(pPlan: EditPlan) -> list[Insertion]:
  """Translate every silence-producing operation into insertion requests."""
  lInsertions: list[Insertion] = []
  for vIndex, vOperation in enumerate(pPlan.mStructuralOperations()):
    if isinstance(vOperation, InsertSilence):
      if vOperation.duration > cEpsilon:
        lInsertions.append(
          Insertion(
            at=min(vOperation.at, pPlan.source_duration),
            duration=vOperation.duration,
            order=vIndex,
          )
        )
    elif isinstance(vOperation, ReplaceRangeWithSilence):
      vSilence = vOperation.effective_silence_duration
      if vSilence > cEpsilon:
        lInsertions.append(
          Insertion(
            at=vOperation.start,
            duration=vSilence,
            order=vIndex,
          )
        )
  return lInsertions


def fBuildTimeline(pPlan: EditPlan, pCrossfadeSeconds: float = 0.0) -> Timeline:
  """Compile the structural part of ``pPlan`` into a :class:`Timeline`."""
  vDuration = float(pPlan.source_duration)
  if vDuration <= 0.0:
    raise ExecutionError("Source duration must be positive to build a timeline", {"plan_id": pPlan.id})

  lRemovals = fMergeRanges(fCollectRemovals(pPlan))
  lInsertions = fCollectInsertions(pPlan)

  # Keep segments are the complement of the removals inside [0, duration].
  lKeep: list[KeepSegment] = []
  vCursor = 0.0
  for vRemoval in lRemovals:
    if vRemoval.start - vCursor > cEpsilon:
      lKeep.append(KeepSegment(vCursor, vRemoval.start))
    vCursor = max(vCursor, vRemoval.end)
  if vDuration - vCursor > cEpsilon:
    lKeep.append(KeepSegment(vCursor, vDuration))

  # Split kept segments at interior insertion points so silence lands exactly
  # where the plan asked for it.
  lSplitPoints = sorted({vInsertion.at for vInsertion in lInsertions})
  if lSplitPoints:
    lKeep = _fSplitSegments(lKeep, lSplitPoints)

  # Order everything by source position. Silence sorts before a kept segment
  # starting at the same instant, so "insert at 0" pads the head.
  lEvents: list[tuple[float, int, int, TimelineItem]] = []
  for vIndex, vSegment in enumerate(lKeep):
    lEvents.append((vSegment.source_start, 1, vIndex, vSegment))
  for vInsertion in lInsertions:
    lEvents.append(
      (
        vInsertion.at,
        0,
        vInsertion.order,
        SilenceSegment(duration=vInsertion.duration),
      )
    )
  lEvents.sort(key=lambda pEvent: (pEvent[0], pEvent[1], pEvent[2]))
  lItems: list[TimelineItem] = [vEvent[3] for vEvent in lEvents]

  if pCrossfadeSeconds > 0.0:
    _fApplySeamFades(lItems, pCrossfadeSeconds)

  return Timeline(vDuration, lItems, lRemovals)


def _fSplitSegments(pSegments: list[KeepSegment], pPoints: list[float]) -> list[KeepSegment]:
  """Split kept segments at every interior point in ``pPoints``."""
  lResult: list[KeepSegment] = []
  for vSegment in pSegments:
    lInterior = [vPoint for vPoint in pPoints if vSegment.source_start + cEpsilon < vPoint < vSegment.source_end - cEpsilon]
    if not lInterior:
      lResult.append(vSegment)
      continue
    vCursor = vSegment.source_start
    for vPoint in lInterior:
      lResult.append(KeepSegment(vCursor, vPoint))
      vCursor = vPoint
    lResult.append(KeepSegment(vCursor, vSegment.source_end))
  return lResult


def _fApplySeamFades(pItems: list[TimelineItem], pCrossfadeSeconds: float) -> None:
  """Mark short fades on the seams between kept segments.

  A hard splice between two arbitrary waveform positions produces a step
  discontinuity, which is audible as a click. A few milliseconds of fade on
  each side of every internal seam removes it without changing the running
  time, which a true cross-fade would.

  Seams that are merely the original audio continuing (segment B starts exactly
  where segment A ended, which happens around an inserted silence) are left
  alone. The very first and very last edges are also left alone: those belong
  to the ``fade_in``/``fade_out`` mastering operations.
  """
  lKeep: list[KeepSegment] = [vItem for vItem in pItems if isinstance(vItem, KeepSegment)]
  for vPosition, vSegment in enumerate(lKeep):
    vBudget = min(pCrossfadeSeconds, vSegment.duration / 2.0)
    if vBudget <= cEpsilon:
      continue
    if vPosition > 0:
      vPrevious = lKeep[vPosition - 1]
      if abs(vPrevious.source_end - vSegment.source_start) > cEpsilon:
        vSegment.fade_in = vBudget
    if vPosition < len(lKeep) - 1:
      vNext = lKeep[vPosition + 1]
      if abs(vSegment.source_end - vNext.source_start) > cEpsilon:
        vSegment.fade_out = vBudget
