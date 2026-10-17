"""Retake detection: the last complete take is the good one.

A podcaster recording alone rarely says a sentence once. They start it, stop,
pause, and start again; they re-record a whole paragraph minutes later; they
comment on the recording ("lo voy a grabar en argentino mejor"). This module
finds all three, plus phrases that trail off and are never finished.

How a retake is recognised
--------------------------

By its opening. A take starts at a phrase boundary - after a pause or after a
sentence ends - and when that opening was already said earlier, the earlier
occurrence is a candidate first take::

    "No me sentí... [2.5 s] No me sentí más identificado con..."
     ^ earlier take          ^ later take: the opening repeats

The candidate is accepted when the later take repeats what the earlier one
said: the share of the earlier take's content words found at the start of the
later take is its *coverage*. A pure restart ("No me sentí...") has nothing
beyond its opening and is accepted outright; a sentence restarted with a new
ending ("la versión es la 12 que funciona... la versión es la 12 que ha
traído") is accepted when the shared opening is long. When the earlier take
says something the later one does not - a paragraph recorded again with
different content - removing it would remove content, so it goes to review
instead of being cut.

Three safeguards keep ordinary speech intact:

* the opening must contain a content word ("y el" is not an opening);
* the later take must follow a pause, trailing punctuation, or a short take:
  fluent repetition is emphasis, not a retake;
* a later match may never start inside an earlier retake and reach past its
  kept take, which would delete the take that was kept.

Every candidate carries its decision: ``auto`` when the evidence is clear,
``review`` when a person or a language model should decide, with a proposed
action and the reason.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from podcast_editor_mcp_server.config import RetakeSettings
from podcast_editor_mcp_server.models.candidates import CandidateType, EditCandidate
from podcast_editor_mcp_server.models.transcript import Transcript

# Function words, folded (lower case, no accents). An opening made only of
# these is not distinctive, and they never count towards coverage.
cStopwords: frozenset[str] = frozenset(
  {
    "a", "al", "algo", "ante", "antes", "asi", "aun", "aunque", "bien", "bueno", "cada", "como", "con",
    "cual", "cuando", "de", "del", "desde", "donde", "e", "el", "ella", "ellas", "ellos", "en", "entonces",
    "entre", "era", "eran", "eres", "es", "esa", "esas", "ese", "eso", "esos", "esta", "estaba", "estan",
    "estar", "estas", "este", "esto", "estos", "estoy", "fue", "ha", "habia", "han", "has", "hasta", "hay",
    "he", "la", "las", "le", "les", "lo", "los", "mas", "me", "mi", "mis", "mucho", "muy", "nada", "ni",
    "no", "nos", "o", "os", "otra", "otro", "para", "pero", "poco", "por", "porque", "pues", "que",
    "quien", "se", "sea", "ser", "si", "sin", "sino", "sobre", "son", "soy", "su", "sus", "tal", "tambien",
    "te", "todo", "todos", "tu", "tus", "u", "un", "una", "uno", "unos", "unas", "va", "vamos", "van",
    "vas", "voy", "vez", "y", "ya", "yo", "eh", "ehh", "mm", "mmm",
    "an", "and", "or", "of", "to", "in", "on", "at", "is", "it", "its", "that", "this", "i", "you", "we",
    "they", "he", "she", "be", "was", "were", "are", "so", "but", "for", "with", "as", "not", "do", "does",
    "did", "have", "had", "my", "your", "our", "the",
  }
)

cMaxAnchorWords: int = 12
# Above this many words an earlier take is a paragraph: it needs a longer
# opening and a higher coverage to be cut without asking.
cLongTakeWords: int = 25
cLongTakeAnchorWords: int = 4
cLongTakeCoverage: float = 0.75
# A long earlier take with at least this much coverage, but not enough to be
# cut, is reported as a retake with unique content.
cReviewMinCoverage: float = 0.45
# A review region mostly inside retakes that are cut anyway is not worth asking about.
cReviewMaxOverlap: float = 0.5
# An opening made only of function words ("pero sí", "a la que era") counts
# when the earlier take is this short and trails off.
cShortFunctionTakeWords: int = 5
# Or when at least this many of them repeat exactly and the earlier take
# hesitated right after them and was then left hanging.
cHesitantOpeningWords: int = 3
# The end of a sentence said again at the start of the next is rhetoric when
# shorter than this; this long, it is a retake.
cRepeatedTailWords: int = 6
# How far a paragraph retake's anchor may sit from the start of its phrase.
cReviewMaxAnchorOffset: int = 5
# The window of the later take compared with the earlier one.
cWindowFactor: float = 1.5
cWindowExtraWords: int = 8
cMinWindowWords: int = 12
# Pauses that bound the phrase an abandoned candidate belongs to.
cAbandonedPhraseGap: float = 0.6
# An abandoned phrase that starts like the unfinished phrase just before it
# ("Pero una de las grandes cosas / Pero uno de los...") restarts it: the
# earlier attempt goes with it when it is this close, this short, and the
# first words match.
cAbandonedRestartGap: float = 3.0
cAbandonedRestartMaxWords: int = 12
cAbandonedRestartMatchWords: int = 2
# An abandoned phrase that only echoes words just said ("... siempre ha sido
# Debian. Y siempre ha sido...") is proposed for cutting: words it must have
# once its opening conjunctions are set aside, and how far back to look.
cEchoOpenings: frozenset[str] = frozenset({"y", "e", "pero", "o", "u", "and", "but", "or"})
cEchoMinimumWords: int = 2
cEchoLookBackWords: int = 20
# Slack around a requested cut when deciding which words fall inside it.
cRequestedSlack: float = 0.05
cTextLimit: int = 300
cTokenSimilarity: float = 0.8
cShortTokenSimilarity: float = 0.6
cTagQuestions: frozenset[str] = frozenset({"no", "eh", "verdad", "si", "vale", "ok", "cierto", "right"})


@dataclass(slots=True)
class Token:
  """One transcript word, folded for comparison."""

  text: str
  raw: str
  start: float
  end: float
  trails_off: bool
  cut_off: bool
  sentence_end: bool


@dataclass(slots=True)
class RetakeRegion:
  """A span of words ``[first, last_exclusive)`` and why it goes."""

  first: int
  last_exclusive: int
  kind: str
  anchor_words: int = 0
  coverage: float = 1.0
  details: dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #


def fFold(pText: str) -> str:
  """Lower-case, strip accents and every character that is not a letter or digit."""
  vDecomposed = unicodedata.normalize("NFKD", pText.lower())
  vStripped = "".join(vChar for vChar in vDecomposed if not unicodedata.combining(vChar))
  return re.sub(r"[^0-9a-zñ]+", "", vStripped)


def fTokens(pTranscript: Transcript) -> list[Token]:
  """Every word with text, in time order, folded."""
  lTokens: list[Token] = []
  for vWord in pTranscript.mAllWords():
    vRaw = vWord.word.strip()
    vText = fFold(vRaw)
    if not vText:
      # Whisper sometimes writes the dash of a word cut off mid-way as a word
      # of its own ("Inta" "-"): it belongs to the word before.
      if lTokens and "-" in vRaw:
        lTokens[-1].cut_off = True
      continue
    # A tag question ("¿no?", "¿eh?") asks for agreement; it does not end
    # what is being said.
    vTag = vRaw.endswith("?") and vText in cTagQuestions
    lTokens.append(
      Token(
        text=vText,
        raw=vRaw,
        start=vWord.start,
        end=max(vWord.end, vWord.start),
        trails_off=vRaw.endswith(("...", "…")),
        cut_off=vRaw.endswith("-") and not vRaw.endswith("--"),
        sentence_end=vRaw.endswith((".", "?", "!", "…")) and not vTag,
      )
    )
  return lTokens


def fIsContent(pToken: str) -> bool:
  return len(pToken) >= 3 and pToken not in cStopwords


def fSameToken(pFirst: str, pSecond: str) -> bool:
  """Equal, or close enough to be the same word said twice ("compra"/"compran")."""
  if pFirst == pSecond:
    return True
  if len(pFirst) >= 4 and len(pSecond) >= 4:
    return SequenceMatcher(None, pFirst, pSecond).ratio() >= cTokenSimilarity
  if pFirst in cStopwords and pSecond in cStopwords and pFirst[:1] == pSecond[:1] and abs(len(pFirst) - len(pSecond)) <= 1:
    # Gender and number variants of the same function word: una/uno, las/los.
    return SequenceMatcher(None, pFirst, pSecond).ratio() >= cShortTokenSimilarity
  return False


def _fGapBefore(pTokens: list[Token], pIndex: int) -> float:
  if pIndex <= 0:
    return 999.0
  return max(0.0, pTokens[pIndex].start - pTokens[pIndex - 1].end)


def _fGapAfter(pTokens: list[Token], pIndex: int) -> float:
  if pIndex >= len(pTokens) - 1:
    return 999.0
  return max(0.0, pTokens[pIndex + 1].start - pTokens[pIndex].end)


def fPhraseStarts(pTokens: list[Token], pGapSeconds: float) -> list[int]:
  """Indices that open a phrase: after a pause, or after punctuation that ends one."""
  lStarts: list[int] = []
  for vIndex in range(len(pTokens)):
    if vIndex == 0:
      lStarts.append(0)
      continue
    vPrevious = pTokens[vIndex - 1]
    if (
      _fGapBefore(pTokens, vIndex) >= pGapSeconds
      or vPrevious.sentence_end
      or vPrevious.trails_off
      or vPrevious.cut_off
    ):
      lStarts.append(vIndex)
  return lStarts


# --------------------------------------------------------------------------- #
# Talk about the recording
# --------------------------------------------------------------------------- #


def fFindMetaRegions(pTokens: list[Token], pSettings: RetakeSettings, pPhraseStarts: list[int]) -> list[RetakeRegion]:
  """The phrases that contain a marker of talk about the recording."""
  lMarkers = [[fFold(vPart) for vPart in vMarker.split() if fFold(vPart)] for vMarker in pSettings.meta_markers]
  lMarkers = [vMarker for vMarker in lMarkers if vMarker]
  sStarts = set(pPhraseStarts)
  lRegions: list[RetakeRegion] = []
  vCount = len(pTokens)
  vIndex = 0
  while vIndex < vCount:
    lMatched: list[str] | None = None
    for lMarker in lMarkers:
      vLength = len(lMarker)
      if vIndex + vLength <= vCount and all(pTokens[vIndex + vOffset].text == lMarker[vOffset] for vOffset in range(vLength)):
        lMatched = lMarker
        break
    if lMatched is None:
      vIndex += 1
      continue
    vFirst = vIndex
    while vFirst > 0 and vFirst not in sStarts:
      vFirst -= 1
    vLast = vIndex + len(lMatched)
    while vLast < vCount and vLast not in sStarts:
      vLast += 1
    lRegions.append(
      RetakeRegion(first=vFirst, last_exclusive=vLast, kind="meta", details={"marker": " ".join(lMatched)})
    )
    vIndex = vLast
  return lRegions


# --------------------------------------------------------------------------- #
# Retakes
# --------------------------------------------------------------------------- #


def _fAnchorLength(pTokens: list[Token], pEarlier: int, pLater: int) -> int:
  vLength = 0
  vCount = len(pTokens)
  while (
    vLength < cMaxAnchorWords
    and pEarlier + vLength < pLater
    and pLater + vLength < vCount
    and fSameToken(pTokens[pEarlier + vLength].text, pTokens[pLater + vLength].text)
  ):
    vLength += 1
  return vLength


def _fCoverage(
  pTokens: list[Token],
  pEarlier: int,
  pLater: int,
  pAnchor: int,
  sExcluded: set[int],
) -> tuple[float, int, int]:
  """Share of the earlier take's content words, after its opening, found early in the later take.

  Returns (share, content words after the opening, how many of them were found).
  """
  lEarlier = [
    pTokens[vIndex]
    for vIndex in range(pEarlier + pAnchor, pLater)
    if vIndex not in sExcluded and fIsContent(pTokens[vIndex].text)
  ]
  if not lEarlier:
    return (1.0, 0, 0)
  vWindow = max(cMinWindowWords, int(cWindowFactor * (pLater - pEarlier)) + cWindowExtraWords)
  lLater = [
    pTokens[vIndex].text
    for vIndex in range(pLater, min(len(pTokens), pLater + vWindow))
    if vIndex not in sExcluded and fIsContent(pTokens[vIndex].text)
  ]
  vCovered = 0
  for vToken in lEarlier:
    vPrefix = vToken.text[:3]
    if any(fSameToken(vToken.text, vOther) for vOther in lLater):
      vCovered += 1
    elif vToken.cut_off and any(vOther.startswith(vPrefix) for vOther in lLater):
      # A word cut off mid-way ("inta-") matches whatever it was going to be.
      vCovered += 1
  return (vCovered / len(lEarlier), len(lEarlier), vCovered)


def _fNearPhraseStart(pIndex: int, sStarts: set[int], pMaxOffset: int) -> bool:
  return any(pIndex - vOffset in sStarts for vOffset in range(0, pMaxOffset + 1))


def _fOnlyTakesInside(pTokens: list[Token], pEarlier: int, pLater: int, pAnchor: int) -> bool:
  """Whether every finished sentence inside ``[pEarlier, pLater)`` is followed by another take.

  A run of takes ("...número 188. Muy buenas a todos... número 189. Muy
  buenas...") restarts the same opening after every full stop. A region that
  finishes a sentence and then says something else ("es mejor escucharla con
  atención, ¿no? Entonces es mejor...") contains finished content, and is not
  a retake as a whole.
  """
  lAnchor = [pTokens[pEarlier + vOffset].text for vOffset in range(min(pAnchor, 3))]

  def fStartsTake(pIndex: int) -> bool:
    return all(
      pIndex + vOffset < pLater and fSameToken(pTokens[pIndex + vOffset].text, lAnchor[vOffset])
      for vOffset in range(len(lAnchor))
    )

  # Sentence ends inside a repeated opening ("Muy buenas a todos. Soy...")
  # belong to the take itself; one on its last word ends the take.
  vSkipUntil = pEarlier + pAnchor - 1
  for vIndex in range(pEarlier, pLater - 1):
    if vIndex >= vSkipUntil and fStartsTake(vIndex):
      vSkipUntil = vIndex + pAnchor - 1
    if vIndex < vSkipUntil:
      continue
    vToken = pTokens[vIndex]
    if not vToken.sentence_end or vToken.trails_off or vToken.cut_off:
      continue
    vNext = vIndex + 1
    if pLater - vNext <= 2 or fStartsTake(vNext):
      continue
    return False
  return True


def fFindRetakeRegions(
  pTokens: list[Token],
  pSettings: RetakeSettings,
  pPhraseStarts: list[int],
  pMeta: list[RetakeRegion],
) -> tuple[list[RetakeRegion], list[RetakeRegion]]:
  """Return (accepted retakes, retakes that need review)."""
  sExcluded = {vIndex for vRegion in pMeta for vIndex in range(vRegion.first, vRegion.last_exclusive)}
  sStarts = set(pPhraseStarts)
  lAccepted: list[RetakeRegion] = []
  lReview: list[RetakeRegion] = []

  for vLater in pPhraseStarts:
    if vLater == 0:
      continue
    vBest: RetakeRegion | None = None
    vReviewCandidate: RetakeRegion | None = None
    vLimit = max(0, vLater - pSettings.max_take_words)

    # Closest earlier take first. A farther one only replaces it when the
    # stretch between them is itself a take of the same opening - a run of
    # takes, like an intro recorded four times - never by a lucky coverage.
    for vEarlier in range(vLater - 1, vLimit - 1, -1):
      if pTokens[vLater].start - pTokens[vEarlier].start > pSettings.max_distance_seconds:
        break
      if not fSameToken(pTokens[vEarlier].text, pTokens[vLater].text):
        continue
      if _fCrossesKeptTake(vEarlier, vLater, lAccepted):
        continue
      vVerdict, vRegion = _fJudgeTake(pTokens, vEarlier, vLater, pSettings, sExcluded, sStarts, True)
      if vVerdict == "accept" and vRegion is not None:
        if vBest is None:
          vBest = vRegion
          continue
        vChain, _ = _fJudgeTake(pTokens, vEarlier, vBest.first, pSettings, sExcluded, sStarts, False)
        if vChain == "accept":
          vBest = RetakeRegion(
            first=vEarlier,
            last_exclusive=vLater,
            kind=vRegion.kind,
            anchor_words=max(vRegion.anchor_words, vBest.anchor_words),
            coverage=min(vRegion.coverage, vBest.coverage),
          )
      elif vVerdict == "review" and vRegion is not None:
        vReviewCandidate = vRegion

    if vBest is not None:
      lAccepted.append(vBest)
    if vReviewCandidate is not None:
      vReviewEnd = vBest.first if vBest is not None else vLater
      if vReviewEnd - vReviewCandidate.first > cLongTakeWords:
        vReviewCandidate.last_exclusive = vReviewEnd
        lReview.append(vReviewCandidate)

  lMerged = _fMergeRegions(lAccepted)
  lReview = [vRegion for vRegion in _fDropContained(lReview) if _fOverlapShare(vRegion, lMerged) < cReviewMaxOverlap]
  return (lMerged, lReview)


def _fOverlapShare(pRegion: RetakeRegion, pOthers: list[RetakeRegion]) -> float:
  """Share of ``pRegion``'s words that other regions already cover."""
  vLength = max(1, pRegion.last_exclusive - pRegion.first)
  vCovered = 0
  for vOther in pOthers:
    vCovered += max(0, min(pRegion.last_exclusive, vOther.last_exclusive) - max(pRegion.first, vOther.first))
  return vCovered / vLength


def _fJudgeTake(
  pTokens: list[Token],
  pEarlier: int,
  pLater: int,
  pSettings: RetakeSettings,
  sExcluded: set[int],
  sStarts: set[int],
  pCheckBoundary: bool,
) -> tuple[str, RetakeRegion | None]:
  """Decide whether ``[pEarlier, pLater)`` is an earlier take of what starts at ``pLater``.

  Returns ``("accept", region)``, ``("review", region)`` or ``("reject", None)``.
  ``pCheckBoundary`` requires the later take to follow a pause, a take left
  hanging or a full stop; it is off when checking a link of a run of takes.
  """
  vPrevious = pTokens[pLater - 1]
  vGap = _fGapBefore(pTokens, pLater)
  vRestartGap = pSettings.restart_gap_ms / 1000.0
  vTrailing = vPrevious.trails_off or vPrevious.cut_off
  # The take before the restart ends a sentence, rather than trailing off.
  vEndsSentence = vPrevious.sentence_end and not vTrailing
  # It ends a clause: one item of an enumeration ("lo puedes meter en un
  # router, lo puedes meter en...") rather than an abandoned attempt.
  vEndsClause = vPrevious.raw.endswith((",", ";", ":")) and not vTrailing

  if pEarlier == pLater - 1:
    # The same word twice in a row across a pause: "Debian... Debian puede".
    # Exactly the same word, and not the end of a sentence: "eres Alberto.
    # Albert." or "No, no, no. No os equivoquéis" are not stumbles.
    if (
      vGap >= vRestartGap
      and len(pTokens[pLater].text) >= 2
      and pTokens[pEarlier].text == pTokens[pLater].text
      and not vEndsSentence
    ):
      return ("accept", RetakeRegion(first=pEarlier, last_exclusive=pLater, kind="restart", anchor_words=1))
    return ("reject", None)

  vAnchor = _fAnchorLength(pTokens, pEarlier, pLater)
  if vAnchor < pSettings.min_anchor_words:
    return ("reject", None)
  vLength = pLater - pEarlier
  # A restart follows a pause, a take left hanging, or a full stop.
  if pCheckBoundary and vGap < vRestartGap and not (vTrailing or vEndsSentence):
    return ("reject", None)
  vAnchorContent = sum(1 for vOffset in range(vAnchor) if fIsContent(pTokens[pEarlier + vOffset].text))
  vCoverage, vContentWords, vCovered = _fCoverage(pTokens, pEarlier, pLater, vAnchor, sExcluded)

  if vAnchorContent < 1:
    # Function words alone only open a take that is short, trails off and
    # repeats exactly: "Pero sí, se ha... Pero sí, llegué hasta". An
    # enumeration ("de las licencias libres y... de la filosofía") does not.
    vExact = all(pTokens[pEarlier + vOffset].text == pTokens[pLater + vOffset].text for vOffset in range(vAnchor))
    if (
      vLength <= cShortFunctionTakeWords
      and vTrailing
      and vExact
      and (vContentWords == 0 or vCoverage >= pSettings.min_coverage)
    ):
      return ("accept", RetakeRegion(first=pEarlier, last_exclusive=pLater, kind="restart", anchor_words=vAnchor))
    # A longer take that hesitated right after the same opening and was then
    # left hanging: "Sobre todo cuando... lo que había era que instalar /
    # Sobre todo cuando... necesitamos instalar servicios".
    if (
      vExact
      and vAnchor >= cHesitantOpeningWords
      and pTokens[pEarlier + vAnchor - 1].trails_off
      and vLength <= cLongTakeWords
      and not (vEndsSentence or vEndsClause)
      and vGap >= vRestartGap
    ):
      return ("accept", RetakeRegion(first=pEarlier, last_exclusive=pLater, kind="retake", anchor_words=vAnchor))
    return ("reject", None)

  vAccept = False
  if vContentWords == 0:
    # Nothing beyond the opening: a pure restart, as long as the earlier take
    # was left hanging. A complete sentence whose opening returns is rhetoric
    # ("¿Con respecto a qué? Con respecto a alguien..."), unless the whole
    # sentence comes back word for word and is long enough not to be emphasis
    # ("Hoy les voy a hablar sobre Debian. Hoy les voy a hablar sobre..."). A
    # few words that end a longer sentence and open the next one ("...no quita
    # esa parte del vídeo. Esa parte del vídeo la tienes que cortar") are
    # rhetoric too; a clause cut short ("si lo desarrollan para Debian, lo
    # desarrollan para Debian va a funcionar") or a long stretch said again
    # ("...de esas funciones extras te duran dos días. De esas funciones
    # extras te duran dos días, cuando...") is a retake.
    vWholeSentence = pEarlier in sStarts or not vEndsSentence or vAnchor >= cRepeatedTailWords
    vAccept = not (vEndsSentence or vEndsClause) or (
      vAnchor == vLength and vAnchor >= 4 and vAnchorContent >= 2 and vWholeSentence
    )
  elif vLength <= cLongTakeWords:
    # Coverage of the whole earlier take, its opening included.
    vCombined = (vAnchorContent + vCovered) / float(vAnchorContent + vContentWords)
    if vEndsSentence or vEndsClause:
      # A finished sentence or clause has to come back with its own content;
      # only a long shared opening excuses a different ending.
      vAccept = vCoverage >= pSettings.min_coverage or (vAnchor >= 8 and vCombined >= pSettings.min_coverage)
    else:
      vAccept = vCombined >= pSettings.min_coverage
  elif vAnchor >= cLongTakeAnchorWords and vCoverage >= max(pSettings.min_coverage, cLongTakeCoverage):
    vAccept = True
  if vAccept and not _fOnlyTakesInside(pTokens, pEarlier, pLater, vAnchor):
    vAccept = False

  if vAccept:
    return (
      "accept",
      RetakeRegion(first=pEarlier, last_exclusive=pLater, kind="retake", anchor_words=vAnchor, coverage=vCoverage),
    )

  if (
    vLength > cLongTakeWords
    and vAnchor >= 3
    and vAnchorContent >= 2
    and vCoverage >= cReviewMinCoverage
    and _fNearPhraseStart(pEarlier, sStarts, cReviewMaxAnchorOffset)
  ):
    # The take starts where its phrase does: "En el episodio de hoy les voy
    # a hablar..." is the earlier take of "Hoy les voy a hablar...", lead-in
    # included.
    vFirst = max(vIndex for vIndex in sStarts if pEarlier - cReviewMaxAnchorOffset <= vIndex <= pEarlier)
    return (
      "review",
      RetakeRegion(first=vFirst, last_exclusive=pLater, kind="unique_content", anchor_words=vAnchor, coverage=vCoverage),
    )
  return ("reject", None)


def _fCrossesKeptTake(pEarlier: int, pLater: int, pAccepted: list[RetakeRegion]) -> bool:
  """Whether ``[pEarlier, pLater)`` starts inside a retake and runs past the take kept after it."""
  for vRegion in pAccepted:
    if vRegion.first < pEarlier < vRegion.last_exclusive and pLater > vRegion.last_exclusive:
      return True
  return False


def _fMergeRegions(pRegions: list[RetakeRegion]) -> list[RetakeRegion]:
  lSorted = sorted(pRegions, key=lambda pRegion: (pRegion.first, pRegion.last_exclusive))
  lMerged: list[RetakeRegion] = []
  for vRegion in lSorted:
    if lMerged and vRegion.first <= lMerged[-1].last_exclusive:
      vLast = lMerged[-1]
      vLast.last_exclusive = max(vLast.last_exclusive, vRegion.last_exclusive)
      vLast.anchor_words = max(vLast.anchor_words, vRegion.anchor_words)
      vLast.coverage = min(vLast.coverage, vRegion.coverage)
      continue
    lMerged.append(
      RetakeRegion(
        first=vRegion.first,
        last_exclusive=vRegion.last_exclusive,
        kind=vRegion.kind,
        anchor_words=vRegion.anchor_words,
        coverage=vRegion.coverage,
      )
    )
  return lMerged


def _fDropContained(pRegions: list[RetakeRegion]) -> list[RetakeRegion]:
  """Keep the widest of review regions that overlap each other."""
  lSorted = sorted(pRegions, key=lambda pRegion: (pRegion.first, -pRegion.last_exclusive))
  lKept: list[RetakeRegion] = []
  for vRegion in lSorted:
    if lKept and vRegion.first < lKept[-1].last_exclusive:
      continue
    lKept.append(vRegion)
  return lKept


# --------------------------------------------------------------------------- #
# Abandoned phrases
# --------------------------------------------------------------------------- #


def fFindAbandonedPhrases(pTokens: list[Token], pSettings: RetakeSettings, sTaken: set[int]) -> list[RetakeRegion]:
  """Short phrases that trail off, or stop dead, before a long pause."""
  if not pSettings.report_abandoned_phrases:
    return []
  lRegions: list[RetakeRegion] = []
  vCount = len(pTokens)
  vMinPause = pSettings.abandoned_min_pause_ms / 1000.0
  vSentencePause = pSettings.abandoned_sentence_min_pause_ms / 1000.0
  for vLast in range(vCount):
    vToken = pTokens[vLast]
    vPause = _fGapAfter(pTokens, vLast)
    if vLast >= vCount - 1:
      continue
    vTrailing = vToken.trails_off or vToken.cut_off
    if not (vTrailing and vPause >= vMinPause) and not (vToken.sentence_end and vPause >= vSentencePause):
      continue
    vFirst = vLast
    while vFirst > 0 and _fGapBefore(pTokens, vFirst) < cAbandonedPhraseGap and not pTokens[vFirst - 1].sentence_end:
      vFirst -= 1
    vWords = vLast - vFirst + 1
    vLimit = pSettings.abandoned_max_words if vTrailing else pSettings.abandoned_sentence_max_words
    if vWords > vLimit:
      continue
    vPhraseFirst = vFirst
    vAttempt = _fRestartedAttemptStart(pTokens, vFirst)
    if vAttempt < vFirst and not any(vIndex in sTaken for vIndex in range(vAttempt, vFirst)):
      vFirst = vAttempt
    if any(vIndex in sTaken for vIndex in range(vFirst, vLast + 1)):
      continue
    lRegions.append(
      RetakeRegion(
        first=vFirst,
        last_exclusive=vLast + 1,
        kind="abandoned",
        details={
          "pause_after": round(vPause, 2),
          "trails_off": vTrailing,
          "restarts_previous": vFirst < vPhraseFirst,
          "echoes_previous": _fEchoesPrevious(pTokens, vFirst, vLast + 1, sTaken),
        },
      )
    )
  return lRegions


def _fEchoesPrevious(pTokens: list[Token], pFirst: int, pLastExclusive: int, sTaken: set[int]) -> bool:
  """Whether an abandoned phrase only repeats words just said before it.

  "Mi elección personal siempre ha sido Debian. Y siempre ha sido..." - the
  fragment says nothing new: once the opening conjunction is set aside, its
  words appear, in the same order and together, among the words kept just
  before it.
  """
  lPhrase = [pTokens[vIndex].text for vIndex in range(pFirst, pLastExclusive)]
  while lPhrase and lPhrase[0] in cEchoOpenings:
    lPhrase = lPhrase[1:]
  if len(lPhrase) < cEchoMinimumWords:
    return False
  lBefore = [
    pTokens[vIndex].text
    for vIndex in range(max(0, pFirst - cEchoLookBackWords), pFirst)
    if vIndex not in sTaken
  ]
  vLength = len(lPhrase)
  for vStart in range(0, len(lBefore) - vLength + 1):
    if all(fSameToken(lBefore[vStart + vOffset], lPhrase[vOffset]) for vOffset in range(vLength)):
      return True
  return False


def _fRestartedAttemptStart(pTokens: list[Token], pFirst: int) -> int:
  """Where the unfinished phrase that ``pFirst`` starts again begins, or ``pFirst``."""
  if pFirst <= 0 or pTokens[pFirst - 1].sentence_end:
    return pFirst
  if _fGapBefore(pTokens, pFirst) >= cAbandonedRestartGap:
    return pFirst
  vStart = pFirst - 1
  while vStart > 0 and _fGapBefore(pTokens, vStart) < cAbandonedPhraseGap and not pTokens[vStart - 1].sentence_end:
    vStart -= 1
  vAttemptWords = pFirst - vStart
  if vAttemptWords < cAbandonedRestartMatchWords or vAttemptWords > cAbandonedRestartMaxWords:
    return pFirst
  if pFirst + cAbandonedRestartMatchWords > len(pTokens):
    return pFirst
  for vOffset in range(cAbandonedRestartMatchWords):
    if not fSameToken(pTokens[vStart + vOffset].text, pTokens[pFirst + vOffset].text):
      return pFirst
  return vStart


# --------------------------------------------------------------------------- #
# Candidates
# --------------------------------------------------------------------------- #


def _fText(pTokens: list[Token], pFirst: int, pLastExclusive: int) -> str:
  vText = " ".join(vToken.raw for vToken in pTokens[pFirst:pLastExclusive])
  if len(vText) > cTextLimit:
    return vText[: cTextLimit - 1] + "…"
  return vText


def _fBoundaryDetails(pTokens: list[Token], pFirst: int, pLastExclusive: int) -> dict[str, object]:
  """The word edges the planner needs to place a clean cut."""
  return {
    "first_word_index": pFirst,
    "removed_start": round(pTokens[pFirst].start, 3),
    "removed_end": round(pTokens[pLastExclusive - 1].end, 3),
    "kept_before_end": round(pTokens[pFirst - 1].end, 3) if pFirst > 0 else None,
    "kept_after_start": round(pTokens[pLastExclusive].start, 3) if pLastExclusive < len(pTokens) else None,
    "word_count": pLastExclusive - pFirst,
  }


def _fCandidate(
  pTranscript: Transcript,
  pTokens: list[Token],
  pRegion: RetakeRegion,
  pIdentifier: str,
  pType: CandidateType,
  pConfidence: float,
  pDetails: dict[str, object],
) -> EditCandidate:
  vStart = pTokens[pRegion.first].start
  # The cut runs up to the next take, taking the pause in between with it;
  # the planner leaves a natural pause back.
  if pRegion.last_exclusive < len(pTokens):
    vEnd = pTokens[pRegion.last_exclusive].start
  else:
    vEnd = pTokens[pRegion.last_exclusive - 1].end
  vEnd = max(vEnd, vStart)
  vBefore, vAfter = pTranscript.mContextText(vStart, vEnd)
  dDetails: dict[str, object] = _fBoundaryDetails(pTokens, pRegion.first, pRegion.last_exclusive)
  dDetails.update(pDetails)
  dDetails.update(pRegion.details)
  return EditCandidate(
    id=pIdentifier,
    type=pType,
    start=vStart,
    end=vEnd,
    text=_fText(pTokens, pRegion.first, pRegion.last_exclusive),
    context_before=vBefore[-240:],
    context_after=vAfter[:240],
    confidence=max(0.0, min(1.0, pConfidence)),
    details=dDetails,
  )


def fRequestedCutCandidates(
  pTranscript: Transcript,
  pCuts: list[dict[str, Any]],
) -> tuple[list[EditCandidate], list[str]]:
  """Turn cuts asked for by time (``{"start", "end", "why"}``) into take cuts.

  Each one removes the words whose middle falls inside its range, and is then
  placed, checked and reported like a retake. Returns the candidates and a
  warning for every range that holds no word.
  """
  if not pCuts or not pTranscript.has_word_timestamps:
    return ([], [])
  lTokens = fTokens(pTranscript)
  lCandidates: list[EditCandidate] = []
  lWarnings: list[str] = []
  for vNumber, dCut in enumerate(pCuts, start=1):
    vStart = float(dCut.get("start", 0.0))
    vEnd = float(dCut.get("end", 0.0))
    lInside = [
      vIndex
      for vIndex, vToken in enumerate(lTokens)
      if vStart - cRequestedSlack <= (vToken.start + vToken.end) / 2.0 <= vEnd + cRequestedSlack
    ]
    if vEnd <= vStart or not lInside:
      lWarnings.append(f"Requested cut {vStart:.2f}-{vEnd:.2f} holds no word and was ignored")
      continue
    vRegion = RetakeRegion(first=lInside[0], last_exclusive=lInside[-1] + 1, kind="requested")
    lCandidates.append(
      _fCandidate(
        pTranscript, lTokens, vRegion, f"requested-{vNumber:04d}", CandidateType.RETAKE, 0.95,
        {
          "decision": "review",
          "review_kind": "requested_cut",
          "proposed_action": "cut",
          "why": str(dCut.get("why") or "requested by the agent"),
        },
      )
    )
  return (lCandidates, lWarnings)


def fFindRetakeCandidates(pTranscript: Transcript, pSettings: RetakeSettings) -> list[EditCandidate]:
  """Every retake, piece of talk about the recording and abandoned phrase."""
  if not pSettings.enabled or not pTranscript.has_word_timestamps:
    return []
  lTokens = fTokens(pTranscript)
  if len(lTokens) < 3:
    return []

  lStarts = fPhraseStarts(lTokens, pSettings.phrase_gap_ms / 1000.0)
  lMeta = fFindMetaRegions(lTokens, pSettings, lStarts)
  lRetakes, lUnique = fFindRetakeRegions(lTokens, pSettings, lStarts, lMeta)
  sTaken = {vIndex for vRegion in lMeta + lRetakes for vIndex in range(vRegion.first, vRegion.last_exclusive)}
  lAbandoned = fFindAbandonedPhrases(lTokens, pSettings, sTaken)

  lCandidates: list[EditCandidate] = []
  for vNumber, vRegion in enumerate(lMeta, start=1):
    lCandidates.append(
      _fCandidate(
        pTranscript, lTokens, vRegion, f"meta-{vNumber:04d}", CandidateType.META_TALK, 0.9,
        {
          "decision": "auto",
          "proposed_action": "cut",
          "why": "talk about the recording itself",
        },
      )
    )

  for vNumber, vRegion in enumerate(lRetakes, start=1):
    vKept = _fText(lTokens, vRegion.last_exclusive, min(len(lTokens), vRegion.last_exclusive + 12))
    lCandidates.append(
      _fCandidate(
        pTranscript, lTokens, vRegion, f"retake-{vNumber:04d}", CandidateType.RETAKE,
        0.7 + 0.25 * vRegion.coverage if vRegion.kind == "retake" else 0.8,
        {
          "decision": "auto",
          "proposed_action": "cut",
          "anchor_words": vRegion.anchor_words,
          "coverage": round(vRegion.coverage, 3),
          "kept_take": vKept,
          "why": "said again later; the last take is kept" if vRegion.kind == "retake" else "word restarted after a pause",
        },
      )
    )

  for vNumber, vRegion in enumerate(lUnique, start=1):
    vKept = _fText(lTokens, vRegion.last_exclusive, min(len(lTokens), vRegion.last_exclusive + 25))
    vPolicy = pSettings.unique_content
    dDetails: dict[str, object] = {
      "decision": "auto" if vPolicy == "cut" else "review",
      "review_kind": "retake_with_unique_content",
      "proposed_action": "cut" if vPolicy == "cut" else "keep",
      "anchor_words": vRegion.anchor_words,
      "coverage": round(vRegion.coverage, 3),
      "kept_take": vKept,
      "why": (
        "an earlier take of a passage that is recorded again later, but it says things the later take "
        "does not; cutting it follows the last-take rule and removes that content"
      ),
    }
    if vPolicy == "keep":
      continue
    lCandidates.append(
      _fCandidate(
        pTranscript, lTokens, vRegion, f"retake-review-{vNumber:04d}", CandidateType.RETAKE, 0.5, dDetails,
      )
    )

  vCutPause = pSettings.abandoned_cut_pause_ms / 1000.0
  for vNumber, vRegion in enumerate(lAbandoned, start=1):
    vPause = float(vRegion.details.get("pause_after", 0.0) or 0.0)
    vWhy = (
      "a short phrase that trails off before a long pause; cut it if the speaker never finished it, "
      "keep it if the next phrase continues it"
    )
    if vRegion.details.get("restarts_previous"):
      vWhy = (
        "an unfinished phrase started again and abandoned before a long pause (both attempts are "
        "included); cut it if the speaker never finished it, keep it if the next phrase continues it"
      )
    if vRegion.details.get("echoes_previous"):
      vWhy = (
        "a phrase that trails off after repeating words just said, and adds nothing; cut it unless "
        "the next phrase continues it"
      )
    vProposeCut = (
      vPause >= vCutPause
      or bool(vRegion.details.get("restarts_previous"))
      or bool(vRegion.details.get("echoes_previous"))
    )
    lCandidates.append(
      _fCandidate(
        pTranscript, lTokens, vRegion, f"abandoned-{vNumber:04d}", CandidateType.ABANDONED_PHRASE,
        0.6 if vProposeCut else 0.35,
        {
          "decision": "review",
          "review_kind": "abandoned_phrase",
          "proposed_action": "cut" if vProposeCut else "keep",
          "why": vWhy,
        },
      )
    )

  lCandidates.sort(key=lambda pCandidate: (pCandidate.start, pCandidate.end))
  return lCandidates
