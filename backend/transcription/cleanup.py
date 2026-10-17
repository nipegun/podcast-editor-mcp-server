"""Transcript post-processing: hallucinations out, corrections in.

Both steps run every time a transcript is read, after the cache, never before
it. The cache keeps exactly what the recogniser produced, so editing the
corrections table or the hallucination list takes effect at once without
transcribing anything again.

* **Hallucinations.** Over silence or noise Whisper writes phrases it saw
  thousands of times in subtitles: "Gracias.", "Suscríbete al canal",
  "Subtítulos realizados por la comunidad de Amara.org". A segment made only of
  one of them is dropped when the recogniser itself doubted it; a real
  "gracias" said with confidence stays.
* **Corrections.** The podcast's own table of what the recogniser keeps getting
  wrong ("género Linux" is "GNU/Linux"). Rules with spaces apply to segment
  text; single-word rules also apply to every word, so the words the detectors
  compare and the text a listener reads say the same thing.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

from podcast_editor_mcp_server.config import CorrectionRule, TranscriptionSettings
from podcast_editor_mcp_server.logging_setup import fGetLogger
from podcast_editor_mcp_server.models.transcript import Segment, Transcript, Word

vLog = fGetLogger("transcription.cleanup")

# A hallucinated phrase is only dropped when the recogniser doubted it.
cHallucinationNoSpeech: float = 0.35
cHallucinationLogProb: float = -0.6
cHallucinationMaxSeconds: float = 3.0
# Real speech is heard word by word with far more confidence than this.
cHallucinationWordProbability: float = 0.75
# Nobody says more than this many words a second: text that fast is invented.
cMaxWordsPerSecond: float = 8.0
cFastSpeechMinWords: int = 3
# Below this average log-probability the recogniser was guessing.
cGuessLogProb: float = -1.5
cPromptEchoMinWords: int = 4
# Share of a segment's words taken from the prompt above which it may be an
# echo, and how many of the prompt's terms it must name to be one.
cPromptEchoVocabulary: float = 0.65
cPromptEchoMinTerms: int = 4
# Two transcriptions of the same stretch this close say the same thing.
cSameSpeechSimilarity: float = 0.5


def fFoldText(pText: str) -> str:
  """Lower-case, strip accents and punctuation, collapse whitespace."""
  vDecomposed = unicodedata.normalize("NFKD", pText.lower())
  vStripped = "".join(vChar for vChar in vDecomposed if not unicodedata.combining(vChar))
  vStripped = re.sub(r"[^\w\s/]", " ", vStripped)
  return " ".join(vStripped.split())


def fIsHallucinationText(pText: str, pPhrases: list[str]) -> bool:
  """Whether a text consists only of one of the known hallucinated phrases."""
  vFolded = fFoldText(pText)
  if not vFolded:
    return False
  for vPhrase in pPhrases:
    if vFolded == fFoldText(vPhrase):
      return True
  return False


def fContainsHallucination(pText: str, pPhrases: list[str]) -> bool:
  """Whether a text contains one of the known hallucinated phrases."""
  vFolded = f" {fFoldText(pText)} "
  return any(f" {fFoldText(vPhrase)} " in vFolded for vPhrase in pPhrases if fFoldText(vPhrase))


def fPromptTerms(pPrompts: list[str]) -> set[str]:
  """The names and terms in the prompts, folded.

  A term is a word written with a capital inside it (NiPeGun, OpenWrt, KDE), a
  digit (hacks4geeks), or a capital that does not open a sentence (Debian,
  Proxmox). Hotwords lists are all terms.
  """
  sTerms: set[str] = set()
  for vPrompt in pPrompts:
    if not vPrompt:
      continue
    vSentenceStart = True
    for vRaw in vPrompt.replace("/", " ").split():
      vWord = vRaw.strip(".,;:!?¡¿\"'()[]…")
      vFolded = fFoldText(vWord)
      if vFolded and (
        any(vChar.isupper() for vChar in vWord[1:])
        or any(vChar.isdigit() for vChar in vWord)
        or (vWord[:1].isupper() and not vSentenceStart)
      ):
        sTerms.update(vFolded.split())
      vSentenceStart = vRaw.endswith((".", "!", "?", "…"))
  return sTerms


def fLooksLikePromptEcho(pText: str, pPrompts: list[str]) -> bool:
  """Whether Whisper wrote the prompt back instead of transcribing.

  It happens on unclear audio: the decoder falls back on the only text it was
  given, typically as a list of the prompt's terms ("Hoy hablamos de Linux,
  GNU, MATE Desktop, OpenWrt, Debian..."). An echo is made almost only of the
  prompt's words and names at least four of its terms. A real intro that uses
  the formula the prompt quotes ("Muy buenas a todos. Soy NiPeGun y esto
  es...") names one or two, so it is not mistaken for one.
  """
  vFolded = fFoldText(pText)
  lWords = vFolded.split()
  if len(lWords) < cPromptEchoMinWords:
    return False
  sVocabulary: set[str] = set()
  for vPrompt in pPrompts:
    sVocabulary.update(fFoldText(vPrompt or "").split())
  if not sVocabulary:
    return False
  vFromPrompt = sum(1 for vWord in lWords if vWord in sVocabulary)
  if vFromPrompt / len(lWords) < cPromptEchoVocabulary:
    return False
  sTerms = fPromptTerms(pPrompts)
  return len({vWord for vWord in lWords if vWord in sTerms}) >= cPromptEchoMinTerms


def fSameSpeech(pFirst: str, pSecond: str) -> bool:
  """Whether two transcriptions of the same audio say roughly the same thing."""
  return SequenceMatcher(None, fFoldText(pFirst), fFoldText(pSecond), autojunk=False).ratio() >= cSameSpeechSimilarity


def fIsInventedSegment(pSegment: Segment, pPhrases: list[str]) -> bool:
  """Whether a segment is text the recogniser made up rather than heard.

  Four signs, any of which is enough: more words than anybody can say in the
  time the segment lasts; a log-probability so low the recogniser was
  guessing; a known hallucinated phrase alone that the recogniser doubted; a
  known hallucinated phrase inside a segment that is doubted or impossibly
  fast.
  """
  vWords = len(fFoldText(pSegment.text).split())
  vDuration = max(pSegment.duration, 0.01)
  vTooFast = vWords >= cFastSpeechMinWords and vWords / vDuration > cMaxWordsPerSecond
  if vTooFast:
    return True
  if pSegment.avg_logprob is not None and pSegment.avg_logprob <= cGuessLogProb:
    return True
  if fIsHallucinationText(pSegment.text, pPhrases) and _fSegmentDoubted(pSegment):
    return True
  if fContainsHallucination(pSegment.text, pPhrases) and (_fSegmentDoubted(pSegment) or vWords / vDuration > cMaxWordsPerSecond / 2):
    return True
  return False


def _fSegmentDoubted(pSegment: Segment) -> bool:
  if pSegment.no_speech_probability is not None and pSegment.no_speech_probability >= cHallucinationNoSpeech:
    return True
  if pSegment.avg_logprob is not None and pSegment.avg_logprob <= cHallucinationLogProb:
    return True
  lProbabilities = [vWord.probability for vWord in pSegment.words if vWord.probability is not None]
  if lProbabilities and sum(lProbabilities) / len(lProbabilities) < cHallucinationWordProbability:
    return True
  return pSegment.duration <= cHallucinationMaxSeconds and (
    pSegment.avg_logprob is None or pSegment.avg_logprob <= -0.35
  )


def fRemoveHallucinations(pTranscript: Transcript, pSettings: TranscriptionSettings) -> Transcript:
  """Drop segments that are only a known hallucinated phrase or an echo of the prompt."""
  if not pSettings.remove_hallucinations:
    return pTranscript
  lPrompts = [vText for vText in (pSettings.initial_prompt, pSettings.hotwords) if vText]
  lKept: list[Segment] = []
  vDropped = 0
  for vSegment in pTranscript.segments:
    if fIsInventedSegment(vSegment, pSettings.hallucination_phrases):
      vDropped += 1
      continue
    if lPrompts and fLooksLikePromptEcho(vSegment.text, lPrompts):
      vDropped += 1
      continue
    lKept.append(vSegment)
  if not vDropped:
    return pTranscript
  vLog.info("Dropped hallucinated segments", extra={"count": vDropped})
  return pTranscript.model_copy(update={"segments": lKept})


# --------------------------------------------------------------------------- #
# Corrections
# --------------------------------------------------------------------------- #


def _fCompile(pRule: CorrectionRule) -> re.Pattern[str]:
  vFlags = re.IGNORECASE if pRule.ignore_case else 0
  if pRule.regex:
    return re.compile(pRule.heard, vFlags)
  # Plain rules match whole words, with any run of spaces between them.
  lParts = [re.escape(vPart) for vPart in pRule.heard.split()]
  return re.compile(r"(?<!\w)" + r"\s+".join(lParts) + r"(?!\w)", vFlags)


def _fReplacer(pRule: CorrectionRule):  # noqa: ANN202 - a re.sub callback
  """The replacement for one rule, capitalised like the text it replaces."""

  def fReplace(pMatch: re.Match[str]) -> str:
    vReplacement = pMatch.expand(pRule.correct) if pRule.regex else pRule.correct
    vFound = pMatch.group(0)
    if pRule.preserve_case and vFound[:1].isupper() and vReplacement[:1].islower():
      return vReplacement[:1].upper() + vReplacement[1:]
    return vReplacement

  return fReplace


def fApplyCorrections(pTranscript: Transcript, pRules: list[CorrectionRule]) -> Transcript:
  """Apply the corrections table to segment text and, for one-word rules, to words."""
  if not pRules:
    return pTranscript
  # A rule that cannot span two words also applies to each word on its own.
  lCompiled = [
    (_fCompile(vRule), _fReplacer(vRule), " " not in vRule.heard.strip() and "\\s" not in vRule.heard)
    for vRule in pRules
  ]
  lSegments: list[Segment] = []
  vChanged = 0
  for vSegment in pTranscript.segments:
    vText = vSegment.text
    for vPattern, vReplacement, _ in lCompiled:
      vText = vPattern.sub(vReplacement, vText)
    lWords: list[Word] = []
    for vWord in vSegment.words:
      vWordText = vWord.word
      for vPattern, vReplacement, vSingle in lCompiled:
        if vSingle:
          vWordText = vPattern.sub(vReplacement, vWordText)
      lWords.append(vWord if vWordText == vWord.word else vWord.model_copy(update={"word": vWordText}))
    if vText != vSegment.text:
      vChanged += 1
    lSegments.append(vSegment.model_copy(update={"text": vText, "words": lWords}))
  if vChanged:
    vLog.debug("Applied transcript corrections", extra={"segments": vChanged})
  return pTranscript.model_copy(update={"segments": lSegments})


def fPostProcessTranscript(pTranscript: Transcript, pSettings: TranscriptionSettings) -> Transcript:
  """Everything that happens to a transcript between the cache and its readers."""
  vTranscript = fRemoveHallucinations(pTranscript, pSettings)
  return fApplyCorrections(vTranscript, pSettings.corrections)
