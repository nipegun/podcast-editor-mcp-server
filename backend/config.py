"""Typed configuration: profiles, workspace layout and runtime settings.

Two independent documents exist:

* A **profile** (``data/profiles/*.yaml``, plus any override) describes *how to
  edit*: silence policy, cut margins, loudness targets, safety limits. Profiles
  are the unit a user picks per job.
* The **runtime configuration** (``<prefix>/etc/users/<user>/config.yaml`` or
  ``<prefix>/etc/config.yaml``) describes *where things live*: workspace roots,
  logging, Audacity integration.

Everything the product owns lives under one prefix, ``/opt/podcast-editor-mcp-server`` by
default: the code, the virtualenv, the launchers, the configuration, the
Whisper models and every user's workspace. Nothing is written to ``/etc``,
``/var`` or the user's home directory. ``PODCAST_EDITOR_MCP_SERVER_PREFIX`` moves the whole
tree, which is what the installer's ``--prefix`` does and what the test suite
uses to stay away from a real installation.

Both documents are validated with Pydantic, so a malformed YAML file produces a
precise, field-level error instead of a crash halfway through a two-hour job.
"""

from __future__ import annotations

import os
import pwd
import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from podcast_editor_mcp_server.errors import ConfigurationError
from podcast_editor_mcp_server.logging_setup import fGetLogger
from podcast_editor_mcp_server.security.paths import PathGuard, WorkspaceRoot

vLog = fGetLogger("config")

cEnvConfigFile: str = "PODCAST_EDITOR_MCP_SERVER_CONFIG"
cEnvWorkspace: str = "PODCAST_EDITOR_MCP_SERVER_WORKSPACE"
cEnvInput: str = "PODCAST_EDITOR_MCP_SERVER_INPUT"
cEnvOutput: str = "PODCAST_EDITOR_MCP_SERVER_OUTPUT"
cEnvProfilesDir: str = "PODCAST_EDITOR_MCP_SERVER_PROFILES_DIR"
# Claude Code puts the folder it was started in into the environment of every
# MCP server it starts.
cEnvClaudeProjectDir: str = "CLAUDE_PROJECT_DIR"
cEnvProfile: str = "PODCAST_EDITOR_MCP_SERVER_PROFILE"
cEnvPrefix: str = "PODCAST_EDITOR_MCP_SERVER_PREFIX"

cDefaultPrefix: Path = Path("/opt/podcast-editor-mcp-server")
cBuiltinProfilesDir: Path = Path(__file__).resolve().parent / "data" / "profiles"
cDefaultProfileName: str = "podcast-default"
# How many profiles an "extends" chain may cross before it is treated as a
# mistake. Real chains are one or two deep (podcast profile -> preset).
cMaxProfileDepth: int = 8

# Written as a template rather than a resolved path so that one system-wide
# configuration file gives every user their own workspace without any
# per-user edit.
cDefaultWorkspaceRoot: str = "${prefix}/var/workspace/${user}"


# --------------------------------------------------------------------------- #
# Installation layout
# --------------------------------------------------------------------------- #


def fInstallPrefix() -> Path:
  """Root of the self-contained installation tree."""
  vEnvironment = os.environ.get(cEnvPrefix)
  if vEnvironment:
    return Path(vEnvironment).expanduser()
  return cDefaultPrefix


def fCurrentUser() -> str:
  """Login name of the account this process actually runs as.

  Read from the password database rather than from ``$USER``, because the
  installer reaches the code through ``runuser`` and a stale environment
  variable would hand one user another user's workspace.
  """
  try:
    return pwd.getpwuid(os.getuid()).pw_name
  except (KeyError, OSError):
    return f"uid-{os.getuid()}"


def fExpandLocation(pText: str) -> Path:
  """Expand ``${prefix}``, ``${user}`` and ``~`` in a configured path."""
  vText = pText.replace("${prefix}", str(fInstallPrefix()))
  vText = vText.replace("${user}", fCurrentUser())
  return Path(vText).expanduser()


def fBaseDirectory() -> Path:
  """Folder that relative paths given through the environment start from.

  An MCP client configured inside a folder starts the server for that folder:
  Claude Code names it in ``CLAUDE_PROJECT_DIR``, other clients start the
  server inside it. Either way a per-folder client configuration can say
  ``input`` and mean that folder's own ``input`` directory.
  """
  vProjectDir = os.environ.get(cEnvClaudeProjectDir)
  if vProjectDir and Path(vProjectDir).is_dir():
    return Path(vProjectDir)
  return Path.cwd()


def fEnvironmentLocation(pText: str) -> Path:
  """Expand a path given through the environment; a relative one starts from :func:`fBaseDirectory`."""
  vPath = fExpandLocation(pText)
  if not vPath.is_absolute():
    vPath = fBaseDirectory() / vPath
  return vPath.resolve()


def fSystemConfigDir() -> Path:
  """Directory holding the system-wide configuration and profile overrides."""
  return fInstallPrefix() / "etc"


def fModelDirectory() -> Path:
  """Directory holding the downloaded speech models, shared by every user."""
  return fInstallPrefix() / "var" / "models"


class StrictModel(BaseModel):
  """Base model: unknown keys are an error, not a silent typo."""

  model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=False, populate_by_name=True)


# --------------------------------------------------------------------------- #
# Profile sections
# --------------------------------------------------------------------------- #


class CorrectionRule(StrictModel):
  """One fix the recogniser keeps needing: what it writes and what was said.

  Applied to every transcript after it is read from the cache, so changing the
  table never forces a new transcription.
  """

  heard: str = Field(min_length=1, max_length=200)
  correct: str = Field(default="", max_length=200)
  regex: bool = False
  ignore_case: bool = True
  # Capitalise the correction when the recogniser capitalised what it heard
  # (start of a sentence). Off for names that are always written one way.
  preserve_case: bool = True

  @model_validator(mode="after")
  def mCheckPattern(self) -> CorrectionRule:
    if self.regex:
      try:
        re.compile(self.heard)
      except re.error as vError:
        raise ValueError(f"invalid regular expression in 'heard': {vError}") from vError
    return self


# Phrases Whisper writes over silence or noise. A segment made only of one of
# them is dropped when the recogniser itself doubted it.
cDefaultHallucinationPhrases: tuple[str, ...] = (
  "gracias", "gracias por ver", "gracias por ver el video", "gracias por ver el vídeo",
  "muchas gracias por ver el video", "suscríbete", "suscríbete al canal",
  "suscríbete al canal y activa la campanita", "no olvides suscribirte",
  "activa la campanita", "dale like", "te suscribas", "nos vemos en el próximo vídeo",
  "subtitulado por", "subtítulos por", "subtítulos realizados por",
  "subtítulos realizados por la comunidad de amara.org",
  "subtítulos por la comunidad de amara.org", "amara.org", "subtitles by",
  "thank you", "thanks for watching", "thank you for watching", "please subscribe",
)


class TranscriptionSettings(StrictModel):
  """Speech-to-text backend configuration."""

  backend: Literal["faster-whisper", "none"] = "faster-whisper"
  # The most accurate Whisper model. On a CUDA GPU it is fast; on a CPU it is
  # slow, and a profile can trade accuracy for speed with "medium" or "small".
  model: str = "large-v3"
  language: str = "es"  # the podcasts are Spanish; "auto" guesses from the first 30 s
  device: Literal["auto", "cpu", "cuda"] = "auto"
  compute_type: str = "auto"
  cpu_threads: int = Field(default=0, ge=0, le=256)
  num_workers: int = Field(default=1, ge=1, le=16)
  batch_size: int = Field(default=0, ge=0, le=64)
  beam_size: int = Field(default=5, ge=1, le=20)
  word_timestamps: bool = True
  vad_filter: bool = True
  condition_on_previous_text: bool = False
  # Text Whisper treats as already said before the recording starts. With
  # condition_on_previous_text off it only reaches the first 30 s window.
  initial_prompt: str | None = None
  # Names and terms Whisper sees before *every* window (faster-whisper's
  # "hotwords"), which is what keeps a podcast's vocabulary right all along.
  hotwords: str | None = Field(default=None, max_length=1_000)
  # "speech" transcribes each stretch of speech on its own. A whole-file pass
  # folds a sentence said twice into one, which hides the retakes; one call per
  # stretch keeps every take, and the prompt reaches every one of them.
  chunking: Literal["none", "speech"] = "none"
  chunk_max_seconds: float = Field(default=12.0, ge=2.0, le=30.0)
  chunk_merge_gap_ms: int = Field(default=600, ge=0, le=5_000)
  chunk_min_silence_ms: int = Field(default=300, ge=100, le=5_000)
  chunk_vad_threshold: float = Field(default=0.5, ge=0.05, le=0.95)
  # Applied after the cache, never part of the cache key.
  remove_hallucinations: bool = True
  hallucination_phrases: list[str] = Field(default_factory=lambda: list(cDefaultHallucinationPhrases))
  corrections: list[CorrectionRule] = Field(default_factory=list)
  model_dir: str | None = None
  local_files_only: bool = False

  @field_validator("hallucination_phrases")
  @classmethod
  def mNormalisePhrases(cls, pValue: list[str]) -> list[str]:
    lPhrases = [vPhrase.strip().lower() for vPhrase in pValue if vPhrase and vPhrase.strip()]
    if len(lPhrases) > 200:
      raise ValueError("hallucination phrase list is limited to 200 entries")
    return lPhrases

  @field_validator("corrections")
  @classmethod
  def mLimitCorrections(cls, pValue: list[CorrectionRule]) -> list[CorrectionRule]:
    if len(pValue) > 500:
      raise ValueError("the corrections table is limited to 500 entries")
    return pValue

  @field_validator("language")
  @classmethod
  def mValidateLanguage(cls, pValue: str) -> str:
    vValue = pValue.strip().lower()
    if not 2 <= len(vValue) <= 8 or not vValue.replace("-", "").isalpha():
      raise ValueError("language must be 'auto' or a language code such as 'es' or 'en'")
    return vValue

  @field_validator("model")
  @classmethod
  def mValidateModel(cls, pValue: str) -> str:
    vValue = pValue.strip()
    if not vValue:
      raise ValueError("model must not be empty")
    return vValue


class VadSettings(StrictModel):
  """Voice activity detection configuration."""

  backend: Literal["silero", "energy", "none"] = "silero"
  threshold: float = Field(default=0.5, ge=0.05, le=0.95)
  min_speech_duration_ms: int = Field(default=150, ge=0, le=10_000)
  min_silence_duration_ms: int = Field(default=400, ge=0, le=60_000)
  speech_pad_ms: int = Field(default=100, ge=0, le=2_000)
  max_speech_duration_s: float = Field(default=30.0, gt=0.0, le=3600.0)


class SilenceSettings(StrictModel):
  """Silence detection and shortening policy."""

  minimum_detectable_ms: int = Field(default=250, ge=20, le=60_000)
  remove_when_longer_than_ms: int = Field(default=2500, ge=100, le=600_000)
  replacement_duration_ms: int = Field(default=600, ge=0, le=60_000)
  preserve_natural_pauses: bool = True
  threshold_mode: Literal["auto", "fixed"] = "auto"
  threshold_db: float = Field(default=-45.0, ge=-120.0, le=0.0)
  # In auto mode the threshold sits this many dB above the measured *peak*
  # noise floor, which is the figure silencedetect compares against.
  auto_margin_db: float = Field(default=6.0, ge=0.0, le=40.0)
  head_target_ms: int = Field(default=500, ge=0, le=60_000)
  tail_target_ms: int = Field(default=1000, ge=0, le=60_000)
  trim_head: bool = True
  trim_tail: bool = True
  # Also treat every gap between the VAD's speech regions as a pause. A pause
  # full of breathing, chair or keyboard noise is not silent, so the silence
  # detector misses it; the VAD still knows nobody is speaking.
  detect_from_vad: bool = False
  # Tight pacing, the way an editor works by hand: every gap between two words
  # longer than word_gap_max_ms is shortened, whatever it holds (silence, a
  # breath, a throat clearing, a click), leaving word_gap_keep_after_ms after
  # the energy of the last word dies away and word_gap_keep_before_ms before
  # the next word starts. Replaces the pause rules above, except head and tail.
  tighten_word_gaps: bool = False
  word_gap_max_ms: int = Field(default=300, ge=50, le=10_000)
  word_gap_keep_after_ms: int = Field(default=70, ge=0, le=2_000)
  word_gap_keep_before_ms: int = Field(default=50, ge=0, le=2_000)
  # A phrase made only of these sounds, standing alone between two gaps, is a
  # noise of the mouth, not speech, and goes with the gap around it.
  vocal_noises: list[str] = Field(
    default_factory=lambda: [
      "eh", "ehh", "ehhh", "em", "emm", "mm", "mmm", "mmmm", "hm", "hmm", "ejem",
      "ps", "pss", "psst", "pssst", "chis", "chist", "chit", "sh", "shh",
    ]
  )

  @model_validator(mode="after")
  def mCheckConsistency(self) -> SilenceSettings:
    if self.replacement_duration_ms >= self.remove_when_longer_than_ms:
      raise ValueError("replacement_duration_ms must be shorter than remove_when_longer_than_ms")
    if self.minimum_detectable_ms > self.remove_when_longer_than_ms:
      raise ValueError("minimum_detectable_ms must not exceed remove_when_longer_than_ms")
    if self.word_gap_keep_after_ms + self.word_gap_keep_before_ms >= self.word_gap_max_ms:
      raise ValueError("word_gap_keep_after_ms + word_gap_keep_before_ms must be shorter than word_gap_max_ms")
    return self


class CutSettings(StrictModel):
  """How a region boundary becomes an actual cut."""

  pre_margin_ms: int = Field(default=40, ge=0, le=2_000)
  post_margin_ms: int = Field(default=80, ge=0, le=2_000)
  crossfade_ms: int = Field(default=10, ge=0, le=500)
  snap_to_low_energy: bool = True
  snap_window_ms: int = Field(default=60, ge=0, le=1_000)
  min_cut_duration_ms: int = Field(default=40, ge=0, le=10_000)
  # Check every cut through speech by transcribing the joined audio, and move
  # the boundary when the seam swallows a kept word or leaves a fragment of a
  # removed one. Word timestamps run early where words are joined, and only
  # listening to the seam tells the difference.
  verify_joins: bool = False
  verify_context_seconds: float = Field(default=3.0, ge=1.0, le=10.0)
  verify_max_attempts: int = Field(default=8, ge=1, le=30)
  # Whisper model used for the check; None uses the transcription model.
  verify_model: str | None = None


class FillerSettings(StrictModel):
  """Filler-word candidate detection."""

  enabled: bool = True
  auto_remove: bool = False
  # "esto", "este" and "bueno" are left out on purpose: in Spanish they are far
  # more often a pronoun, a demonstrative or an answer than a filler.
  words: list[str] = Field(
    default_factory=lambda: ["eh", "ehh", "ehhh", "em", "mmm", "mm", "hmm", "o sea", "a ver"]
  )
  # Regular expressions matched against a whole normalised token, so that
  # "ehhhh" or "mmmmm" are found however long the speaker held them.
  patterns: list[str] = Field(default_factory=lambda: ["e+h+", "e+m+", "m{2,}", "h+m+"])
  # A filler followed by one of these words is part of the sentence:
  # "o sea que", "a ver que".
  keep_when_followed_by: list[str] = Field(default_factory=lambda: ["que"])
  max_duration_ms: int = Field(default=900, ge=50, le=10_000)
  min_probability: float = Field(default=0.4, ge=0.0, le=1.0)
  require_isolation: bool = True
  isolation_gap_ms: int = Field(default=120, ge=0, le=3_000)
  max_per_minute: float = Field(default=12.0, ge=0.0, le=600.0)

  @field_validator("words", "keep_when_followed_by")
  @classmethod
  def mNormaliseWords(cls, pValue: list[str]) -> list[str]:
    lWords = [vWord.strip().lower() for vWord in pValue if vWord and vWord.strip()]
    if len(lWords) > 500:
      raise ValueError("filler word lists are limited to 500 entries")
    return lWords

  @field_validator("patterns")
  @classmethod
  def mCheckPatterns(cls, pValue: list[str]) -> list[str]:
    lPatterns = [vPattern.strip() for vPattern in pValue if vPattern and vPattern.strip()]
    if len(lPatterns) > 50:
      raise ValueError("filler pattern list is limited to 50 entries")
    for vPattern in lPatterns:
      try:
        re.compile(vPattern)
      except re.error as vError:
        raise ValueError(f"invalid filler pattern '{vPattern}': {vError}") from vError
    return lPatterns


class RepetitionSettings(StrictModel):
  """Accidental repetition candidate detection."""

  enabled: bool = True
  auto_remove: bool = False
  min_words: int = Field(default=2, ge=1, le=20)
  max_words: int = Field(default=12, ge=1, le=60)
  max_gap_seconds: float = Field(default=6.0, gt=0.0, le=120.0)
  similarity: float = Field(default=0.82, ge=0.3, le=1.0)

  @model_validator(mode="after")
  def mCheckWordWindow(self) -> RepetitionSettings:
    if self.min_words > self.max_words:
      raise ValueError("min_words must not exceed max_words")
    return self


class FalseStartSettings(StrictModel):
  """False-start candidate detection."""

  enabled: bool = True
  auto_remove: bool = False
  markers: list[str] = Field(
    default_factory=lambda: [
      "perdón", "perdon", "perdona", "digo", "mejor dicho", "quiero decir",
      "es decir no", "sorry", "i mean", "rather",
    ]
  )
  lookback_seconds: float = Field(default=8.0, gt=0.0, le=60.0)
  max_removed_seconds: float = Field(default=12.0, gt=0.0, le=300.0)

  @field_validator("markers")
  @classmethod
  def mNormaliseMarkers(cls, pValue: list[str]) -> list[str]:
    lMarkers = [vMarker.strip().lower() for vMarker in pValue if vMarker and vMarker.strip()]
    if len(lMarkers) > 200:
      raise ValueError("false-start marker list is limited to 200 entries")
    return lMarkers


# Talk about the recording itself. Only phrases that cannot be ordinary content
# belong here: "otra vez" or "corta" alone are far too common in real speech.
cDefaultMetaMarkers: tuple[str, ...] = (
  "lo voy a grabar", "lo vuelvo a grabar", "lo grabo de nuevo", "lo grabo otra vez",
  "voy a grabar de nuevo", "voy a grabarlo de nuevo", "voy a volver a grabar", "vuelvo a grabar",
  "lo voy a volver a grabar", "lo repito desde", "desde el principio otra vez", "otra toma",
  "vuelvo a empezar", "empiezo de nuevo", "empiezo otra vez", "lo empiezo de nuevo",
  "esto lo corto", "esto se corta", "eso lo corto", "eso se corta", "corto esto", "corta esto",
  "lo voy a cortar", "esto lo voy a cortar", "esto lo quito", "eso lo quito",
  "mejor en argentino", "en argentino mejor", "mejor en castellano", "en castellano mejor",
  "let me start over", "let me start again", "i'll record that again", "i will record that again",
  "cut that out", "cut this out", "from the top", "take two",
)


class RetakeSettings(StrictModel):
  """Retakes: the same thing said again, talk about the recording, abandoned phrases.

  The rule is that the last complete take is the good one. A take is
  recognised by its opening: when a later phrase starts with the same words as
  an earlier one, and the later take covers what the earlier one said, the
  earlier take is a retake.
  """

  enabled: bool = True
  auto_remove: bool = False
  # How far back an earlier take may start, in time and in words.
  max_distance_seconds: float = Field(default=300.0, gt=0.0, le=3_600.0)
  max_take_words: int = Field(default=400, ge=5, le=5_000)
  # Words two takes must open with. A single word only counts when it is said
  # twice in a row across a pause ("Debian... Debian puede ser").
  min_anchor_words: int = Field(default=2, ge=2, le=10)
  # Share of the earlier take's content words the later take must repeat.
  min_coverage: float = Field(default=0.6, ge=0.0, le=1.0)
  # A pause this long, or punctuation, ends a phrase.
  phrase_gap_ms: int = Field(default=300, ge=0, le=5_000)
  # The pause in front of a restart. Fluent repetition is emphasis, not a retake.
  restart_gap_ms: int = Field(default=450, ge=0, le=10_000)
  # What to do when an earlier take says something the later one does not:
  # "review" asks, "keep" keeps it, "cut" applies the last-take rule anyway.
  unique_content: Literal["review", "keep", "cut"] = "review"
  # Report phrases that trail off ("Pero uno de los...") and are followed by a
  # long pause, so a person or a model can decide whether they were abandoned.
  report_abandoned_phrases: bool = True
  abandoned_max_words: int = Field(default=7, ge=1, le=30)
  abandoned_min_pause_ms: int = Field(default=1500, ge=0, le=60_000)
  # Short sentences followed by a pause this long are reported too.
  abandoned_sentence_max_words: int = Field(default=5, ge=0, le=30)
  abandoned_sentence_min_pause_ms: int = Field(default=6000, ge=0, le=120_000)
  # When an abandoned phrase is followed by a pause this long the proposal is
  # to cut it; below it the proposal is to keep it.
  abandoned_cut_pause_ms: int = Field(default=8000, ge=0, le=120_000)
  meta_markers: list[str] = Field(default_factory=lambda: list(cDefaultMetaMarkers))

  @field_validator("meta_markers")
  @classmethod
  def mNormaliseMarkers(cls, pValue: list[str]) -> list[str]:
    lMarkers = [vMarker.strip().lower() for vMarker in pValue if vMarker and vMarker.strip()]
    if len(lMarkers) > 300:
      raise ValueError("meta marker list is limited to 300 entries")
    return lMarkers


class LoudnessSettings(StrictModel):
  """Podcast mastering targets."""

  integrated_lufs: float = Field(default=-16.0, ge=-40.0, le=-5.0)
  true_peak_dbtp: float = Field(default=-1.0, ge=-12.0, le=0.0)
  loudness_range: float = Field(default=11.0, gt=0.0, le=30.0)
  normalize: bool = True
  apply_limiter: bool = True
  limiter_attack_ms: float = Field(default=5.0, gt=0.0, le=100.0)
  limiter_release_ms: float = Field(default=50.0, gt=0.0, le=5_000.0)
  dual_pass: bool = True


class CompressorSettings(StrictModel):
  """Parameters handed to FFmpeg's ``acompressor``."""

  threshold_db: float = Field(default=-18.0, ge=-60.0, le=0.0)
  ratio: float = Field(default=3.0, ge=1.0, le=20.0)
  attack_ms: float = Field(default=20.0, ge=0.01, le=2_000.0)
  release_ms: float = Field(default=250.0, ge=0.01, le=9_000.0)
  makeup_db: float = Field(default=0.0, ge=0.0, le=36.0)
  # Knee width in dB. FFmpeg's acompressor takes a linear ratio (1..8); the
  # backend converts, so 9 dB here is acompressor's own 2.82843 default.
  knee_db: float = Field(default=9.0, ge=0.0, le=18.0)


class CleanupSettings(StrictModel):
  """Spectral clean-up applied before mastering."""

  high_pass_hz: float | None = Field(default=80.0, ge=10.0, le=1_000.0)
  low_pass_hz: float | None = Field(default=None, ge=1_000.0, le=24_000.0)
  denoise: bool = False
  denoise_reduction_db: float = Field(default=12.0, ge=0.01, le=97.0)
  denoise_floor_offset_db: float = Field(default=0.0, ge=-40.0, le=40.0)
  compress: bool = False
  compressor: CompressorSettings = Field(default_factory=CompressorSettings)


class OutputFormat(StrEnum):
  """Delivery formats this build can export."""

  WAV = "wav"
  FLAC = "flac"
  MP3 = "mp3"


class TranscriptFormat(StrEnum):
  """Transcript artifact formats."""

  JSON = "json"
  TXT = "txt"
  SRT = "srt"
  VTT = "vtt"


class OutputSettings(StrictModel):
  """Final rendering and export settings."""

  sample_rate: int = Field(default=48_000, ge=8_000, le=192_000)
  channels: Literal[0, 1, 2] = 1  # 0 keeps the source channel count
  formats: list[OutputFormat] = Field(default_factory=lambda: [OutputFormat("wav"), OutputFormat("mp3")])
  wav_codec: Literal["pcm_s16le", "pcm_s24le", "pcm_f32le"] = "pcm_s24le"
  mp3_bitrate_kbps: int = Field(default=128, ge=32, le=320)
  mp3_mode: Literal["cbr", "vbr"] = "cbr"
  mp3_vbr_quality: int = Field(default=2, ge=0, le=9)
  flac_compression: int = Field(default=5, ge=0, le=12)
  fade_in_ms: int = Field(default=0, ge=0, le=60_000)
  fade_out_ms: int = Field(default=0, ge=0, le=60_000)
  audacity_project: bool = True
  transcript_formats: list[TranscriptFormat] = Field(
    default_factory=lambda: [TranscriptFormat("json"), TranscriptFormat("txt"), TranscriptFormat("srt")]
  )
  # Where the delivered transcript comes from. "master" transcribes the edited
  # episode, so its timestamps and its text match what listeners hear; "source"
  # maps the source transcript through the edit instead, which is instant.
  # "master" falls back to "source" when transcription is unavailable.
  transcript_source: Literal["master", "source"] = "master"

  @field_validator("formats")
  @classmethod
  def mCheckFormats(cls, pValue: list[OutputFormat]) -> list[OutputFormat]:
    if not pValue:
      raise ValueError("at least one output format is required")
    lSeen: list[OutputFormat] = []
    for vFormat in pValue:
      if vFormat not in lSeen:
        lSeen.append(vFormat)
    return lSeen


class SafetySettings(StrictModel):
  """Hard limits the validator enforces on every EditPlan."""

  max_deleted_fraction: float = Field(default=0.40, ge=0.0, le=0.95)
  max_single_delete_seconds: float = Field(default=120.0, gt=0.0, le=36_000.0)
  # A paragraph recorded again minutes later is one long removal by nature, so
  # removals that include a retake have their own, larger limit.
  max_single_retake_seconds: float = Field(default=600.0, gt=0.0, le=36_000.0)
  reject_overlapping_operations: bool = True
  max_operations: int = Field(default=5_000, ge=1, le=200_000)
  min_output_duration_seconds: float = Field(default=1.0, ge=0.0, le=3_600.0)
  max_inserted_seconds: float = Field(default=600.0, ge=0.0, le=36_000.0)
  max_gain_db: float = Field(default=24.0, ge=0.0, le=60.0)


class Profile(StrictModel):
  """A complete, named editing policy."""

  name: str = Field(default=cDefaultProfileName, min_length=1, max_length=80)
  description: str = ""
  transcription: TranscriptionSettings = Field(default_factory=TranscriptionSettings)
  vad: VadSettings = Field(default_factory=VadSettings)
  silence: SilenceSettings = Field(default_factory=SilenceSettings)
  cuts: CutSettings = Field(default_factory=CutSettings)
  fillers: FillerSettings = Field(default_factory=FillerSettings)
  repetitions: RepetitionSettings = Field(default_factory=RepetitionSettings)
  false_starts: FalseStartSettings = Field(default_factory=FalseStartSettings)
  retakes: RetakeSettings = Field(default_factory=RetakeSettings)
  loudness: LoudnessSettings = Field(default_factory=LoudnessSettings)
  cleanup: CleanupSettings = Field(default_factory=CleanupSettings)
  output: OutputSettings = Field(default_factory=OutputSettings)
  safety: SafetySettings = Field(default_factory=SafetySettings)

  @field_validator("name")
  @classmethod
  def mValidateName(cls, pValue: str) -> str:
    vValue = pValue.strip()
    if not vValue.replace("-", "").replace("_", "").replace(".", "").isalnum():
      raise ValueError("profile name may only contain letters, digits, '-', '_' and '.'")
    return vValue

  def mFingerprint(self) -> dict[str, Any]:
    """Deterministic representation used in cache keys and provenance."""
    return self.model_dump(mode="json")


# --------------------------------------------------------------------------- #
# Runtime configuration
# --------------------------------------------------------------------------- #


class ExtraRoot(StrictModel):
  """An additional authorised directory."""

  role: str = Field(min_length=1, max_length=40)
  path: str
  writable: bool = False


class WorkspaceSettings(StrictModel):
  """Where the product is allowed to read and write.

  Every path accepts ``${prefix}`` (the installation root) and ``${user}``
  (the account the process runs as), so the system-wide configuration file
  stays correct for every user of the machine.
  """

  root: str = cDefaultWorkspaceRoot
  input: str | None = None
  work: str | None = None
  output: str | None = None
  jobs: str | None = None
  cache: str | None = None
  logs: str | None = None
  extra_roots: list[ExtraRoot] = Field(default_factory=list)
  allow_symlinks: bool = False


class LoggingSettings(StrictModel):
  """Logging behaviour. Console output always goes to stderr."""

  level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
  # Named json_format because "json" shadows BaseModel.json; the YAML key stays
  # "json" through the alias.
  json_format: bool = Field(default=False, alias="json")
  file_name: str = "podcast-editor-mcp-server.log"
  per_job_log: bool = True


class AudacitySettings(StrictModel):
  """Audacity / mod-script-pipe integration."""

  enabled: bool = True
  executable: str = "audacity"
  # "auto" starts Xvfb only when there is no DISPLAY; "always" keeps Audacity
  # off the user's desktop even on a graphical session; "never" refuses to
  # start one.
  use_xvfb: Literal["auto", "always", "never"] = "auto"
  display: str | None = None
  xvfb_screen: str = "1280x1024x24"
  startup_timeout_seconds: float = Field(default=90.0, gt=0.0, le=600.0)
  command_timeout_seconds: float = Field(default=180.0, gt=0.0, le=3_600.0)
  connect_timeout_seconds: float = Field(default=10.0, gt=0.0, le=300.0)
  pipe_directory: str = "/tmp"  # noqa: S108 - Audacity hard-codes this location
  project_mode: Literal["master", "replay"] = "master"


class JobSettings(StrictModel):
  """Job execution policy."""

  keep_work_dir: bool = False
  max_concurrent: int = Field(default=2, ge=1, le=16)
  # Background jobs that may wait for a free worker. A request beyond the
  # running ones plus this many is refused before anything is recorded.
  max_queued: int = Field(default=8, ge=0, le=1_000)
  transcription_cache: bool = True
  retention_days: int = Field(default=90, ge=0, le=3_650)


class RuntimeConfig(StrictModel):
  """The whole runtime configuration document."""

  workspace: WorkspaceSettings = Field(default_factory=WorkspaceSettings)
  logging: LoggingSettings = Field(default_factory=LoggingSettings)
  audacity: AudacitySettings = Field(default_factory=AudacitySettings)
  jobs: JobSettings = Field(default_factory=JobSettings)
  default_profile: str = cDefaultProfileName
  profile_directories: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #


def fLoadYaml(pPath: Path) -> dict[str, Any]:
  """Read a YAML mapping, with clear errors for the usual mistakes."""
  try:
    vText = pPath.read_text(encoding="utf-8")
  except OSError as vError:
    raise ConfigurationError(f"Cannot read {pPath}", {"reason": str(vError)}) from vError
  try:
    vParsed = yaml.safe_load(vText)
  except yaml.YAMLError as vError:
    raise ConfigurationError(f"Invalid YAML in {pPath}", {"reason": str(vError)}) from vError
  if vParsed is None:
    return {}
  if not isinstance(vParsed, dict):
    raise ConfigurationError(f"{pPath} must contain a YAML mapping at the top level")
  return vParsed


def fFormatValidationError(pError: ValidationError, pSource: str) -> ConfigurationError:
  """Turn a Pydantic error into a short, actionable message."""
  lProblems: list[str] = []
  for dItem in pError.errors():
    vLocation = ".".join(str(vPart) for vPart in dItem["loc"]) or "<root>"
    lProblems.append(f"{vLocation}: {dItem['msg']}")
  return ConfigurationError(
    f"Invalid configuration in {pSource}",
    {"problems": lProblems[:25], "problem_count": len(lProblems)},
  )


def fUserConfigDir() -> Path:
  """Return this user's own configuration directory inside the prefix.

  Per-user rather than per-home: the product keeps nothing in ``$HOME``, so
  one user's overrides live in ``<prefix>/etc/users/<user>/``.
  """
  return fSystemConfigDir() / "users" / fCurrentUser()


def fProfileSearchPath(pConfig: RuntimeConfig | None = None) -> list[Path]:
  """Ordered list of directories searched for profiles (first match wins)."""
  lDirectories: list[Path] = []
  vEnvironmentDir = os.environ.get(cEnvProfilesDir)
  if vEnvironmentDir:
    lDirectories.append(fEnvironmentLocation(vEnvironmentDir))
  if pConfig is not None:
    lDirectories.extend(fExpandLocation(vDir) for vDir in pConfig.profile_directories)
  lDirectories.append(fUserConfigDir() / "profiles")
  lDirectories.append(fSystemConfigDir() / "profiles")
  lDirectories.append(cBuiltinProfilesDir)
  lUnique: list[Path] = []
  sSeen: set[Path] = set()
  for vDirectory in lDirectories:
    if vDirectory not in sSeen:
      sSeen.add(vDirectory)
      lUnique.append(vDirectory)
  return lUnique


def fListProfiles(pConfig: RuntimeConfig | None = None) -> list[dict[str, str]]:
  """Enumerate every discoverable profile, nearest search path first."""
  ldProfiles: list[dict[str, str]] = []
  sSeenNames: set[str] = set()
  for vDirectory in fProfileSearchPath(pConfig):
    if not vDirectory.is_dir():
      continue
    for vFile in sorted(vDirectory.glob("*.yaml")):
      vName = vFile.stem
      if vName in sSeenNames:
        continue
      sSeenNames.add(vName)
      vDescription = ""
      try:
        dRaw = fLoadYaml(vFile)
        vDescription = str(dRaw.get("description", "") or "")
      except ConfigurationError:
        vDescription = "<unreadable>"
      ldProfiles.append({"name": vName, "path": str(vFile), "description": vDescription})
  return ldProfiles


def fCheckProfiles(pConfig: RuntimeConfig | None = None) -> list[dict[str, Any]]:
  """Load every discoverable profile and report the ones that fail.

  Listing a profile only reads its file name. This resolves each one fully,
  its ``extends`` chain included, so a wrong key shows up as soon as the file
  is written instead of when an episode is edited with it.
  """
  ldErrors: list[dict[str, Any]] = []
  for dProfile in fListProfiles(pConfig):
    vName = dProfile["name"]
    try:
      vProfile = fLoadProfile(vName, pConfig)
    except ConfigurationError as vError:
      lProblems = list(vError.details.get("problems", []))
      if not lProblems:
        lProblems = [f"{vKey}: {vValue}" for vKey, vValue in vError.details.items()]
      ldErrors.append({"name": vName, "path": dProfile["path"], "error": vError.message, "problems": lProblems})
      continue
    if vProfile.name != vName:
      ldErrors.append(
        {
          "name": vName,
          "path": dProfile["path"],
          "error": "The profile's name does not match its file name",
          "problems": [f"name: '{vProfile.name}' should be '{vName}'"],
        }
      )
  return ldErrors


def fDeepMerge(pBase: dict[str, Any], pOverride: dict[str, Any]) -> dict[str, Any]:
  """Return ``pBase`` with ``pOverride`` laid over it.

  Nested mappings merge key by key; anything else in ``pOverride`` replaces
  the base value outright, lists included.
  """
  dResult = dict(pBase)
  for vKey, vValue in pOverride.items():
    if isinstance(vValue, dict) and isinstance(dResult.get(vKey), dict):
      dResult[vKey] = fDeepMerge(dResult[vKey], vValue)
    else:
      dResult[vKey] = vValue
  return dResult


def fExtendProfile(
  pRaw: dict[str, Any],
  pBaseName: str,
  pName: str,
  pConfig: RuntimeConfig | None,
  pChain: tuple[str, ...],
) -> dict[str, Any]:
  """Lay a profile's own keys over the complete profile it ``extends``.

  This is what lets a per-podcast profile (kept in a profile directory of its
  own, outside the installation) say only "podcast-default, plus this
  vocabulary" instead of copying every setting and drifting from the preset.
  """
  vBaseName = pBaseName.strip()
  vChain = (*pChain, pName)
  if vBaseName in vChain:
    raise ConfigurationError(
      "Profile inheritance loops back on itself; a profile cannot extend itself or one of its descendants",
      {"chain": [*vChain, vBaseName]},
    )
  if len(vChain) >= cMaxProfileDepth:
    raise ConfigurationError(
      "Profile inheritance is too deep",
      {"chain": list(vChain), "limit": cMaxProfileDepth},
    )
  vBase = fLoadProfile(vBaseName, pConfig, vChain)
  dBase = vBase.model_dump(mode="json")
  # The name and the description belong to the profile being loaded, never to
  # the one it starts from.
  dBase.pop("name", None)
  dBase.pop("description", None)
  return fDeepMerge(dBase, pRaw)


def fLoadProfile(
  pName: str | None = None,
  pConfig: RuntimeConfig | None = None,
  pChain: tuple[str, ...] = (),
) -> Profile:
  """Load a profile by name from the search path.

  A profile may name another one in ``extends``; it then starts from that
  profile, fully resolved, and lists only what it changes. ``pChain`` is the
  list of profiles already being resolved, used to catch loops.
  """
  vName = pName or os.environ.get(cEnvProfile) or (pConfig.default_profile if pConfig else cDefaultProfileName)
  vName = str(vName).strip()
  if not vName:
    vName = cDefaultProfileName
  if "/" in vName or "\\" in vName or vName.startswith("."):
    raise ConfigurationError("Profile name must not contain path separators", {"profile": vName})

  for vDirectory in fProfileSearchPath(pConfig):
    vCandidate = vDirectory / f"{vName}.yaml"
    if vCandidate.is_file():
      dRaw = fLoadYaml(vCandidate)
      dRaw.setdefault("name", vName)
      vBaseName = dRaw.pop("extends", None)
      if vBaseName is not None:
        dRaw = fExtendProfile(dRaw, str(vBaseName), vName, pConfig, pChain)
      try:
        vProfile = Profile.model_validate(dRaw)
      except ValidationError as vError:
        raise fFormatValidationError(vError, str(vCandidate)) from vError
      vLog.debug("Loaded profile", extra={"profile": vName, "path": str(vCandidate)})
      return vProfile

  raise ConfigurationError(
    f"Profile '{vName}' not found",
    {"searched": [str(vDir) for vDir in fProfileSearchPath(pConfig)]},
  )


def fConfigFileCandidates() -> list[Path]:
  """Ordered list of runtime configuration files (first existing one wins)."""
  lCandidates: list[Path] = []
  vEnvironmentFile = os.environ.get(cEnvConfigFile)
  if vEnvironmentFile:
    lCandidates.append(fExpandLocation(vEnvironmentFile))
  lCandidates.append(fUserConfigDir() / "config.yaml")
  lCandidates.append(fSystemConfigDir() / "config.yaml")
  return lCandidates


def fLoadRuntimeConfig(pPath: Path | None = None) -> RuntimeConfig:
  """Load the runtime configuration, falling back to built-in defaults."""
  lCandidates = [pPath] if pPath is not None else fConfigFileCandidates()
  for vCandidate in lCandidates:
    if vCandidate is None or not vCandidate.is_file():
      continue
    dRaw = fLoadYaml(vCandidate)
    try:
      vConfig = RuntimeConfig.model_validate(dRaw)
    except ValidationError as vError:
      raise fFormatValidationError(vError, str(vCandidate)) from vError
    vLog.debug("Loaded runtime configuration", extra={"path": str(vCandidate)})
    return vConfig
  vLog.debug("No runtime configuration file found; using defaults")
  return RuntimeConfig()


# --------------------------------------------------------------------------- #
# Workspace
# --------------------------------------------------------------------------- #


class Workspace:
  """Resolved directory layout plus the :class:`PathGuard` that protects it."""

  def __init__(self, pConfig: RuntimeConfig) -> None:
    self.config = pConfig
    vSettings = pConfig.workspace
    vRootText = os.environ.get(cEnvWorkspace)
    if vRootText:
      self.root: Path = fEnvironmentLocation(vRootText)
    else:
      self.root = fExpandLocation(vSettings.root).resolve()
    # Recordings and deliveries can live in the folder the client works in,
    # while work files, jobs, caches and logs stay in the workspace.
    vInputText = os.environ.get(cEnvInput)
    if vInputText:
      self.input: Path = fEnvironmentLocation(vInputText)
    else:
      self.input = self._mSub(vSettings.input, "input")
    self.work: Path = self._mSub(vSettings.work, "work")
    vOutputText = os.environ.get(cEnvOutput)
    if vOutputText:
      self.output: Path = fEnvironmentLocation(vOutputText)
    else:
      self.output = self._mSub(vSettings.output, "output")
    self.jobs: Path = self._mSub(vSettings.jobs, "jobs")
    self.cache: Path = self._mSub(vSettings.cache, "cache")
    self.logs: Path = self._mSub(vSettings.logs, "logs")

    lRoots: list[WorkspaceRoot] = [
      WorkspaceRoot(role="input", path=self.input, writable=False),
      WorkspaceRoot(role="work", path=self.work, writable=True),
      WorkspaceRoot(role="output", path=self.output, writable=True),
      WorkspaceRoot(role="jobs", path=self.jobs, writable=True),
      WorkspaceRoot(role="cache", path=self.cache, writable=True),
      WorkspaceRoot(role="logs", path=self.logs, writable=True),
    ]
    for vExtra in vSettings.extra_roots:
      lRoots.append(
        WorkspaceRoot(
          role=vExtra.role,
          path=fExpandLocation(vExtra.path),
          writable=vExtra.writable,
        )
      )
    self.guard = PathGuard(lRoots, pAllowSymlinks=vSettings.allow_symlinks)

  def _mSub(self, pConfigured: str | None, pDefaultName: str) -> Path:
    if pConfigured:
      return fExpandLocation(pConfigured).resolve()
    return self.root / pDefaultName

  def mEnsure(self) -> None:
    """Create every workspace directory."""
    self.root.mkdir(parents=True, exist_ok=True)
    self.guard.mEnsureRoots()

  def mGetLogFile(self) -> Path:
    return self.logs / self.config.logging.file_name

  log_file = property(mGetLogFile)

  def mJobWorkDir(self, pJobId: str) -> Path:
    """Per-job scratch directory. The name is a UUID, never caller-controlled."""
    return self.work / pJobId

  def mDescribe(self) -> dict[str, Any]:
    return {
      "root": str(self.root),
      "input": str(self.input),
      "work": str(self.work),
      "output": str(self.output),
      "jobs": str(self.jobs),
      "cache": str(self.cache),
      "logs": str(self.logs),
      "allow_symlinks": self.config.workspace.allow_symlinks,
      "extra_roots": [
        {"role": vRoot.role, "path": str(vRoot.path), "writable": vRoot.writable}
        for vRoot in self.guard.roots
        if vRoot.role not in {"input", "work", "output", "jobs", "cache", "logs"}
      ],
    }


class AppContext:
  """Everything a command needs: configuration, workspace and active profile."""

  def __init__(
    self,
    pConfig: RuntimeConfig | None = None,
    pProfile: Profile | None = None,
    pEnsureDirectories: bool = True,
  ) -> None:
    self.config = pConfig if pConfig is not None else fLoadRuntimeConfig()
    self.workspace = Workspace(self.config)
    if pEnsureDirectories:
      self.workspace.mEnsure()
    self.profile = pProfile if pProfile is not None else fLoadProfile(None, self.config)

  def mWithProfile(self, pName: str | None) -> Profile:
    """Load another profile without rebuilding the whole context."""
    if pName is None or pName == self.profile.name:
      return self.profile
    return fLoadProfile(pName, self.config)
