"""faster-whisper transcription backend.

Device selection is automatic and never fatal: CUDA is used only when it can
actually be initialised, and any failure falls back to CPU rather than aborting
the job. A machine with no NVIDIA card transcribes perfectly well, just slower.

Two ways to transcribe:

* **Whole file** (``chunking: none``). One pass with faster-whisper's own VAD
  filter. The text reads best, because every sentence sees the one before it.
* **By stretch of speech** (``chunking: speech``). Silero splits the recording
  into stretches of speech and each one is transcribed on its own. A whole-file
  pass folds "No me sentí... No me sentí más identificado" into one sentence,
  which hides exactly the retakes an editor must find; one call per stretch
  keeps every take, with its own timestamps, and the prompt reaches all of them.
"""

from __future__ import annotations

import bisect
import datetime as dt
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from podcast_editor_mcp_server.audio.samples import fDecodeMono, fFrameLevelsDb, fLoudRuns, fSpeechThresholdDb
from podcast_editor_mcp_server.backends.base import BackendStatus
from podcast_editor_mcp_server.config import TranscriptionSettings, fExpandLocation, fModelDirectory
from podcast_editor_mcp_server.errors import PodcastEditorMcpServerError, TranscriptionError
from podcast_editor_mcp_server.logging_setup import fGetLogger
from podcast_editor_mcp_server.models.transcript import Segment, Transcript, Word
from podcast_editor_mcp_server.transcription.base import TranscriptionBackend
from podcast_editor_mcp_server.transcription.cleanup import (
  fContainsHallucination,
  fFoldText,
  fIsHallucinationText,
  fLooksLikePromptEcho,
  fSameSpeech,
)

vLog = fGetLogger("transcription.faster_whisper")

# Whisper models kept loaded at once: the one that transcribes and the one
# that listens to the joins. Loading another drops the least recently used.
cMaxResidentModels: int = 2
# Padding Silero adds around each stretch of speech before it is transcribed.
cChunkPadMs: int = 150
cChunkProgressEvery: int = 50
# A stretch transcribed with the prompt is transcribed again without it when a
# segment's average log-probability is this low or lower: the prompt can make
# Whisper write a subtitle formula ("¡Suscríbete al canal!") over real speech.
cRetryLogProb: float = -0.5
# Score given to a segment without a log-probability.
cUnknownLogProb: float = -1.0
# Sounds no word covers (see _mFillHoles): frame of the level analysis, dips
# that do not split a sound, shortest sound worth hearing again, slack around
# the words, how far a misplaced word may sit from its sound, padding of the
# excerpt, and what the excerpt must hold to be added.
cHoleFrameSeconds: float = 0.01
cHoleBridgeFrames: int = 5
cHoleMinimumSeconds: float = 0.4
cHoleGroupGap: float = 0.3
cHoleCoverMargin: float = 0.05
cHoleMisplacedReach: float = 0.5
cHoleLongestWord: float = 3.0
cHolePadding: float = 0.15
cHoleMinimumWords: int = 2
cHoleMinimumLogProb: float = -0.55
cHoleMaximumNoSpeech: float = 0.8

cEnvModelDir: str = "PODCAST_EDITOR_MCP_SERVER_MODEL_DIR"
cDefaultCpuComputeType: str = "int8"
cDefaultCudaComputeType: str = "float16"

# Every Whisper model works on 16 kHz mono.
cWhisperSampleRate: int = 16_000

# Shared objects CTranslate2 loads lazily on the first CUDA inference. Each
# entry is a list of acceptable sonames for one dependency. Only the majors
# CTranslate2 is actually built against count: every 4.x wheel links CUDA 12,
# so a system cuBLAS 13 (or an unversioned libcublas.so pointing at it) is no
# substitute and would only make the check pass before the first inference
# fails. The same goes for cuDNN 9, which every wheel since 4.5 links.
# podcast-editor-mcp-server/deploy/install.sh puts cuBLAS 12 and cuDNN 9 into the virtualenv, and the
# launchers add them to LD_LIBRARY_PATH.
cCudaLibraries: tuple[tuple[str, ...], ...] = (
  ("libcublas.so.12",),
  ("libcudnn.so.9",),
)

# Substrings that identify a CUDA runtime failure in CTranslate2's messages.
cCudaErrorMarkers: tuple[str, ...] = ("libcublas", "libcudnn", "cuda", "cublas", "cudnn", "gpu")


def fMissingCudaLibraries() -> list[str]:
  """Return the CUDA runtime sonames that cannot be loaded on this machine."""
  import ctypes

  lMissing: list[str] = []
  for vAlternatives in cCudaLibraries:
    vLoaded = False
    for vName in vAlternatives:
      try:
        ctypes.CDLL(vName)
      except OSError:
        continue
      vLoaded = True
      break
    if not vLoaded:
      lMissing.append(vAlternatives[0])
  return lMissing


def fCudaDeviceCount() -> int:
  """How many CUDA devices CTranslate2 sees; 0 when it cannot even ask."""
  try:
    import ctranslate2

    return int(ctranslate2.get_cuda_device_count())
  except Exception as vError:  # noqa: BLE001 - a broken CUDA install must not crash us
    vLog.debug("CUDA device probe failed", extra={"reason": str(vError)})
    return 0


def fLooksLikeCudaFailure(pMessage: str) -> bool:
  """Whether an error message points at the CUDA runtime rather than the model."""
  vLower = pMessage.lower()
  return any(vMarker in vLower for vMarker in cCudaErrorMarkers)


def fDefaultModelDirectory() -> Path:
  """Where models are cached when the profile does not say otherwise.

  Inside the installation prefix and shared by every user of the machine: a
  ``large-v3`` download is 3 GB, and one copy per home directory is 3 GB
  wasted. Write access comes from the ``podcast-editor-mcp-server`` group.
  """
  vEnvironment = os.environ.get(cEnvModelDir)
  if vEnvironment:
    return fExpandLocation(vEnvironment)
  return fModelDirectory()


class FasterWhisperBackend(TranscriptionBackend):
  """Local speech-to-text through CTranslate2."""

  name = "faster-whisper"

  def __init__(self) -> None:
    self._vImportError: str | None = None
    # In order of use, the most recent last.
    self._dModelCache: dict[tuple[str, str, str, int], Any] = {}
    self._vCacheLock = threading.Lock()

  # ----------------------------------------------------------- availability --

  def _mImportModule(self) -> Any | None:
    try:
      import faster_whisper
    except ImportError as vError:
      self._vImportError = str(vError)
      return None
    return faster_whisper

  def mStatus(self) -> BackendStatus:
    vModule = self._mImportModule()
    if vModule is None:
      return BackendStatus(
        name=self.name,
        available=False,
        reason="faster-whisper is not installed (pip install 'podcast-editor-mcp-server[whisper]')",
        details={"import_error": self._vImportError or ""},
      )
    vVersion = getattr(getattr(vModule, "version", None), "__version__", None) or "unknown"
    return BackendStatus(
      name=self.name,
      available=True,
      version=str(vVersion),
      details={"model_directory": str(fDefaultModelDirectory())},
    )

  def mModelDirectory(self, pSettings: TranscriptionSettings) -> Path:
    if pSettings.model_dir:
      return fExpandLocation(pSettings.model_dir)
    return fDefaultModelDirectory()

  # ------------------------------------------------------------ device pick --

  def mCudaAvailable(self) -> bool:
    """Whether CUDA is genuinely usable for inference.

    A present NVIDIA card is not enough. ``get_cuda_device_count`` counts
    devices the driver exposes, but CTranslate2 also needs the cuBLAS and cuDNN
    runtime libraries, and those are a separate install. When they are missing
    the device loads fine and then the *first inference* dies with "Library
    libcublas.so.12 is not found". Loading them here turns that into a clean
    decision to use the CPU instead.
    """
    try:
      import ctranslate2

      if int(ctranslate2.get_cuda_device_count()) <= 0:
        return False
    except Exception as vError:  # noqa: BLE001 - a broken CUDA install must not crash us
      vLog.debug("CUDA device probe failed", extra={"reason": str(vError)})
      return False

    lMissing = fMissingCudaLibraries()
    if lMissing:
      vLog.info(
        "A CUDA device is present but its runtime libraries are not installed; using the CPU",
        extra={"missing": lMissing},
      )
      return False
    return True

  def mResolveDevice(self, pSettings: TranscriptionSettings) -> tuple[str, str]:
    """Return the (device, compute_type) pair that will actually be used."""
    vRequested = pSettings.device
    if vRequested == "cuda":
      vDevice = "cuda" if self.mCudaAvailable() else "cpu"
      if vDevice == "cpu":
        vLog.warning("CUDA was requested but is unavailable; using the CPU")
    elif vRequested == "cpu":
      vDevice = "cpu"
    else:
      vDevice = "cuda" if self.mCudaAvailable() else "cpu"

    vComputeType = pSettings.compute_type
    if vComputeType in {"", "auto", "default"}:
      vComputeType = cDefaultCudaComputeType if vDevice == "cuda" else cDefaultCpuComputeType
    return (vDevice, vComputeType)

  # ------------------------------------------------------------ model load --

  def mLoadModel(self, pSettings: TranscriptionSettings) -> Any:
    """Load (and cache) a Whisper model, falling back to CPU if CUDA fails."""
    vModule = self._mImportModule()
    if vModule is None:
      raise TranscriptionError(
        "faster-whisper is not installed",
        {"install": "pip install 'podcast-editor-mcp-server[whisper]'"},
      )
    vDevice, vComputeType = self.mResolveDevice(pSettings)
    vDirectory = self.mModelDirectory(pSettings)
    vKey = (pSettings.model, vDevice, vComputeType, pSettings.cpu_threads)
    with self._vCacheLock:
      vCached = self._dModelCache.pop(vKey, None)
      if vCached is not None:
        self._dModelCache[vKey] = vCached
        return vCached

    vDirectory.mkdir(parents=True, exist_ok=True)
    vLog.info(
      "Loading transcription model",
      extra={"model": pSettings.model, "device": vDevice, "compute_type": vComputeType},
    )
    try:
      vModel = vModule.WhisperModel(
        pSettings.model,
        device=vDevice,
        compute_type=vComputeType,
        cpu_threads=pSettings.cpu_threads,
        num_workers=pSettings.num_workers,
        download_root=str(vDirectory),
        local_files_only=pSettings.local_files_only,
      )
    except Exception as vError:  # noqa: BLE001 - CTranslate2 raises RuntimeError subtypes
      if vDevice == "cuda":
        vLog.warning("CUDA model load failed; retrying on the CPU", extra={"reason": str(vError)})
        vCpuSettings = pSettings.model_copy(update={"device": "cpu", "compute_type": "auto"})
        return self.mLoadModel(vCpuSettings)
      raise TranscriptionError(
        "Could not load the transcription model",
        {
          "model": pSettings.model,
          "device": vDevice,
          "compute_type": vComputeType,
          "reason": str(vError),
          "hint": "Run 'podcast-editor-mcp-server-cli doctor' to check the model cache and network access",
        },
      ) from vError

    with self._vCacheLock:
      self._dModelCache[vKey] = vModel
      while len(self._dModelCache) > cMaxResidentModels:
        vOldest = next(iter(self._dModelCache))
        self._dModelCache.pop(vOldest)
        vLog.info("Unloading transcription model", extra={"model": vOldest[0], "device": vOldest[1]})
    return vModel

  def mEnsureModel(self, pSettings: TranscriptionSettings) -> Path:
    """Download the model if it is not cached yet, and return its directory."""
    vModule = self._mImportModule()
    if vModule is None:
      raise TranscriptionError("faster-whisper is not installed")
    vDirectory = self.mModelDirectory(pSettings)
    vDirectory.mkdir(parents=True, exist_ok=True)
    try:
      vPath = vModule.download_model(pSettings.model, cache_dir=str(vDirectory))
    except Exception as vError:  # noqa: BLE001 - huggingface_hub raises many error types
      raise TranscriptionError(
        "Could not download the transcription model",
        {"model": pSettings.model, "reason": str(vError)},
      ) from vError
    return Path(vPath)

  def mIsModelCached(self, pSettings: TranscriptionSettings) -> bool:
    """Whether the model is already on disk (no network access)."""
    vModule = self._mImportModule()
    if vModule is None:
      return False
    try:
      vModule.download_model(
        pSettings.model,
        cache_dir=str(self.mModelDirectory(pSettings)),
        local_files_only=True,
      )
    except Exception:  # noqa: BLE001 - "not cached" surfaces as several error types
      return False
    return True

  # ----------------------------------------------------------- transcription --

  def mTranscribe(self, pPath: Path, pSettings: TranscriptionSettings) -> Transcript:
    vModel = self.mLoadModel(pSettings)
    vDevice, vComputeType = self.mResolveDevice(pSettings)
    vLanguage = None if pSettings.language == "auto" else pSettings.language

    dArguments: dict[str, Any] = {
      "language": vLanguage,
      "task": "transcribe",
      "beam_size": pSettings.beam_size,
      "word_timestamps": pSettings.word_timestamps,
      "vad_filter": pSettings.vad_filter,
      "condition_on_previous_text": pSettings.condition_on_previous_text,
    }
    if pSettings.initial_prompt:
      dArguments["initial_prompt"] = pSettings.initial_prompt
    if pSettings.hotwords:
      dArguments["hotwords"] = pSettings.hotwords

    vEngine: Any = vModel
    if pSettings.batch_size > 0 and pSettings.chunking == "none":
      vEngine = self._mImportModule().BatchedInferencePipeline(model=vModel)
      dArguments["batch_size"] = pSettings.batch_size

    # Decoded here with FFmpeg and handed over as samples: given a path,
    # faster-whisper would decode it with PyAV, which a current PyAV breaks.
    try:
      aAudio = fDecodeMono(pPath, cWhisperSampleRate)
    except PodcastEditorMcpServerError as vError:
      raise TranscriptionError(
        "Could not decode the audio for transcription",
        {"path": pPath.name, "reason": str(vError)},
      ) from vError

    vLog.info(
      "Transcribing",
      extra={"path": pPath.name, "model": pSettings.model, "device": vDevice, "chunking": pSettings.chunking},
    )
    try:
      if pSettings.chunking == "speech":
        lSegments, vInfo = self._mTranscribeBySpeech(vEngine, aAudio, dArguments, pSettings)
      else:
        vSegments, vInfo = vEngine.transcribe(aAudio, **dArguments)
        lSegments = list(vSegments)
    except Exception as vError:  # noqa: BLE001 - the stack raises many error types
      # CTranslate2 loads the CUDA maths libraries on the first inference, not
      # when the model is created, so a missing cuBLAS only shows up here.
      # Fall back to the CPU rather than failing the job.
      if vDevice == "cuda" and fLooksLikeCudaFailure(str(vError)):
        vLog.warning(
          "CUDA inference failed; retrying on the CPU",
          extra={"reason": str(vError)[:200]},
        )
        vCpuSettings = pSettings.model_copy(update={"device": "cpu", "compute_type": "auto"})
        return self.mTranscribe(pPath, vCpuSettings)
      raise TranscriptionError(
        "Transcription failed",
        {"path": pPath.name, "model": pSettings.model, "device": vDevice, "reason": str(vError)},
      ) from vError

    return self._mNormalise(lSegments, vInfo, pSettings, vDevice, vComputeType)

  def _mTranscribeBySpeech(
    self,
    pEngine: Any,
    pAudio: np.ndarray,
    pArguments: dict[str, Any],
    pSettings: TranscriptionSettings,
  ) -> tuple[list[Any], Any]:
    """Transcribe every stretch of speech on its own and stitch the results.

    Each call starts from nothing, so a stretch can never be merged into the
    one before it, and the prompt reaches every stretch. Only one prompt is
    given - the initial prompt, or the hotwords when there is none - because a
    short stretch of unclear audio with a list in front of it is answered with
    the list. Even a natural sentence sometimes makes Whisper write something
    else over real speech: an echo of the prompt, or a subtitle formula
    ("¡Suscríbete al canal y activa la campanita!"). So a stretch that comes
    back as an echo, with a known hallucinated phrase or with a segment Whisper
    was unsure of is transcribed again without the prompt. If both versions
    say the same, the speaker really said it (an intro quotes the prompt on
    purpose) and the prompted spelling is kept; otherwise the version Whisper
    is surer of wins, and an echo always loses.
    """
    lChunks = fSpeechChunks(pAudio, pSettings)
    vLog.info("Transcribing by stretch of speech", extra={"chunks": len(lChunks)})
    dPlain = {vKey: vValue for vKey, vValue in pArguments.items() if vKey not in {"initial_prompt", "hotwords"}}
    dPlain["vad_filter"] = False
    dArguments = dict(dPlain)
    vPrompt = pSettings.initial_prompt or pSettings.hotwords
    if vPrompt:
      dArguments["initial_prompt"] = vPrompt
    lPrompts = [vText for vText in (pSettings.initial_prompt, pSettings.hotwords) if vText]

    lSegments: list[Any] = []
    vInfo: Any = None
    vRetries = 0
    vReplaced = 0
    for vIndex, (vStart, vEnd) in enumerate(lChunks):
      aSamples = pAudio[int(vStart * cWhisperSampleRate):int(vEnd * cWhisperSampleRate)]
      if aSamples.size == 0:
        continue
      vResult, vChunkInfo = pEngine.transcribe(aSamples, **dArguments)
      lChunkSegments = list(vResult)
      if lPrompts and _fPromptedChunkDoubtful(lChunkSegments, lPrompts, pSettings.hallucination_phrases):
        vRetries += 1
        vPlainResult, vPlainInfo = pEngine.transcribe(aSamples, **dPlain)
        lPlainSegments = list(vPlainResult)
        if _fPreferPlain(lChunkSegments, lPlainSegments, lPrompts):
          vReplaced += 1
          lChunkSegments, vChunkInfo = lPlainSegments, vPlainInfo
      if vInfo is None:
        vInfo = vChunkInfo
        # The language detected on the first stretch holds for the rest.
        if dArguments.get("language") is None:
          dArguments["language"] = getattr(vChunkInfo, "language", None)
          dPlain["language"] = dArguments["language"]
      for vItem in lChunkSegments:
        lSegments.append(_fShiftSegment(vItem, vStart))
      if (vIndex + 1) % cChunkProgressEvery == 0:
        vLog.info("Transcription progress", extra={"chunk": vIndex + 1, "of": len(lChunks), "at": round(vStart, 1)})

    if vRetries:
      vLog.info("Stretches transcribed again without the prompt", extra={"count": vRetries, "replaced": vReplaced})
    lFilled = self._mFillHoles(pEngine, pAudio, lSegments, dPlain, pSettings)
    if lFilled:
      vLog.info("Sounds without words transcribed on their own", extra={"count": len(lFilled)})
      lSegments = sorted(lSegments + lFilled, key=lambda pItem: float(getattr(pItem, "start", 0.0) or 0.0))
    vDuration = pAudio.size / float(cWhisperSampleRate)
    vSummary = SimpleNamespace(
      language=getattr(vInfo, "language", None) or pSettings.language,
      language_probability=getattr(vInfo, "language_probability", None),
      duration=vDuration,
    )
    return (lSegments, vSummary)

  def _mFillHoles(
    self,
    pEngine: Any,
    pAudio: np.ndarray,
    pSegments: list[Any],
    pPlain: dict[str, Any],
    pSettings: TranscriptionSettings,
  ) -> list[Any]:
    """Transcribe on its own every sound that no word of the transcript covers.

    Whisper drops a phrase said twice in a row ("en el que Proxmox está
    basado, en el que Proxmox está basado"): it writes it once. The copy it
    leaves out is exactly what an editor needs, because it is usually the
    good, last take. Every stretch of sound of at least ``cHoleMinimumSeconds``
    that no word overlaps, and that no misplaced word (one whose timestamp
    holds no sound) sits next to, is transcribed again without the prompt;
    what it says is added when it is made of real words.
    """
    aLevels = fFrameLevelsDb(pAudio, cWhisperSampleRate, cHoleFrameSeconds)
    if aLevels.size == 0:
      return []
    vThreshold = fSpeechThresholdDb(aLevels)
    aLoud = aLevels > vThreshold
    aFrameStarts = np.arange(aLevels.size) * cHoleFrameSeconds
    lRuns = fLoudRuns(aLoud, aFrameStarts, aFrameStarts + cHoleFrameSeconds, cHoleBridgeFrames)

    lSpans: list[tuple[float, float, bool]] = []
    for vSegment in pSegments:
      for vWord in getattr(vSegment, "words", None) or []:
        vStart = float(getattr(vWord, "start", 0.0) or 0.0)
        vEnd = float(getattr(vWord, "end", vStart) or vStart)
        aSlice = aLoud[int(vStart / cHoleFrameSeconds):int(vEnd / cHoleFrameSeconds) + 1]
        lSpans.append((vStart, vEnd, bool(aSlice.any())))
    lSpans.sort()
    lSpanStarts = [vSpan[0] for vSpan in lSpans]
    vDuration = pAudio.size / float(cWhisperSampleRate)

    def fNear(pFrom: float, pTo: float, pOnlyMisplaced: bool) -> bool:
      vIndex = max(0, bisect.bisect_left(lSpanStarts, pFrom - cHoleLongestWord))
      while vIndex < len(lSpans) and lSpans[vIndex][0] < pTo:
        vStart, vEnd, vHasSound = lSpans[vIndex]
        if vEnd > pFrom and (not pOnlyMisplaced or not vHasSound):
          return True
        vIndex += 1
      return False

    # Sounds no word covers, joined into phrases: the stops of "que es
    # importante" split one phrase into several short sounds.
    lGroups: list[list[float]] = []
    vPreviousUncovered = False
    for vRunStart, vRunEnd in lRuns:
      if fNear(vRunStart - cHoleCoverMargin, vRunEnd + cHoleCoverMargin, False):
        vPreviousUncovered = False
        continue
      if vPreviousUncovered and vRunStart - lGroups[-1][1] < cHoleGroupGap:
        lGroups[-1][1] = vRunEnd
      else:
        lGroups.append([vRunStart, vRunEnd])
      vPreviousUncovered = True

    lFilled: list[Any] = []
    for vRunStart, vRunEnd in lGroups:
      if vRunEnd - vRunStart < cHoleMinimumSeconds:
        continue
      if fNear(vRunStart - cHoleMisplacedReach, vRunEnd + cHoleMisplacedReach, True):
        continue
      vFrom = max(0.0, vRunStart - cHolePadding)
      vTo = min(vDuration, vRunEnd + cHolePadding)
      aSamples = pAudio[int(vFrom * cWhisperSampleRate):int(vTo * cWhisperSampleRate)]
      vResult, _ = pEngine.transcribe(aSamples, **pPlain)
      lHeard = list(vResult)
      vText = " ".join(str(getattr(vItem, "text", "")) for vItem in lHeard).strip()
      if len(fFoldText(vText).split()) < cHoleMinimumWords:
        continue
      if fIsHallucinationText(vText, pSettings.hallucination_phrases) or fContainsHallucination(vText, pSettings.hallucination_phrases):
        continue
      # Over a breath or a noise Whisper writes a sign-off ("un saludo", "y
      # ya está") it is not sure of, and it says so: the probability that
      # nobody is speaking stays high.
      if _fChunkScore(lHeard) <= cHoleMinimumLogProb or _fNoSpeech(lHeard) >= cHoleMaximumNoSpeech:
        continue
      lFilled.extend(_fShiftSegment(vItem, vFrom) for vItem in lHeard)
    return lFilled

  def mRecognizeSamples(self, pSamples: np.ndarray, pSettings: TranscriptionSettings, pModel: str | None = None) -> str:
    """Return the plain text Whisper hears in a short buffer of 16 kHz samples.

    Used to listen to the seam of a cut. No VAD, no word timestamps, no
    initial prompt (a few seconds of audio would only echo it back), but the
    hotwords, so the podcast's names are heard as names.
    """
    vSettings = pSettings if not pModel else pSettings.model_copy(update={"model": pModel})
    vModel = self.mLoadModel(vSettings)
    vLanguage = None if vSettings.language == "auto" else vSettings.language
    dArguments: dict[str, Any] = {
      "language": vLanguage,
      "task": "transcribe",
      "beam_size": vSettings.beam_size,
      "word_timestamps": False,
      "vad_filter": False,
      "condition_on_previous_text": False,
    }
    if vSettings.hotwords:
      dArguments["hotwords"] = vSettings.hotwords
    vDevice, _ = self.mResolveDevice(vSettings)
    try:
      vSegments, _ = vModel.transcribe(np.asarray(pSamples, dtype=np.float32), **dArguments)
      return " ".join(str(getattr(vItem, "text", "")).strip() for vItem in vSegments).strip()
    except Exception as vError:  # noqa: BLE001 - the stack raises many error types
      if vDevice == "cuda" and fLooksLikeCudaFailure(str(vError)):
        vCpuSettings = vSettings.model_copy(update={"device": "cpu", "compute_type": "auto"})
        return self.mRecognizeSamples(pSamples, vCpuSettings)
      raise TranscriptionError("Could not recognise a short excerpt", {"reason": str(vError)[:300]}) from vError

  def _mNormalise(
    self,
    pSegments: list[Any],
    pInfo: Any,
    pSettings: TranscriptionSettings,
    pDevice: str,
    pComputeType: str,
  ) -> Transcript:
    """Convert faster-whisper objects into our own models."""
    lSegments: list[Segment] = []
    vHasWords = False
    for vIndex, vRaw in enumerate(pSegments, start=1):
      lWords: list[Word] = []
      for vWord in getattr(vRaw, "words", None) or []:
        vStart = float(getattr(vWord, "start", 0.0) or 0.0)
        vEnd = float(getattr(vWord, "end", vStart) or vStart)
        lWords.append(
          Word(
            word=str(getattr(vWord, "word", "")),
            start=max(0.0, vStart),
            end=max(0.0, max(vEnd, vStart)),
            probability=_fClampProbability(getattr(vWord, "probability", None)),
          )
        )
      if lWords:
        vHasWords = True
      vSegmentStart = max(0.0, float(getattr(vRaw, "start", 0.0) or 0.0))
      vSegmentEnd = max(vSegmentStart, float(getattr(vRaw, "end", vSegmentStart) or vSegmentStart))
      lSegments.append(
        Segment(
          id=vIndex,
          start=vSegmentStart,
          end=vSegmentEnd,
          text=str(getattr(vRaw, "text", "")).strip(),
          words=lWords,
          avg_logprob=_fFloatOrNone(getattr(vRaw, "avg_logprob", None)),
          no_speech_probability=_fClampProbability(getattr(vRaw, "no_speech_prob", None)),
          compression_ratio=_fFloatOrNone(getattr(vRaw, "compression_ratio", None)),
          temperature=_fFloatOrNone(getattr(vRaw, "temperature", None)),
        )
      )

    return Transcript(
      language=str(getattr(pInfo, "language", "") or "unknown"),
      language_probability=_fClampProbability(getattr(pInfo, "language_probability", None)),
      duration=max(0.0, float(getattr(pInfo, "duration", 0.0) or 0.0)),
      backend=self.name,
      model=pSettings.model,
      device=pDevice,
      compute_type=pComputeType,
      generated_at=dt.datetime.now(dt.UTC).isoformat(),
      segments=lSegments,
      has_word_timestamps=vHasWords,
    )


def fSpeechChunks(pAudio: np.ndarray, pSettings: TranscriptionSettings) -> list[tuple[float, float]]:
  """Split 16 kHz samples into stretches of speech, in seconds.

  Silero finds the speech; stretches closer than ``chunk_merge_gap_ms`` are
  joined while the result stays under ``chunk_max_seconds``, which keeps each
  stretch well inside Whisper's 30 s window.
  """
  try:
    from faster_whisper.vad import VadOptions, get_speech_timestamps
  except ImportError as vError:
    raise TranscriptionError("faster-whisper is not installed", {"reason": str(vError)}) from vError

  vOptions = VadOptions(
    threshold=pSettings.chunk_vad_threshold,
    min_silence_duration_ms=pSettings.chunk_min_silence_ms,
    speech_pad_ms=cChunkPadMs,
  )
  ldRegions = get_speech_timestamps(pAudio, vOptions, sampling_rate=cWhisperSampleRate)
  vMergeGap = pSettings.chunk_merge_gap_ms / 1000.0
  vMaxLength = pSettings.chunk_max_seconds
  lChunks: list[list[float]] = []
  for dRegion in ldRegions:
    vStart = float(dRegion["start"]) / cWhisperSampleRate
    vEnd = float(dRegion["end"]) / cWhisperSampleRate
    if vEnd <= vStart:
      continue
    if lChunks and vStart - lChunks[-1][1] < vMergeGap and vEnd - lChunks[-1][0] <= vMaxLength:
      lChunks[-1][1] = vEnd
      continue
    # A single stretch longer than the window is split, never truncated.
    while vEnd - vStart > vMaxLength:
      lChunks.append([vStart, vStart + vMaxLength])
      vStart += vMaxLength
    lChunks.append([vStart, vEnd])
  return [(vStart, vEnd) for vStart, vEnd in lChunks]


def _fPromptedChunkDoubtful(pSegments: list[Any], pPrompts: list[str], pPhrases: list[str]) -> bool:
  """Whether a stretch transcribed with the prompt must be heard again without it."""
  for vSegment in pSegments:
    vText = str(getattr(vSegment, "text", ""))
    if fLooksLikePromptEcho(vText, pPrompts) or fContainsHallucination(vText, pPhrases):
      return True
    vLogProb = getattr(vSegment, "avg_logprob", None)
    if vLogProb is not None and vLogProb <= cRetryLogProb:
      return True
  return False


def _fPreferPlain(pPrompted: list[Any], pPlain: list[Any], pPrompts: list[str]) -> bool:
  """Whether the transcription without the prompt replaces the prompted one."""
  vPromptedText = " ".join(str(getattr(vItem, "text", "")) for vItem in pPrompted)
  vPlainText = " ".join(str(getattr(vItem, "text", "")) for vItem in pPlain)
  if fSameSpeech(vPromptedText, vPlainText):
    return False
  if any(fLooksLikePromptEcho(str(getattr(vItem, "text", "")), pPrompts) for vItem in pPrompted):
    return True
  if not pPlain:
    return False
  return _fChunkScore(pPlain) > _fChunkScore(pPrompted)


def _fChunkScore(pSegments: list[Any]) -> float:
  """Average log-probability of a stretch, weighted by each segment's words."""
  vTotal = 0.0
  vWeight = 0
  for vSegment in pSegments:
    vWords = max(1, len(fFoldText(str(getattr(vSegment, "text", ""))).split()))
    vLogProb = getattr(vSegment, "avg_logprob", None)
    vTotal += (cUnknownLogProb if vLogProb is None else float(vLogProb)) * vWords
    vWeight += vWords
  return vTotal / vWeight if vWeight else cUnknownLogProb


def _fNoSpeech(pSegments: list[Any]) -> float:
  """Mean probability that nobody is speaking, over a stretch's segments."""
  lValues = [float(vValue) for vValue in (getattr(vItem, "no_speech_prob", None) for vItem in pSegments) if vValue is not None]
  return sum(lValues) / len(lValues) if lValues else 0.0


def _fShiftSegment(pSegment: Any, pOffset: float) -> Any:
  """Copy a faster-whisper segment, moving its times by ``pOffset`` seconds."""
  lWords = [
    SimpleNamespace(
      word=getattr(vWord, "word", ""),
      start=float(getattr(vWord, "start", 0.0) or 0.0) + pOffset,
      end=float(getattr(vWord, "end", 0.0) or 0.0) + pOffset,
      probability=getattr(vWord, "probability", None),
    )
    for vWord in (getattr(pSegment, "words", None) or [])
  ]
  return SimpleNamespace(
    start=float(getattr(pSegment, "start", 0.0) or 0.0) + pOffset,
    end=float(getattr(pSegment, "end", 0.0) or 0.0) + pOffset,
    text=getattr(pSegment, "text", ""),
    words=lWords,
    avg_logprob=getattr(pSegment, "avg_logprob", None),
    no_speech_prob=getattr(pSegment, "no_speech_prob", None),
    compression_ratio=getattr(pSegment, "compression_ratio", None),
    temperature=getattr(pSegment, "temperature", None),
  )


def _fFloatOrNone(pValue: Any) -> float | None:
  if pValue is None:
    return None
  try:
    vNumber = float(pValue)
  except (TypeError, ValueError):
    return None
  if vNumber != vNumber:  # NaN
    return None
  return vNumber


def _fClampProbability(pValue: Any) -> float | None:
  vNumber = _fFloatOrNone(pValue)
  if vNumber is None:
    return None
  return float(min(1.0, max(0.0, vNumber)))
