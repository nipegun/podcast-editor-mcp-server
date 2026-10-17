"""MCP tool surface.

Design rules for everything in this file:

* **Semantic, not general.** Tools name product operations. There is no
  ``run_command``, no ``eval``, no arbitrary path access and no way to reach an
  Audacity command string.
* **Typed and bounded.** Every argument is annotated, every enum is checked,
  every number is clamped, every path goes through the workspace guard.
* **Same code as the CLI.** Each tool is a thin wrapper over
  :class:`~podcast_editor_mcp_server.pipeline.PodcastPipeline`. Nothing is implemented twice.
* **Bounded responses.** Transcripts and candidate lists are paginated, because
  a two-hour episode does not fit in a tool result.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field, ValidationError

from podcast_editor_mcp_server.analysis.false_starts import fFindFalseStartCandidates
from podcast_editor_mcp_server.analysis.fillers import fFindFillerCandidates
from podcast_editor_mcp_server.analysis.repetitions import fFindRepetitionCandidates
from podcast_editor_mcp_server.analysis.retakes import fFindRetakeCandidates
from podcast_editor_mcp_server.backends.audacity import AudacityBackend, fAudacityIsRunning, fEnableScriptPipeModule
from podcast_editor_mcp_server.config import fCheckProfiles, fListProfiles
from podcast_editor_mcp_server.errors import PodcastEditorMcpServerError
from podcast_editor_mcp_server.jobs.models import Job, JobState
from podcast_editor_mcp_server.logging_setup import fGetLogger
from podcast_editor_mcp_server.mcp.naming import fMcpParameters
from podcast_editor_mcp_server.mcp.state import ServerState
from podcast_editor_mcp_server.models.edit_plan import EditPlan, cMasteringTypes, cStructuralTypes
from podcast_editor_mcp_server.models.transcript import Transcript
from podcast_editor_mcp_server.pipeline import ProcessOptions
from podcast_editor_mcp_server.security.paths import cAudioWriteExtensions
from podcast_editor_mcp_server.transcription.formats import fRender
from podcast_editor_mcp_server.validation.validator import SafetyOverride, fIssueDicts
from podcast_editor_mcp_server.version import cProductName, cVersion

vLog = fGetLogger("mcp.tools")

cMaxSegmentsPerResponse: int = 200
cMaxCandidatesPerResponse: int = 200
cMaxOperationsInResponse: int = 200
cMaxTextChars: int = 60_000
# Analysis scratch directories older than this are swept away. They only hold a
# decoded copy of a source file, so losing one costs a re-decode.
cScratchMaxAgeSeconds: float = 24 * 3600.0
# All of them together are kept under this size, the least recently used going
# first.
cScratchMaxBytes: int = 16 * 1024**3

RootName = Literal["input", "output", "work"]

# A Whisper model a request may name: a size such as "large-v3", or an
# "organisation/name" repository. Never a path.
cModelNamePattern = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?")


class RequestedCut(BaseModel):
  """A cut asked for by time, in seconds of the source recording."""

  start: float = Field(ge=0.0, description="Start of the first word to remove (seconds of the source)")
  end: float = Field(gt=0.0, description="End of the last word to remove (seconds of the source)")
  why: str = Field(default="", max_length=300, description="Why it goes, for the report")


def fGuardTool[ReturnT](pFunction: Callable[..., ReturnT]) -> Callable[..., ReturnT]:
  """Translate product errors into clean MCP tool errors.

  Anticipated failures become a ``ToolError`` carrying the short message and
  the structured details. Unexpected ones are logged with their traceback and
  reported without internals.
  """

  @functools.wraps(pFunction)
  def fWrapper(*pArgs: Any, **pKeywordArgs: Any) -> ReturnT:
    try:
      return pFunction(*pArgs, **pKeywordArgs)
    except ToolError:
      raise
    except PodcastEditorMcpServerError as vError:
      vLog.info(
        "Tool rejected a request",
        extra={"tool": pFunction.__name__, "error": vError.cCode, "reason": vError.message},
      )
      raise ToolError(_fFormatError(vError)) from vError
    except ValidationError as vError:
      raise ToolError(f"Invalid input: {_fFirstValidationProblem(vError)}") from vError
    except Exception as vError:  # noqa: BLE001 - the model must not see internals
      vLog.exception("Tool crashed", extra={"tool": pFunction.__name__})
      raise ToolError(f"{pFunction.__name__} failed unexpectedly: {type(vError).__name__}") from vError

  return fWrapper


def _fFormatError(pError: PodcastEditorMcpServerError) -> str:
  vMessage = f"{pError.message} [{pError.cCode}]"
  if pError.details:
    lParts = [f"{vKey}={vValue}" for vKey, vValue in list(pError.details.items())[:6]]
    vMessage += " (" + "; ".join(lParts) + ")"
  return vMessage[:2000]


def _fFirstValidationProblem(pError: ValidationError) -> str:
  ldErrors = pError.errors()
  if not ldErrors:
    return "validation failed"
  dFirst = ldErrors[0]
  vLocation = ".".join(str(vPart) for vPart in dFirst["loc"]) or "<root>"
  return f"{vLocation}: {dFirst['msg']}"


def _fTruncate(pText: str, pLimit: int = cMaxTextChars) -> tuple[str, bool]:
  if len(pText) <= pLimit:
    return (pText, False)
  return (pText[:pLimit], True)


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


def fRegisterTools(pServer: MCPServer, pServerState: ServerState) -> None:
  """Register every tool on ``pServer``."""

  # ------------------------------------------------------------------ info --

  @pServer.tool(title="Server information")
  @fGuardTool
  @fMcpParameters('get_server_info')
  def fToolGetServerInfo() -> dict[str, Any]:
    """Describe this server: version, backends, profiles and workspace roots.

    Call this first. It tells you which optional pieces (transcription,
    Audacity) are usable right now, so you can plan the session accordingly.
    """
    vPipeline = pServerState.pipeline
    vContext = pServerState.context
    vFfmpeg = vPipeline.ffmpeg.mStatus()
    vAudacity = AudacityBackend(vContext.config.audacity).mStatus()

    from podcast_editor_mcp_server.transcription.base import fCreateTranscriptionBackend

    dTranscription: dict[str, Any] = {"backend": vContext.profile.transcription.backend, "available": False}
    try:
      vBackend = fCreateTranscriptionBackend(vContext.profile.transcription.backend)
      if vBackend is not None:
        dTranscription = vBackend.mStatus().mAsDict()
        dTranscription["model"] = vContext.profile.transcription.model
    except PodcastEditorMcpServerError as vError:
      dTranscription["reason"] = vError.message

    return {
      "product": cProductName,
      "version": cVersion,
      "transport": "stdio",
      "active_profile": vContext.profile.name,
      "profiles": [dProfile["name"] for dProfile in fListProfiles(vContext.config)],
      "profile_errors": fCheckProfiles(vContext.config),
      "workspace": vContext.workspace.mDescribe(),
      "backends": {
        "ffmpeg": vFfmpeg.mAsDict(),
        "audacity": vAudacity.mAsDict(),
        "transcription": dTranscription,
      },
      "supported_operations": {
        "structural": sorted(cStructuralTypes),
        "mastering": sorted(cMasteringTypes),
      },
      "notes": [
        "All paths are relative to the workspace roles above; absolute paths outside them are rejected.",
        "The source file is never modified.",
      ],
    }

  @pServer.tool(title="Health check")
  @fGuardTool
  @fMcpParameters('health_check')
  def fToolHealthCheck(
    pCheckModel: Annotated[bool, Field(description="Also check whether the Whisper model is cached")] = False,
  ) -> dict[str, Any]:
    """Run the diagnostics and report whether this machine can process audio."""
    from podcast_editor_mcp_server.doctor import fRunDoctor

    vReport = fRunDoctor(pServerState.context, pCheckModel=pCheckModel, pCheckMcp=False)
    dPayload = vReport.mAsDict()
    dPayload["checks"] = [
      dCheck for dCheck in dPayload["checks"] if dCheck["status"] in {"error", "warn"}
    ]
    dPayload["note"] = "Only failing and warning checks are listed; run 'podcast-editor-mcp-server-doctor' for the full report."
    return dPayload

  @pServer.tool(title="List input files")
  @fGuardTool
  @fMcpParameters('list_input_files')
  def fToolListInputFiles(
    pRoot: Annotated[RootName, Field(description="Which workspace role to list")] = "input",
    pLimit: Annotated[int, Field(ge=1, le=500, description="Maximum number of files to return")] = 100,
  ) -> dict[str, Any]:
    """List the audio files available in a workspace directory."""
    vRoot = pServerState.pipeline.guard.mRootFor(pRoot)
    ldFiles: list[dict[str, Any]] = []
    for vPath in sorted(vRoot.rglob("*")):
      if len(ldFiles) >= pLimit:
        break
      if not vPath.is_file() or vPath.is_symlink():
        continue
      if vPath.suffix.lower() not in cAudioWriteExtensions and vPath.suffix.lower() not in {".wave", ".m4a", ".aac", ".oga", ".wv", ".aiff", ".aif", ".caf", ".mp4"}:
        continue
      ldFiles.append(
        {
          "path": str(vPath.relative_to(vRoot)),
          "label": pServerState.pipeline.guard.mLabel(vPath),
          "size_bytes": vPath.stat().st_size,
        }
      )
    return {"root": pRoot, "directory": str(vRoot), "count": len(ldFiles), "files": ldFiles}

  # ----------------------------------------------------------------- audio --

  @pServer.tool(title="Probe audio")
  @fGuardTool
  @fMcpParameters('probe_audio')
  def fToolProbeAudio(
    pPath: Annotated[str, Field(description="Audio file, relative to the input directory or an absolute path inside the workspace")],
    pRoot: RootName = "input",
  ) -> dict[str, Any]:
    """Return format, duration, sample rate, channels and the SHA-256 of a file.

    Cheap and safe: it reads metadata and hashes the bytes, nothing else.
    """
    vSource = _fResolveAudio(pServerState, pPath, pRoot)
    return pServerState.pipeline.mProbe(vSource).model_dump(mode="json")

  @pServer.tool(title="Analyse audio")
  @fGuardTool
  @fMcpParameters('analyze_audio')
  def fToolAnalyzeAudio(
    pPath: Annotated[str, Field(description="Audio file to analyse")],
    pProfile: Annotated[str | None, Field(description="Profile name; defaults to the active profile")] = None,
    pRoot: RootName = "input",
    pIncludeRegions: Annotated[bool, Field(description="Include the full silence and speech region lists")] = False,
  ) -> dict[str, Any]:
    """Measure loudness, peaks, noise floor, silence and speech regions.

    This is the factual basis for every editing decision. Prefer it over
    guessing from a transcript.
    """
    vPipeline = pServerState.pipeline
    vSource = _fResolveAudio(pServerState, pPath, pRoot)
    vProfile = vPipeline.mResolveProfile(pProfile)
    vWorkDir = _fScratchDir(pServerState, vSource)
    vReport = vPipeline.mAnalyze(vSource, vProfile, vWorkDir)
    dPayload = vReport.mSummary()
    dPayload["profile"] = vProfile.name
    dPayload["silence_threshold_db"] = round(vReport.silence_threshold_db, 2)
    dPayload["vad_backend"] = vReport.vad_backend
    if pIncludeRegions:
      dPayload["silences"] = [vRegion.model_dump(mode="json") for vRegion in vReport.silences[:500]]
      dPayload["speech_regions"] = [vRegion.model_dump(mode="json") for vRegion in vReport.speech_regions[:500]]
    return dPayload

  @pServer.tool(title="Measure loudness")
  @fGuardTool
  @fMcpParameters('measure_loudness')
  def fToolMeasureLoudness(
    pPath: Annotated[str, Field(description="Audio file to measure")],
    pRoot: RootName = "input",
  ) -> dict[str, Any]:
    """Measure integrated loudness, loudness range and true peak (EBU R128)."""
    vSource = _fResolveAudio(pServerState, pPath, pRoot)
    vMeasurement = pServerState.pipeline.ffmpeg.mMeasureEbur128(vSource)
    dPayload = vMeasurement.model_dump(mode="json")
    dPayload["path"] = pServerState.pipeline.guard.mLabel(vSource)
    return dPayload

  @pServer.tool(title="Measure noise floor")
  @fGuardTool
  @fMcpParameters('measure_noise_floor')
  def fToolMeasureNoiseFloor(
    pPath: Annotated[str, Field(description="Audio file to measure")],
    pRoot: RootName = "input",
  ) -> dict[str, Any]:
    """Measure the noise floor, RMS, peak and DC offset of a file."""
    vSource = _fResolveAudio(pServerState, pPath, pRoot)
    vStatistics = pServerState.pipeline.ffmpeg.mMeasureAstats(vSource)
    dPayload = vStatistics.model_dump(mode="json")
    dPayload["path"] = pServerState.pipeline.guard.mLabel(vSource)
    return dPayload

  @pServer.tool(title="Detect silences")
  @fGuardTool
  @fMcpParameters('detect_silences')
  def fToolDetectSilences(
    pPath: Annotated[str, Field(description="Audio file to scan")],
    pProfile: str | None = None,
    pThresholdDb: Annotated[float | None, Field(ge=-90.0, le=0.0, description="Override the silence threshold in dBFS")] = None,
    pMinimumDurationMs: Annotated[int | None, Field(ge=20, le=60_000, description="Override the shortest silence to report")] = None,
    pRoot: RootName = "input",
    pLimit: Annotated[int, Field(ge=1, le=2000)] = 500,
  ) -> dict[str, Any]:
    """Find silent regions, using the profile's threshold unless you override it."""
    from podcast_editor_mcp_server.audio.analysis import fPeakNoiseFloorDb, fResolveSilenceThresholdDb

    vPipeline = pServerState.pipeline
    vSource = _fResolveAudio(pServerState, pPath, pRoot)
    vProfile = vPipeline.mResolveProfile(pProfile)
    vStatistics = vPipeline.ffmpeg.mMeasureAstats(vSource)
    vThreshold = pThresholdDb
    if vThreshold is None:
      vEnvelope = vPipeline.mEnvelope(vSource, _fScratchDir(pServerState, vSource))
      vThreshold = fResolveSilenceThresholdDb(
        vProfile, fPeakNoiseFloorDb(vEnvelope, vStatistics.noise_floor_dbfs)
      )
    vMinimum = (pMinimumDurationMs if pMinimumDurationMs is not None else vProfile.silence.minimum_detectable_ms) / 1000.0
    lRegions = vPipeline.ffmpeg.mDetectSilence(vSource, vThreshold, vMinimum)
    return {
      "path": vPipeline.guard.mLabel(vSource),
      "threshold_db": round(vThreshold, 2),
      "minimum_duration_seconds": round(vMinimum, 3),
      "noise_floor_dbfs": vStatistics.noise_floor_dbfs,
      "count": len(lRegions),
      "total_seconds": round(sum(vRegion.duration for vRegion in lRegions), 3),
      "silences": [vRegion.model_dump(mode="json") for vRegion in lRegions[:pLimit]],
      "truncated": len(lRegions) > pLimit,
    }

  @pServer.tool(title="Detect speech regions")
  @fGuardTool
  @fMcpParameters('detect_speech_regions')
  def fToolDetectSpeechRegions(
    pPath: Annotated[str, Field(description="Audio file to scan")],
    pProfile: str | None = None,
    pRoot: RootName = "input",
    pLimit: Annotated[int, Field(ge=1, le=2000)] = 500,
  ) -> dict[str, Any]:
    """Find the regions where somebody is speaking, using the VAD backend."""
    from podcast_editor_mcp_server.vad.silero import fCreateVadBackend

    vPipeline = pServerState.pipeline
    vSource = _fResolveAudio(pServerState, pPath, pRoot)
    vProfile = vPipeline.mResolveProfile(pProfile)
    vBackend = fCreateVadBackend(vProfile.vad.backend)
    if vBackend is None:
      raise ToolError("Voice activity detection is disabled in this profile")
    lRegions = vBackend.mDetectSpeech(vSource, vProfile.vad)
    vDuration = vPipeline.ffmpeg.mDuration(vSource)
    vSpeech = sum(vRegion.duration for vRegion in lRegions)
    return {
      "path": vPipeline.guard.mLabel(vSource),
      "backend": vBackend.name,
      "count": len(lRegions),
      "speech_seconds": round(vSpeech, 3),
      "speech_ratio": round(vSpeech / vDuration, 4) if vDuration > 0 else 0.0,
      "regions": [vRegion.model_dump(mode="json") for vRegion in lRegions[:pLimit]],
      "truncated": len(lRegions) > pLimit,
    }

  @pServer.tool(title="Detect clipping")
  @fGuardTool
  @fMcpParameters('detect_clipping')
  def fToolDetectClipping(
    pPath: Annotated[str, Field(description="Audio file to scan")],
    pRoot: RootName = "input",
    pLimit: Annotated[int, Field(ge=1, le=1000)] = 200,
  ) -> dict[str, Any]:
    """Find runs of samples at or beyond full scale."""
    from podcast_editor_mcp_server.audio.analysis import fEnsureReadableCopy
    from podcast_editor_mcp_server.audio.samples import fClippedSampleFraction, fFindClippingRuns, fReadSamples

    vPipeline = pServerState.pipeline
    vSource = _fResolveAudio(pServerState, pPath, pRoot)
    vWorkDir = _fScratchDir(pServerState, vSource)
    vReadable = fEnsureReadableCopy(vSource, vPipeline.ffmpeg, vWorkDir)
    vBlock = fReadSamples(vReadable)
    lRuns = fFindClippingRuns(vBlock)
    return {
      "path": vPipeline.guard.mLabel(vSource),
      "clipped_sample_fraction": round(fClippedSampleFraction(vBlock), 8),
      "region_count": len(lRuns),
      "regions": [
        {"start": round(vStart, 4), "end": round(vEnd, 4), "sample_count": vCount}
        for vStart, vEnd, vCount in lRuns[:pLimit]
      ],
      "truncated": len(lRuns) > pLimit,
    }

  # --------------------------------------------------------- transcription --

  @pServer.tool(title="Transcribe audio")
  @fGuardTool
  @fMcpParameters('transcribe_audio')
  def fToolTranscribeAudio(
    pPath: Annotated[str, Field(description="Audio file to transcribe")],
    pProfile: str | None = None,
    pLanguage: Annotated[str | None, Field(max_length=8, description="Language code, or 'auto' to detect")] = None,
    pModel: Annotated[
      str | None,
      Field(
        max_length=100,
        description="Whisper model to use, e.g. 'small' or 'large-v3'; one other than the profile's must already be downloaded",
      ),
    ] = None,
    pRoot: RootName = "input",
    pForce: Annotated[bool, Field(description="Ignore any cached transcript for this file")] = False,
  ) -> dict[str, Any]:
    """Transcribe a file locally and cache the result.

    Long files take a long time; the transcript is cached against the audio
    hash plus the settings, so calling this again is instant. Use
    ``get_transcript`` to read the text back in pages.
    """
    vPipeline = pServerState.pipeline
    vSource = _fResolveAudio(pServerState, pPath, pRoot)
    vProfile = vPipeline.mResolveProfile(pProfile)
    if pLanguage or pModel:
      dUpdates: dict[str, Any] = {}
      if pLanguage:
        dUpdates["language"] = pLanguage
      if pModel:
        if not cModelNamePattern.fullmatch(pModel) or ".." in pModel:
          raise ToolError("model must be a Whisper model name such as 'small' or 'large-v3', or an 'organisation/name' repository")
        dUpdates["model"] = pModel
        if pModel != vProfile.transcription.model:
          # A request may use any model already in the cache, but never make
          # the server download one: models are provisioned through profiles.
          dUpdates["local_files_only"] = True
      vProfile = vProfile.model_copy(
        update={"transcription": vProfile.transcription.model_copy(update=dUpdates)}
      )

    vProbe = vPipeline.mProbe(vSource)
    vTranscript = vPipeline.mTranscribe(vSource, vProfile, vProbe.sha256, pUseCache=not pForce)
    vCachePath = vPipeline.cache.mPathFor(vProbe.sha256, vProfile.transcription)
    return {
      "path": vPipeline.guard.mLabel(vSource),
      "source_sha256": vProbe.sha256,
      "language": vTranscript.language,
      "language_probability": vTranscript.language_probability,
      "backend": vTranscript.backend,
      "model": vTranscript.model,
      "device": vTranscript.device,
      "segment_count": len(vTranscript.segments),
      "word_count": vTranscript.word_count,
      "has_word_timestamps": vTranscript.has_word_timestamps,
      "duration_seconds": round(vTranscript.duration, 3),
      "cached_at": vPipeline.guard.mLabel(vCachePath) if vCachePath.exists() else None,
      "next_step": "Call get_transcript to read the text, or find_*_candidates to look for edits.",
    }

  @pServer.tool(title="Get transcript")
  @fGuardTool
  @fMcpParameters('get_transcript')
  def fToolGetTranscript(
    pPath: Annotated[str, Field(description="Audio file whose transcript to read")],
    pProfile: str | None = None,
    pFormat: Annotated[Literal["json", "txt", "srt", "vtt"], Field(description="Output format")] = "json",
    pOffset: Annotated[int, Field(ge=0, description="First segment to return")] = 0,
    pLimit: Annotated[int, Field(ge=1, le=cMaxSegmentsPerResponse, description="How many segments to return")] = 100,
    pIncludeWords: Annotated[bool, Field(description="Include per-word timestamps (large)")] = False,
    pRoot: RootName = "input",
  ) -> dict[str, Any]:
    """Read a cached transcript, one page at a time.

    Fails if the file has not been transcribed yet: call ``transcribe_audio``
    first. Text formats are truncated to keep the response usable.
    """
    vTranscript, vPipeline, vSource = _fLoadCachedTranscript(pServerState, pPath, pProfile, pRoot)
    if pFormat != "json":
      vText, vTruncated = _fTruncate(fRender(vTranscript, pFormat))
      return {
        "path": vPipeline.guard.mLabel(vSource),
        "format": pFormat,
        "language": vTranscript.language,
        "segment_count": len(vTranscript.segments),
        "text": vText,
        "truncated": vTruncated,
      }

    lSlice = vTranscript.segments[pOffset : pOffset + pLimit]
    ldSegments: list[dict[str, Any]] = []
    for vSegment in lSlice:
      dSegment = vSegment.model_dump(mode="json", exclude={"words"} if not pIncludeWords else set())
      ldSegments.append(dSegment)
    return {
      "path": vPipeline.guard.mLabel(vSource),
      "language": vTranscript.language,
      "segment_count": len(vTranscript.segments),
      "offset": pOffset,
      "returned": len(ldSegments),
      "has_more": pOffset + len(ldSegments) < len(vTranscript.segments),
      "has_word_timestamps": vTranscript.has_word_timestamps,
      "segments": ldSegments,
    }

  @pServer.tool(title="Get transcript segment")
  @fGuardTool
  @fMcpParameters('get_transcript_segment')
  def fToolGetTranscriptSegment(
    pPath: Annotated[str, Field(description="Audio file whose transcript to read")],
    pStart: Annotated[float, Field(ge=0.0, description="Start of the window, in seconds")],
    pEnd: Annotated[float, Field(ge=0.0, description="End of the window, in seconds")],
    pProfile: str | None = None,
    pIncludeWords: bool = True,
    pRoot: RootName = "input",
  ) -> dict[str, Any]:
    """Read the transcript covering one time window, with word timestamps.

    This is what you use before proposing a cut: it shows exactly which words
    live inside the region you are about to delete.
    """
    if pEnd <= pStart:
      raise ToolError("end must be greater than start")
    if pEnd - pStart > 600.0:
      raise ToolError("The window is limited to 600 seconds; ask for a narrower range")

    vTranscript, vPipeline, vSource = _fLoadCachedTranscript(pServerState, pPath, pProfile, pRoot)
    lSegments = vTranscript.mSegmentsBetween(pStart, pEnd)
    vBefore, vAfter = vTranscript.mContextText(pStart, pEnd)
    return {
      "path": vPipeline.guard.mLabel(vSource),
      "start": round(pStart, 3),
      "end": round(pEnd, 3),
      "context_before": vBefore[-400:],
      "context_after": vAfter[:400],
      "text": " ".join(vSegment.text.strip() for vSegment in lSegments),
      "segments": [
        vSegment.model_dump(mode="json", exclude=set() if pIncludeWords else {"words"})
        for vSegment in lSegments[:cMaxSegmentsPerResponse]
      ],
    }

  # ------------------------------------------------------------ candidates --

  @pServer.tool(title="Find filler-word candidates")
  @fGuardTool
  @fMcpParameters('find_filler_word_candidates')
  def fToolFindFillerWordCandidates(
    pPath: Annotated[str, Field(description="Audio file to inspect")],
    pProfile: str | None = None,
    pMinConfidence: Annotated[float, Field(ge=0.0, le=1.0, description="Drop candidates below this score")] = 0.0,
    pRoot: RootName = "input",
    pLimit: Annotated[int, Field(ge=1, le=cMaxCandidatesPerResponse)] = 100,
  ) -> dict[str, Any]:
    """Find filler words that *might* be worth removing.

    These are candidates, not decisions. Each one carries its duration, the
    surrounding pauses and the recogniser's confidence so you can judge whether
    it is a real filler or an ordinary word.
    """
    vTranscript, vPipeline, vSource = _fLoadCachedTranscript(pServerState, pPath, pProfile, pRoot)
    vProfile = vPipeline.mResolveProfile(pProfile)
    lCandidates = fFindFillerCandidates(vTranscript, vProfile.fillers)
    return _fCandidateResponse(pServerState, vSource, lCandidates, pMinConfidence, pLimit, "filler_word")

  @pServer.tool(title="Find repetition candidates")
  @fGuardTool
  @fMcpParameters('find_repetition_candidates')
  def fToolFindRepetitionCandidates(
    pPath: Annotated[str, Field(description="Audio file to inspect")],
    pProfile: str | None = None,
    pMinConfidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0,
    pRoot: RootName = "input",
    pLimit: Annotated[int, Field(ge=1, le=cMaxCandidatesPerResponse)] = 100,
  ) -> dict[str, Any]:
    """Find phrases the speaker appears to have repeated.

    Both occurrences are reported. Deciding which one to keep is your call:
    a repetition can be a stumble or deliberate emphasis.
    """
    vTranscript, vPipeline, vSource = _fLoadCachedTranscript(pServerState, pPath, pProfile, pRoot)
    vProfile = vPipeline.mResolveProfile(pProfile)
    lCandidates = fFindRepetitionCandidates(vTranscript, vProfile.repetitions)
    return _fCandidateResponse(pServerState, vSource, lCandidates, pMinConfidence, pLimit, "possible_repetition")

  @pServer.tool(title="Find false-start candidates")
  @fGuardTool
  @fMcpParameters('find_false_start_candidates')
  def fToolFindFalseStartCandidates(
    pPath: Annotated[str, Field(description="Audio file to inspect")],
    pProfile: str | None = None,
    pMinConfidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0,
    pRoot: RootName = "input",
    pLimit: Annotated[int, Field(ge=1, le=cMaxCandidatesPerResponse)] = 100,
  ) -> dict[str, Any]:
    """Find abandoned attempts followed by a self-correction marker."""
    vTranscript, vPipeline, vSource = _fLoadCachedTranscript(pServerState, pPath, pProfile, pRoot)
    vProfile = vPipeline.mResolveProfile(pProfile)
    lCandidates = fFindFalseStartCandidates(vTranscript, vProfile.false_starts)
    return _fCandidateResponse(pServerState, vSource, lCandidates, pMinConfidence, pLimit, "possible_false_start")

  @pServer.tool(title="Find retake candidates")
  @fGuardTool
  @fMcpParameters('find_retake_candidates')
  def fToolFindRetakeCandidates(
    pPath: Annotated[str, Field(description="Audio file to inspect")],
    pProfile: str | None = None,
    pMinConfidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0,
    pRoot: RootName = "input",
    pLimit: Annotated[int, Field(ge=1, le=cMaxCandidatesPerResponse)] = 100,
  ) -> dict[str, Any]:
    """Find retakes, talk about the recording and phrases that trail off.

    A retake is an earlier take of something said again later; the last take
    is the good one. Each candidate says whether it is clear (``decision:
    auto``) or needs a decision (``decision: review``), with a proposed action
    and the reason. Transcribe with ``chunking: speech`` first: a whole-file
    transcription folds retakes said close together into one sentence.
    """
    vTranscript, vPipeline, vSource = _fLoadCachedTranscript(pServerState, pPath, pProfile, pRoot)
    vProfile = vPipeline.mResolveProfile(pProfile)
    lCandidates = fFindRetakeCandidates(vTranscript, vProfile.retakes)
    return _fCandidateResponse(pServerState, vSource, lCandidates, pMinConfidence, pLimit, "retake")

  # ------------------------------------------------------- automatic editing --

  @pServer.tool(title="Prepare an episode edit")
  @fGuardTool
  @fMcpParameters('prepare_episode_edit')
  def fToolPrepareEpisodeEdit(
    pPath: Annotated[str, Field(description="Audio file to edit")],
    pProfile: Annotated[str | None, Field(description="The podcast's profile")] = None,
    pRoot: RootName = "input",
  ) -> dict[str, Any]:
    """Plan the whole edit of an episode automatically, and list what needs a decision.

    Analyses, transcribes (cached), detects retakes, talk about the recording,
    fillers and long pauses, builds the plan, listens to every join and fixes
    the ones that bite, then validates and simulates. Nothing is rendered.

    The answer has the cut list and ``review_items``: candidates the server
    would not decide alone (a phrase that may have been abandoned, an earlier
    take with content the later one lacks). Decide each one, then call
    ``finish_episode_edit`` with ``decisions={id: "cut" | "keep"}``. Undecided
    items are kept.
    """
    vPipeline = pServerState.pipeline
    vPrepared = vPipeline.mPrepareEdit(_fResolveAudio(pServerState, pPath, pRoot), pProfile)
    vPlan = vPrepared.plan
    dPayload: dict[str, Any] = {
      "path": vPrepared.probe.label,
      "profile": vPlan.provenance.profile,
      "plan_path": vPipeline.guard.mLabel(vPrepared.plan_path) if vPrepared.plan_path else None,
      "source_duration_seconds": round(vPrepared.probe.duration_seconds, 3),
      "estimated_output_seconds": round(vPrepared.validation.estimated_output_seconds, 3),
      "removed_seconds": round(vPrepared.validation.removed_seconds, 3),
      "removed_percent": round(vPrepared.validation.removed_fraction * 100.0, 2),
      "transcription": _fTranscriptSummary(vPrepared.transcript),
      "cuts": _fCutList(vPlan, vPrepared.transcript),
      "pauses": _fPauseSummary(vPlan),
      "review_items": vPlan.metadata.get("review_items", []),
      "joins": _fJoinSummary(vPlan),
      "valid": vPrepared.validation.valid,
      "errors": fIssueDicts(vPrepared.validation.errors),
      "warnings": vPrepared.warnings[:50] + [vIssue.message for vIssue in vPrepared.validation.warnings][:20],
      "next_step": (
        "Decide every review item (read its text, context and kept_take), then call finish_episode_edit "
        "with decisions={item id: 'cut' or 'keep'} and the same profile."
      ),
    }
    return dPayload

  @pServer.tool(title="Finish an episode edit")
  @fGuardTool
  @fMcpParameters('finish_episode_edit')
  def fToolFinishEpisodeEdit(
    pPath: Annotated[str, Field(description="Audio file being edited")],
    pDecisions: Annotated[
      dict[str, Literal["cut", "keep"]] | None,
      Field(description="Review item id -> 'cut' or 'keep'; undecided items are kept"),
    ] = None,
    pProfile: Annotated[str | None, Field(description="The podcast's profile, the same as in prepare_episode_edit")] = None,
    pExtraCuts: Annotated[
      list[RequestedCut] | None,
      Field(
        description=(
          "Further cuts the detectors missed, by the word timestamps of the source transcript: stumbles, "
          "repeated fragments, self-corrections, muttered asides. The words whose middle falls inside each "
          "range are removed, and the cut is placed and checked like a retake."
        ),
        max_length=500,
      ),
    ] = None,
    pOutputName: Annotated[str | None, Field(max_length=120, description="Base name for the artifacts")] = None,
    pFormats: Annotated[list[Literal["wav", "flac", "mp3"]] | None, Field(description="Delivery formats; defaults to the profile")] = None,
    pAudacityProject: Annotated[bool | None, Field(description="Also write an .aup3 project when Audacity is reachable")] = None,
    pOverwrite: Annotated[bool, Field(description="Replace artifacts that already exist")] = False,
    pWait: Annotated[bool, Field(description="Block until the job finishes; set false for long recordings")] = True,
    pMastering: Annotated[
      bool,
      Field(
        description=(
          "Also clean up and master the audio (filters, noise reduction, compressor, loudness, limiter, fades). "
          "False delivers only the cuts, as <name>.wav in lossless formats, to master later with master_audio"
        ),
      ),
    ] = True,
    pRoot: RootName = "input",
  ) -> dict[str, Any]:
    """Apply the decisions, render the episode and deliver it with its transcript.

    Rebuilds exactly the plan ``prepare_episode_edit`` showed (same profile,
    same cached transcript), adds the cuts decided here and the ``extra_cuts``,
    checks the joins again, validates and renders. The delivered transcript is
    a transcription of the edited episode, so its timestamps match what
    listeners hear. With ``mastering=false`` the audio and the Audacity
    project hold the edit alone, as recorded; ``master_audio`` masters it.
    """
    vPipeline = pServerState.pipeline
    vSource = _fResolveAudio(pServerState, pPath, pRoot)
    vOptions = ProcessOptions(
      profile=pProfile,
      overwrite=pOverwrite,
      transcribe=True,
      audacity_project=pAudacityProject,
      formats=list(pFormats) if pFormats else None,
      output_name=pOutputName,
      decisions=dict(pDecisions or {}),
      apply_proposals=False,
      extra_cuts=[vCut.model_dump() for vCut in pExtraCuts or []],
      mastering=pMastering,
    )
    if not pWait:
      vJobId = _fStartBackgroundJob(pServerState, vSource, vOptions)
      return {
        "job_id": vJobId,
        "state": "queued",
        "waiting": False,
        "next_step": f"Poll get_job with job_id='{vJobId}'.",
      }
    vResult = vPipeline.mProcess(vSource, vOptions)
    dPayload = _fProcessPayload(pServerState, vResult)
    if vResult.plan is not None:
      dPayload["cuts"] = _fCutList(vResult.plan, vResult.transcript)
      dPayload["pauses"] = _fPauseSummary(vResult.plan)
      dPayload["review_items"] = vResult.plan.metadata.get("review_items", [])
      dPayload["joins"] = _fJoinSummary(vResult.plan)
    return dPayload

  # ----------------------------------------------------------------- plans --

  @pServer.tool(title="Create an automatic edit plan")
  @fGuardTool
  @fMcpParameters('create_automatic_edit_plan')
  def fToolCreateAutomaticEditPlan(
    pPath: Annotated[str, Field(description="Audio file to plan an edit for")],
    pProfile: str | None = None,
    pUseTranscript: Annotated[bool, Field(description="Use the cached transcript when one exists")] = True,
    pRoot: RootName = "input",
    pSave: Annotated[bool, Field(description="Write the plan into the workspace and return its path")] = True,
  ) -> dict[str, Any]:
    """Build an EditPlan from the measurements and the active profile.

    The plan is deterministic and always expressed in coordinates of the
    original file. Review it, adjust it, then validate and execute it.
    """
    vPipeline = pServerState.pipeline
    vSource = _fResolveAudio(pServerState, pPath, pRoot)
    vProfile = vPipeline.mResolveProfile(pProfile)
    vWorkDir = _fScratchDir(pServerState, vSource)
    vProbe = vPipeline.mProbe(vSource)
    vAnalysis = vPipeline.mAnalyze(vSource, vProfile, vWorkDir, vProbe)

    vTranscript: Transcript | None = None
    if pUseTranscript:
      vTranscript = vPipeline.mLoadCachedTranscript(vProbe.sha256, vProfile)

    vCandidates = vPipeline.mFindCandidates(vAnalysis, vProfile, vTranscript)
    vEnvelope = vPipeline.mEnvelope(vSource, vWorkDir)
    vPlan = vPipeline.mCreatePlan(vAnalysis, vProfile, vCandidates, vTranscript, vEnvelope)
    vValidation = vPipeline.mValidate(vPlan, vProfile, vProbe)

    vPlanPath: Path | None = None
    if pSave:
      vPlanPath = vPipeline.mSavePlan(vPlan)

    return {
      "plan": _fPlanPayload(vPlan),
      "plan_path": vPipeline.guard.mLabel(vPlanPath) if vPlanPath else None,
      "valid": vValidation.valid,
      "estimated_output_seconds": round(vValidation.estimated_output_seconds, 3),
      "removed_seconds": round(vValidation.removed_seconds, 3),
      "removed_percent": round(vValidation.removed_fraction * 100.0, 2),
      "errors": fIssueDicts(vValidation.errors),
      "warnings": fIssueDicts(vValidation.warnings),
      "transcript_used": vTranscript is not None,
      "next_step": "Call simulate_edit_plan to preview, then execute_edit_plan to render.",
    }

  @pServer.tool(title="Get an edit plan")
  @fGuardTool
  @fMcpParameters('get_edit_plan')
  def fToolGetEditPlan(
    pPlanPath: Annotated[str | None, Field(description="Path of a saved plan inside the workspace")] = None,
    pJobId: Annotated[str | None, Field(description="Job whose plan to read")] = None,
    pIncludeOperations: bool = True,
  ) -> dict[str, Any]:
    """Read a saved EditPlan, either by path or by the job that produced it."""
    vPlan = _fResolvePlan(pServerState, None, pPlanPath, pJobId)
    dPayload = _fPlanPayload(vPlan, pIncludeOperations)
    return dPayload

  @pServer.tool(title="Validate an edit plan")
  @fGuardTool
  @fMcpParameters('validate_edit_plan')
  def fToolValidateEditPlan(
    pPlan: Annotated[dict[str, Any] | None, Field(description="The plan document itself")] = None,
    pPlanPath: Annotated[str | None, Field(description="Path of a saved plan inside the workspace")] = None,
    pJobId: Annotated[str | None, Field(description="Job whose plan to validate")] = None,
    pProfile: str | None = None,
    pSource: Annotated[str | None, Field(description="Audio file to check the plan against")] = None,
    pRoot: RootName = "input",
    pAllowOverlappingOperations: Annotated[bool, Field(description="Merge overlapping ranges instead of rejecting them")] = False,
    pMaxDeletedFraction: Annotated[float | None, Field(ge=0.0, le=0.99, description="Raise or lower the deletion budget")] = None,
    pMaxSingleDeleteSeconds: Annotated[float | None, Field(gt=0.0, le=36_000.0)] = None,
  ) -> dict[str, Any]:
    """Check a plan before it is allowed anywhere near the audio.

    Safety limits can be relaxed through the explicit, bounded arguments here;
    there is no way to switch validation off.
    """
    vPipeline = pServerState.pipeline
    vPlan = _fResolvePlan(pServerState, pPlan, pPlanPath, pJobId)
    vProfile = vPipeline.mResolveProfile(pProfile)
    vProbe = None
    if pSource:
      vProbe = vPipeline.mProbe(_fResolveAudio(pServerState, pSource, pRoot))
    vOverride = SafetyOverride(
      max_deleted_fraction=pMaxDeletedFraction,
      max_single_delete_seconds=pMaxSingleDeleteSeconds,
      allow_overlapping_operations=pAllowOverlappingOperations,
    )
    vValidation = vPipeline.mValidate(vPlan, vProfile, vProbe, vOverride)
    dPayload = vValidation.mAsDict()
    dPayload["plan_id"] = vPlan.id
    dPayload["profile"] = vProfile.name
    return dPayload

  @pServer.tool(title="Simulate an edit plan")
  @fGuardTool
  @fMcpParameters('simulate_edit_plan')
  def fToolSimulateEditPlan(
    pPlan: dict[str, Any] | None = None,
    pPlanPath: str | None = None,
    pJobId: str | None = None,
    pProfile: str | None = None,
    pSource: Annotated[str | None, Field(description="Audio file to check the plan against")] = None,
    pRoot: RootName = "input",
    pIncludeRegions: Annotated[bool, Field(description="List every removed region")] = True,
  ) -> dict[str, Any]:
    """Report what a plan would do, without touching any audio.

    Returns the estimated final duration, the removed regions, the percentage
    removed and any conflicts. Always cheap.
    """
    vPipeline = pServerState.pipeline
    vPlan = _fResolvePlan(pServerState, pPlan, pPlanPath, pJobId)
    vProfile = vPipeline.mResolveProfile(pProfile)
    vProbe = None
    if pSource:
      vProbe = vPipeline.mProbe(_fResolveAudio(pServerState, pSource, pRoot))
    vSimulation = vPipeline.mSimulate(vPlan, vProfile, vProbe)
    dPayload = vSimulation.model_dump(mode="json")
    if not pIncludeRegions:
      dPayload.pop("removed_regions", None)
    elif len(vSimulation.removed_regions) > cMaxOperationsInResponse:
      dPayload["removed_regions"] = vSimulation.removed_regions[:cMaxOperationsInResponse]
      dPayload["removed_regions_truncated"] = True
    return dPayload

  @pServer.tool(title="Execute an edit plan")
  @fGuardTool
  @fMcpParameters('execute_edit_plan')
  def fToolExecuteEditPlan(
    pSource: Annotated[str, Field(description="Audio file the plan was built for")],
    pPlan: dict[str, Any] | None = None,
    pPlanPath: str | None = None,
    pJobId: Annotated[str | None, Field(description="Reuse the plan from this job")] = None,
    pProfile: str | None = None,
    pRoot: RootName = "input",
    pFormats: Annotated[list[Literal["wav", "flac", "mp3"]] | None, Field(description="Delivery formats; defaults to the profile")] = None,
    pAudacityProject: Annotated[bool | None, Field(description="Also write an .aup3 project when Audacity is reachable")] = None,
    pOverwrite: Annotated[bool, Field(description="Replace artifacts that already exist")] = False,
    pOutputName: Annotated[str | None, Field(max_length=120, description="Base name for the artifacts")] = None,
    pMastering: Annotated[
      bool,
      Field(description="Render the plan's mastering operations too; false renders only its cuts, as <name>.wav"),
    ] = True,
  ) -> dict[str, Any]:
    """Validate a plan and render it into final audio.

    The plan is re-validated against this exact file - including its SHA-256 -
    before anything is written. The source is never modified.
    """
    vPipeline = pServerState.pipeline
    vPlan = _fResolvePlan(pServerState, pPlan, pPlanPath, pJobId)
    vOptions = ProcessOptions(
      profile=pProfile,
      dry_run=False,
      overwrite=pOverwrite,
      transcribe=False,
      audacity_project=pAudacityProject,
      formats=list(pFormats) if pFormats else None,
      plan=vPlan,
      output_name=pOutputName,
      mastering=pMastering,
    )
    vResult = vPipeline.mProcess(_fResolveAudio(pServerState, pSource, pRoot), vOptions)
    return _fProcessPayload(pServerState, vResult)

  # ------------------------------------------------------- direct rendering --

  @pServer.tool(title="Normalise audio")
  @fGuardTool
  @fMcpParameters('normalize_audio')
  def fToolNormalizeAudio(
    pPath: Annotated[str, Field(description="Audio file to normalise")],
    pOutputName: Annotated[str | None, Field(max_length=120, description="Base name for the artifacts")] = None,
    pTargetLufs: Annotated[float | None, Field(ge=-40.0, le=-5.0, description="Integrated loudness target")] = None,
    pTruePeakDbtp: Annotated[float | None, Field(ge=-12.0, le=0.0, description="True-peak ceiling")] = None,
    pProfile: str | None = None,
    pRoot: RootName = "input",
    pOverwrite: bool = False,
  ) -> dict[str, Any]:
    """Master a file to a loudness target without editing its content.

    Builds a plan containing only the mastering chain, validates it and renders
    it, so this shares exactly the same code path as a full edit.
    """
    from podcast_editor_mcp_server.models.edit_plan import EditReason, LimitTruePeak, NormalizeLoudness, PlanProvenance

    vPipeline = pServerState.pipeline
    vSource = _fResolveAudio(pServerState, pPath, pRoot)
    vProfile = vPipeline.mResolveProfile(pProfile)
    vTargetLufs = pTargetLufs if pTargetLufs is not None else vProfile.loudness.integrated_lufs
    vTargetPeak = pTruePeakDbtp if pTruePeakDbtp is not None else vProfile.loudness.true_peak_dbtp
    vProbe = vPipeline.mProbe(vSource)

    vPlan = EditPlan(
      source=vProbe.label,
      source_path=vProbe.path,
      source_sha256=vProbe.sha256,
      source_duration=vProbe.duration_seconds,
      sample_rate=vProbe.sample_rate,
      channels=vProbe.channels,
      provenance=PlanProvenance(profile=vProfile.name),
      operations=[
        NormalizeLoudness(
          integrated_lufs=vTargetLufs,
          true_peak_dbtp=vTargetPeak,
          loudness_range=vProfile.loudness.loudness_range,
          reason=EditReason.MASTERING,
        ),
        LimitTruePeak(true_peak_dbtp=vTargetPeak, reason=EditReason.MASTERING),
      ],
      notes="Mastering only; no content was removed.",
    )
    vOptions = ProcessOptions(
      profile=pProfile,
      transcribe=False,
      audacity_project=False,
      overwrite=pOverwrite,
      plan=vPlan,
      output_name=pOutputName,
      export_transcript=False,
    )
    vResult = vPipeline.mProcess(vSource, vOptions)
    return _fProcessPayload(pServerState, vResult)

  @pServer.tool(title="Master edited audio")
  @fGuardTool
  @fMcpParameters('master_audio')
  def fToolMasterAudio(
    pPath: Annotated[str, Field(description="Edited audio file to master, e.g. '<name>/<name>.wav' under output")],
    pProfile: Annotated[str | None, Field(description="Profile whose clean-up, loudness and output settings apply")] = None,
    pOutputName: Annotated[str | None, Field(max_length=120, description="Base name for the artifacts; defaults to the file name")] = None,
    pFormats: Annotated[list[Literal["wav", "flac", "mp3"]] | None, Field(description="Delivery formats; defaults to the profile")] = None,
    pOverwrite: Annotated[bool, Field(description="Replace artifacts that already exist")] = False,
    pRoot: RootName = "output",
  ) -> dict[str, Any]:
    """Clean up and master an audio file that is already edited, without cutting anything.

    Applies the profile's chain in a fixed order - high-pass and low-pass
    filters, noise reduction, compressor, EBU R128 loudness normalisation,
    true-peak limiter, fades - and exports every delivery format into
    ``<output>/<name>/``: ``<name>.master.wav``, ``<name>.mp3``... Meant for the
    audio ``finish_episode_edit`` delivers with ``mastering=false``; its plan,
    report and transcript are left untouched.
    """
    vPipeline = pServerState.pipeline
    vSource = _fResolveAudio(pServerState, pPath, pRoot)
    vResult = vPipeline.mMasterFile(
      vSource, pProfile, pOutputName, list(pFormats) if pFormats else None, pOverwrite,
    )
    vOutcome = vResult.outcome
    return {
      "source": vPipeline.guard.mLabel(vSource),
      "profile": vResult.plan.provenance.profile,
      "artifacts": {vFormat: vPipeline.guard.mLabel(vPath) for vFormat, vPath in vOutcome.exports.items()},
      "duration_seconds": round(vOutcome.output_duration, 3),
      "applied": [
        vOperation.model_dump(mode="json", exclude={"id", "source", "confidence", "note"})
        for vOperation in vResult.plan.mMasteringOperations()
        if vOperation.id in vOutcome.applied_operations
      ],
      "loudness_before": _fLoudnessSummary(vOutcome.loudness_before),
      "loudness_after": _fLoudnessSummary(vOutcome.loudness_after),
      "warnings": vOutcome.warnings[:50],
    }

  @pServer.tool(title="Export audio")
  @fGuardTool
  @fMcpParameters('export_audio')
  def fToolExportAudio(
    pPath: Annotated[str, Field(description="Audio file to convert")],
    pFormat: Annotated[Literal["wav", "flac", "mp3"], Field(description="Target format")],
    pOutputName: Annotated[str | None, Field(max_length=120, description="File name without extension")] = None,
    pSampleRate: Annotated[int | None, Field(ge=8000, le=192_000)] = None,
    pChannels: Annotated[int | None, Field(ge=1, le=2)] = None,
    pMp3BitrateKbps: Annotated[int, Field(ge=32, le=320)] = 128,
    pProfile: str | None = None,
    pRoot: RootName = "output",
    pOverwrite: bool = False,
  ) -> dict[str, Any]:
    """Convert a file into another delivery format, without editing it."""
    vPipeline = pServerState.pipeline
    vSource = _fResolveAudio(pServerState, pPath, pRoot)
    vProfile = vPipeline.mResolveProfile(pProfile)
    vProbe = vPipeline.mProbe(vSource)
    vName = pOutputName or vSource.stem
    vTarget = vPipeline.guard.mResolveOutput(
      f"{vName}.{pFormat}",
      "output",
      pAllowedExtensions=cAudioWriteExtensions,
      pOverwrite=pOverwrite,
      pSource=vSource,
    )
    with vPipeline.mReserveOutput(vTarget):
      vPipeline.ffmpeg.mExport(
        vSource,
        vTarget,
        pFormat,
        pSampleRate or vProbe.sample_rate,
        pChannels or vProbe.channels,
        pWavCodec=vProfile.output.wav_codec,
        pMp3BitrateKbps=pMp3BitrateKbps,
        pFlacCompression=vProfile.output.flac_compression,
        pOverwrite=pOverwrite,
      )
    return {
      "source": vPipeline.guard.mLabel(vSource),
      "output": vPipeline.guard.mLabel(vTarget),
      "format": pFormat,
      "size_bytes": vTarget.stat().st_size,
      "duration_seconds": round(vPipeline.ffmpeg.mDuration(vTarget), 3),
    }

  # -------------------------------------------------------------- Audacity --

  @pServer.tool(title="Audacity status")
  @fGuardTool
  @fMcpParameters('get_audacity_status')
  def fToolGetAudacityStatus() -> dict[str, Any]:
    """Report whether Audacity can be driven right now, and why not if not."""
    from podcast_editor_mcp_server.backends.audacity import cModuleStatusNames, fAudacityConfigPath, fReadModuleStatus

    vBackend = AudacityBackend(pServerState.context.config.audacity)
    vStatus = vBackend.mStatus()
    vModule = fReadModuleStatus()
    return {
      "available": vStatus.available,
      "reason": vStatus.reason,
      "version": vStatus.version,
      "running": fAudacityIsRunning(),
      "module_status": cModuleStatusNames.get(vModule, "not configured") if vModule is not None else "not configured",
      "config_file": str(fAudacityConfigPath()),
      "details": vStatus.details,
      "how_to_fix": [
        "Close Audacity.",
        "Run 'podcast-editor-mcp-server-cli audacity-status --enable-module'.",
        "Start Audacity again (or run 'podcast-editor-mcp-server-cli audacity-start' for a headless instance).",
      ]
      if not vStatus.available
      else [],
    }

  @pServer.tool(title="Create an Audacity project")
  @fGuardTool
  @fMcpParameters('create_audacity_project')
  def fToolCreateAudacityProject(
    pPath: Annotated[str, Field(description="Audio file to import into the project")],
    pOutputName: Annotated[str | None, Field(max_length=120, description="Project name without extension")] = None,
    pRoot: RootName = "output",
  ) -> dict[str, Any]:
    """Build an editable ``.aup3`` project around an audio file.

    Requires a reachable Audacity with ``mod-script-pipe``; call
    ``get_audacity_status`` first if you are not sure.
    """
    vPipeline = pServerState.pipeline
    vSource = _fResolveAudio(pServerState, pPath, pRoot)
    # Audacity opens the file with its own importers, so it is first checked to
    # be one of the accepted audio containers.
    vPipeline.ffmpeg.mProbeRaw(vSource)
    vBackend = AudacityBackend(pServerState.context.config.audacity)
    vStatus = vBackend.mStatus()
    if not vStatus.available:
      raise ToolError(f"Audacity is not available: {vStatus.reason}")
    vName = pOutputName or vSource.stem
    vProject = vPipeline.guard.mResolveOutput(
      f"{vName}.aup3", "output", pAllowedExtensions=frozenset({".aup3"}), pOverwrite=True, pSource=vSource,
    )
    vResult = vBackend.mCreateProject(pAudio=vSource, pProject=vProject, pMode="master")
    return {
      "project": vPipeline.guard.mLabel(vResult.project),
      "mode": vResult.mode,
      "imported": vResult.imported,
      "size_bytes": vResult.project.stat().st_size,
      "warnings": vResult.warnings,
    }

  @pServer.tool(title="Open a file in Audacity")
  @fGuardTool
  @fMcpParameters('open_in_audacity')
  def fToolOpenInAudacity(
    pPath: Annotated[str, Field(description="Audio file to import into a new Audacity window")],
    pRoot: RootName = "output",
  ) -> dict[str, Any]:
    """Import a file into a fresh Audacity project so a human can look at it.

    Nothing is saved; this just puts the audio in front of the user.
    """
    from podcast_editor_mcp_server.backends.audacity import AudacityCommand

    vPipeline = pServerState.pipeline
    vSource = _fResolveAudio(pServerState, pPath, pRoot)
    # Audacity opens the file with its own importers, so it is first checked to
    # be one of the accepted audio containers.
    vPipeline.ffmpeg.mProbeRaw(vSource)
    vBackend = AudacityBackend(pServerState.context.config.audacity)
    vStatus = vBackend.mStatus()
    if not vStatus.available:
      raise ToolError(f"Audacity is not available: {vStatus.reason}")
    vBackend.client.mAcquireProcessLock()
    try:
      vBackend.client.mConnect(pServerState.context.config.audacity.connect_timeout_seconds)
      vBackend.client.mSend(AudacityCommand("New"))
      vBackend.client.mSend(AudacityCommand("Import2", {"Filename": str(vSource)}))
      vBackend.client.mSend(AudacityCommand("FitInWindow"))
    finally:
      vBackend.client.mClose()
    return {"opened": vPipeline.guard.mLabel(vSource), "note": "The project was not saved."}

  @pServer.tool(title="Save the Audacity project")
  @fGuardTool
  @fMcpParameters('save_audacity_project')
  def fToolSaveAudacityProject(
    pOutputName: Annotated[str, Field(max_length=120, description="Project name without extension")],
  ) -> dict[str, Any]:
    """Save whatever Audacity currently has open as an ``.aup3`` project.

    Use this after ``open_in_audacity`` and any manual edits the user made.
    """
    from podcast_editor_mcp_server.backends.audacity import AudacityCommand

    vPipeline = pServerState.pipeline
    vBackend = AudacityBackend(pServerState.context.config.audacity)
    vStatus = vBackend.mStatus()
    if not vStatus.available:
      raise ToolError(f"Audacity is not available: {vStatus.reason}")
    vProject = vPipeline.guard.mResolveOutput(
      f"{pOutputName}.aup3", "output", pAllowedExtensions=frozenset({".aup3"}), pOverwrite=True
    )
    if vProject.exists():
      vProject.unlink()
    vBackend.client.mAcquireProcessLock()
    try:
      vBackend.client.mConnect(pServerState.context.config.audacity.connect_timeout_seconds)
      vBackend.client.mSend(AudacityCommand("SaveProject2", {"Filename": str(vProject), "AddToHistory": 0}))
    finally:
      vBackend.client.mClose()
    if not vProject.exists():
      raise ToolError("Audacity did not write the project file")
    return {"project": vPipeline.guard.mLabel(vProject), "size_bytes": vProject.stat().st_size}

  # ------------------------------------------------------------------ jobs --

  @pServer.tool(title="Process a podcast")
  @fGuardTool
  @fMcpParameters('process_podcast')
  def fToolProcessPodcast(
    pPath: Annotated[str, Field(description="Audio file to process")],
    pProfile: Annotated[str | None, Field(description="Profile name; defaults to the active profile")] = None,
    pDryRun: Annotated[bool, Field(description="Plan and simulate only; write no audio")] = False,
    pTranscribe: Annotated[bool, Field(description="Run transcription (needed for filler and repetition detection)")] = True,
    pFormats: Annotated[list[Literal["wav", "flac", "mp3"]] | None, Field(description="Delivery formats; defaults to the profile")] = None,
    pAudacityProject: Annotated[bool | None, Field(description="Also write an .aup3 project when Audacity is reachable")] = None,
    pOverwrite: Annotated[bool, Field(description="Replace artifacts that already exist")] = False,
    pOutputName: Annotated[str | None, Field(max_length=120, description="Base name for the artifacts")] = None,
    pDecisions: Annotated[
      dict[str, Literal["cut", "keep"]] | None,
      Field(description="Candidate id -> 'cut' or 'keep'; undecided review candidates follow their proposal"),
    ] = None,
    pExtraCuts: Annotated[
      list[RequestedCut] | None,
      Field(description="Further cuts by the word timestamps of the source transcript (see finish_episode_edit)", max_length=500),
    ] = None,
    pWait: Annotated[bool, Field(description="Block until the job finishes; set false for long recordings")] = True,
    pRoot: RootName = "input",
  ) -> dict[str, Any]:
    """Run the whole pipeline: analyse, transcribe, plan, validate, render, export.

    This is the high-level path. It uses exactly the same layers and the same
    validator as the individual tools, so nothing is bypassed. Candidates that
    need a decision follow their proposed action unless ``decisions`` says
    otherwise; use ``prepare_episode_edit`` first to see them.

    With ``wait=false`` the job runs in the background and this returns a job
    id immediately; poll it with ``get_job``.
    """
    vPipeline = pServerState.pipeline
    vSource = _fResolveAudio(pServerState, pPath, pRoot)
    vOptions = ProcessOptions(
      profile=pProfile,
      dry_run=pDryRun,
      overwrite=pOverwrite,
      transcribe=pTranscribe,
      audacity_project=pAudacityProject,
      formats=list(pFormats) if pFormats else None,
      output_name=pOutputName,
      decisions=dict(pDecisions or {}),
      apply_proposals=True,
      extra_cuts=[vCut.model_dump() for vCut in pExtraCuts or []],
    )

    if pWait:
      vResult = vPipeline.mProcess(vSource, vOptions)
      return _fProcessPayload(pServerState, vResult)

    vJobId = _fStartBackgroundJob(pServerState, vSource, vOptions)
    return {
      "job_id": vJobId,
      "state": "queued",
      "waiting": False,
      "next_step": f"Poll get_job with job_id='{vJobId}'.",
    }

  @pServer.tool(title="Get a job")
  @fGuardTool
  @fMcpParameters('get_job')
  def fToolGetJob(
    pJobId: Annotated[str, Field(max_length=64, description="Job identifier")],
    pIncludeMessages: Annotated[bool, Field(description="Include the job's message log")] = True,
  ) -> dict[str, Any]:
    """Read a job's state, progress, artifacts and metrics."""
    vJob = pServerState.pipeline.jobs.mLoad(pJobId)
    dPayload = vJob.mSummary()
    dPayload["artifacts"] = [
      {"kind": vArtifact.kind, "path": vArtifact.label, "size_bytes": vArtifact.size_bytes}
      for vArtifact in vJob.artifacts
    ]
    dPayload["metrics"] = vJob.metrics
    dPayload["errors"] = [vError.model_dump(mode="json") for vError in vJob.errors]
    if pIncludeMessages:
      dPayload["messages"] = [vMessage.model_dump(mode="json") for vMessage in vJob.messages[-50:]]
    return dPayload

  @pServer.tool(title="List jobs")
  @fGuardTool
  @fMcpParameters('list_jobs')
  def fToolListJobs(
    pState: Annotated[str | None, Field(description="Filter by state, e.g. 'completed' or 'failed'")] = None,
    pLimit: Annotated[int, Field(ge=1, le=200)] = 20,
  ) -> dict[str, Any]:
    """List recent jobs, newest first."""
    vState: JobState | None = None
    if pState:
      try:
        vState = JobState(pState)
      except ValueError as vError:
        raise ToolError(
          f"Unknown job state '{pState}'. Valid states: {', '.join(vItem.value for vItem in JobState)}"
        ) from vError
    lJobs = pServerState.pipeline.jobs.mList(vState, pLimit)
    return {
      "count": len(lJobs),
      "running_in_background": pServerState.mBackgroundJobIds(),
      "jobs": [vJob.mSummary() for vJob in lJobs],
    }

  @pServer.tool(title="Get job artifacts")
  @fGuardTool
  @fMcpParameters('get_job_artifacts')
  def fToolGetJobArtifacts(
    pJobId: Annotated[str, Field(max_length=64, description="Job identifier")],
  ) -> dict[str, Any]:
    """List everything a job produced, with sizes and hashes."""
    vJob = pServerState.pipeline.jobs.mLoad(pJobId)
    return {
      "job_id": vJob.id,
      "state": str(vJob.state),
      "output_dir": vJob.output_dir,
      "count": len(vJob.artifacts),
      "artifacts": [vArtifact.model_dump(mode="json") for vArtifact in vJob.artifacts],
    }

  @pServer.tool(title="Cancel a job")
  @fGuardTool
  @fMcpParameters('cancel_job')
  def fToolCancelJob(
    pJobId: Annotated[str, Field(max_length=64, description="Job identifier")],
  ) -> dict[str, Any]:
    """Cancel a queued job or ask a running job to stop at its next phase boundary.

    Cancellation is cooperative: the current phase finishes, then the job stops
    and is marked cancelled. A job that has already completed cannot be
    cancelled.
    """
    vPipeline = pServerState.pipeline
    vJob = vPipeline.jobs.mLoad(pJobId)
    if vJob.is_terminal:
      return {"job_id": pJobId, "state": str(vJob.state), "cancelled": False, "reason": "The job already finished"}
    pServerState.mRequestCancel(pJobId)
    vJob = vPipeline.jobs.mLoad(pJobId)
    return {
      "job_id": pJobId,
      "state": str(vJob.state),
      "cancelled": True,
      "note": (
        "The queued job was cancelled before processing."
        if vJob.state == JobState.CANCELLED else "The job will stop at its next phase boundary."
      ),
    }

  # ------------------------------------------------------------ maintenance --

  @pServer.tool(title="Reload configuration")
  @fGuardTool
  @fMcpParameters('reload_configuration')
  def fToolReloadConfiguration() -> dict[str, Any]:
    """Re-read the configuration and profiles from disk.

    Every profile is loaded, so ``profile_errors`` names the ones that would
    fail and the keys that are wrong. A profile listed in ``profiles`` but also
    in ``profile_errors`` cannot be used until its file is fixed.
    """
    pServerState.mReload()
    vContext = pServerState.context
    return {
      "active_profile": vContext.profile.name,
      "profiles": [dProfile["name"] for dProfile in fListProfiles(vContext.config)],
      "profile_errors": fCheckProfiles(vContext.config),
      "workspace": vContext.workspace.mDescribe(),
    }

  @pServer.tool(title="Enable Audacity scripting")
  @fGuardTool
  @fMcpParameters('enable_audacity_scripting')
  def fToolEnableAudacityScripting() -> dict[str, Any]:
    """Set ``mod-script-pipe=1`` in ``audacity.cfg``.

    This writes the same key Audacity's own Preferences > Modules page writes.
    Audacity must be closed: it rewrites the file when it exits and would undo
    the change.
    """
    if fAudacityIsRunning():
      raise ToolError("Audacity is running; close it first, because it rewrites audacity.cfg on exit")
    dResult = fEnableScriptPipeModule()
    if "error" in dResult:
      raise ToolError(str(dResult["error"]))
    dResult["next_step"] = "Start Audacity again; the script pipes appear a few seconds later."
    return dResult


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _fResolveAudio(pState: ServerState, pPath: str, pRoot: str) -> Path:
  """Resolve a caller-supplied audio path inside the workspace."""
  if pRoot not in {"input", "output", "work"}:
    raise ToolError(f"Unknown workspace root '{pRoot}'")
  return pState.pipeline.guard.mResolveInput(pPath, pRoot)


def _fScratchDir(pState: ServerState, pSource: Path) -> Path:
  """A stable scratch directory for read-only analysis of one file.

  Stable so that analysing the same file twice reuses the decoded copy instead
  of making another one. Old ones are swept here rather than at exit, because a
  stdio server can be killed at any moment and would never get the chance.
  Using a directory marks it as recently used, which is what the size quota
  goes by.
  """
  import os

  from podcast_editor_mcp_server.util.hashing import fHashText, fShortHash

  vWork = pState.context.workspace.work
  vName = f"scratch-{fShortHash(fHashText(str(pSource)), 16)}"
  vPath = vWork / vName
  vPath.mkdir(parents=True, exist_ok=True)
  os.utime(vPath)
  _fSweepScratchDirs(vWork, vPath)
  return vPath


def _fSweepScratchDirs(pWorkDir: Path, pKeep: Path) -> None:
  """Delete old analysis scratch directories, then the least recently used ones
  until all of them fit in ``cScratchMaxBytes``."""
  import shutil
  import time

  vCutoff = time.time() - cScratchMaxAgeSeconds
  lCandidates: list[tuple[float, int, Path]] = []
  vTotal = 0
  for vCandidate in pWorkDir.glob("scratch-*"):
    if not vCandidate.is_dir() or vCandidate.is_symlink():
      continue
    try:
      vUsed = vCandidate.stat().st_mtime
      vSize = sum(vFile.stat().st_size for vFile in vCandidate.iterdir() if vFile.is_file())
    except OSError:
      continue
    if vCandidate != pKeep and vUsed < vCutoff:
      shutil.rmtree(vCandidate, ignore_errors=True)
      continue
    vTotal += vSize
    if vCandidate != pKeep:
      lCandidates.append((vUsed, vSize, vCandidate))
  for vUsed, vSize, vCandidate in sorted(lCandidates):
    if vTotal <= cScratchMaxBytes:
      break
    shutil.rmtree(vCandidate, ignore_errors=True)
    vTotal -= vSize


def _fLoadCachedTranscript(
  pState: ServerState,
  pPath: str,
  pProfile: str | None,
  pRoot: str,
) -> tuple[Transcript, Any, Path]:
  """Load the cached transcript for a file, with a clear error when absent."""
  vPipeline = pState.pipeline
  vSource = _fResolveAudio(pState, pPath, pRoot)
  vProfile = vPipeline.mResolveProfile(pProfile)
  vProbe = vPipeline.mProbe(vSource)
  vTranscript = vPipeline.mLoadCachedTranscript(vProbe.sha256, vProfile)
  if vTranscript is None:
    raise ToolError(
      f"No transcript is cached for '{vPipeline.guard.mLabel(vSource)}' with profile '{vProfile.name}'. "
      "Call transcribe_audio first."
    )
  return (vTranscript, vPipeline, vSource)


def _fCandidateResponse(
  pState: ServerState,
  pSource: Path,
  pCandidates: list[Any],
  pMinConfidence: float,
  pLimit: int,
  pType: str,
) -> dict[str, Any]:
  lFiltered = [vCandidate for vCandidate in pCandidates if vCandidate.confidence >= pMinConfidence]
  return {
    "path": pState.pipeline.guard.mLabel(pSource),
    "type": pType,
    "count": len(lFiltered),
    "total_seconds": round(sum(vCandidate.duration for vCandidate in lFiltered), 3),
    "candidates": [vCandidate.model_dump(mode="json") for vCandidate in lFiltered[:pLimit]],
    "truncated": len(lFiltered) > pLimit,
    "note": "These are candidates, not decisions. Build an EditPlan from the ones you judge to be real.",
  }


def _fResolvePlan(
  pState: ServerState,
  pPlan: dict[str, Any] | None,
  pPlanPath: str | None,
  pJobId: str | None,
) -> EditPlan:
  """Load a plan from a document, a saved file or a job."""
  lGiven = [vItem for vItem in (pPlan, pPlanPath, pJobId) if vItem is not None]
  if len(lGiven) != 1:
    raise ToolError("Provide exactly one of: plan, plan_path, job_id")

  if pPlan is not None:
    try:
      return EditPlan.model_validate(pPlan)
    except ValidationError as vError:
      raise ToolError(f"The plan document is invalid: {_fFirstValidationProblem(vError)}") from vError

  if pPlanPath is not None:
    return pState.pipeline.mLoadPlan(pPlanPath)

  vPlan = pState.pipeline.mFindPlanForJob(str(pJobId))
  if vPlan is None:
    raise ToolError(f"Job '{pJobId}' has no readable plan artifact")
  return vPlan


def _fPlanPayload(pPlan: EditPlan, pIncludeOperations: bool = True) -> dict[str, Any]:
  dPayload = pPlan.mSummary()
  if pIncludeOperations:
    ldOperations = [vOperation.model_dump(mode="json") for vOperation in pPlan.operations[:cMaxOperationsInResponse]]
    dPayload["operations"] = ldOperations
    dPayload["operations_truncated"] = len(pPlan.operations) > cMaxOperationsInResponse
  dPayload["created_at"] = pPlan.created_at
  dPayload["notes"] = pPlan.notes
  return dPayload


cSpeechReasons: frozenset[str] = frozenset({"mistake", "repetition", "false_start", "filler_word", "manual", "other"})


def _fRemovedText(pTranscript: Transcript | None, pStart: float, pEnd: float) -> str:
  if pTranscript is None:
    return ""
  lWords = [
    vWord.word.strip()
    for vWord in pTranscript.mAllWords()
    if pStart <= (vWord.start + vWord.end) / 2.0 <= pEnd
  ]
  vText = " ".join(vWord for vWord in lWords if vWord)
  return vText if len(vText) <= 200 else vText[:199] + "…"


def _fCutList(pPlan: EditPlan, pTranscript: Transcript | None) -> list[dict[str, Any]]:
  """Every cut through speech, with the words it removes."""
  dJoins = {
    str(dItem.get("operation_id")): dItem
    for dItem in (pPlan.metadata.get("joins", {}) or {}).get("unverified", [])
  }
  vChecked = bool(pPlan.metadata.get("joins"))
  ldCuts: list[dict[str, Any]] = []
  for vOperation in pPlan.mStructuralOperations():
    if vOperation.type != "delete_range" or _fIsGapCut(vOperation):
      continue
    if len(ldCuts) >= cMaxOperationsInResponse:
      break
    vReason = str(vOperation.reason)
    ldCuts.append(
      {
        "id": vOperation.id,
        "start": round(vOperation.start, 3),  # type: ignore[attr-defined]
        "end": round(vOperation.end, 3),  # type: ignore[attr-defined]
        "duration": round(vOperation.end - vOperation.start, 3),  # type: ignore[attr-defined]
        "reason": vReason,
        "candidate": vOperation.note,
        "text": _fRemovedText(pTranscript, vOperation.start, vOperation.end),  # type: ignore[attr-defined]
        "join_verified": (vOperation.id not in dJoins) if (vChecked and vReason in cSpeechReasons) else None,
      }
    )
  return ldCuts[:cMaxOperationsInResponse]


def _fIsGapCut(pOperation: Any) -> bool:
  """Whether an operation is one of tight pacing's cuts between words."""
  return pOperation.type == "delete_range" and str(getattr(pOperation, "note", "") or "").startswith("gap-")


def _fPauseSummary(pPlan: EditPlan) -> dict[str, Any]:
  lPauses = [vOperation for vOperation in pPlan.mStructuralOperations() if vOperation.type == "shorten_silence"]
  lGaps = [vOperation for vOperation in pPlan.mStructuralOperations() if _fIsGapCut(vOperation)]
  vRemoved = sum(
    max(0.0, vOperation.end - vOperation.start - vOperation.target_duration)  # type: ignore[attr-defined]
    for vOperation in lPauses
  )
  vRemoved += sum(vOperation.end - vOperation.start for vOperation in lGaps)  # type: ignore[attr-defined]
  dPacing = pPlan.metadata.get("pacing") or {}
  dSummary: dict[str, Any] = {"count": len(lPauses) + len(lGaps), "removed_seconds": round(vRemoved, 3)}
  if dPacing.get("tight"):
    dSummary["tight_pacing"] = True
    dSummary["gaps_between_words_shortened"] = len(lGaps)
    dSummary["gaps_kept_because_words_were_heard"] = dPacing.get("kept_for_speech", [])[:30]
  return dSummary


def _fJoinSummary(pPlan: EditPlan) -> dict[str, Any] | None:
  dJoins = pPlan.metadata.get("joins")
  if not dJoins:
    return None
  return {
    "checked": dJoins.get("checked", 0),
    "verified": dJoins.get("verified", 0),
    "adjusted": dJoins.get("adjusted", 0),
    "unverified": dJoins.get("unverified", [])[:30],
  }


def _fTranscriptSummary(pTranscript: Transcript | None) -> dict[str, Any] | None:
  if pTranscript is None:
    return None
  return {
    "language": pTranscript.language,
    "model": pTranscript.model,
    "device": pTranscript.device,
    "segment_count": len(pTranscript.segments),
    "word_count": pTranscript.word_count,
  }


def _fLoudnessSummary(pMeasurement: Any) -> dict[str, float | None] | None:
  """The loudness figures worth reporting from a measurement."""
  if pMeasurement is None:
    return None
  return {
    "integrated_lufs": pMeasurement.integrated_lufs,
    "loudness_range": pMeasurement.loudness_range,
    "true_peak_dbfs": pMeasurement.true_peak_dbfs,
  }


def _fProcessPayload(pState: ServerState, pResult: Any) -> dict[str, Any]:
  """Shape a :class:`~podcast_editor_mcp_server.pipeline.ProcessResult` for a tool response."""
  vJob = pResult.job
  dPayload: dict[str, Any] = {
    "job_id": vJob.id,
    "state": str(vJob.state),
    "profile": vJob.profile,
    "source": vJob.source_label,
    "source_sha256": vJob.source_sha256,
    "artifacts": {vKind: pState.pipeline.guard.mLabel(vPath) for vKind, vPath in pResult.artifacts.items()},
    "warnings": pResult.warnings[:50],
  }
  if pResult.report is not None:
    dPayload["report"] = pResult.report.mSummary()
  if pResult.simulation is not None:
    dPayload["simulation"] = {
      "estimated_output_seconds": pResult.simulation.estimated_output_seconds,
      "removed_seconds": pResult.simulation.removed_seconds,
      "removed_percent": round(pResult.simulation.removed_fraction * 100.0, 2),
      "operation_count": pResult.simulation.operation_count,
      "valid": pResult.simulation.valid,
    }
  if pResult.plan is not None:
    dPayload["plan_id"] = pResult.plan.id
    dPayload["plan_summary"] = pResult.plan.mCountByType()
  return dPayload


def _fStartBackgroundJob(pState: ServerState, pSource: Path, pOptions: ProcessOptions) -> str:
  """Kick off a run on the executor and return the job id immediately.

  The job id is created here, before the thread starts, so the caller can poll
  ``get_job`` straight away. The pipeline is told to reuse it. A request beyond
  ``jobs.max_concurrent`` running jobs plus ``jobs.max_queued`` waiting ones is
  refused before any record is written.
  """
  with pState.admission_lock:
    pState.mCheckBackgroundCapacity()
    return _fSubmitBackgroundJob(pState, pSource, pOptions)


def _fSubmitBackgroundJob(pState: ServerState, pSource: Path, pOptions: ProcessOptions) -> str:
  """Record a queued job and hand it to the executor; the caller holds the admission lock."""
  # Capture this pipeline so a configuration reload cannot move an accepted
  # job into a different workspace halfway through its lifecycle.
  vPipeline = pState.pipeline
  vQueued = Job(
    source_path=str(pSource),
    source_label=vPipeline.guard.mLabel(pSource),
    profile=pOptions.profile or vPipeline.context.profile.name,
    options={"dry_run": pOptions.dry_run, "transcribe": pOptions.transcribe},
  )
  vPipeline.jobs.mSave(vQueued)
  vJobId = vQueued.id
  pOptions.job_id = vJobId
  pOptions.cancel_check = lambda: pState.mIsCancelled(vJobId)

  def fRecordFailure(pError: Exception) -> None:
    vJob = vPipeline.jobs.mLoad(vJobId)
    if not vJob.is_terminal:
      if isinstance(pError, PodcastEditorMcpServerError):
        vJob.mAddError(pError.cCode, pError.message, pError.details)
      else:
        vJob.mAddError("internal_error", str(pError))
      vJob.mSetState(JobState.FAILED, "Background job failed")
      vPipeline.jobs.mSave(vJob)

  def fRecordCancellation() -> None:
    vJob = vPipeline.jobs.mLoad(vJobId)
    if not vJob.is_terminal:
      vJob.mSetState(JobState.CANCELLED, "Cancelled before processing")
      vPipeline.jobs.mSave(vJob)

  def fRun() -> None:
    try:
      if pState.mIsCancelled(vJobId):
        fRecordCancellation()
        return
      vPipeline.mProcess(pSource, pOptions)
    except PodcastEditorMcpServerError as vError:
      fRecordFailure(vError)
      vLog.warning("Background job failed", extra={"job_id": vJobId, "error": vError.cCode})
    except Exception as vError:  # noqa: BLE001 - persist failures before pipeline initialization too
      fRecordFailure(vError)
      vLog.exception("Background job crashed", extra={"job_id": vJobId})
    finally:
      pState.mClearCancel(vJobId)

  def fOnDone(pFuture: Future[Any]) -> None:
    if pFuture.cancelled():
      try:
        fRecordCancellation()
      finally:
        pState.mClearCancel(vJobId)

  try:
    vFuture = pState.executor.submit(fRun)
  except Exception as vError:  # noqa: BLE001 - an accepted record must never remain queued on submission failure
    fRecordFailure(vError)
    pState.mClearCancel(vJobId)
    raise
  vFuture.add_done_callback(fOnDone)
  pState.mRegisterBackground(vJobId, vFuture)
  return vJobId
