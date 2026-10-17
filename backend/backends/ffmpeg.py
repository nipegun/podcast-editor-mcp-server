"""FFmpeg editing backend.

This is the workhorse. It is always available (FFmpeg is a hard dependency),
and every other part of the product can fall back to it.

Rendering strategy
------------------

1. **Prepare.** The source is decoded once into a working WAV with a known
   sample rate, channel count and sample format (``pcm_f32le``). Duration is
   preserved, so every timestamp in the EditPlan stays valid. The original file
   is only ever opened for reading.
2. **Structural render.** The compiled timeline becomes a single deterministic
   ``filter_complex``: one ``atrim`` branch per kept segment, one ``aevalsrc``
   branch per inserted silence, all joined by ``concat``. Very long timelines
   are rendered in bounded passes and joined with the ``concat`` demuxer, so a
   two-hour podcast with a thousand cuts cannot blow up the filter graph.
3. **Mastering.** Spectral clean-up and dynamics first, then two-pass
   ``loudnorm``, then the limiter, then fades.
4. **Export.** One encode per requested format.

No command is ever built by string concatenation, and ``shell=False`` is used
everywhere. Every media input goes through :func:`fInputArguments`: local files
only, read by the demuxer of an accepted audio container and by nothing else.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from podcast_editor_mcp_server.backends.base import BackendStatus, RenderRequest, RenderResult
from podcast_editor_mcp_server.errors import ExportError, FfmpegError
from podcast_editor_mcp_server.logging_setup import fGetLogger
from podcast_editor_mcp_server.models.audio import DynamicsStatistics, LoudnessMeasurement, SilenceRegion
from podcast_editor_mcp_server.util.atomic import fAtomicPath, fDefaultFileMode
from podcast_editor_mcp_server.util.proc import ProcessResult, fRunProcess, fWhich
from podcast_editor_mcp_server.util.timeline import KeepSegment, SilenceSegment, TimelineItem

vLog = fGetLogger("backends.ffmpeg")

cEnvFfmpeg: str = "PODCAST_EDITOR_MCP_SERVER_FFMPEG"
cEnvFfprobe: str = "PODCAST_EDITOR_MCP_SERVER_FFPROBE"

# Keeping each filter graph bounded avoids pathological memory use and command
# lines that exceed what the kernel will accept.
cMaxSegmentsPerPass: int = 120
cWorkingCodec: str = "pcm_f32le"

# alimiter caps sample peaks, not inter-sample (true) peaks. Aiming this far
# below the requested ceiling absorbs the usual reconstruction overshoot.
cTruePeakHeadroomDb: float = 0.3

# Media is opened from local files only, and only by the demuxers of the audio
# containers the workspace accepts. Left to itself FFmpeg picks the demuxer from
# the content, so a playlist or a concat script saved under an audio extension
# would make it open other files, or URLs, that the workspace guard never saw.
cInputProtocols: str = "file"
cInputFormats: str = "wav,w64,flac,mp3,mov,aac,ogg,wv,aiff,caf"

cDefaultProbeTimeout: float = 120.0
cDefaultRenderTimeout: float = 7_200.0
cDefaultMeasureTimeout: float = 3_600.0


@dataclass(slots=True)
class PreparedAudio:
  """A decoded working copy of a source file."""

  path: Path
  duration_seconds: float


def fInputArguments(pPath: Path) -> list[str]:
  """The input arguments for a media file, restricted to local audio containers."""
  return ["-protocol_whitelist", cInputProtocols, "-format_whitelist", cInputFormats, "-i", str(pPath)]


def fDbToLinear(pDecibels: float) -> float:
  """Convert a dB value into a linear amplitude ratio."""
  return float(10.0 ** (pDecibels / 20.0))


def fChannelLayout(pChannels: int) -> str:
  """FFmpeg channel-layout name for a channel count."""
  if pChannels <= 1:
    return "mono"
  if pChannels == 2:
    return "stereo"
  return f"{pChannels}c"


def fFormatNumber(pValue: float, pDigits: int = 6) -> str:
  """Format a float for a filter argument without scientific notation."""
  vText = f"{pValue:.{pDigits}f}".rstrip("0").rstrip(".")
  return vText if vText and vText not in {"-", "-0"} else "0"


class FfmpegBackend:
  """Everything this product does with FFmpeg lives here."""

  name = "ffmpeg"

  def __init__(self) -> None:
    self._vFfmpeg = os.environ.get(cEnvFfmpeg) or "ffmpeg"
    self._vFfprobe = os.environ.get(cEnvFfprobe) or "ffprobe"
    self._vFfmpegPath: str | None = None
    self._vFfprobePath: str | None = None
    self._dVersionCache: dict[str, str] = {}

  # ----------------------------------------------------------- availability --

  def mGetFfmpegPath(self) -> str:
    if self._vFfmpegPath is None:
      vFound = fWhich(self._vFfmpeg) if "/" not in self._vFfmpeg else self._vFfmpeg
      if not vFound or not Path(vFound).exists():
        raise FfmpegError("ffmpeg executable not found", {"looked_for": self._vFfmpeg})
      self._vFfmpegPath = vFound
    return self._vFfmpegPath

  ffmpeg_path = property(mGetFfmpegPath)

  def mGetFfprobePath(self) -> str:
    if self._vFfprobePath is None:
      vFound = fWhich(self._vFfprobe) if "/" not in self._vFfprobe else self._vFfprobe
      if not vFound or not Path(vFound).exists():
        raise FfmpegError("ffprobe executable not found", {"looked_for": self._vFfprobe})
      self._vFfprobePath = vFound
    return self._vFfprobePath

  ffprobe_path = property(mGetFfprobePath)

  def mStatus(self) -> BackendStatus:
    try:
      vFfmpeg = self.ffmpeg_path
      vFfprobe = self.ffprobe_path
    except FfmpegError as vError:
      return BackendStatus(name=self.name, available=False, reason=vError.message)
    return BackendStatus(
      name=self.name,
      available=True,
      version=self.mVersion(),
      details={"ffmpeg": vFfmpeg, "ffprobe": vFfprobe},
    )

  def mVersion(self) -> str:
    if "ffmpeg" not in self._dVersionCache:
      vResult = self._mRunFfmpeg(["-version"], pTimeout=30.0, pCheck=False)
      vFirstLine = (vResult.stdout or vResult.stderr).splitlines()
      self._dVersionCache["ffmpeg"] = vFirstLine[0].strip() if vFirstLine else "unknown"
    return self._dVersionCache["ffmpeg"]

  def mProbeVersion(self) -> str:
    if "ffprobe" not in self._dVersionCache:
      vResult = fRunProcess([self.ffprobe_path, "-version"], pTimeout=30.0)
      vFirstLine = (vResult.stdout or vResult.stderr).splitlines()
      self._dVersionCache["ffprobe"] = vFirstLine[0].strip() if vFirstLine else "unknown"
    return self._dVersionCache["ffprobe"]

  def mToolVersions(self) -> dict[str, str]:
    return {"ffmpeg": self.mVersion(), "ffprobe": self.mProbeVersion()}

  def mHasEncoder(self, pEncoder: str) -> bool:
    """Whether FFmpeg was built with a given encoder."""
    vResult = self._mRunFfmpeg(["-hide_banner", "-encoders"], pTimeout=60.0, pCheck=False)
    vPattern = re.compile(rf"^\s*\S+\s+{re.escape(pEncoder)}\s", re.MULTILINE)
    return bool(vPattern.search(vResult.stdout or ""))

  def mHasFilter(self, pFilter: str) -> bool:
    """Whether FFmpeg was built with a given filter."""
    vResult = self._mRunFfmpeg(["-hide_banner", "-filters"], pTimeout=60.0, pCheck=False)
    vPattern = re.compile(rf"^\s*\S+\s+{re.escape(pFilter)}\s", re.MULTILINE)
    return bool(vPattern.search(vResult.stdout or ""))

  # ------------------------------------------------------------- invocation --

  def _mRunFfmpeg(
    self,
    pArguments: list[str],
    pTimeout: float | None = cDefaultRenderTimeout,
    pCheck: bool = True,
    pContext: str = "ffmpeg",
  ) -> ProcessResult:
    """Run ffmpeg with stdin detached so it can never steal the MCP stream."""
    lCommand = [self.ffmpeg_path, "-nostdin", *pArguments]
    vResult = fRunProcess(lCommand, pTimeout=pTimeout)
    if pCheck and not vResult.ok:
      raise FfmpegError(
        f"{pContext} failed",
        {
          "returncode": vResult.returncode,
          "timed_out": vResult.timed_out,
          "stderr_tail": vResult.mTail(25),
        },
      )
    return vResult

  def _mRunFfprobe(self, pArguments: list[str], pTimeout: float = cDefaultProbeTimeout) -> ProcessResult:
    lCommand = [self.ffprobe_path, *pArguments]
    return fRunProcess(lCommand, pTimeout=pTimeout)

  # ------------------------------------------------------------------ probe --

  def mProbeRaw(self, pSource: Path) -> dict[str, Any]:
    """Return ffprobe's JSON description of a file."""
    lArguments = [
      "-hide_banner",
      "-loglevel", "error",
      "-print_format", "json",
      "-show_format",
      "-show_streams",
      *fInputArguments(pSource),
    ]
    vResult = self._mRunFfprobe(lArguments)
    if not vResult.ok:
      raise FfmpegError(
        "ffprobe could not read the file",
        {"path": pSource.name, "returncode": vResult.returncode, "stderr_tail": vResult.mTail(10)},
      )
    try:
      dPayload = json.loads(vResult.stdout)
    except json.JSONDecodeError as vError:
      raise FfmpegError("ffprobe returned malformed JSON", {"path": pSource.name}) from vError
    if not isinstance(dPayload, dict):
      raise FfmpegError("ffprobe returned an unexpected document", {"path": pSource.name})
    return dPayload

  # ---------------------------------------------------------------- prepare --

  def mPrepare(
    self,
    pSource: Path,
    pTarget: Path,
    pSampleRate: int,
    pChannels: int,
  ) -> PreparedAudio:
    """Decode the source into a deterministic working WAV.

    Duration is preserved, so EditPlan timestamps remain valid against the
    prepared copy. The source is opened read-only and never modified.
    """
    lArguments = [
      "-hide_banner",
      "-loglevel", "error",
      "-y",
      *fInputArguments(pSource),
      "-map", "0:a:0",
      "-vn", "-sn", "-dn",
      "-ar", str(pSampleRate),
      "-ac", str(pChannels),
      "-c:a", cWorkingCodec,
      str(pTarget),
    ]
    self._mRunFfmpeg(lArguments, pContext="preparing the working copy")
    if not pTarget.exists() or pTarget.stat().st_size == 0:
      raise FfmpegError("Preparing the working copy produced no audio", {"target": pTarget.name})
    vDuration = self.mDuration(pTarget)
    return PreparedAudio(path=pTarget, duration_seconds=vDuration)

  def mDecode(self, pSource: Path, pTarget: Path, pMaxBytes: int | None = None) -> Path:
    """Decode a file to uncompressed WAV, keeping its sample rate and channel count.

    With ``pMaxBytes`` FFmpeg stops writing once the file reaches that size,
    so the caller can tell a decode that ran into the limit by its size.
    """
    lArguments = [
      "-hide_banner", "-loglevel", "error", "-y",
      *fInputArguments(pSource),
      "-map", "0:a:0",
      "-vn", "-sn", "-dn",
      "-c:a", cWorkingCodec,
    ]
    if pMaxBytes is not None:
      lArguments.extend(["-fs", str(int(pMaxBytes))])
    lArguments.append(str(pTarget))
    self._mRunFfmpeg(lArguments, pContext="decoding audio")
    if not pTarget.exists() or pTarget.stat().st_size == 0:
      raise FfmpegError("Decoding produced no audio", {"target": pTarget.name})
    return pTarget

  def mDuration(self, pSource: Path) -> float:
    """Return the duration of a file in seconds."""
    lArguments = [
      "-hide_banner",
      "-loglevel", "error",
      "-print_format", "json",
      "-show_entries", "format=duration:stream=duration",
      *fInputArguments(pSource),
    ]
    vResult = self._mRunFfprobe(lArguments)
    if not vResult.ok:
      raise FfmpegError("ffprobe could not measure the duration", {"path": pSource.name})
    try:
      dPayload = json.loads(vResult.stdout)
    except json.JSONDecodeError as vError:
      raise FfmpegError("ffprobe returned malformed duration JSON", {"path": pSource.name}) from vError
    vDuration = dPayload.get("format", {}).get("duration")
    if vDuration is None:
      for dStream in dPayload.get("streams", []):
        if dStream.get("duration") is not None:
          vDuration = dStream["duration"]
          break
    try:
      return float(vDuration)
    except (TypeError, ValueError) as vError:
      raise FfmpegError("File has no usable duration", {"path": pSource.name}) from vError

  # ------------------------------------------------------- structural render --

  def mRenderStructural(self, pRequest: RenderRequest) -> RenderResult:
    """Render the timeline of ``pRequest`` into ``pRequest.target``."""
    vTimeline = pRequest.timeline
    if vTimeline.is_empty:
      raise FfmpegError("The edit plan removes the entire recording", {"plan_id": pRequest.plan.id})

    lItems = list(vTimeline.items)
    lApplied = [
      vOperation.id for vOperation in pRequest.plan.mStructuralOperations()
    ]

    if vTimeline.is_identity:
      # Nothing structural to do: copy the prepared audio straight through.
      self._mCopyAudio(pRequest.source, pRequest.target, pRequest.sample_rate, pRequest.channels)
      return RenderResult(path=pRequest.target, applied_operations=lApplied)

    lChunks = [lItems[vIndex : vIndex + cMaxSegmentsPerPass] for vIndex in range(0, len(lItems), cMaxSegmentsPerPass)]
    vLog.debug(
      "Rendering timeline",
      extra={"items": len(lItems), "passes": len(lChunks), "target": pRequest.target.name},
    )

    if len(lChunks) == 1:
      self._mRenderChunk(pRequest, lChunks[0], pRequest.target)
    else:
      lPartials: list[Path] = []
      for vIndex, lChunk in enumerate(lChunks):
        vPartial = pRequest.work_dir / f"structural-part-{vIndex:04d}.wav"
        self._mRenderChunk(pRequest, lChunk, vPartial)
        lPartials.append(vPartial)
      self._mConcatFiles(lPartials, pRequest.target, pRequest.work_dir, pRequest.sample_rate, pRequest.channels)
      for vPartial in lPartials:
        vPartial.unlink(missing_ok=True)

    return RenderResult(path=pRequest.target, applied_operations=lApplied)

  def _mCopyAudio(self, pSource: Path, pTarget: Path, pSampleRate: int, pChannels: int) -> None:
    lArguments = [
      "-hide_banner", "-loglevel", "error", "-y",
      *fInputArguments(pSource),
      "-ar", str(pSampleRate),
      "-ac", str(pChannels),
      "-c:a", cWorkingCodec,
      str(pTarget),
    ]
    self._mRunFfmpeg(lArguments, pContext="copying audio")

  def _mRenderChunk(self, pRequest: RenderRequest, pItems: list[TimelineItem], pTarget: Path) -> None:
    """Render one bounded group of timeline items with a single filter graph."""
    vGraph = fBuildFilterGraph(pItems, pRequest.sample_rate, pRequest.channels)
    lArguments = [
      "-hide_banner", "-loglevel", "error", "-y",
      *fInputArguments(pRequest.source),
      "-filter_complex", vGraph,
      "-map", "[out]",
      "-ar", str(pRequest.sample_rate),
      "-ac", str(pRequest.channels),
      "-c:a", cWorkingCodec,
      str(pTarget),
    ]
    self._mRunFfmpeg(lArguments, pContext="rendering the timeline")
    if not pTarget.exists() or pTarget.stat().st_size == 0:
      raise FfmpegError("Rendering produced no audio", {"target": pTarget.name})

  def _mConcatFiles(
    self,
    pParts: list[Path],
    pTarget: Path,
    pWorkDir: Path,
    pSampleRate: int,
    pChannels: int,
  ) -> None:
    """Join rendered passes with the concat demuxer."""
    vListFile = pWorkDir / "concat-list.txt"
    lLines: list[str] = []
    for vPart in pParts:
      # The concat demuxer escapes a single quote as '\''.
      vEscaped = str(vPart).replace("'", "'\\''")
      lLines.append(f"file '{vEscaped}'")
    vListFile.write_text("\n".join(lLines) + "\n", encoding="utf-8")

    lArguments = [
      "-hide_banner", "-loglevel", "error", "-y",
      "-protocol_whitelist", cInputProtocols,
      "-f", "concat",
      "-safe", "0",
      "-i", str(vListFile),
      "-ar", str(pSampleRate),
      "-ac", str(pChannels),
      "-c:a", cWorkingCodec,
      str(pTarget),
    ]
    self._mRunFfmpeg(lArguments, pContext="joining rendered passes")
    vListFile.unlink(missing_ok=True)

  # --------------------------------------------------------------- filtering --

  def mApplyFilters(
    self,
    pSource: Path,
    pTarget: Path,
    pFilters: list[str],
    pSampleRate: int,
    pChannels: int,
    pContext: str = "applying filters",
  ) -> Path:
    """Apply an ordered list of audio filters in one pass."""
    if not pFilters:
      self._mCopyAudio(pSource, pTarget, pSampleRate, pChannels)
      return pTarget
    lArguments = [
      "-hide_banner", "-loglevel", "error", "-y",
      *fInputArguments(pSource),
      "-af", ",".join(pFilters),
      "-ar", str(pSampleRate),
      "-ac", str(pChannels),
      "-c:a", cWorkingCodec,
      str(pTarget),
    ]
    self._mRunFfmpeg(lArguments, pContext=pContext)
    if not pTarget.exists() or pTarget.stat().st_size == 0:
      raise FfmpegError("Filtering produced no audio", {"target": pTarget.name})
    return pTarget

  def mBuildHighPass(self, pFrequency: float, pPoles: int = 2) -> str:
    return f"highpass=frequency={fFormatNumber(pFrequency, 3)}:poles={int(pPoles)}"

  def mBuildLowPass(self, pFrequency: float, pPoles: int = 2) -> str:
    return f"lowpass=frequency={fFormatNumber(pFrequency, 3)}:poles={int(pPoles)}"

  def mBuildGain(self, pGainDb: float) -> str:
    return f"volume={fFormatNumber(pGainDb, 4)}dB"

  def mBuildDenoise(self, pReductionDb: float, pNoiseFloorDb: float | None = None) -> str:
    vReduction = min(97.0, max(0.01, pReductionDb))
    lParts = [f"afftdn=noise_reduction={fFormatNumber(vReduction, 3)}"]
    if pNoiseFloorDb is not None:
      vFloor = min(-20.0, max(-80.0, pNoiseFloorDb))
      lParts.append(f"noise_floor={fFormatNumber(vFloor, 3)}")
    return ":".join(lParts)

  def mBuildCompressor(
    self,
    pThresholdDb: float,
    pRatio: float,
    pAttackMs: float,
    pReleaseMs: float,
    pMakeupDb: float,
    pKneeDb: float,
  ) -> str:
    """Build an ``acompressor`` expression from dB-denominated settings.

    ``acompressor`` takes threshold, makeup and knee as linear ratios, so the
    conversion happens here rather than being guessed by the caller.
    """
    vThreshold = min(1.0, max(0.000976563, fDbToLinear(pThresholdDb)))
    vMakeup = min(64.0, max(1.0, fDbToLinear(pMakeupDb)))
    vKnee = min(8.0, max(1.0, fDbToLinear(pKneeDb)))
    vAttack = min(2000.0, max(0.01, pAttackMs))
    vRelease = min(9000.0, max(0.01, pReleaseMs))
    return (
      f"acompressor=threshold={fFormatNumber(vThreshold, 8)}"
      f":ratio={fFormatNumber(pRatio, 3)}"
      f":attack={fFormatNumber(vAttack, 3)}"
      f":release={fFormatNumber(vRelease, 3)}"
      f":makeup={fFormatNumber(vMakeup, 5)}"
      f":knee={fFormatNumber(vKnee, 5)}"
    )

  def mBuildLimiter(
    self,
    pTruePeakDbtp: float,
    pAttackMs: float = 5.0,
    pReleaseMs: float = 50.0,
    pHeadroomDb: float = cTruePeakHeadroomDb,
  ) -> str:
    """Build an ``alimiter`` expression aimed at a true-peak ceiling.

    ``alimiter`` caps *sample* peaks, but the ceiling we are asked for is a
    *true* peak: the inter-sample maxima of the reconstructed waveform sit
    above the sample peaks, typically by a few tenths of a dB. The limiter is
    therefore aimed a little lower than the target, and the engine verifies the
    measured true peak afterwards.
    """
    vLimit = min(1.0, max(0.0625, fDbToLinear(pTruePeakDbtp - max(0.0, pHeadroomDb))))
    vAttack = min(80.0, max(0.1, pAttackMs))
    vRelease = min(8000.0, max(1.0, pReleaseMs))
    return (
      f"alimiter=limit={fFormatNumber(vLimit, 8)}"
      f":attack={fFormatNumber(vAttack, 3)}"
      f":release={fFormatNumber(vRelease, 3)}"
      f":level=false"
    )

  def mBuildFade(self, pDirection: str, pStart: float, pDuration: float, pCurve: str = "tri") -> str:
    vType = "in" if pDirection == "in" else "out"
    return (
      f"afade=type={vType}:start_time={fFormatNumber(max(0.0, pStart))}"
      f":duration={fFormatNumber(max(0.001, pDuration))}:curve={pCurve}"
    )

  # -------------------------------------------------------------- loudness --

  def mMeasureLoudnorm(
    self,
    pSource: Path,
    pIntegratedLufs: float,
    pTruePeakDbtp: float,
    pLoudnessRange: float,
  ) -> dict[str, float]:
    """First ``loudnorm`` pass: measure the file against a target."""
    vFilter = (
      f"loudnorm=I={fFormatNumber(pIntegratedLufs, 3)}"
      f":TP={fFormatNumber(pTruePeakDbtp, 3)}"
      f":LRA={fFormatNumber(pLoudnessRange, 3)}"
      f":print_format=json"
    )
    lArguments = [
      "-hide_banner", "-nostats",
      *fInputArguments(pSource),
      "-af", vFilter,
      "-f", "null", "-",
    ]
    vResult = self._mRunFfmpeg(lArguments, pTimeout=cDefaultMeasureTimeout, pCheck=True, pContext="measuring loudness")
    dMeasured = fParseLoudnormJson(vResult.stderr)
    if dMeasured is None:
      raise FfmpegError("loudnorm did not report a measurement", {"path": pSource.name})
    return dMeasured

  def mBuildLoudnorm(
    self,
    pIntegratedLufs: float,
    pTruePeakDbtp: float,
    pLoudnessRange: float,
    pMeasured: dict[str, float] | None = None,
  ) -> str:
    """Build the ``loudnorm`` expression, optionally in second-pass form."""
    lParts = [
      f"loudnorm=I={fFormatNumber(pIntegratedLufs, 3)}",
      f"TP={fFormatNumber(pTruePeakDbtp, 3)}",
      f"LRA={fFormatNumber(pLoudnessRange, 3)}",
    ]
    if pMeasured:
      lParts.extend(
        [
          f"measured_I={fFormatNumber(pMeasured['input_i'], 4)}",
          f"measured_TP={fFormatNumber(pMeasured['input_tp'], 4)}",
          f"measured_LRA={fFormatNumber(pMeasured['input_lra'], 4)}",
          f"measured_thresh={fFormatNumber(pMeasured['input_thresh'], 4)}",
          f"offset={fFormatNumber(pMeasured['target_offset'], 4)}",
          "linear=true",
        ]
      )
    lParts.append("print_format=summary")
    return ":".join(lParts)

  def mMeasureEbur128(self, pSource: Path) -> LoudnessMeasurement:
    """Run the ``ebur128`` filter and parse its summary."""
    lArguments = [
      "-hide_banner", "-nostats",
      *fInputArguments(pSource),
      "-filter_complex", "ebur128=peak=true:framelog=quiet",
      "-f", "null", "-",
    ]
    vResult = self._mRunFfmpeg(lArguments, pTimeout=cDefaultMeasureTimeout, pCheck=True, pContext="measuring EBU R128")
    return fParseEbur128(vResult.stderr)

  def mMeasureAstats(self, pSource: Path) -> DynamicsStatistics:
    """Run the ``astats`` filter and parse the overall section."""
    lArguments = [
      "-hide_banner", "-nostats",
      *fInputArguments(pSource),
      "-af", "astats=measure_perchannel=none:reset=0",
      "-f", "null", "-",
    ]
    vResult = self._mRunFfmpeg(lArguments, pTimeout=cDefaultMeasureTimeout, pCheck=True, pContext="measuring statistics")
    return fParseAstats(vResult.stderr)

  def mDetectSilence(self, pSource: Path, pThresholdDb: float, pMinimumDuration: float) -> list[SilenceRegion]:
    """Run ``silencedetect`` and return the regions it reported."""
    vFilter = (
      f"silencedetect=noise={fFormatNumber(pThresholdDb, 3)}dB"
      f":duration={fFormatNumber(max(0.001, pMinimumDuration), 4)}"
    )
    lArguments = [
      "-hide_banner", "-nostats",
      *fInputArguments(pSource),
      "-af", vFilter,
      "-f", "null", "-",
    ]
    vResult = self._mRunFfmpeg(lArguments, pTimeout=cDefaultMeasureTimeout, pCheck=True, pContext="detecting silence")
    vDuration = self.mDuration(pSource)
    return fParseSilenceDetect(vResult.stderr, vDuration)

  # ----------------------------------------------------------------- export --

  def mExport(
    self,
    pSource: Path,
    pTarget: Path,
    pFormat: str,
    pSampleRate: int,
    pChannels: int,
    pWavCodec: str = "pcm_s24le",
    pMp3BitrateKbps: int = 128,
    pMp3Mode: str = "cbr",
    pMp3VbrQuality: int = 2,
    pFlacCompression: int = 5,
    pMetadata: dict[str, str] | None = None,
    pOverwrite: bool = True,
  ) -> Path:
    """Encode the mastered audio into a delivery format.

    The file is encoded beside the target and moved into place when complete,
    so a failed encode leaves nothing behind and, without ``pOverwrite``, a
    file that appeared in the meantime is never replaced. FFmpeg's own ``-n``
    cannot do this: it exits successfully when it finds the target there.
    """
    lArguments = [
      "-hide_banner", "-loglevel", "error", "-y",
      *fInputArguments(pSource),
      "-map", "0:a:0",
      "-ar", str(pSampleRate),
      "-ac", str(pChannels),
    ]
    if pMetadata:
      for vKey, vValue in pMetadata.items():
        lArguments.extend(["-metadata", f"{vKey}={vValue}"])

    if pFormat == "wav":
      lArguments.extend(["-c:a", pWavCodec])
    elif pFormat == "flac":
      lArguments.extend(["-c:a", "flac", "-compression_level", str(pFlacCompression)])
    elif pFormat == "mp3":
      lArguments.extend(["-c:a", "libmp3lame"])
      if pMp3Mode == "vbr":
        lArguments.extend(["-q:a", str(pMp3VbrQuality)])
      else:
        lArguments.extend(["-b:a", f"{pMp3BitrateKbps}k"])
    else:
      raise ExportError(f"Unsupported export format: {pFormat}", {"format": pFormat})

    with fAtomicPath(pTarget, pSuffix=f".partial{pTarget.suffix}", pReplace=pOverwrite) as vPartial:
      lArguments.append(str(vPartial))
      self._mRunFfmpeg(lArguments, pContext=f"exporting {pFormat}")
      if not vPartial.exists() or vPartial.stat().st_size == 0:
        raise ExportError("Export produced no file", {"target": pTarget.name, "format": pFormat})
      # The temporary file is private; a delivery gets the usual permissions.
      vPartial.chmod(fDefaultFileMode())
    return pTarget
# --------------------------------------------------------------------------- #
# Output parsers
# --------------------------------------------------------------------------- #

cLoudnormKeys: tuple[str, ...] = (
  "input_i", "input_tp", "input_lra", "input_thresh",
  "target_offset",
)


def fParseLoudnormJson(pStderr: str) -> dict[str, float] | None:
  """Extract the JSON block ``loudnorm`` prints on stderr."""
  vStart = pStderr.rfind("{")
  vEnd = pStderr.rfind("}")
  if vStart == -1 or vEnd == -1 or vEnd <= vStart:
    return None
  try:
    dRaw = json.loads(pStderr[vStart : vEnd + 1])
  except json.JSONDecodeError:
    return None
  if not isinstance(dRaw, dict):
    return None
  dParsed: dict[str, float] = {}
  for vKey in cLoudnormKeys:
    vValue = dRaw.get(vKey)
    if vValue is None:
      continue
    try:
      vNumber = float(vValue)
    except (TypeError, ValueError):
      continue
    if math.isnan(vNumber) or math.isinf(vNumber):
      # loudnorm reports -inf for digital silence; clamp so the second pass
      # still receives a usable number.
      vNumber = -70.0 if vKey.endswith(("_i", "_thresh")) else -99.0
    dParsed[vKey] = vNumber
  for vRequired in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset"):
    if vRequired not in dParsed:
      return None
  return dParsed


cEbur128Patterns: dict[str, re.Pattern[str]] = {
  "integrated_lufs": re.compile(r"^\s*I:\s*(-?[\d.]+|-inf)\s*LUFS", re.MULTILINE),
  "threshold_lufs": re.compile(r"Integrated loudness:.*?Threshold:\s*(-?[\d.]+|-inf)\s*LUFS", re.DOTALL),
  "loudness_range": re.compile(r"^\s*LRA:\s*(-?[\d.]+|-inf)\s*LU", re.MULTILINE),
  "lra_low_lufs": re.compile(r"^\s*LRA low:\s*(-?[\d.]+|-inf)\s*LUFS", re.MULTILINE),
  "lra_high_lufs": re.compile(r"^\s*LRA high:\s*(-?[\d.]+|-inf)\s*LUFS", re.MULTILINE),
  "true_peak_dbfs": re.compile(r"True peak:.*?Peak:\s*(-?[\d.]+|-inf)\s*dBFS", re.DOTALL),
}


def _fFloatOrNone(pText: str | None) -> float | None:
  if pText is None:
    return None
  vText = pText.strip()
  if vText in {"-inf", "inf", "nan", "-nan"}:
    return None
  try:
    vValue = float(vText)
  except ValueError:
    return None
  if math.isnan(vValue) or math.isinf(vValue):
    return None
  return vValue


def fParseEbur128(pStderr: str) -> LoudnessMeasurement:
  """Parse the summary block that ``ebur128`` prints at the end of a run."""
  vSummaryIndex = pStderr.rfind("Summary:")
  vText = pStderr[vSummaryIndex:] if vSummaryIndex != -1 else pStderr
  dValues: dict[str, float | None] = {}
  for vKey, vPattern in cEbur128Patterns.items():
    vMatch = vPattern.search(vText)
    dValues[vKey] = _fFloatOrNone(vMatch.group(1)) if vMatch else None
  return LoudnessMeasurement(
    integrated_lufs=dValues.get("integrated_lufs"),
    loudness_range=dValues.get("loudness_range"),
    threshold_lufs=dValues.get("threshold_lufs"),
    lra_low_lufs=dValues.get("lra_low_lufs"),
    lra_high_lufs=dValues.get("lra_high_lufs"),
    true_peak_dbfs=dValues.get("true_peak_dbfs"),
    source="ebur128",
  )


cAstatsFieldMap: dict[str, str] = {
  "DC offset": "dc_offset",
  "Peak level dB": "peak_dbfs",
  "RMS level dB": "rms_dbfs",
  "RMS peak dB": "rms_peak_dbfs",
  "RMS trough dB": "rms_trough_dbfs",
  "Flat factor": "flat_factor",
  "Peak count": "peak_count",
  "Noise floor dB": "noise_floor_dbfs",
  "Noise floor count": "noise_floor_count",
  "Entropy": "entropy",
  "Dynamic range": "dynamic_range_db",
  "Crest factor": "crest_factor",
  "Zero crossings rate": "zero_crossings_rate",
  "Number of samples": "number_of_samples",
}

cAstatsIntegerFields: frozenset[str] = frozenset({"peak_count", "noise_floor_count", "number_of_samples"})
cAstatsLine = re.compile(r"^\s*\[Parsed_astats[^\]]*\]\s*(?P<key>[^:]+):\s*(?P<value>.+?)\s*$", re.MULTILINE)


def fParseAstats(pStderr: str) -> DynamicsStatistics:
  """Parse the ``Overall`` section of an ``astats`` run."""
  vOverallIndex = pStderr.rfind("Overall")
  vText = pStderr[vOverallIndex:] if vOverallIndex != -1 else pStderr
  dFields: dict[str, Any] = {}
  for vMatch in cAstatsLine.finditer(vText):
    vKey = vMatch.group("key").strip()
    vRaw = vMatch.group("value").strip()
    vField = cAstatsFieldMap.get(vKey)
    if vField is None:
      if vKey == "Bit depth":
        # Reported as "used/total"; the used depth is the informative half.
        vFirst = vRaw.split("/")[0].strip()
        vValue = _fFloatOrNone(vFirst)
        if vValue is not None:
          dFields["bit_depth"] = vValue
      continue
    if vField in cAstatsIntegerFields:
      try:
        dFields[vField] = int(float(vRaw))
      except ValueError:
        continue
    else:
      vValue = _fFloatOrNone(vRaw)
      if vValue is not None:
        dFields[vField] = vValue
  return DynamicsStatistics(**dFields)


cSilenceStart = re.compile(r"silence_start:\s*(-?[\d.]+)")
cSilenceEnd = re.compile(r"silence_end:\s*(-?[\d.]+)")


def fParseSilenceDetect(pStderr: str, pDuration: float) -> list[SilenceRegion]:
  """Turn ``silencedetect`` log lines into regions.

  A silence that runs to the end of the file has a ``silence_start`` with no
  matching ``silence_end``; it is closed at the file duration.
  """
  lRegions: list[SilenceRegion] = []
  vPendingStart: float | None = None
  for vLine in pStderr.splitlines():
    if "silence_start" in vLine:
      vMatch = cSilenceStart.search(vLine)
      if vMatch:
        vPendingStart = max(0.0, float(vMatch.group(1)))
    elif "silence_end" in vLine:
      vMatch = cSilenceEnd.search(vLine)
      if vMatch and vPendingStart is not None:
        vEnd = min(pDuration, float(vMatch.group(1)))
        if vEnd > vPendingStart:
          lRegions.append(SilenceRegion(start=vPendingStart, end=vEnd, duration=vEnd - vPendingStart))
        vPendingStart = None
  if vPendingStart is not None and pDuration > vPendingStart:
    lRegions.append(
      SilenceRegion(start=vPendingStart, end=pDuration, duration=pDuration - vPendingStart)
    )
  return lRegions


def fBuildFilterGraph(pItems: list[TimelineItem], pSampleRate: int, pChannels: int) -> str:
  """Compile timeline items into one deterministic ``filter_complex`` string.

  This is the single source of truth for how a timeline becomes FFmpeg
  instructions: the backend runs exactly what this function returns, and the
  tests assert against it without spawning a process.
  """
  if not pItems:
    raise FfmpegError("Cannot build a filter graph from an empty timeline")

  lFilters: list[str] = []
  lLabels: list[str] = []
  vLayout = fChannelLayout(pChannels)
  vFormat = f"aformat=sample_fmts=fltp:sample_rates={pSampleRate}:channel_layouts={vLayout}"
  vSingle = len(pItems) == 1

  for vIndex, vItem in enumerate(pItems):
    vLabel = "out" if vSingle else f"s{vIndex}"
    if isinstance(vItem, KeepSegment):
      lParts = [
        f"atrim=start={fFormatNumber(vItem.source_start)}:end={fFormatNumber(vItem.source_end)}",
        "asetpts=PTS-STARTPTS",
      ]
      if vItem.fade_in > 0.0:
        lParts.append(f"afade=type=in:start_time=0:duration={fFormatNumber(vItem.fade_in)}:curve=tri")
      if vItem.fade_out > 0.0:
        vFadeStart = max(0.0, vItem.duration - vItem.fade_out)
        lParts.append(
          f"afade=type=out:start_time={fFormatNumber(vFadeStart)}:duration={fFormatNumber(vItem.fade_out)}:curve=tri"
        )
      lParts.append(vFormat)
      lFilters.append(f"[0:a]{','.join(lParts)}[{vLabel}]")
    elif isinstance(vItem, SilenceSegment):
      lFilters.append(
        f"aevalsrc=0:duration={fFormatNumber(vItem.duration)}:sample_rate={pSampleRate}"
        f":channel_layout={vLayout},{vFormat}[{vLabel}]"
      )
    else:  # pragma: no cover - the timeline only produces the two kinds above
      raise FfmpegError(f"Unknown timeline item: {type(vItem).__name__}")
    lLabels.append(f"[{vLabel}]")

  if vSingle:
    return ";".join(lFilters)
  lFilters.append(f"{''.join(lLabels)}concat=n={len(lLabels)}:v=0:a=1[out]")
  return ";".join(lFilters)
