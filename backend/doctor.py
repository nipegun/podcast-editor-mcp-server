"""Environment diagnostics.

``podcast-editor-mcp-server-doctor`` (or ``podcast-editor-mcp-server-cli doctor``) answers one question:
*will this machine actually process a podcast?* Every check reports what it
found, not what it assumes, and nothing is reported as OK unless it was really
exercised.

Exit codes: ``0`` when nothing failed, ``1`` when at least one check errored.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from podcast_editor_mcp_server.backends.audacity import (
  AudacityBackend,
  AudacityPipeClient,
  cAllowedCommands,
  cModuleEnabled,
  cModuleLibraryPath,
  cModuleStatusNames,
  fAudacityConfigPath,
  fAudacityIsRunning,
  fReadModuleStatus,
)
from podcast_editor_mcp_server.backends.ffmpeg import FfmpegBackend
from podcast_editor_mcp_server.config import (
  AppContext,
  fInstallPrefix,
  fListProfiles,
  fLoadProfile,
  fLoadRuntimeConfig,
  fModelDirectory,
  fSystemConfigDir,
  fUserConfigDir,
)
from podcast_editor_mcp_server.errors import PodcastEditorMcpServerError
from podcast_editor_mcp_server.logging_setup import fSetupLogging
from podcast_editor_mcp_server.transcription.faster_whisper_backend import (
  FasterWhisperBackend,
  fCudaDeviceCount,
  fMissingCudaLibraries,
)
from podcast_editor_mcp_server.util.proc import fWhich
from podcast_editor_mcp_server.vad.silero import SileroVadBackend
from podcast_editor_mcp_server.version import cProductName, cVersion

cRequiredEncoders: tuple[str, ...] = ("pcm_s16le", "pcm_s24le", "libmp3lame", "flac")
cRequiredFilters: tuple[str, ...] = (
  "loudnorm", "alimiter", "silencedetect", "ebur128", "astats",
  "afade", "atrim", "aresample", "highpass", "lowpass", "volume",
  "aevalsrc", "concat", "afftdn", "acompressor", "aformat",
)
cMinimumFreeBytes: int = 2 * 1024 * 1024 * 1024


class CheckStatus(StrEnum):
  OK = "ok"
  WARN = "warn"
  ERROR = "error"
  SKIP = "skip"


cStatusLabel: dict[CheckStatus, str] = {
  CheckStatus.OK: "OK",
  CheckStatus.WARN: "WARN",
  CheckStatus.ERROR: "ERROR",
  CheckStatus.SKIP: "SKIP",
}


@dataclass(slots=True)
class CheckResult:
  """The outcome of one diagnostic check."""

  name: str
  status: CheckStatus
  message: str = ""
  group: str = "core"
  details: dict[str, Any] = field(default_factory=dict)

  def mAsDict(self) -> dict[str, Any]:
    dResult: dict[str, Any] = {
      "name": self.name,
      "group": self.group,
      "status": str(self.status),
      "message": self.message,
    }
    if self.details:
      dResult["details"] = self.details
    return dResult

  def mLine(self) -> str:
    return f"[{cStatusLabel[self.status]:>5}] {self.name}" + (f" - {self.message}" if self.message else "")


@dataclass(slots=True)
class DoctorReport:
  """Every check, plus the overall verdict."""

  version: str
  checks: list[CheckResult] = field(default_factory=list)

  def mGetFailed(self) -> bool:
    return any(vCheck.status == CheckStatus.ERROR for vCheck in self.checks)

  failed = property(mGetFailed)

  def mCounts(self) -> dict[str, int]:
    dCounts = {str(vStatus): 0 for vStatus in CheckStatus}
    for vCheck in self.checks:
      dCounts[str(vCheck.status)] += 1
    return dCounts

  def mAsDict(self) -> dict[str, Any]:
    return {
      "product": cProductName,
      "version": self.version,
      "ok": not self.failed,
      "counts": self.mCounts(),
      "checks": [vCheck.mAsDict() for vCheck in self.checks],
    }

  def mRender(self) -> str:
    lLines: list[str] = [f"{cProductName} {self.version} - diagnostics", ""]
    vCurrentGroup = ""
    for vCheck in self.checks:
      if vCheck.group != vCurrentGroup:
        vCurrentGroup = vCheck.group
        lLines.append(f"{vCurrentGroup}:")
      lLines.append("  " + vCheck.mLine())
    dCounts = self.mCounts()
    lLines.extend(
      [
        "",
        f"{dCounts['ok']} ok, {dCounts['warn']} warning(s), {dCounts['error']} error(s), {dCounts['skip']} skipped",
      ]
    )
    return "\n".join(lLines)


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def fCheckOperatingSystem() -> list[CheckResult]:
  lResults: list[CheckResult] = []
  vRelease = Path("/etc/os-release")
  dValues: dict[str, str] = {}
  if vRelease.is_file():
    for vLine in vRelease.read_text(encoding="utf-8", errors="replace").splitlines():
      if "=" in vLine:
        vKey, vValue = vLine.split("=", 1)
        dValues[vKey.strip()] = vValue.strip().strip('"')
  vName = dValues.get("PRETTY_NAME", platform.platform())
  vId = dValues.get("ID", "")
  vVersionId = dValues.get("VERSION_ID", "")

  if vId == "debian" and vVersionId == "13":
    lResults.append(CheckResult("Operating system", CheckStatus.OK, vName))
  elif vId == "debian":
    lResults.append(
      CheckResult(
        "Operating system",
        CheckStatus.WARN,
        f"{vName} (this product targets Debian 13)",
      )
    )
  else:
    lResults.append(
      CheckResult(
        "Operating system",
        CheckStatus.WARN,
        f"{vName} (untested; Debian 13 is the supported platform)",
      )
    )
  lResults.append(
    CheckResult("Architecture", CheckStatus.OK, platform.machine(), details={"machine": platform.machine()})
  )
  return lResults


def fCheckPython() -> list[CheckResult]:
  lResults: list[CheckResult] = []
  vVersion = sys.version.split()[0]
  lResults.append(CheckResult("Python", CheckStatus.OK, vVersion))

  vInVirtualenv = sys.prefix != sys.base_prefix
  lResults.append(
    CheckResult(
      "Python environment",
      CheckStatus.OK,
      f"{'virtualenv' if vInVirtualenv else 'system interpreter'} at {sys.prefix}",
      details={"executable": sys.executable, "virtualenv": vInVirtualenv},
    )
  )

  for vModule in ("mcp", "pydantic", "yaml", "numpy", "soundfile"):
    try:
      __import__(vModule)
    except ImportError as vError:
      lResults.append(CheckResult(f"Python module '{vModule}'", CheckStatus.ERROR, str(vError)))
    else:
      lResults.append(CheckResult(f"Python module '{vModule}'", CheckStatus.OK))
  return lResults


def fCheckFfmpeg() -> list[CheckResult]:
  lResults: list[CheckResult] = []
  vBackend = FfmpegBackend()
  vStatus = vBackend.mStatus()
  if not vStatus.available:
    lResults.append(
      CheckResult("FFmpeg", CheckStatus.ERROR, vStatus.reason or "not found", details={"hint": "sudo apt-get install ffmpeg"})
    )
    return lResults

  lResults.append(CheckResult("FFmpeg", CheckStatus.OK, vStatus.version or "", details=vStatus.details))
  lResults.append(CheckResult("ffprobe", CheckStatus.OK, vBackend.mProbeVersion()))

  lMissingEncoders = [vEncoder for vEncoder in cRequiredEncoders if not vBackend.mHasEncoder(vEncoder)]
  if lMissingEncoders:
    lResults.append(
      CheckResult(
        "FFmpeg encoders",
        CheckStatus.ERROR,
        "missing: " + ", ".join(lMissingEncoders),
        details={"missing": lMissingEncoders},
      )
    )
  else:
    lResults.append(CheckResult("FFmpeg encoders", CheckStatus.OK, ", ".join(cRequiredEncoders)))

  lMissingFilters = [vFilter for vFilter in cRequiredFilters if not vBackend.mHasFilter(vFilter)]
  if lMissingFilters:
    lResults.append(
      CheckResult(
        "FFmpeg filters",
        CheckStatus.ERROR,
        "missing: " + ", ".join(lMissingFilters),
        details={"missing": lMissingFilters},
      )
    )
  else:
    lResults.append(CheckResult("FFmpeg filters", CheckStatus.OK, f"{len(cRequiredFilters)} required filters present"))
  return lResults


def fCheckTranscription(pContext: AppContext, pCheckModel: bool = True) -> list[CheckResult]:
  lResults: list[CheckResult] = []
  vSettings = pContext.profile.transcription

  if vSettings.backend == "none":
    lResults.append(
      CheckResult("Transcription backend", CheckStatus.SKIP, "disabled in the active profile", group="transcription")
    )
    return lResults

  vBackend = FasterWhisperBackend()

  vStatus = vBackend.mStatus()
  if not vStatus.available:
    lResults.append(
      CheckResult(
        "Transcription backend",
        CheckStatus.ERROR,
        vStatus.reason or "unavailable",
        group="transcription",
        details=vStatus.details,
      )
    )
    return lResults

  lResults.append(
    CheckResult(
      "Transcription backend",
      CheckStatus.OK,
      f"{vBackend.name} {vStatus.version or ''}".strip(),
      group="transcription",
    )
  )

  vCuda = vBackend.mCudaAvailable()
  vDevice, vComputeType = vBackend.mResolveDevice(vSettings)
  # A card the driver exposes but CTranslate2 cannot use deserves its own
  # message: the fix is the installer, not the configuration.
  lMissingCuda: list[str] = []
  if not vCuda and fCudaDeviceCount() > 0:
    lMissingCuda = fMissingCudaLibraries()
  if vCuda:
    lResults.append(
      CheckResult("CUDA", CheckStatus.OK, "available through CTranslate2", group="transcription")
    )
  elif lMissingCuda:
    lResults.append(
      CheckResult(
        "CUDA",
        CheckStatus.WARN,
        f"an NVIDIA GPU is present but {', '.join(lMissingCuda)} cannot be loaded; using the CPU. "
        "Re-run podcast-editor-mcp-server/deploy/install.sh, which installs them into the virtualenv",
        group="transcription",
        details={"missing": lMissingCuda},
      )
    )
  elif vSettings.device == "cuda":
    lResults.append(
      CheckResult("CUDA", CheckStatus.WARN, "requested but unavailable; the CPU will be used", group="transcription")
    )
  else:
    lResults.append(
      CheckResult("CUDA", CheckStatus.WARN, "unavailable; using the CPU", group="transcription")
    )
  lResults.append(
    CheckResult(
      "Transcription device",
      CheckStatus.OK,
      f"{vDevice} / {vComputeType}",
      group="transcription",
      details={"model_directory": str(vBackend.mModelDirectory(vSettings))},
    )
  )

  if pCheckModel:
    if vBackend.mIsModelCached(vSettings):
      lResults.append(
        CheckResult(
          f"Whisper model '{vSettings.model}'",
          CheckStatus.OK,
          "cached locally",
          group="transcription",
        )
      )
    else:
      lResults.append(
        CheckResult(
          f"Whisper model '{vSettings.model}'",
          CheckStatus.WARN,
          "not downloaded yet; the first transcription will fetch it",
          group="transcription",
          details={"directory": str(vBackend.mModelDirectory(vSettings))},
        )
      )
  else:
    lResults.append(
      CheckResult(f"Whisper model '{vSettings.model}'", CheckStatus.SKIP, "not checked", group="transcription")
    )

  vSilero = SileroVadBackend()
  vVadStatus = vSilero.mStatus()
  if vVadStatus.available:
    lResults.append(CheckResult("Silero VAD", CheckStatus.OK, vVadStatus.version or "", group="transcription"))
  else:
    lResults.append(
      CheckResult(
        "Silero VAD",
        CheckStatus.WARN,
        f"{vVadStatus.reason}; the energy VAD will be used",
        group="transcription",
      )
    )
  return lResults


def fCheckAudacity(pContext: AppContext) -> list[CheckResult]:
  lResults: list[CheckResult] = []
  vSettings = pContext.config.audacity

  if not vSettings.enabled:
    lResults.append(
      CheckResult("Audacity", CheckStatus.SKIP, "integration disabled in the configuration", group="audacity")
    )
    return lResults

  vExecutable = fWhich(vSettings.executable) if "/" not in vSettings.executable else vSettings.executable
  if not vExecutable:
    lResults.append(
      CheckResult(
        "Audacity",
        CheckStatus.WARN,
        "not installed; FFmpeg handles everything except .aup3 projects",
        group="audacity",
        details={"hint": "sudo apt-get install audacity"},
      )
    )
    return lResults

  vBackend = AudacityBackend(vSettings)
  vVersion = vBackend.mVersion()
  lResults.append(
    CheckResult("Audacity", CheckStatus.OK, vVersion or vExecutable, group="audacity", details={"path": vExecutable})
  )

  if cModuleLibraryPath.is_file():
    lResults.append(
      CheckResult("mod-script-pipe library", CheckStatus.OK, str(cModuleLibraryPath), group="audacity")
    )
  else:
    lResults.append(
      CheckResult(
        "mod-script-pipe library",
        CheckStatus.ERROR,
        f"not found at {cModuleLibraryPath}",
        group="audacity",
      )
    )

  vModuleStatus = fReadModuleStatus()
  vConfigPath = fAudacityConfigPath()
  if vModuleStatus is None:
    lResults.append(
      CheckResult(
        "mod-script-pipe setting",
        CheckStatus.WARN,
        f"not present in {vConfigPath}; run 'podcast-editor-mcp-server-cli audacity-status --enable-module'",
        group="audacity",
      )
    )
  elif vModuleStatus == cModuleEnabled:
    lResults.append(CheckResult("mod-script-pipe setting", CheckStatus.OK, "enabled", group="audacity"))
  else:
    lResults.append(
      CheckResult(
        "mod-script-pipe setting",
        CheckStatus.WARN,
        f"status is '{cModuleStatusNames.get(vModuleStatus, vModuleStatus)}'; "
        "run 'podcast-editor-mcp-server-cli audacity-status --enable-module' while Audacity is closed",
        group="audacity",
      )
    )

  vXvfb = fWhich("Xvfb")
  vDisplay = vSettings.display or os.environ.get("DISPLAY")
  if vDisplay:
    lResults.append(CheckResult("X display", CheckStatus.OK, vDisplay, group="audacity"))
  elif vXvfb:
    lResults.append(
      CheckResult("X display", CheckStatus.OK, "none, but Xvfb is available for a virtual one", group="audacity")
    )
  else:
    lResults.append(
      CheckResult(
        "X display",
        CheckStatus.WARN,
        "no DISPLAY and no Xvfb; Audacity cannot be started (sudo apt-get install xvfb)",
        group="audacity",
      )
    )

  vRunning = fAudacityIsRunning()
  vClient = AudacityPipeClient(Path(vSettings.pipe_directory))
  if vClient.mPipesExist():
    vStatus = vBackend.mStatus()
    if vStatus.available:
      lResults.append(
        CheckResult("Audacity scripting", CheckStatus.OK, "connected and answering", group="audacity")
      )
      lResults.extend(fCheckAudacityCommands(vClient))
    else:
      lResults.append(
        CheckResult("Audacity scripting", CheckStatus.WARN, vStatus.reason or "not responding", group="audacity")
      )
  else:
    lResults.append(
      CheckResult(
        "Audacity scripting",
        CheckStatus.WARN,
        "script pipes absent" + (" (Audacity is running without the module)" if vRunning else " (Audacity is not running)"),
        group="audacity",
        details={"expected": [str(vClient.to_pipe), str(vClient.from_pipe)]},
      )
    )
  return lResults


def fCheckAudacityCommands(pClient: AudacityPipeClient) -> list[CheckResult]:
  """Verify the command whitelist against the Audacity that is actually running.

  The whitelist was built against Audacity 3.7.3. A different build may not
  have every command, and finding that out here is far better than finding it
  out halfway through writing a project.
  """
  try:
    pClient.mConnect(5.0)
    try:
      ldCommands = pClient.mListCommands(30.0)
      vMenus = pClient.mRun("GetInfo", {"Type": "Menus", "Format": "JSON"}, 30.0).body
    finally:
      pClient.mClose()
  except Exception as vError:  # noqa: BLE001 - a diagnostic must never raise
    return [
      CheckResult(
        "Audacity command whitelist",
        CheckStatus.WARN,
        f"could not be verified: {vError}",
        group="audacity",
      )
    ]

  sAvailable = {str(dItem.get("id", "")) for dItem in ldCommands}
  lMissing = [
    vName
    for vName in sorted(cAllowedCommands)
    if vName not in sAvailable and f'"id":"{vName}"' not in vMenus.replace(" ", "")
  ]
  if lMissing:
    return [
      CheckResult(
        "Audacity command whitelist",
        CheckStatus.WARN,
        "this build does not provide: " + ", ".join(lMissing),
        group="audacity",
        details={"missing": lMissing},
      )
    ]
  return [
    CheckResult(
      "Audacity command whitelist",
      CheckStatus.OK,
      f"all {len(cAllowedCommands)} commands exist in this build",
      group="audacity",
    )
  ]


def fCheckInstallation() -> list[CheckResult]:
  """Verify the self-contained tree under the installation prefix.

  Everything the product owns lives under one prefix, so a broken permission
  here is the single most likely reason for a job to fail before it starts.
  The model directory is shared between users and gets its write access from
  the ``podcast-editor-mcp-server`` group, which a login started before the installer ran
  will not have yet.
  """
  lResults: list[CheckResult] = []
  vPrefix = fInstallPrefix()

  if vPrefix.is_dir():
    lResults.append(CheckResult("Installation prefix", CheckStatus.OK, str(vPrefix), group="installation"))
  else:
    lResults.append(
      CheckResult(
        "Installation prefix",
        CheckStatus.ERROR,
        f"missing: {vPrefix}; re-run podcast-editor-mcp-server/deploy/install.sh",
        group="installation",
      )
    )
    return lResults

  vSystemConfig = fSystemConfigDir() / "config.yaml"
  if vSystemConfig.is_file():
    lResults.append(CheckResult("System configuration", CheckStatus.OK, str(vSystemConfig), group="installation"))
  else:
    lResults.append(
      CheckResult(
        "System configuration",
        CheckStatus.WARN,
        f"not present at {vSystemConfig}; built-in defaults are in use",
        group="installation",
      )
    )

  vUserConfig = fUserConfigDir()
  if vUserConfig.is_dir() and os.access(vUserConfig, os.W_OK):
    lResults.append(CheckResult("User configuration", CheckStatus.OK, str(vUserConfig), group="installation"))
  elif vUserConfig.is_dir():
    lResults.append(
      CheckResult(
        "User configuration",
        CheckStatus.WARN,
        f"not writable: {vUserConfig}; run podcast-editor-mcp-server/deploy/install.sh --user {vUserConfig.name}",
        group="installation",
      )
    )
  else:
    lResults.append(
      CheckResult(
        "User configuration",
        CheckStatus.WARN,
        f"not set up yet: {vUserConfig}",
        group="installation",
      )
    )

  vModels = fModelDirectory()
  if not vModels.is_dir():
    lResults.append(
      CheckResult("Model directory", CheckStatus.WARN, f"missing: {vModels}", group="installation")
    )
  elif os.access(vModels, os.W_OK):
    lResults.append(CheckResult("Model directory", CheckStatus.OK, str(vModels), group="installation"))
  else:
    lResults.append(
      CheckResult(
        "Model directory",
        CheckStatus.WARN,
        f"read-only for this user: {vModels}; cached models still work, new downloads do not. "
        "Log in again to pick up the 'podcast-editor-mcp-server' group",
        group="installation",
      )
    )
  return lResults


def fCheckWorkspace(pContext: AppContext) -> list[CheckResult]:
  lResults: list[CheckResult] = []
  dDescription = pContext.workspace.mDescribe()

  for vRole in ("input", "work", "output", "jobs", "cache", "logs"):
    vPath = Path(dDescription[vRole])
    if not vPath.is_dir():
      lResults.append(
        CheckResult(f"Directory '{vRole}'", CheckStatus.ERROR, f"missing: {vPath}", group="workspace")
      )
      continue
    vWritable = os.access(vPath, os.W_OK) if vRole != "input" else os.access(vPath, os.R_OK)
    if vWritable:
      lResults.append(CheckResult(f"Directory '{vRole}'", CheckStatus.OK, str(vPath), group="workspace"))
    else:
      lResults.append(
        CheckResult(
          f"Directory '{vRole}'",
          CheckStatus.ERROR,
          f"not {'readable' if vRole == 'input' else 'writable'}: {vPath}",
          group="workspace",
        )
      )

  try:
    vUsage = shutil.disk_usage(pContext.workspace.root)
    vFreeGb = vUsage.free / (1024**3)
    if vUsage.free < cMinimumFreeBytes:
      lResults.append(
        CheckResult("Free space", CheckStatus.WARN, f"{vFreeGb:.1f} GiB available", group="workspace")
      )
    else:
      lResults.append(CheckResult("Free space", CheckStatus.OK, f"{vFreeGb:.1f} GiB available", group="workspace"))
  except OSError as vError:
    lResults.append(CheckResult("Free space", CheckStatus.WARN, str(vError), group="workspace"))

  vProbe = pContext.workspace.work / ".podcast-editor-mcp-server-write-test"
  try:
    vProbe.write_text("ok", encoding="utf-8")
    vProbe.unlink()
    lResults.append(CheckResult("Write access", CheckStatus.OK, "work directory is writable", group="workspace"))
  except OSError as vError:
    lResults.append(CheckResult("Write access", CheckStatus.ERROR, str(vError), group="workspace"))

  if os.getuid() == 0:
    lResults.append(
      CheckResult(
        "Effective user",
        CheckStatus.WARN,
        "running as root; podcasts should be processed as an ordinary user",
        group="workspace",
      )
    )
  else:
    lResults.append(
      CheckResult("Effective user", CheckStatus.OK, f"uid {os.getuid()}", group="workspace")
    )
  return lResults


def fCheckConfiguration(pContext: AppContext) -> list[CheckResult]:
  lResults: list[CheckResult] = []
  try:
    fLoadRuntimeConfig()
    lResults.append(CheckResult("Runtime configuration", CheckStatus.OK, "valid", group="configuration"))
  except PodcastEditorMcpServerError as vError:
    lResults.append(
      CheckResult("Runtime configuration", CheckStatus.ERROR, vError.message, group="configuration", details=vError.details)
    )

  ldProfiles = fListProfiles(pContext.config)
  if not ldProfiles:
    lResults.append(CheckResult("Profiles", CheckStatus.ERROR, "no profiles found", group="configuration"))
    return lResults

  lBroken: list[str] = []
  for dProfile in ldProfiles:
    try:
      fLoadProfile(dProfile["name"], pContext.config)
    except PodcastEditorMcpServerError as vError:
      lBroken.append(f"{dProfile['name']}: {vError.message}")
  if lBroken:
    lResults.append(
      CheckResult("Profiles", CheckStatus.ERROR, "; ".join(lBroken[:3]), group="configuration", details={"broken": lBroken})
    )
  else:
    lResults.append(
      CheckResult(
        "Profiles",
        CheckStatus.OK,
        ", ".join(dProfile["name"] for dProfile in ldProfiles),
        group="configuration",
      )
    )
  lResults.append(
    CheckResult("Active profile", CheckStatus.OK, pContext.profile.name, group="configuration")
  )
  return lResults


def fCheckMcpServer() -> list[CheckResult]:
  """Build the MCP server object and count its tools without serving anything."""
  try:
    from podcast_editor_mcp_server.server import fBuildServer

    vServer = fBuildServer()
    import anyio

    lTools = anyio.run(vServer.list_tools)
    return [
      CheckResult(
        "MCP server",
        CheckStatus.OK,
        f"initialises with {len(lTools)} tools",
        group="mcp",
        details={"tools": [vTool.name for vTool in lTools]},
      )
    ]
  except Exception as vError:  # noqa: BLE001 - any failure here is a real diagnostic
    return [CheckResult("MCP server", CheckStatus.ERROR, str(vError), group="mcp")]


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def fRunDoctor(
  pContext: AppContext | None = None,
  pCheckModel: bool = True,
  pCheckMcp: bool = True,
) -> DoctorReport:
  """Run every check and return the report."""
  vReport = DoctorReport(version=cVersion)
  vReport.checks.extend(fCheckOperatingSystem())
  vReport.checks.extend(fCheckPython())
  vReport.checks.extend(fCheckFfmpeg())
  # Before the context: a missing or unreadable prefix is the reason the
  # configuration below would fail to load, and saying so first is kinder
  # than a validation error with no cause attached.
  vReport.checks.extend(fCheckInstallation())

  try:
    vContext = pContext if pContext is not None else AppContext()
  except PodcastEditorMcpServerError as vError:
    vReport.checks.append(
      CheckResult("Configuration", CheckStatus.ERROR, vError.message, group="configuration", details=vError.details)
    )
    return vReport

  vReport.checks.extend(fCheckConfiguration(vContext))
  vReport.checks.extend(fCheckWorkspace(vContext))
  vReport.checks.extend(fCheckTranscription(vContext, pCheckModel))
  vReport.checks.extend(fCheckAudacity(vContext))
  if pCheckMcp:
    vReport.checks.extend(fCheckMcpServer())
  return vReport


def fMain(pArgv: list[str] | None = None) -> int:
  """Entry point for ``podcast-editor-mcp-server-doctor``."""
  import argparse

  vParser = argparse.ArgumentParser(
    prog="podcast-editor-mcp-server-doctor",
    description="Diagnose a podcast-editor-mcp-server installation.",
  )
  vParser.add_argument("--json", action="store_true", help="print the report as JSON")
  vParser.add_argument("--no-model-check", action="store_true", help="skip the Whisper model cache check")
  vParser.add_argument("--no-mcp-check", action="store_true", help="skip building the MCP server")
  vParser.add_argument("--version", action="version", version=f"{cProductName} {cVersion}")
  vArguments = vParser.parse_args(pArgv)

  fSetupLogging(pLevel="WARNING")
  vReport = fRunDoctor(pCheckModel=not vArguments.no_model_check, pCheckMcp=not vArguments.no_mcp_check)

  if vArguments.json:
    sys.stdout.write(json.dumps(vReport.mAsDict(), indent=2, ensure_ascii=False) + "\n")
  else:
    sys.stdout.write(vReport.mRender() + "\n")
  return 1 if vReport.failed else 0


if __name__ == "__main__":  # pragma: no cover - module entry point
  raise SystemExit(fMain())
