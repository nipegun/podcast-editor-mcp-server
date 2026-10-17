#!/bin/bash

# podcast-editor-mcp-server installer for Debian 13 (trixie).
#
# Run it as root, straight from GitHub:
# curl -sL https://raw.githubusercontent.com/nipegun/podcast-editor-mcp-server/refs/heads/main/deploy/install.sh | bash
# Options go after 'bash -s --', for example '| bash -s -- --user alice --yes'.
# From a downloaded repository: bash deploy/install.sh. When there is no
# adjacent source tree, the main branch is downloaded from GitHub.
#
# Without --user, the working user is the one who called sudo, the one who
# logged in, the only account an earlier run set up or the only ordinary
# account; with several, it asks. Started by an ordinary user, it runs itself
# again through sudo.
#
# Idempotent by construction: run it as often as you like. It inspects what is
# already there before changing anything, never duplicates a configuration line,
# and never overwrites your profiles, your podcasts or your output.
#
# Everything lives under one prefix, /opt/podcast-editor-mcp-server by default:
#
#   /opt/podcast-editor-mcp-server/bin                 three small launchers
#   /opt/podcast-editor-mcp-server/venv                the interpreter and the dependencies,
#                                               plus CUDA 12 when there is an NVIDIA GPU
#   /opt/podcast-editor-mcp-server/lib/systemd/user    the optional Audacity unit
#   /opt/podcast-editor-mcp-server/etc/config.yaml     system-wide defaults you may edit
#   /opt/podcast-editor-mcp-server/etc/profiles        system-wide profile overrides
#   /opt/podcast-editor-mcp-server/etc/users/<user>    one user's own configuration
#   /opt/podcast-editor-mcp-server/var/log             this installer's log
#   /opt/podcast-editor-mcp-server/var/models          the Whisper models, shared
#   /opt/podcast-editor-mcp-server/var/workspace/<user>  input, output, jobs, cache and logs
#
# Nothing is written to /etc, /var or anybody's home directory. The only things
# placed outside the prefix are symbolic links: three in /usr/local/bin so the
# commands are on PATH, and one in the systemd user unit directory so systemd
# can find the Audacity unit. Both point back into the prefix.
#
# Audacity refuses to run as root, so podcasts are processed by an ordinary
# user. That user needs to write inside the prefix, which is what the
# 'podcast-editor-mcp-server' group is for: the shared directories are group-writable, and
# each user's own workspace belongs to them.

set -euo pipefail

cProductName="podcast-editor-mcp-server"
cVersionRequiredDebian="13"
cPrefixDefault="/opt/podcast-editor-mcp-server"
cSymlinkDir="/usr/local/bin"
cSystemdUserDir="/usr/lib/systemd/user"
cGroupName="podcast-editor-mcp-server"
cRepositoryUrl="https://github.com/nipegun/podcast-editor-mcp-server"
cSourceArchiveUrl="https://codeload.github.com/nipegun/podcast-editor-mcp-server/tar.gz/refs/heads/main"
cInstallerUrl="https://raw.githubusercontent.com/nipegun/podcast-editor-mcp-server/refs/heads/main/deploy/install.sh"
# Capture this outside a function: stdin execution has no script filename.
cScriptPath="${BASH_SOURCE[0]:-}"
# The most accurate Whisper model (about 3 GB). It is what the profiles use, so
# pre-downloading it is what lets the first transcription start straight away.
cWhisperModelDefault="large-v3"

cAptPackages="python3 python3-venv python3-pip ffmpeg libsndfile1"
cAptPackagesAudacity="audacity xvfb"
cAptPackagesDownload="ca-certificates curl tar gzip"

# CTranslate2's wheels are built against CUDA 12 and load cuBLAS 12 and cuDNN 9
# by soname on the first GPU inference. NVIDIA publishes both as wheels, so they
# go into the venv and never clash with whatever CUDA toolkit the system has
# (a system CUDA 13, for example, is of no use to CTranslate2).
aPipPackagesCuda=("nvidia-cublas-cu12" "nvidia-cudnn-cu12==9.*")

cLaunchers="podcast-editor-mcp-server podcast-editor-mcp-server-cli podcast-editor-mcp-server-doctor"

vPrefix="$cPrefixDefault"
vUser=""
vWhisperModel="$cWhisperModelDefault"
vInstallModel="yes"
vInstallAudacity="ask"
vInstallTranscription="yes"
vInstallCuda="auto"
vRunApt="yes"
vCheckOnly="no"
vAssumeYes="no"
vSourceDir=""
vDownloadSource="no"
vTempDir=""
vFailed="no"
vUserHome=""
vUserGroup=""
vLogWritable="no"
vGroupAdded="no"
# Set when this run only handed over to sudo, which reports its own outcome.
vElevated="no"

# Derived from vPrefix once the arguments have been parsed, because --prefix
# moves every one of them.
vBinDir=""
vConfigDir=""
vUserConfigDir=""
vSystemdDir=""
vVarDir=""
vLogDir=""
vLogFile=""
vModelDir=""
vWorkspaceBase=""
vWorkspace=""

fResolveLayout() {
  vBinDir="${vPrefix}/bin"
  vConfigDir="${vPrefix}/etc"
  vSystemdDir="${vPrefix}/lib/systemd/user"
  vVarDir="${vPrefix}/var"
  vLogDir="${vVarDir}/log"
  vLogFile="${vLogDir}/install.log"
  vModelDir="${vVarDir}/models"
  vWorkspaceBase="${vVarDir}/workspace"
  return 0
}

# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #

fOpenLog() {
  # Only root can write inside the prefix. Non-root runs (--help, a mistaken
  # invocation) must not spew redirection errors, so logging simply stays off.
  #
  # --check must not create, touch or append to an existing installation log.
  if [ "$vCheckOnly" = "yes" ]; then
    vLogWritable="no"
    return 0
  fi
  mkdir -p "$vLogDir" 2>/dev/null || true
  chmod 0750 "$vLogDir" 2>/dev/null || true
  if touch "$vLogFile" 2>/dev/null; then
    chmod 0600 "$vLogFile" 2>/dev/null || true
    vLogWritable="yes"
  else
    vLogWritable="no"
  fi
}

fLog() {
  if [ "$vLogWritable" = "yes" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$vLogFile" 2>/dev/null || true
  fi
}

fSay() {
  echo "$*"
  fLog "$*"
}

fStep() {
  echo ""
  echo "==> $*"
  fLog "STEP $*"
}

fWarn() {
  echo "WARNING: $*" >&2
  fLog "WARNING $*"
}

fError() {
  echo "ERROR: $*" >&2
  fLog "ERROR $*"
}

fUsage() {
  echo "Run as root: curl -sL ${cInstallerUrl} | bash"
  echo "With options: curl -sL ${cInstallerUrl} | bash -s -- [options]"
  echo "From a downloaded repository: bash deploy/install.sh [options]"
  echo ""
  echo "Installs podcast-editor-mcp-server on Debian ${cVersionRequiredDebian}. Safe to re-run."
  echo "Started by an ordinary user, it runs again through sudo for that user."
  echo ""
  echo "Options:"
  echo "  --user NAME            Set up the workspace for this user (never root;"
  echo "                         default: the user who called sudo, the login user,"
  echo "                         the only account already set up here or the only"
  echo "                         ordinary account; with several, it asks)"
  echo "  --prefix DIR           Install everything here (default: ${cPrefixDefault}):"
  echo "                         code, configuration, models and workspaces"
  echo "  --whisper-model NAME   Model to pre-download: tiny, base, small,"
  echo "                         medium, large-v3 (default: ${cWhisperModelDefault})"
  echo "  --no-model             Do not download any Whisper model now"
  echo "  --no-transcription     Do not install faster-whisper at all"
  echo "  --with-cuda            Install the CUDA 12 libraries (cuBLAS, cuDNN 9)"
  echo "                         into the venv even if no NVIDIA GPU is detected"
  echo "                         (default: only when one is detected)"
  echo "  --no-cuda              Do not install them; transcribe on the CPU"
  echo "  --with-audacity        Install Audacity and Xvfb without asking"
  echo "  --no-audacity          Do not install Audacity"
  echo "  --no-apt               Do not touch APT; assume the packages are present"
  echo "  --check                Report what would be done, change nothing"
  echo "  --yes                  Never ask anything"
  echo "  --help                 Show this text"
  echo ""
  echo "Examples:"
  echo "  curl -sL ${cInstallerUrl} | bash"
  echo "  curl -sL ${cInstallerUrl} | bash -s -- --user alice --yes"
  echo "  bash deploy/install.sh --user alice --whisper-model medium"
  echo "  bash deploy/install.sh --user alice --no-model --no-audacity --yes"
  echo "  bash deploy/install.sh --user alice --check"
  echo ""
  echo "Local execution uses the source tree next to this script."
  echo "Standalone or piped execution downloads the main branch from ${cRepositoryUrl}."
  echo "--check never downloads sources or changes the installation."
}

fCleanup() {
  if [ -n "$vTempDir" ] && [ -d "$vTempDir" ]; then
    rm -rf -- "$vTempDir"
    # Remove only empty staging parents; another installer may still use them.
    rmdir "${vPrefix}/_/temp" "${vPrefix}/_" 2>/dev/null || true
  fi
}

trap fCleanup EXIT

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #

fParseArguments() {
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --user)
        if [ "$#" -lt 2 ]; then
          fError "--user needs a value"
          exit 2
        fi
        vUser="$2"
        shift 2
        ;;
      --prefix)
        if [ "$#" -lt 2 ]; then
          fError "--prefix needs a value"
          exit 2
        fi
        vPrefix="$2"
        shift 2
        ;;
      --whisper-model)
        if [ "$#" -lt 2 ]; then
          fError "--whisper-model needs a value"
          exit 2
        fi
        vWhisperModel="$2"
        shift 2
        ;;
      --no-model)
        vInstallModel="no"
        shift
        ;;
      --no-transcription)
        vInstallTranscription="no"
        vInstallModel="no"
        shift
        ;;
      --with-cuda)
        vInstallCuda="yes"
        shift
        ;;
      --no-cuda)
        vInstallCuda="no"
        shift
        ;;
      --with-audacity)
        vInstallAudacity="yes"
        shift
        ;;
      --no-audacity)
        vInstallAudacity="no"
        shift
        ;;
      --no-apt)
        vRunApt="no"
        shift
        ;;
      --check)
        vCheckOnly="yes"
        shift
        ;;
      --yes)
        vAssumeYes="yes"
        shift
        ;;
      --help|-h)
        fUsage
        exit 0
        ;;
      *)
        fError "Unknown option: $1"
        echo ""
        fUsage
        exit 2
        ;;
    esac
  done
  return 0
}

# --------------------------------------------------------------------------- #
# Preconditions
# --------------------------------------------------------------------------- #

fRunAsRoot() {
  # Started by an ordinary user: the same installer runs again through sudo,
  # for that user unless --user names another. A piped installer has no file
  # to hand over, so it is fetched again from the same address.
  vElevated="yes"
  if ! command -v sudo > /dev/null 2>&1; then
    fError "This installer must run as root, and sudo is not installed. Run it from a root shell."
    return 1
  fi
  local aArguments=("$@")
  if [ -z "$vUser" ]; then
    aArguments+=("--user" "$(id -un)")
  fi
  fSay "This installer needs root; running it again through sudo..."
  # The password is asked for first, on the terminal, so a refusal never
  # leaves a download half-written into a sudo that already gave up.
  if ! sudo -v; then
    fError "sudo did not grant root. Run the installer from a root shell instead."
    return 1
  fi
  if [ -n "$cScriptPath" ] && [ -f "$cScriptPath" ]; then
    if ! sudo bash "$cScriptPath" "${aArguments[@]}"; then
      return 1
    fi
    return 0
  fi
  if ! command -v curl > /dev/null 2>&1; then
    fError "curl is needed to fetch the installer again for sudo. Run it from a root shell instead."
    return 1
  fi
  if ! curl --fail --silent --show-error --location "$cInstallerUrl" | sudo bash -s -- "${aArguments[@]}"; then
    return 1
  fi
  return 0
}

fCheckDebian() {
  if [ ! -r /etc/os-release ]; then
    fError "/etc/os-release is missing; this does not look like Debian"
    return 1
  fi
  local vId
  local vVersion
  vId=$(sed -n 's/^ID=//p' /etc/os-release | tr -d '"')
  vVersion=$(sed -n 's/^VERSION_ID=//p' /etc/os-release | tr -d '"')
  if [ "$vId" != "debian" ]; then
    fWarn "This is '${vId}', not Debian. The installer targets Debian ${cVersionRequiredDebian} and may not work."
    if [ "$vAssumeYes" != "yes" ] && [ "$vCheckOnly" != "yes" ]; then
      if ! fConfirm "Continue anyway?"; then
        return 1
      fi
    fi
  elif [ "$vVersion" != "$cVersionRequiredDebian" ]; then
    fWarn "This is Debian ${vVersion}; the installer targets Debian ${cVersionRequiredDebian}."
  else
    fSay "Debian ${vVersion} detected."
  fi
  return 0
}

fListWorkspaceUsers() {
  # The accounts an earlier run set up, so that a plain re-run updates them.
  if [ ! -d "$vWorkspaceBase" ]; then
    return 0
  fi
  local vDirectory
  local vName
  for vDirectory in "$vWorkspaceBase"/*/; do
    if [ ! -d "$vDirectory" ]; then
      continue
    fi
    vName=$(basename "$vDirectory")
    if id "$vName" > /dev/null 2>&1 && [ "$(id -u "$vName")" -ne 0 ]; then
      echo "$vName"
    fi
  done
  return 0
}

fListOrdinaryUsers() {
  # People, as login.defs defines them: a UID in the range useradd gives to
  # ordinary accounts, a home directory and a shell one can log in with.
  local vUidMin="1000"
  local vUidMax="60000"
  local vValue
  if [ -r /etc/login.defs ]; then
    vValue=$(sed -n 's/^[[:space:]]*UID_MIN[[:space:]]\{1,\}\([0-9]\{1,\}\).*/\1/p' /etc/login.defs | tail -n 1)
    if [ -n "$vValue" ]; then
      vUidMin="$vValue"
    fi
    vValue=$(sed -n 's/^[[:space:]]*UID_MAX[[:space:]]\{1,\}\([0-9]\{1,\}\).*/\1/p' /etc/login.defs | tail -n 1)
    if [ -n "$vValue" ]; then
      vUidMax="$vValue"
    fi
  fi
  local vName
  local vUid
  local vHome
  local vShell
  # The password, group and comment fields are read into '_' and ignored.
  while IFS=: read -r vName _ vUid _ _ vHome vShell; do
    # NIS compatibility entries ('+', '-') carry no numeric UID.
    case "$vUid" in
      ""|*[!0-9]*) continue ;;
    esac
    if [ "$vUid" -lt "$vUidMin" ] || [ "$vUid" -gt "$vUidMax" ] || [ ! -d "$vHome" ]; then
      continue
    fi
    case "$vShell" in
      ""|*/nologin|*/false) continue ;;
    esac
    echo "$vName"
  done < <(getent passwd)
  return 0
}

fChooseUser() {
  # $1 and $2 name the candidates in the singular and the plural; the rest are
  # the candidates. One is taken as it is; several are offered on the terminal.
  local vSingular="$1"
  local vPlural="$2"
  shift 2
  local aCandidates=("$@")
  if [ "${#aCandidates[@]}" -eq 1 ]; then
    vUser="${aCandidates[0]}"
    fSay "Setting up ${vUser}, the only ${vSingular}."
    return 0
  fi
  if [ "$vAssumeYes" = "yes" ] || ! fHasTerminal; then
    fError "There are several ${vPlural}: ${aCandidates[*]}. Pass --user NAME to pick one."
    return 1
  fi
  echo ""
  echo "There are several ${vPlural}. Which one will process the recordings?"
  local vIndex
  for vIndex in "${!aCandidates[@]}"; do
    echo "  $((vIndex + 1))) ${aCandidates[$vIndex]}"
  done
  local vAnswer=""
  echo -n "Number or name: "
  read -r vAnswer < /dev/tty || true
  for vIndex in "${!aCandidates[@]}"; do
    if [ "$vAnswer" = "$((vIndex + 1))" ] || [ "$vAnswer" = "${aCandidates[$vIndex]}" ]; then
      vUser="${aCandidates[$vIndex]}"
      return 0
    fi
  done
  fError "'${vAnswer}' is not one of them. Pass --user NAME to pick one."
  return 1
}

fResolveUser() {
  if [ -z "$vUser" ]; then
    if [ -n "${SUDO_USER:-}" ] && [ "${SUDO_USER}" != "root" ]; then
      vUser="$SUDO_USER"
    elif [ -n "${PKEXEC_UID:-}" ] && [ "${PKEXEC_UID}" != "0" ]; then
      vUser=$(getent passwd "$PKEXEC_UID" | cut -d: -f1)
    else
      local vLogname
      vLogname=$(logname 2>/dev/null || true)
      if [ -n "$vLogname" ] && [ "$vLogname" != "root" ]; then
        vUser="$vLogname"
      fi
    fi
  fi

  if [ -z "$vUser" ]; then
    # A root shell (a console, ssh as root, a container) names nobody: take an
    # account an earlier run set up, otherwise an ordinary account.
    local aCandidates=()
    mapfile -t aCandidates < <(fListWorkspaceUsers)
    if [ "${#aCandidates[@]}" -gt 0 ]; then
      if ! fChooseUser "account already set up here" "accounts already set up here" "${aCandidates[@]}"; then
        return 1
      fi
    else
      mapfile -t aCandidates < <(fListOrdinaryUsers)
      if [ "${#aCandidates[@]}" -eq 0 ]; then
        fError "There is no ordinary account to process the recordings, and root cannot do it."
        fError "Create one (for example: adduser alice) and run the installer again, or pass --user NAME."
        return 1
      fi
      if ! fChooseUser "ordinary account" "ordinary accounts" "${aCandidates[@]}"; then
        return 1
      fi
    fi
  fi

  if [ -z "$vUser" ]; then
    fError "Could not work out which user to set up. Pass --user NAME."
    return 1
  fi
  if ! id "$vUser" > /dev/null 2>&1; then
    fError "User '${vUser}' does not exist."
    return 1
  fi
  if [ "$(id -u "$vUser")" -eq 0 ]; then
    fError "Refusing to set up the workspace for root. Podcast processing must run as an ordinary user."
    return 1
  fi

  # The product itself stores nothing in the home directory, but Audacity
  # keeps its own settings there and refuses to start without one.
  vUserHome=$(getent passwd "$vUser" | cut -d: -f6)
  if [ -z "$vUserHome" ] || [ ! -d "$vUserHome" ]; then
    fError "User '${vUser}' has no usable home directory; Audacity needs one to start."
    return 1
  fi
  vUserGroup=$(id -gn "$vUser")
  fSay "Working user: ${vUser} (${vUserHome})"
  return 0
}

fHasTerminal() {
  # /dev/tty always exists, but it only opens when there is a controlling
  # terminal: a piped run started by a plain 'ssh host command' has none.
  ( : < /dev/tty ) 2> /dev/null
}

fConfirm() {
  if [ "$vAssumeYes" = "yes" ]; then
    return 0
  fi
  local vAnswer=""
  echo -n "$1 [y/N] "
  # Read from the terminal, never from a redirected stdin, so the script can
  # never advance on input that was not typed by a person.
  if fHasTerminal; then
    read -r vAnswer < /dev/tty || true
  else
    echo ""
    fSay "(no terminal to answer on; taking that as no)"
  fi
  case "$vAnswer" in
    y|Y|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

# --------------------------------------------------------------------------- #
# APT
# --------------------------------------------------------------------------- #

fPackageInstalled() {
  dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -q "ok installed"
}

fMissingPackages() {
  local vPackage
  local vMissing=""
  for vPackage in $1; do
    if ! fPackageInstalled "$vPackage"; then
      vMissing="${vMissing} ${vPackage}"
    fi
  done
  echo "$vMissing" | sed 's/^ *//'
}

fInstallPackages() {
  local vWanted="$cAptPackages"
  if [ "$vDownloadSource" = "yes" ]; then
    vWanted="${vWanted} ${cAptPackagesDownload}"
  fi
  if [ "$vInstallAudacity" = "yes" ]; then
    vWanted="${vWanted} ${cAptPackagesAudacity}"
  fi

  local vMissing
  vMissing=$(fMissingPackages "$vWanted")

  if [ -z "$vMissing" ]; then
    fSay "All required packages are already installed."
    return 0
  fi

  fSay "Missing packages:${vMissing}"
  if [ "$vCheckOnly" = "yes" ]; then
    fSay "(--check: not installing)"
    return 0
  fi
  if [ "$vRunApt" = "no" ]; then
    fError "Packages are missing and --no-apt was given:${vMissing}"
    return 1
  fi

  fSay "Updating the package lists..."
  if ! DEBIAN_FRONTEND=noninteractive apt-get update >> "$vLogFile" 2>&1; then
    fError "apt-get update failed. See ${vLogFile}."
    return 1
  fi

  fSay "Installing:${vMissing}"
  # shellcheck disable=SC2086
  if ! DEBIAN_FRONTEND=noninteractive apt-get install -y $vMissing >> "$vLogFile" 2>&1; then
    fError "apt-get install failed. See ${vLogFile}."
    return 1
  fi

  local vStillMissing
  vStillMissing=$(fMissingPackages "$vWanted")
  if [ -n "$vStillMissing" ]; then
    fError "These packages are still missing after the install:${vStillMissing}"
    return 1
  fi
  fSay "Packages installed."
  return 0
}

fDecideAudacity() {
  if [ "$vInstallAudacity" != "ask" ]; then
    return 0
  fi
  if fPackageInstalled "audacity"; then
    vInstallAudacity="yes"
    fSay "Audacity is already installed; keeping the integration enabled."
    return 0
  fi
  if [ "$vAssumeYes" = "yes" ] || [ "$vCheckOnly" = "yes" ]; then
    vInstallAudacity="no"
    fSay "Audacity is not installed and no choice was given; skipping it."
    fSay "Everything except the .aup3 project works without Audacity."
    return 0
  fi
  echo ""
  echo "Audacity gives you an editable .aup3 project alongside the audio."
  echo "It is optional: every other feature works through FFmpeg."
  if fConfirm "Install Audacity and Xvfb (about 60 MB)?"; then
    vInstallAudacity="yes"
  else
    vInstallAudacity="no"
  fi
  return 0
}

# --------------------------------------------------------------------------- #
# Installation tree
# --------------------------------------------------------------------------- #

fEnsureGroup() {
  # Everything lives under the prefix, but Audacity refuses to run as root, so
  # the working user has to be able to write there. One group grants exactly
  # that, and only for the shared model cache: each workspace belongs to its
  # own user.
  if getent group "$cGroupName" > /dev/null 2>&1; then
    fSay "Group '${cGroupName}' already exists."
  elif [ "$vCheckOnly" = "yes" ]; then
    fSay "(--check: would create the system group '${cGroupName}')"
  else
    if ! groupadd --system "$cGroupName" >> "$vLogFile" 2>&1; then
      fError "Could not create the system group '${cGroupName}'. See ${vLogFile}."
      return 1
    fi
    fSay "Created the system group '${cGroupName}'."
  fi

  # Compared without a pipeline on purpose: 'grep -q' exits on the first match
  # and would leave the writer with a SIGPIPE, which 'pipefail' then reports as
  # a failure, and the membership would read as missing.
  local vGroups
  vGroups=$(id -nG "$vUser" 2>/dev/null || true)
  case " ${vGroups} " in
    *" ${cGroupName} "*)
      fSay "User ${vUser} is already in the '${cGroupName}' group."
      return 0
      ;;
  esac
  if [ "$vCheckOnly" = "yes" ]; then
    fSay "(--check: would add ${vUser} to the '${cGroupName}' group)"
    return 0
  fi
  if ! usermod -aG "$cGroupName" "$vUser" >> "$vLogFile" 2>&1; then
    fError "Could not add ${vUser} to the '${cGroupName}' group. See ${vLogFile}."
    return 1
  fi
  vGroupAdded="yes"
  fSay "Added ${vUser} to the '${cGroupName}' group."
  return 0
}

fCreateTree() {
  if [ "$vCheckOnly" = "yes" ]; then
    fSay "(--check: would create the directory tree under ${vPrefix})"
    return 0
  fi

  # Owned by root and read-only to everyone else: code, launchers,
  # configuration and the systemd unit.
  mkdir -p "$vPrefix" "$vBinDir" "$vSystemdDir" "${vConfigDir}/profiles" "${vConfigDir}/users"
  chown root:root "$vPrefix" "$vBinDir" "$vSystemdDir" "$vConfigDir" "${vConfigDir}/profiles" "${vConfigDir}/users"
  chmod 0755 "$vPrefix" "$vBinDir" "$vSystemdDir" "$vConfigDir" "${vConfigDir}/profiles" "${vConfigDir}/users"

  mkdir -p "$vVarDir" "$vWorkspaceBase"
  chown root:root "$vVarDir" "$vWorkspaceBase"
  chmod 0755 "$vVarDir" "$vWorkspaceBase"

  # The only shared writable directory. A large-v3 model is 3 GB, so one copy
  # per user would be 3 GB wasted; setgid keeps the group on what is
  # downloaded later.
  mkdir -p "$vModelDir"
  chown "root:${cGroupName}" "$vModelDir"
  chmod 2775 "$vModelDir"

  # The log names package versions and paths, and nobody but root needs it.
  mkdir -p "$vLogDir"
  chown root:root "$vLogDir"
  chmod 0750 "$vLogDir"
  if [ -f "$vLogFile" ]; then
    chmod 0600 "$vLogFile"
  fi

  fSay "Directory tree ready under ${vPrefix}"
  return 0
}

# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #

fLocateSource() {
  # A script read from stdin has no local source location. Do not infer one
  # from $0 (usually 'bash') or from an unrelated current working directory.
  if [ -n "$cScriptPath" ] && [ -f "$cScriptPath" ]; then
    local vScriptFile
    if ! vScriptFile=$(readlink -f -- "$cScriptPath"); then
      fError "Could not resolve the installer path: ${cScriptPath}"
      return 1
    fi
    if ! vSourceDir=$(cd -- "$(dirname -- "$vScriptFile")/.." && pwd -P); then
      fError "Could not locate the directory above install.sh."
      return 1
    fi
    if [ -f "${vSourceDir}/pyproject.toml" ]; then
      if ! fValidateSource; then
        return 1
      fi
      fSay "Local source: ${vSourceDir}"
      return 0
    fi
  fi
  vSourceDir=""
  vDownloadSource="yes"
  fSay "Source: main branch of ${cRepositoryUrl} (download required)"
  return 0
}

fValidateSource() {
  if [ ! -f "${vSourceDir}/pyproject.toml" ] || \
     [ ! -f "${vSourceDir}/README.md" ] || \
     [ ! -f "${vSourceDir}/backend/server.py" ] || \
     [ ! -f "${vSourceDir}/backend/data/profiles/podcast-default.yaml" ] || \
     [ ! -f "${vSourceDir}/deploy/systemd/podcast-editor-mcp-server-audacity.service" ]; then
    fError "The ${cProductName} source tree is incomplete: ${vSourceDir}"
    return 1
  fi
  if ! grep -Eq '^name[[:space:]]*=[[:space:]]*"podcast-editor-mcp-server"[[:space:]]*$' "${vSourceDir}/pyproject.toml"; then
    fError "The source tree does not identify the expected package: ${vSourceDir}"
    return 1
  fi
  return 0
}

fDownloadSource() {
  if [ "$vDownloadSource" != "yes" ]; then
    return 0
  fi
  if [ "$vCheckOnly" = "yes" ]; then
    fSay "(--check: would download ${cSourceArchiveUrl})"
    return 0
  fi
  if ! mkdir -p "${vPrefix}/_/temp"; then
    fError "Could not create the source staging directory."
    return 1
  fi
  if ! vTempDir=$(mktemp -d "${vPrefix}/_/temp/source.XXXXXXXX"); then
    fError "Could not create a temporary source directory."
    return 1
  fi
  local vArchive="${vTempDir}/source.tar.gz"
  vSourceDir="${vTempDir}/source"
  if ! mkdir -p "$vSourceDir"; then
    fError "Could not create the source extraction directory."
    return 1
  fi
  fSay "Downloading ${cSourceArchiveUrl}..."
  if ! curl --fail --show-error --silent --location --retry 3 --connect-timeout 30 \
      --output "$vArchive" "$cSourceArchiveUrl" >> "$vLogFile" 2>&1; then
    fError "Could not download the repository. Check that its main branch is public and reachable. See ${vLogFile}."
    return 1
  fi
  if ! tar --extract --gzip --file "$vArchive" --directory "$vSourceDir" \
      --strip-components=1 --no-same-owner --no-same-permissions >> "$vLogFile" 2>&1; then
    fError "Could not extract the downloaded repository. See ${vLogFile}."
    return 1
  fi
  if ! fValidateSource; then
    return 1
  fi
  fSay "Downloaded source ready: ${vSourceDir}"
  return 0
}

fStageLocalSource() {
  # pip builds inside the tree it installs from and leaves build/ and
  # *.egg-info behind, so local sources are installed from a copy.
  if ! mkdir -p "${vPrefix}/_/temp"; then
    fError "Could not create the source staging directory."
    return 1
  fi
  if ! vTempDir=$(mktemp -d "${vPrefix}/_/temp/source.XXXXXXXX"); then
    fError "Could not create a temporary source directory."
    return 1
  fi
  if ! mkdir -p "${vTempDir}/source"; then
    fError "Could not create the source copy directory."
    return 1
  fi
  if ! cp -R "${vSourceDir}/pyproject.toml" "${vSourceDir}/README.md" "${vSourceDir}/backend" "${vTempDir}/source/" >> "$vLogFile" 2>&1; then
    fError "Could not copy the source tree into ${vTempDir}. See ${vLogFile}."
    return 1
  fi
  return 0
}

fInstallApplication() {
  if [ "$vCheckOnly" = "yes" ]; then
    fSay "(--check: would install into ${vPrefix})"
    return 0
  fi

  local vVenv="${vPrefix}/venv"
  local vPython="${vVenv}/bin/python"

  if [ -x "$vPython" ] && "$vPython" -c "import sys" > /dev/null 2>&1; then
    fSay "Reusing the existing virtualenv at ${vVenv}"
  else
    if [ -e "$vVenv" ]; then
      fWarn "The virtualenv at ${vVenv} is broken; rebuilding it."
      rm -rf "$vVenv"
    fi
    fSay "Creating the virtualenv..."
    if ! python3 -m venv "$vVenv" >> "$vLogFile" 2>&1; then
      fError "Could not create the virtualenv. See ${vLogFile}."
      return 1
    fi
  fi

  fSay "Updating pip..."
  if ! "$vPython" -m pip install --upgrade pip >> "$vLogFile" 2>&1; then
    fWarn "Could not update pip; continuing with the bundled version."
  fi

  local vExtras=""
  if [ "$vInstallTranscription" = "yes" ]; then
    vExtras="[whisper]"
  fi

  local vBuildSource="$vSourceDir"
  if [ "$vDownloadSource" != "yes" ]; then
    if ! fStageLocalSource; then
      return 1
    fi
    vBuildSource="${vTempDir}/source"
  fi

  fSay "Installing ${cProductName}${vExtras}..."
  if ! "$vPython" -m pip install --upgrade "${vBuildSource}${vExtras}" >> "$vLogFile" 2>&1; then
    fError "pip install failed. See ${vLogFile}."
    return 1
  fi

  # Record a durable origin, not the temporary directory removed on exit.
  if ! mkdir -p "${vPrefix}/share"; then
    return 1
  fi
  if [ "$vDownloadSource" = "yes" ]; then
    if ! echo "$cSourceArchiveUrl" > "${vPrefix}/share/source-url"; then
      return 1
    fi
    rm -f "${vPrefix}/share/source-directory" || return 1
  else
    if ! echo "$vSourceDir" > "${vPrefix}/share/source-directory"; then
      return 1
    fi
    rm -f "${vPrefix}/share/source-url" || return 1
  fi

  fSay "Application installed in ${vPrefix}"
  return 0
}

fDecideCuda() {
  # Without faster-whisper there is nothing that could use the GPU.
  if [ "$vInstallTranscription" != "yes" ]; then
    vInstallCuda="no"
    return 0
  fi
  if [ "$vInstallCuda" != "auto" ]; then
    return 0
  fi
  # The driver creates the device node; nvidia-smi confirms the card on systems
  # where the node only appears once something has used it.
  if [ -e /dev/nvidia0 ] || nvidia-smi -L > /dev/null 2>&1; then
    vInstallCuda="yes"
    fSay "NVIDIA GPU detected; the CUDA 12 libraries will go into the virtualenv."
  else
    vInstallCuda="no"
    fSay "No NVIDIA GPU detected; transcription will run on the CPU."
  fi
  return 0
}

fInstallCudaLibraries() {
  if [ "$vInstallCuda" != "yes" ]; then
    return 0
  fi
  if [ "$vCheckOnly" = "yes" ]; then
    fSay "(--check: would install ${aPipPackagesCuda[*]} into ${vPrefix}/venv)"
    return 0
  fi

  local vPython="${vPrefix}/venv/bin/python"

  # No --upgrade: they are about 1 GB, and a re-run must not download them
  # again just because NVIDIA published a newer build.
  fSay "Installing the CUDA 12 libraries (cuBLAS, cuDNN 9; about 1 GB the first time)..."
  if ! "$vPython" -m pip install "${aPipPackagesCuda[@]}" >> "$vLogFile" 2>&1; then
    # Not fatal: the CPU path is fully supported.
    fWarn "Could not install the CUDA libraries; transcription will use the CPU."
    fWarn "See ${vLogFile}."
    return 0
  fi
  fSay "CUDA libraries installed."
  return 0
}

fWriteLauncher() {
  local vName="$1"
  local vModule="$2"
  local vTarget="${vBinDir}/${vName}"
  local vTemporary="${vTarget}.new"

  echo "#!/bin/bash" > "$vTemporary"
  echo "" >> "$vTemporary"
  echo "# Generated by the ${cProductName} installer. Do not edit; re-run podcast-editor-mcp-server/deploy/install.sh." >> "$vTemporary"
  echo "# The venv interpreter is used directly, so nothing has to be activated." >> "$vTemporary"
  echo "# Exporting the prefix is what keeps an installation moved with --prefix" >> "$vTemporary"
  echo "# from reading the configuration of the default one." >> "$vTemporary"
  echo "" >> "$vTemporary"
  echo "set -euo pipefail" >> "$vTemporary"
  echo "" >> "$vTemporary"
  echo "cPrefix=\"${vPrefix}\"" >> "$vTemporary"
  echo "cPython=\"\${cPrefix}/venv/bin/python\"" >> "$vTemporary"
  echo "" >> "$vTemporary"
  echo "if [ ! -x \"\$cPython\" ]; then" >> "$vTemporary"
  echo "  echo \"${cProductName}: the installation at ${vPrefix} is broken; re-run podcast-editor-mcp-server/deploy/install.sh\" >&2" >> "$vTemporary"
  echo "  exit 1" >> "$vTemporary"
  echo "fi" >> "$vTemporary"
  echo "" >> "$vTemporary"
  echo "export PODCAST_EDITOR_MCP_SERVER_PREFIX=\"\$cPrefix\"" >> "$vTemporary"
  echo "" >> "$vTemporary"
  echo "# CTranslate2 loads cuBLAS 12 and cuDNN 9 by soname. When the installer put" >> "$vTemporary"
  echo "# them in the venv (an NVIDIA GPU was detected), their directories go on the" >> "$vTemporary"
  echo "# library path; without them nothing changes and the CPU is used." >> "$vTemporary"
  echo "for vCudaLibDir in \"\${cPrefix}\"/venv/lib/python3*/site-packages/nvidia/*/lib; do" >> "$vTemporary"
  echo "  if [ -d \"\$vCudaLibDir\" ]; then" >> "$vTemporary"
  echo "    export LD_LIBRARY_PATH=\"\${vCudaLibDir}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}\"" >> "$vTemporary"
  echo "  fi" >> "$vTemporary"
  echo "done" >> "$vTemporary"
  echo "" >> "$vTemporary"
  echo "exec \"\$cPython\" -m ${vModule} \"\$@\"" >> "$vTemporary"

  chmod 0755 "$vTemporary"
  mv "$vTemporary" "$vTarget"
  fSay "  ${vTarget}"
}

fInstallLaunchers() {
  if [ "$vCheckOnly" = "yes" ]; then
    fSay "(--check: would install launchers into ${vBinDir})"
    return 0
  fi
  mkdir -p "$vBinDir"
  fSay "Installing launchers:"
  fWriteLauncher "podcast-editor-mcp-server" "podcast_editor_mcp_server.server"
  fWriteLauncher "podcast-editor-mcp-server-cli" "podcast_editor_mcp_server.cli"
  fWriteLauncher "podcast-editor-mcp-server-doctor" "podcast_editor_mcp_server.doctor"
  return 0
}

fInstallSymlinks() {
  # The only thing this installer puts outside the prefix. A link, not a copy,
  # so an upgrade never leaves a stale launcher behind on PATH.
  if [ "$vCheckOnly" = "yes" ]; then
    fSay "(--check: would link the launchers into ${cSymlinkDir})"
    return 0
  fi
  mkdir -p "$cSymlinkDir"
  fSay "Linking the commands onto PATH:"
  local vName
  local vLink
  local vTarget
  for vName in $cLaunchers; do
    vLink="${cSymlinkDir}/${vName}"
    vTarget="${vBinDir}/${vName}"
    if [ -e "$vLink" ] && [ ! -L "$vLink" ]; then
      fWarn "${vLink} exists and was not created by this installer; leaving it alone."
      continue
    fi
    ln -sfn "$vTarget" "$vLink"
    fSay "  ${vLink} -> ${vTarget}"
  done
  return 0
}

fWriteSystemConfig() {
  if [ "$vCheckOnly" = "yes" ]; then
    fSay "(--check: would ensure ${vConfigDir})"
    return 0
  fi

  mkdir -p "${vConfigDir}/profiles"
  chmod 0755 "$vConfigDir" "${vConfigDir}/profiles"

  local vConfig="${vConfigDir}/config.yaml"
  if [ -f "$vConfig" ]; then
    fSay "Keeping the existing ${vConfig}"
  else
    fSay "Writing ${vConfig}"
    echo "# System-wide defaults for podcast-editor-mcp-server." > "$vConfig"
    echo "# A user's own etc/users/<user>/config.yaml takes precedence." >> "$vConfig"
    echo "# This file is never overwritten by the installer once it exists." >> "$vConfig"
    echo "" >> "$vConfig"
    echo "# Paths accept two placeholders, so one file serves every user:" >> "$vConfig"
    echo "#   \${prefix}  the installation root, ${vPrefix} here" >> "$vConfig"
    echo "#   \${user}    the account the process runs as" >> "$vConfig"
    echo "workspace:" >> "$vConfig"
    echo '  root: ${prefix}/var/workspace/${user}' >> "$vConfig"
    echo "  allow_symlinks: false" >> "$vConfig"
    echo "" >> "$vConfig"
    echo "logging:" >> "$vConfig"
    echo "  level: INFO" >> "$vConfig"
    echo "  json: false" >> "$vConfig"
    echo "  per_job_log: true" >> "$vConfig"
    echo "" >> "$vConfig"
    echo "audacity:" >> "$vConfig"
    if [ "$vInstallAudacity" = "yes" ]; then
      echo "  enabled: true" >> "$vConfig"
    else
      echo "  enabled: false" >> "$vConfig"
    fi
    echo "  executable: audacity" >> "$vConfig"
    echo "  use_xvfb: auto" >> "$vConfig"
    echo "  project_mode: master" >> "$vConfig"
    echo "" >> "$vConfig"
    echo "jobs:" >> "$vConfig"
    echo "  keep_work_dir: false" >> "$vConfig"
    echo "  max_concurrent: 2" >> "$vConfig"
    echo "  max_queued: 8" >> "$vConfig"
    echo "  transcription_cache: true" >> "$vConfig"
    echo "" >> "$vConfig"
    echo "default_profile: podcast-default" >> "$vConfig"
    chmod 0644 "$vConfig"
  fi

  local vReadme="${vConfigDir}/profiles/README"
  echo "Drop your own profile YAML files here to add or override profiles." > "$vReadme"
  echo "" >> "$vReadme"
  echo "A file called podcast-default.yaml in this directory replaces the" >> "$vReadme"
  echo "built-in profile of that name. A file only needs the keys it changes;" >> "$vReadme"
  echo "everything else falls back to the built-in defaults." >> "$vReadme"
  echo "" >> "$vReadme"
  echo "A profile here applies to everybody. For yourself only, use" >> "$vReadme"
  echo "${vConfigDir}/users/<your user>/profiles/ instead." >> "$vReadme"
  echo "" >> "$vReadme"
  echo "See the built-in profiles with: podcast-editor-mcp-server-cli profiles NAME" >> "$vReadme"
  chmod 0644 "$vReadme"
  return 0
}

fInstallSystemdUnit() {
  if [ "$vInstallAudacity" != "yes" ]; then
    return 0
  fi
  if [ "$vCheckOnly" = "yes" ]; then
    fSay "(--check: would install the systemd --user unit)"
    return 0
  fi

  local vUnit="${vSystemdDir}/podcast-editor-mcp-server-audacity.service"
  local vSource="${vSourceDir}/deploy/systemd/podcast-editor-mcp-server-audacity.service"

  mkdir -p "$vSystemdDir"
  sed "s|@BINDIR@|${vBinDir}|g" "$vSource" > "${vUnit}.new"
  chmod 0644 "${vUnit}.new"
  mv "${vUnit}.new" "$vUnit"
  fSay "Installed ${vUnit}"

  # The unit file stays in the prefix; systemd only gets a link, because it
  # looks for units in its own directories and nowhere else.
  if [ -d "$cSystemdUserDir" ]; then
    local vLink="${cSystemdUserDir}/podcast-editor-mcp-server-audacity.service"
    ln -sfn "$vUnit" "$vLink"
    fSay "Linked ${vLink} -> ${vUnit}"
    fSay "Start it as the working user with:"
    fSay "  systemctl --user enable --now podcast-editor-mcp-server-audacity.service"
  else
    fSay "No systemd user unit directory; enable the unit by hand with:"
    fSay "  systemctl --user link ${vUnit}"
  fi
  return 0
}

# --------------------------------------------------------------------------- #
# User workspace
# --------------------------------------------------------------------------- #

fCreateWorkspace() {
  vWorkspace="${vWorkspaceBase}/${vUser}"
  vUserConfigDir="${vConfigDir}/users/${vUser}"
  if [ "$vCheckOnly" = "yes" ]; then
    fSay "(--check: would ensure ${vWorkspace} and ${vUserConfigDir})"
    return 0
  fi

  local vDirectory
  for vDirectory in input work output jobs cache logs; do
    mkdir -p "${vWorkspace}/${vDirectory}"
  done
  mkdir -p "${vUserConfigDir}/profiles"

  # The recordings, the edited masters and the personal settings belong to the
  # person, never to root, even though they live under the prefix. 0750
  # keeps one user's podcasts out of another user's reach.
  chown -R "${vUser}:${vUserGroup}" "$vWorkspace"
  chown -R "${vUser}:${vUserGroup}" "$vUserConfigDir"
  chmod 0750 "$vWorkspace" "$vUserConfigDir"

  fSay "Workspace ready at ${vWorkspace}"
  fSay "Personal configuration at ${vUserConfigDir}"
  return 0
}

fRunAsUser() {
  # runuser keeps the environment predictable and does not need a password. It
  # also re-reads the group list, so the 'podcast-editor-mcp-server' group added moments ago
  # already applies here. The prefix is passed explicitly, because a --prefix
  # installation must not read the default one's configuration.
  runuser -u "$vUser" -- env "PODCAST_EDITOR_MCP_SERVER_PREFIX=${vPrefix}" "$@"
}

fDownloadModel() {
  if [ "$vInstallModel" != "yes" ] || [ "$vInstallTranscription" != "yes" ]; then
    fSay "Skipping the Whisper model download."
    return 0
  fi
  if [ "$vCheckOnly" = "yes" ]; then
    fSay "(--check: would ensure the '${vWhisperModel}' model is cached)"
    return 0
  fi

  local vPython="${vPrefix}/venv/bin/python"
  fSay "Checking the '${vWhisperModel}' Whisper model in ${vModelDir}..."

  if fRunAsUser "$vPython" -c "from podcast_editor_mcp_server.transcription.faster_whisper_backend import FasterWhisperBackend; from podcast_editor_mcp_server.config import TranscriptionSettings; import sys; sys.exit(0 if FasterWhisperBackend().mIsModelCached(TranscriptionSettings(model='${vWhisperModel}')) else 1)" > /dev/null 2>&1; then
    fSay "The model is already cached; not downloading it again."
    return 0
  fi

  fSay "Downloading it (this happens once and needs network access)..."
  if fRunAsUser "$vPython" -c "from podcast_editor_mcp_server.transcription.faster_whisper_backend import FasterWhisperBackend; from podcast_editor_mcp_server.config import TranscriptionSettings; print(FasterWhisperBackend().mEnsureModel(TranscriptionSettings(model='${vWhisperModel}')))" >> "$vLogFile" 2>&1; then
    fSay "Model '${vWhisperModel}' is ready."
  else
    fWarn "Could not download the model. The first transcription will try again."
    fWarn "See ${vLogFile}. To pick another model, edit the profile or pass --whisper-model."
  fi
  return 0
}

fEnableAudacityModule() {
  if [ "$vInstallAudacity" != "yes" ] || [ "$vCheckOnly" = "yes" ]; then
    return 0
  fi
  if ! command -v audacity > /dev/null 2>&1; then
    return 0
  fi
  if pgrep -u "$(id -u "$vUser")" -x audacity > /dev/null 2>&1; then
    fWarn "Audacity is running as ${vUser}; not touching audacity.cfg."
    fWarn "Close it and run: podcast-editor-mcp-server-cli audacity-status --enable-module"
    return 0
  fi

  fSay "Enabling mod-script-pipe in ${vUser}'s Audacity configuration..."
  if fRunAsUser "${vBinDir}/podcast-editor-mcp-server-cli" audacity-status --enable-module >> "$vLogFile" 2>&1; then
    fSay "mod-script-pipe is enabled."
  else
    fWarn "Could not enable mod-script-pipe automatically."
    fWarn "Run as ${vUser}: podcast-editor-mcp-server-cli audacity-status --enable-module"
  fi
  return 0
}

fRestartAudacityUnit() {
  # A unit that is already running keeps the previous installation loaded
  # until it restarts. Root reaches the working user's systemd through
  # --machine=<user>@; runuser does not, because that user's bus is not in its
  # environment. When the user's systemd is not running there is nothing to
  # restart, and the calls fail harmlessly.
  if [ "$vInstallAudacity" != "yes" ] || [ "$vCheckOnly" = "yes" ]; then
    return 0
  fi
  if ! command -v systemctl > /dev/null 2>&1; then
    return 0
  fi
  if ! systemctl --user --machine="${vUser}@" daemon-reload > /dev/null 2>&1; then
    return 0
  fi
  if systemctl --user --machine="${vUser}@" is-active --quiet podcast-editor-mcp-server-audacity.service > /dev/null 2>&1; then
    fSay "Restarting ${vUser}'s Audacity unit so that it runs this version..."
    if ! systemctl --user --machine="${vUser}@" restart podcast-editor-mcp-server-audacity.service >> "$vLogFile" 2>&1; then
      fWarn "Could not restart it. Run as ${vUser}: systemctl --user restart podcast-editor-mcp-server-audacity.service"
    fi
  fi
  return 0
}

# --------------------------------------------------------------------------- #
# Smoke tests
# --------------------------------------------------------------------------- #

vSmokePython="unknown"
vSmokeImport="unknown"
vSmokeCli="unknown"
vSmokeFfmpeg="unknown"
vSmokeFfprobe="unknown"
vSmokeConfig="unknown"
vSmokeDirectories="unknown"
vSmokeTranscription="unknown"
vSmokeModel="unknown"
vSmokeCuda="unknown"
vSmokeAudacity="unknown"
vSmokeModule="unknown"
vSmokeServer="unknown"

fRunSmokeTests() {
  if [ "$vCheckOnly" = "yes" ]; then
    return 0
  fi
  local vPython="${vPrefix}/venv/bin/python"

  if "$vPython" --version > /dev/null 2>&1; then
    vSmokePython=$("$vPython" --version 2>&1 | sed 's/^Python //')
  else
    vSmokePython="FAILED"
    vFailed="yes"
  fi

  if fRunAsUser "$vPython" -c "import podcast_editor_mcp_server.version" > /dev/null 2>&1; then
    vSmokeImport="OK"
  else
    vSmokeImport="FAILED"
    vFailed="yes"
  fi

  if fRunAsUser "${vBinDir}/podcast-editor-mcp-server-cli" --version > /dev/null 2>&1; then
    vSmokeCli="OK"
  else
    vSmokeCli="FAILED"
    vFailed="yes"
  fi

  if command -v ffmpeg > /dev/null 2>&1; then
    vSmokeFfmpeg=$(ffmpeg -version 2>/dev/null | sed -n '1s/^ffmpeg version \([^ ]*\).*/\1/p')
    if [ -z "$vSmokeFfmpeg" ]; then
      vSmokeFfmpeg="OK"
    fi
  else
    vSmokeFfmpeg="FAILED"
    vFailed="yes"
  fi

  if command -v ffprobe > /dev/null 2>&1; then
    vSmokeFfprobe="OK"
  else
    vSmokeFfprobe="FAILED"
    vFailed="yes"
  fi

  if fRunAsUser "${vBinDir}/podcast-editor-mcp-server-cli" profiles > /dev/null 2>&1; then
    vSmokeConfig="OK"
  else
    vSmokeConfig="FAILED"
    vFailed="yes"
  fi

  if fRunAsUser "${vBinDir}/podcast-editor-mcp-server-cli" workspace > /dev/null 2>&1; then
    vSmokeDirectories="OK"
  else
    vSmokeDirectories="FAILED"
    vFailed="yes"
  fi

  if [ "$vInstallTranscription" = "yes" ]; then
    if fRunAsUser "$vPython" -c "import faster_whisper" > /dev/null 2>&1; then
      vSmokeTranscription="OK"
    else
      vSmokeTranscription="FAILED"
      vFailed="yes"
    fi
    if [ "$vInstallModel" = "yes" ]; then
      if fRunAsUser "$vPython" -c "from podcast_editor_mcp_server.transcription.faster_whisper_backend import FasterWhisperBackend; from podcast_editor_mcp_server.config import TranscriptionSettings; import sys; sys.exit(0 if FasterWhisperBackend().mIsModelCached(TranscriptionSettings(model='${vWhisperModel}')) else 1)" > /dev/null 2>&1; then
        vSmokeModel="OK"
      else
        vSmokeModel="not downloaded"
      fi
    else
      vSmokeModel="skipped"
    fi
  else
    vSmokeTranscription="not installed"
    vSmokeModel="skipped"
  fi

  if [ "$vInstallCuda" = "yes" ]; then
    # Asked through the doctor launcher, so the library path is exactly the one
    # the server gets. Not a failure when it says no: the CPU takes over.
    local vDoctorJson
    vDoctorJson=$(fRunAsUser "${vBinDir}/podcast-editor-mcp-server-doctor" --json --no-model-check --no-mcp-check 2>/dev/null || true)
    if echo "$vDoctorJson" | "$vPython" -c "import json, sys; aChecks = json.load(sys.stdin)['checks']; sys.exit(0 if any(dCheck['name'] == 'CUDA' and dCheck['status'] == 'ok' for dCheck in aChecks) else 1)" > /dev/null 2>&1; then
      vSmokeCuda="OK"
    else
      vSmokeCuda="unavailable; the CPU will be used (run podcast-editor-mcp-server-doctor)"
    fi
  else
    vSmokeCuda="not installed; the CPU will be used"
  fi

  if command -v audacity > /dev/null 2>&1; then
    vSmokeAudacity=$(dpkg-query -W -f='${Version}' audacity 2>/dev/null || echo "OK")
    if fRunAsUser "$vPython" -c "from podcast_editor_mcp_server.backends.audacity import fReadModuleStatus, cModuleEnabled; import sys; sys.exit(0 if fReadModuleStatus() == cModuleEnabled else 1)" > /dev/null 2>&1; then
      vSmokeModule="enabled"
    else
      vSmokeModule="not enabled"
    fi
  else
    vSmokeAudacity="not installed"
    vSmokeModule="n/a"
  fi

  # Build the MCP server object and list its tools, without serving anything.
  if fRunAsUser "${vBinDir}/podcast-editor-mcp-server" --print-tools > /dev/null 2>&1; then
    vSmokeServer="OK"
  else
    vSmokeServer="FAILED"
    vFailed="yes"
  fi
  return 0
}

fPrintSummary() {
  echo ""
  echo "podcast-editor-mcp-server installation"
  echo ""
  echo "Core:"
  echo "  Python:               ${vSmokePython}"
  echo "  Package import:       ${vSmokeImport}"
  echo "  CLI:                  ${vSmokeCli}"
  echo "  MCP server:           ${vSmokeServer}"
  echo "  FFmpeg:               ${vSmokeFfmpeg}"
  echo "  ffprobe:              ${vSmokeFfprobe}"
  echo "  Configuration:        ${vSmokeConfig}"
  echo "  Directories:          ${vSmokeDirectories}"
  echo "  Transcription:        ${vSmokeTranscription}"
  printf '  %-22s%s\n' "Model (${vWhisperModel}):" "${vSmokeModel}"
  echo "  CUDA:                 ${vSmokeCuda}"
  echo ""
  echo "Audacity:"
  echo "  Audacity:             ${vSmokeAudacity}"
  echo "  mod-script-pipe:      ${vSmokeModule}"
  echo ""
  echo "Everything is under ${vPrefix}:"
  echo "  Commands:             ${vBinDir}"
  echo "  Configuration:        ${vConfigDir}/config.yaml"
  echo "  Your configuration:   ${vUserConfigDir}"
  echo "  Models:               ${vModelDir}"
  echo "  Install log:          ${vLogFile}"
  echo ""
  echo "Commands (linked into ${cSymlinkDir}, so they are on PATH):"
  echo "  podcast-editor-mcp-server"
  echo "  podcast-editor-mcp-server-cli"
  echo "  podcast-editor-mcp-server-doctor"
  echo ""
  echo "Input:"
  echo "  ${vWorkspace}/input"
  echo ""
  echo "Output:"
  echo "  ${vWorkspace}/output"
  echo ""
  echo "Next steps, as ${vUser}:"
  echo "  podcast-editor-mcp-server-doctor"
  echo "  podcast-editor-mcp-server-cli demo"
  echo "  podcast-editor-mcp-server-cli process demo.wav"
  if [ "$vGroupAdded" = "yes" ]; then
    echo ""
    echo "${vUser} was just added to the '${cGroupName}' group. Sessions that were"
    echo "already open do not have it yet: log in again before downloading a new"
    echo "Whisper model. Everything else works right away."
  fi
  return 0
}

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

fMain() {
  if ! fParseArguments "$@"; then
    return 1
  fi
  # Every directory hangs off --prefix, so the layout can only be resolved
  # once the arguments are known.
  fResolveLayout

  if [ "$(id -u)" -ne 0 ]; then
    if ! fRunAsRoot "$@"; then
      return 1
    fi
    return 0
  fi

  fOpenLog
  fLog "=== install.sh started: $* ==="
  if ! fLocateSource; then
    return 1
  fi
  if ! fCheckDebian; then
    return 1
  fi
  if ! fResolveUser; then
    return 1
  fi

  fStep "Packages"
  if ! fDecideAudacity; then
    return 1
  fi
  if ! fInstallPackages; then
    return 1
  fi

  fStep "Source code"
  if ! fDownloadSource; then
    return 1
  fi

  fStep "Installation tree"
  if ! fEnsureGroup; then
    return 1
  fi
  if ! fCreateTree; then
    return 1
  fi

  fStep "Application"
  if ! fInstallApplication; then
    return 1
  fi
  if ! fDecideCuda; then
    return 1
  fi
  if ! fInstallCudaLibraries; then
    return 1
  fi
  if ! fInstallLaunchers; then
    return 1
  fi
  if ! fInstallSymlinks; then
    return 1
  fi

  fStep "Configuration"
  if ! fWriteSystemConfig; then
    return 1
  fi
  if ! fInstallSystemdUnit; then
    return 1
  fi

  fStep "Workspace for ${vUser}"
  if ! fCreateWorkspace; then
    return 1
  fi

  fStep "Transcription model"
  if ! fDownloadModel; then
    return 1
  fi

  fStep "Audacity scripting"
  if ! fEnableAudacityModule; then
    return 1
  fi
  if ! fRestartAudacityUnit; then
    return 1
  fi

  if [ "$vCheckOnly" = "yes" ]; then
    echo ""
    fSay "--check finished; nothing was changed."
    return 0
  fi

  fStep "Smoke tests"
  if ! fRunSmokeTests; then
    return 1
  fi
  fPrintSummary

  if [ "$vFailed" = "yes" ]; then
    echo ""
    fError "Some smoke tests failed. Run 'podcast-editor-mcp-server-doctor' for details."
    return 1
  fi
  return 0
}

# Bash owns stdin when this file is piped to it. Child commands must not read
# that stream; interactive confirmations explicitly read /dev/tty instead.
if ! fMain "$@" < /dev/null; then
  # A run that handed over to sudo leaves the report to the run it started.
  if [ "$vElevated" != "yes" ]; then
    echo "" >&2
    if [ "$vLogWritable" = "yes" ]; then
      echo "Installation did not complete. See ${vLogFile}." >&2
    else
      echo "Installation did not complete." >&2
    fi
  fi
  exit 1
fi
