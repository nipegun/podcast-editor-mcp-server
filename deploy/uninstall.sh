#!/bin/bash

# podcast-editor-mcp-server uninstaller for Debian 13 (trixie).
#
# Removes the application, its virtualenv, the launchers and the systemd unit.
#
# It never touches your podcasts. Everything lives under one prefix, so
# this script removes the parts of that prefix it installed and leaves the
# rest alone: recordings, edited masters, transcripts, jobs, personal settings
# and downloaded models stay exactly where they are unless you ask for them
# with --purge-data, and that asks for confirmation first. The prefix itself is
# only removed once nothing of yours is left inside it.
#
# APT packages (ffmpeg, audacity, xvfb) are left installed: other software on
# the machine probably uses them.

set -euo pipefail

cProductName="podcast-editor-mcp-server"
cPrefixDefault="/opt/podcast-editor-mcp-server"
cSymlinkDir="/usr/local/bin"
cSystemdUserDir="/usr/lib/systemd/user"
cGroupName="podcast-editor-mcp-server"

cLaunchers="podcast-editor-mcp-server podcast-editor-mcp-server-cli podcast-editor-mcp-server-doctor"

vPrefix="$cPrefixDefault"
vUser=""
vPurgeData="no"
vPurgeConfig="no"
vDryRun="no"
vAssumeYes="no"
vRemoved=0

# Derived from vPrefix once the arguments have been parsed.
vBinDir=""
vConfigDir=""
vSystemdDir=""
vVarDir=""
vLogDir=""
vModelDir=""
vWorkspaceBase=""

fResolveLayout() {
  vBinDir="${vPrefix}/bin"
  vConfigDir="${vPrefix}/etc"
  vSystemdDir="${vPrefix}/lib/systemd/user"
  vVarDir="${vPrefix}/var"
  vLogDir="${vVarDir}/log"
  vModelDir="${vVarDir}/models"
  vWorkspaceBase="${vVarDir}/workspace"
  return 0
}

fSay() {
  echo "$*"
}

fWarn() {
  echo "WARNING: $*" >&2
}

fError() {
  echo "ERROR: $*" >&2
}

fUsage() {
  echo "Usage: sudo ./podcast-editor-mcp-server/deploy/uninstall.sh [options]"
  echo ""
  echo "Removes the podcast-editor-mcp-server application. Your podcasts are kept."
  echo ""
  echo "Options:"
  echo "  --user NAME       User whose workspace to consider"
  echo "                    (default: the user who invoked sudo)"
  echo "  --prefix DIR      Installation directory (default: ${cPrefixDefault})"
  echo "  --purge-config    Also remove the system configuration and profiles"
  echo "  --purge-data      Also DELETE every workspace in the prefix, the"
  echo "                    personal settings and the downloaded models."
  echo "                    This destroys recordings and edited output."
  echo "  --dry-run         Show what would be removed, remove nothing"
  echo "  --yes             Do not ask for confirmation"
  echo "  --help            Show this text"
  echo ""
  echo "APT packages (ffmpeg, audacity, xvfb) are never removed."
}

fCleanup() {
  :
}

trap fCleanup EXIT

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
      --purge-config)
        vPurgeConfig="yes"
        shift
        ;;
      --purge-data)
        vPurgeData="yes"
        vPurgeConfig="yes"
        shift
        ;;
      --dry-run)
        vDryRun="yes"
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

fConfirm() {
  if [ "$vAssumeYes" = "yes" ]; then
    return 0
  fi
  local vAnswer=""
  echo -n "$1 [y/N] "
  if [ -r /dev/tty ]; then
    read -r vAnswer < /dev/tty || true
  fi
  case "$vAnswer" in
    y|Y|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

fCheckRoot() {
  if [ "$(id -u)" -ne 0 ]; then
    fError "This uninstaller must run as root: sudo ./podcast-editor-mcp-server/deploy/uninstall.sh"
    return 1
  fi
  return 0
}

fResolveUser() {
  if [ -z "$vUser" ]; then
    if [ -n "${SUDO_USER:-}" ] && [ "${SUDO_USER}" != "root" ]; then
      vUser="$SUDO_USER"
    else
      local vLogname
      vLogname=$(logname 2>/dev/null || true)
      if [ -n "$vLogname" ] && [ "$vLogname" != "root" ]; then
        vUser="$vLogname"
      fi
    fi
  fi
  # The user is only needed for their systemd unit: the links that enable it
  # are the only thing of ours in their home directory.
  if [ -n "$vUser" ] && ! id "$vUser" > /dev/null 2>&1; then
    vUser=""
  fi
  return 0
}

fRemovePath() {
  local vPath="$1"
  local vLabel="$2"
  # -e follows symbolic links, so a link whose target is already gone would
  # look absent and be left behind. -L catches exactly that case.
  if [ ! -e "$vPath" ] && [ ! -L "$vPath" ]; then
    return 0
  fi
  if [ "$vDryRun" = "yes" ]; then
    fSay "  would remove ${vLabel}: ${vPath}"
    return 0
  fi
  rm -rf "$vPath"
  fSay "  removed ${vLabel}: ${vPath}"
  vRemoved=$((vRemoved + 1))
  return 0
}

fStopAudacityUnit() {
  local vUnit="${vSystemdDir}/podcast-editor-mcp-server-audacity.service"
  local vLink="${cSystemdUserDir}/podcast-editor-mcp-server-audacity.service"
  if [ ! -e "$vUnit" ] && [ ! -e "$vLink" ] && [ ! -L "$vLink" ]; then
    return 0
  fi
  if [ -n "$vUser" ] && command -v systemctl > /dev/null 2>&1 && [ "$vDryRun" != "yes" ]; then
    # Root reaches the user's systemd through --machine=<user>@; runuser does
    # not, because that user's bus is not in its environment. It fails
    # harmlessly when the user's systemd is not running.
    systemctl --user --machine="${vUser}@" disable --now podcast-editor-mcp-server-audacity.service > /dev/null 2>&1 || true
  fi
  fRemoveUserUnitLinks
  fRemovePath "$vLink" "systemd unit link"
  fRemovePath "$vUnit" "systemd unit"
  return 0
}

fRemoveUserUnitLinks() {
  # The links 'systemctl --user enable' made into the prefix stay behind when
  # the user's systemd was not running to disable the unit.
  if [ -z "$vUser" ]; then
    return 0
  fi
  local vHome
  vHome=$(getent passwd "$vUser" | cut -d: -f6)
  local vUserUnits="${vHome}/.config/systemd/user"
  if [ -z "$vHome" ] || [ ! -d "$vUserUnits" ]; then
    return 0
  fi
  local vLink
  while IFS= read -r vLink; do
    fRemovePath "$vLink" "user unit link"
  done < <(find "$vUserUnits" -maxdepth 2 -type l -lname "${vSystemdDir}/*")
  return 0
}

fRemoveSymlinks() {
  # Only links that point back into this prefix. Anything else on PATH belongs
  # to somebody else.
  local vLauncher
  local vLink
  local vTarget
  for vLauncher in $cLaunchers; do
    vLink="${cSymlinkDir}/${vLauncher}"
    if [ -L "$vLink" ]; then
      vTarget=$(readlink -f "$vLink" 2>/dev/null || true)
      case "$vTarget" in
        "${vPrefix}"/*)
          fRemovePath "$vLink" "command link"
          ;;
        *)
          fWarn "${vLink} points outside ${vPrefix}; leaving it alone."
          ;;
      esac
    fi
  done
  return 0
}

fRemoveApplication() {
  fSay "Application:"
  # The unit goes first: stopping it needs the launcher it points at, and
  # removing lib/ would leave systemd holding a dangling link.
  fStopAudacityUnit
  fRemoveSymlinks
  fRemovePath "$vBinDir" "launchers"
  fRemovePath "${vPrefix}/venv" "virtualenv"
  fRemovePath "${vPrefix}/lib" "library files"
  fRemovePath "${vPrefix}/share" "installation metadata"
  fRemovePath "$vLogDir" "install log"
  return 0
}

fRemoveConfiguration() {
  local vConfigFile="${vConfigDir}/config.yaml"
  local vProfiles="${vConfigDir}/profiles"

  if [ "$vPurgeConfig" != "yes" ]; then
    if [ -e "$vConfigFile" ] || [ -d "$vProfiles" ]; then
      fSay "Keeping ${vConfigDir} (pass --purge-config to remove it)."
    fi
    return 0
  fi
  fSay "Configuration:"
  fRemovePath "$vConfigFile" "system configuration"
  fRemovePath "$vProfiles" "system profiles"
  return 0
}

fRemoveUserData() {
  local vUserConfigs="${vConfigDir}/users"

  if [ "$vPurgeData" != "yes" ]; then
    fSay ""
    fSay "Your data has been left alone:"
    if [ -d "$vWorkspaceBase" ]; then
      fSay "  ${vWorkspaceBase}"
    fi
    if [ -d "$vUserConfigs" ]; then
      fSay "  ${vUserConfigs}"
    fi
    if [ -d "$vModelDir" ]; then
      fSay "  ${vModelDir}"
    fi
    fSay ""
    fSay "To delete it too: sudo ./podcast-editor-mcp-server/deploy/uninstall.sh --purge-data"
    return 0
  fi

  fSay ""
  fSay "--purge-data will permanently delete:"
  if [ -d "$vWorkspaceBase" ]; then
    fSay "  ${vWorkspaceBase}   (recordings, edited masters, transcripts, jobs)"
  fi
  if [ -d "$vUserConfigs" ]; then
    fSay "  ${vUserConfigs}   (every user's own settings and profiles)"
  fi
  if [ -d "$vModelDir" ]; then
    fSay "  ${vModelDir}   (downloaded Whisper models)"
  fi
  fSay ""

  if [ "$vDryRun" != "yes" ]; then
    if ! fConfirm "This cannot be undone. Delete them?"; then
      fSay "Keeping your data."
      return 0
    fi
  fi

  fSay "User data:"
  fRemovePath "$vWorkspaceBase" "workspaces"
  fRemovePath "$vUserConfigs" "user configuration"
  fRemovePath "$vModelDir" "model cache"
  return 0
}

fRemoveEmptyDirectories() {
  # rmdir refuses to delete a directory that still holds something, which is
  # exactly the safety net wanted here: the prefix disappears only once
  # nothing of yours is left in it.
  if [ "$vDryRun" = "yes" ]; then
    return 0
  fi
  local vDirectory
  for vDirectory in "$vVarDir" "$vConfigDir" "$vPrefix"; do
    if [ -d "$vDirectory" ] && rmdir "$vDirectory" 2>/dev/null; then
      fSay "  removed empty directory: ${vDirectory}"
      vRemoved=$((vRemoved + 1))
    fi
  done
  return 0
}

fRemoveGroup() {
  # Only once there is nothing left for it to grant access to.
  if [ "$vPurgeData" != "yes" ] || [ "$vDryRun" = "yes" ]; then
    return 0
  fi
  if [ -d "$vPrefix" ]; then
    return 0
  fi
  if ! getent group "$cGroupName" > /dev/null 2>&1; then
    return 0
  fi
  if groupdel "$cGroupName" 2>/dev/null; then
    fSay "  removed group: ${cGroupName}"
    vRemoved=$((vRemoved + 1))
  else
    fWarn "Could not remove the '${cGroupName}' group; it is still somebody's primary group."
  fi
  return 0
}

fMain() {
  if ! fParseArguments "$@"; then
    return 1
  fi
  fResolveLayout
  if ! fCheckRoot; then
    return 1
  fi
  if ! fResolveUser; then
    return 1
  fi

  if [ "$vDryRun" = "yes" ]; then
    fSay "Dry run: nothing will be removed."
    fSay ""
  fi

  if [ ! -e "$vPrefix" ] && [ ! -e "${cSymlinkDir}/podcast-editor-mcp-server" ]; then
    fWarn "No installation found at ${vPrefix}."
  fi

  if ! fRemoveApplication; then
    return 1
  fi
  if ! fRemoveConfiguration; then
    return 1
  fi
  if ! fRemoveUserData; then
    return 1
  fi

  fRemoveEmptyDirectories
  fRemoveGroup

  fSay ""
  if [ "$vDryRun" = "yes" ]; then
    fSay "Dry run finished."
  else
    fSay "${cProductName} removed (${vRemoved} item(s))."
    fSay "APT packages such as ffmpeg and audacity were left installed."
  fi
  return 0
}

if ! fMain "$@"; then
  echo "" >&2
  echo "Uninstall did not complete." >&2
  exit 1
fi
