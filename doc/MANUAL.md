# User manual

## Contents

- [Quick start](#quick-start)
- [Installation and updates](#installation-and-updates)
- [Run from source](#run-from-source)
- [Files and configuration](#files-and-configuration)
- [Edit a recording](#edit-a-recording)
- [Connect an MCP client](#connect-an-mcp-client)
- [Audacity and troubleshooting](#audacity-and-troubleshooting)
- [Uninstall](#uninstall)

## Quick start

Five steps take you from a Debian 13 machine to an edited episode. Each one links to the section with the details.

1. **Install**, as root. `--with-audacity` also installs Audacity, which the editable project needs; leave it out if you only want audio:

   ```bash
   curl -sL https://raw.githubusercontent.com/nipegun/podcast-editor-mcp-server/refs/heads/main/deploy/install.sh | bash -s -- --with-audacity
   ```

   The installer sets up the ordinary account that will process the recordings (see [Installation and updates](#installation-and-updates)) and ends with a summary of where everything is. On a machine with an NVIDIA GPU it also installs CUDA, so a long episode is transcribed in minutes instead of hours.

2. **Check the installation**, as that ordinary account, never as root:

   ```bash
   podcast-editor-mcp-server-doctor
   podcast-editor-mcp-server-cli workspace
   ```

   `doctor` lists what works and what is missing, and `workspace` shows your `input/` and `output/` directories. If the installer has just added your account to the application group, log in again first.

3. **Register the server in your MCP client**, as the same account. For Claude Code:

   ```bash
   claude mcp add-json --scope user podcast-editor-mcp-server '{"type":"stdio","command":"/usr/local/bin/podcast-editor-mcp-server","args":[],"timeout":14400000}'
   claude mcp list
   ```

   Codex and other clients are covered in [Connect an MCP client](#connect-an-mcp-client).

4. **Start Audacity** on your desktop if you want the editable `.aup3` project, or enable its service once (see [Audacity and troubleshooting](#audacity-and-troubleshooting)). Without it you still get the audio and the transcripts.

5. **Edit**: copy the recording to `input/`, start the client (`claude`) and ask, for example, "Edit episode-42.wav: remove the retakes and the long pauses, then master it". A long episode takes minutes on a GPU and far longer on a CPU. The results appear in `output/<name>/` (see [Results](#results)).

## Installation and updates

Use Debian 13 and an existing ordinary account with a home directory, the one that will process the recordings. Install `ca-certificates` and `curl` with APT before using the remote method if they are missing. As root:

```bash
curl -sL https://raw.githubusercontent.com/nipegun/podcast-editor-mcp-server/refs/heads/main/deploy/install.sh | bash
```

Options go after `bash -s --`. Examples use `alice`; replace it with your working account. With `-f` and `pipefail`, a failed download stops the pipeline instead of handing an error page to bash:

```bash
set -o pipefail
curl -fsSL https://raw.githubusercontent.com/nipegun/podcast-editor-mcp-server/refs/heads/main/deploy/install.sh | bash -s -- --user alice --yes
```

Without `--user`, the working account is the one that called sudo, the one that logged in (`su -` keeps it), the only account an earlier run set up, or the only ordinary account on the machine. With several and a terminal, the installer lists them and asks; without a terminal, as in `ssh host 'curl ... | bash'`, it stops and asks for `--user`. Root is never the working account, so a machine with no ordinary account needs one first (`adduser alice`). The same command also works from an ordinary account with sudo: the installer asks for the password and runs itself again as root, for that account. Without a terminal every question is answered no, so Audacity is only installed with `--with-audacity`.

Or, from the root of a downloaded and extracted repository:

```bash
bash deploy/install.sh --user alice --yes
```

Both forms accept the same arguments after the script name or `bash -s --`. The local method installs the adjacent source tree. The remote method fetches `main` and requires public GitHub access. A downloaded standalone installer also fetches the repository. Neither method requires Git, and sudo is only used when an ordinary account starts the installer.

Rerun either command to update; without `--user`, a rerun updates the account set up the first time. Download a newer repository before a local update. Existing settings, profiles, recordings, jobs, output and cached models are preserved. An update migrates nothing: a setting or profile key the new version no longer has stops that file from loading until you delete the key (the error names it), and job records or cached transcripts written by an older version are skipped or transcribed again. The virtual environment is reused when valid. A failed source download or extraction stops the installation before replacing the application; temporary source files are removed on exit.

Use `--with-audacity` for editable Audacity projects, `--no-model` to defer the initial model download, `--no-transcription` to omit transcription, or `--no-cuda` to skip GPU dependencies. `--whisper-model small` only selects the model to pre-download: change your editing profile too if you want it used for transcription. With `--yes`, Audacity is skipped if absent unless explicitly requested. `--no-apt` requires all system dependencies to exist, including curl, certificates, tar and gzip for remote sources.

Append `--check` to inspect planned actions without writing files, downloading sources or installing packages. It does not prove that a remote archive is reachable or valid. `--help` lists every option.

## Run from source

The repository stores its Python modules directly in `backend/`, including `server.py`, `cli.py`, `config.py` and the module subdirectories. The installed package is still named `podcast_editor_mcp_server`, and the installed commands keep their names.

For development, use an ordinary account with Python 3.13 or later and run these commands from the repository root:

```bash
python3 -m venv _/temp/venv
_/temp/venv/bin/python3 -m pip install --break-system-packages -e .
export PODCAST_EDITOR_MCP_SERVER_PREFIX="$PWD/_/temp/prefix"
_/temp/venv/bin/python3 -m podcast_editor_mcp_server.cli --help
```

The editable installation maps the package name to `backend/`, so subsequent edits to Python files are used by this environment. Rerun the editable installation after changing package metadata or adding module directories. Keep using the installer above for deployment; it installs an independent copy and does not depend on the downloaded source tree afterward.

## Files and configuration

Default installation prefix: `/opt/podcast-editor-mcp-server`. `--prefix` changes it; installed launchers pass the matching prefix to the application.

| Location under the prefix | Purpose |
| --- | --- |
| `venv/`, `bin/` | Application, dependencies and launchers |
| `etc/config.yaml` | Existing system configuration is kept during updates |
| `etc/users/alice/config.yaml` | Optional personal runtime configuration |
| `etc/profiles/`, `etc/users/alice/profiles/` | Shared and personal editing profiles |
| `var/workspace/alice/input/` | Original recordings |
| `var/workspace/alice/output/` | Edited audio, reports, transcripts and optional projects |
| `var/workspace/alice/jobs/`, `work/`, `cache/`, `logs/` | Job state, work files, caches and logs |
| `var/models/` | Shared Whisper models |
| `var/log/install.log` | Installation log, readable by root |
| `share/source-directory` or `share/source-url` | Local source path or remote archive URL |
| `_/temp/source.*/` | Source staging (the downloaded archive or a copy of local sources), removed on installer exit |

Runtime configuration uses the explicit configuration path if supplied, otherwise the first existing file from `PODCAST_EDITOR_MCP_SERVER_CONFIG`, personal settings and system settings. Missing fields use built-in defaults. A personal runtime file is not merged with the system file. Profiles support `extends` and personal overrides; keep custom profiles outside the virtual environment.

The `jobs` section bounds background work: `max_concurrent` jobs run at once (2 by default) and up to `max_queued` more wait for a free worker (8 by default). A request beyond that is refused, with a message to try again when a job finishes, and nothing is recorded for it. Each time the server starts, finished job records older than `retention_days` (90 by default) are deleted, as `podcast-editor-mcp-server-cli jobs --purge` does; their artifacts are kept.

Environment overrides use these exact names in MCP client settings and scripts:

| Variable | Purpose |
| --- | --- |
| `PODCAST_EDITOR_MCP_SERVER_PREFIX` | Installation root |
| `PODCAST_EDITOR_MCP_SERVER_CONFIG` | Runtime configuration file |
| `PODCAST_EDITOR_MCP_SERVER_WORKSPACE` | Workspace root |
| `PODCAST_EDITOR_MCP_SERVER_INPUT` | Directory the original recordings are read from, instead of the workspace's `input/` |
| `PODCAST_EDITOR_MCP_SERVER_OUTPUT` | Directory the results are written to, instead of the workspace's `output/` |
| `PODCAST_EDITOR_MCP_SERVER_PROFILES_DIR` | Additional profile directory |
| `PODCAST_EDITOR_MCP_SERVER_PROFILE` | Active editing profile |
| `PODCAST_EDITOR_MCP_SERVER_MODEL_DIR` | Whisper model directory |
| `PODCAST_EDITOR_MCP_SERVER_FFMPEG` | FFmpeg executable |
| `PODCAST_EDITOR_MCP_SERVER_FFPROBE` | ffprobe executable |

A relative value in `PODCAST_EDITOR_MCP_SERVER_WORKSPACE`, `PODCAST_EDITOR_MCP_SERVER_INPUT`, `PODCAST_EDITOR_MCP_SERVER_OUTPUT` or `PODCAST_EDITOR_MCP_SERVER_PROFILES_DIR` starts from the folder in `CLAUDE_PROJECT_DIR`, which Claude Code sets for the servers it starts, and otherwise from the directory the server was started in.

## Edit a recording

Run processing commands as the ordinary account. Copy a recording to that account's `input/` directory; use `podcast-editor-mcp-server-cli workspace` to see the exact paths.

```bash
podcast-editor-mcp-server-cli probe episode.wav
podcast-editor-mcp-server-cli analyze episode.wav
podcast-editor-mcp-server-cli process episode.wav --dry-run
podcast-editor-mcp-server-cli process episode.wav
```

The dry run prepares and simulates the plan without rendering audio. Normal processing writes results under `output/`. Existing output is protected unless you explicitly pass `--overwrite`. Use `--no-transcribe` and `--no-audacity` on processing commands when those optional components are not wanted.

Two runs that write the same artifacts never overlap: while one is writing `output/<name>/`, another run with the same name is refused at once, even with `--overwrite`; wait for the first one or choose another output name. Without `--overwrite`, a file that appears in the output while a run is working is not replaced either.

Recordings are read only from local files in the accepted audio formats: WAV, FLAC, MP3, M4A/MP4, AAC, Ogg (Vorbis and Opus), WavPack, AIFF and CAF. A file whose content is something else, such as a playlist or a concat list saved with an audio extension, is refused. Sample-level analysis, transcription and voice detection hold at most 3 GiB of decoded audio in memory, more than four hours at 48 kHz; a longer recording is refused by those steps, and analysis reports it as a warning. A compressed recording is decoded once into the work directory for analysis, and the decode is refused when it would leave less than 2 GiB free on that disk; those copies are kept under 16 GiB in total, the least recently used being deleted first.

```bash
podcast-editor-mcp-server-cli profiles
podcast-editor-mcp-server-cli profiles podcast-default
podcast-editor-mcp-server-cli process episode.wav --profile podcast-tight
```

Editing and mastering can also be two separate steps. `process --no-mastering` renders only the cuts, without filters, loudness normalisation, limiter or fades, and writes `output/<name>/<name>.wav`. `master` then applies the profile's clean-up and mastering chain to that file and writes `<name>.master.wav` and the other delivery formats next to it:

```bash
podcast-editor-mcp-server-cli process episode.wav --no-mastering
podcast-editor-mcp-server-cli master episode/episode.wav
```

When the true-peak ceiling keeps FFmpeg's loudness normalisation short of the profile's target, the master is measured and the missing gain is added in front of the limiter, up to twice, until it is within 0.2 LU of the target.

`podcast-editor-mcp-server-cli jobs` lists jobs. `plan`, `validate`, `simulate` and `execute` provide explicit control over saved plans; consult each command's `--help` for its arguments.

### Results

Every edit writes its files to `output/<name>/`, where `<name>` is the output name you chose or, by default, the name of the recording without its extension. The recording in `input/` is never modified.

| File | What it is |
| --- | --- |
| `<name>.master.wav`, `<name>.mp3` | The edited and mastered episode, in the profile's delivery formats |
| `<name>.wav` | The edited episode without mastering, when editing and mastering are separate steps; mastering it later adds the two files above |
| `<name>.aup3` | Editable Audacity project, only when Audacity was reachable |
| `<name>.transcript.json`, `<name>.transcript.txt`, `<name>.srt` | Transcript of the edited episode |
| `<name>.labels.txt` | The same transcript as an Audacity label track |
| `<name>.edit-plan.json`, `<name>.edit-report.json` | The cuts that were made and the report of the edit |
| `<name>.analysis.json`, `<name>.candidates.json`, `<name>.job.json`, `<name>.job.log` | Measurements, detected candidates, job record and log |

Which audio and transcript formats appear depends on the profile's `output` section; the presets deliver WAV and MP3, and the transcript as JSON, text and SRT.

## Connect an MCP client

Run the MCP client under the same ordinary account, not with sudo. The client starts the server over stdio. There is no HTTP listener or network port.

Transcribing, preparing or finishing a long episode takes minutes on a GPU and far longer on a CPU, more than these clients wait for a tool by default, so each configuration below raises that limit to four hours.

**Claude Code.** Register the server for the whole account:

```bash
claude mcp add-json --scope user podcast-editor-mcp-server '{"type":"stdio","command":"/usr/local/bin/podcast-editor-mcp-server","args":[],"timeout":14400000}'
```

`timeout` is in milliseconds and applies only to this server. Setting `MCP_TOOL_TIMEOUT` instead is not enough: Claude Code still stops a call to a local server after 30 minutes without progress messages, and this server sends none. Start a new `claude` session to load the server; `claude mcp list` shows whether it connects.

**Codex.** Add to `~/.codex/config.toml`:

```toml
[mcp_servers.podcast-editor-mcp-server]
command = "/usr/local/bin/podcast-editor-mcp-server"
args = []
startup_timeout_sec = 30
tool_timeout_sec = 14400
env_vars = ["DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR"]
```

`codex mcp add podcast-editor-mcp-server -- /usr/local/bin/podcast-editor-mcp-server` writes the first two lines; add the others below them. Codex starts the server with an almost empty environment, and `env_vars` passes it the desktop variables it needs to start Audacity. `codex mcp list` shows the server.

**Other clients.** For clients using an `mcpServers` JSON configuration, add the following, and raise their tool timeout if they have one:

```json
{
  "mcpServers": {
    "podcast-editor-mcp-server": {
      "command": "/usr/local/bin/podcast-editor-mcp-server",
      "args": []
    }
  }
}
```

Copy the recordings to the account's `input/` directory (`podcast-editor-mcp-server-cli workspace` shows it) and ask the assistant something like "Edit episode-42.wav, remove the silences and filler words and master it for podcast". The results appear in `output/`.

Begin with the `get_server_info` tool to inspect available features and paths. The `prepare_episode_edit` and `finish_episode_edit` tools support the episode editing workflow. With `mastering=false`, `finish_episode_edit` delivers the edited audio unmastered (`<name>.wav`), and the `master_audio` tool cleans it up and masters it afterwards.

`transcribe_audio` accepts a `model` such as `small` or `large-v3`, or an `organisation/name` repository, never a path. A model other than the profile's is used only if it is already downloaded: a request never makes the server download one. At most two Whisper models stay loaded at once. Validation answers list at most 100 errors and 100 warnings; `error_count` and `warning_count` give the totals.

## Audacity and troubleshooting

The editable project (`<name>.aup3`) is only created when Audacity is running and answering on its script pipes at the moment the edit is rendered. The installer enables Audacity's `mod-script-pipe` module for the working account; `podcast-editor-mcp-server-cli audacity-status --enable-module` does the same by hand, with Audacity closed. Then, before editing, either open Audacity on your desktop and leave it open, or, after installing with `--with-audacity`, enable the optional user service once as the working user, so that a hidden Audacity is always running:

```bash
systemctl --user enable --now podcast-editor-mcp-server-audacity.service
```

When Audacity cannot be reached, the edit still delivers the audio and the transcripts and reports `Audacity project not created` as a warning. The `create_audacity_project` tool builds the project later around the edited audio, once Audacity is running.

Reinstalling or updating restarts that unit when it is running, so Audacity runs the new version. Uninstalling stops and disables it, and removes the links that enabled it.

Audacity stores its own preferences in the user's home directory. `podcast-editor-mcp-server-cli audacity-status` reports its integration status. Processing must not run as root.

The server drives Audacity through its script pipes in `/tmp` only when they are FIFOs owned by your account and closed to other users, as Audacity creates them. Pipes created by another account are refused, and so is a reply that never ends.

Run `podcast-editor-mcp-server-doctor` to inspect dependencies. Review `var/log/install.log` as root if installation fails. If the installer just added your account to the application group, log in again before downloading models from an existing session. A 404 while downloading means the repository, branch or file is not publicly available at the documented URL.

## Uninstall

From a downloaded repository, as root:

```bash
bash deploy/uninstall.sh --user alice --yes
```

This keeps recordings, output, models and configuration. Add `--purge-config` to remove shared configuration. To permanently remove all workspaces, personal settings, models and shared configuration as well:

```bash
bash deploy/uninstall.sh --user alice --purge-data --yes
```

System packages such as FFmpeg and Audacity are retained.
