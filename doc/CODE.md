# Code map

## Contents

- [Architecture](#architecture)
- [Modules](#modules)
- [Key symbols](#symbols)
- [Call paths](#flows)
- [Entry points](#entry-points)
- [Impact analysis](#impact)
- [Extension points](#extensions)

<a id="architecture"></a>
## Architecture

The Python package is `podcast_editor_mcp_server`; its distribution and installed commands are named `podcast-editor-mcp-server`. The CLI and MCP tools share `PodcastPipeline`, keeping editing, validation and rendering behavior consistent. The MCP transport is stdio, with no HTTP listener. FFmpeg renders audio; faster-whisper and Audacity are optional integrations.

The source package lives directly in `backend/`: entry modules and the `audio/`, `mcp/`, `models/` and other module directories share that root. `pyproject.toml` maps `podcast_editor_mcp_server` to `backend/` through `tool.setuptools.package-dir`; setuptools discovers its implicit namespace subpackages under that mapping. Imports, console entry points and the packaged `data/profiles/` keep their installed names. This removes a redundant source directory without changing the installed package layout or introducing `__init__.py` files.

Source execution uses an editable installation in `_/temp/venv/`, as described in the manual. The virtual environment then resolves the installed package name to these source files. Reinstall the editable package after changing metadata or adding module directories. Deployment continues to use a regular installation, so removing the downloaded sources is safe.

The Debian installer owns a self-contained prefix with a virtual environment, launchers, settings, models and workspaces. Only command and systemd discovery links live outside that tree, in addition to the system group and APT dependencies. Audacity manages its own user preferences.

The installer captures `BASH_SOURCE[0]` before entering functions. If the script's parent source tree has package metadata, it validates and uses that tree. Otherwise it downloads the public GitHub `main` archive. It does not use an arbitrary working directory as a substitute for local sources. A malformed local tree with `pyproject.toml` fails validation instead of silently using remote code. Piped into bash (`curl -sL .../refs/heads/main/deploy/install.sh | bash`), `BASH_SOURCE[0]` is empty, so remote mode is chosen.

Without `--user`, `fResolveUser` takes `SUDO_USER`, `PKEXEC_UID` or `logname`; from a root shell that names nobody, it takes the only account with a workspace under the prefix (an earlier run's), otherwise the only ordinary account (`fListOrdinaryUsers`: the UID range of `/etc/login.defs`, a home directory and a login shell). Several candidates are offered on the terminal by `fChooseUser`, and refused without one or with `--yes`. Started by an ordinary user, `fRunAsRoot` validates sudo first and runs the installer again as root for that user: the same file, or, when it was piped, the same URL fetched again. `fHasTerminal` tells whether `/dev/tty` really opens, so a run without a terminal answers no instead of failing on it.

Downloaded sources are staged at `<prefix>/_/temp/source.*/`, after download dependencies have been installed. curl must finish successfully before tar extraction and source validation. Installation uses the ordinary non-editable pip path, so deleting the staging tree on exit leaves the installed package usable. Provenance is recorded as `share/source-url` for remote sources or `share/source-directory` for local sources, removing the obsolete counterpart. Local sources are copied into the same staging area before pip installs them, because pip builds inside the tree it is given and would leave `build/` and `*.egg-info` in the sources.

Reinstallation preserves existing settings, profiles and user data, reuses a working virtual environment, and checks cached models. `--check` does not download or extract sources and does not create or append to the install log. The call to `fMain` reads stdin from `/dev/null`, preventing child commands from consuming a piped script; confirmations read `/dev/tty`. When the working user's Audacity unit is running, the installer reloads that user's systemd and restarts the unit through `systemctl --user --machine=<user>@`, so it runs the new version; `runuser` cannot reach the user's bus. The uninstaller disables the unit the same way and removes the links left in `~/.config/systemd/user`.

An MCP client configured inside a folder can give the server that folder's directories with relative paths in `PODCAST_EDITOR_MCP_SERVER_INPUT`, `PODCAST_EDITOR_MCP_SERVER_OUTPUT` and `PODCAST_EDITOR_MCP_SERVER_PROFILES_DIR`. `fEnvironmentLocation` resolves them from `CLAUDE_PROJECT_DIR` when Claude Code sets it, otherwise from the server's working directory, the folder the client was started in. Only the input and output roles move into the folder: work files, jobs, the transcript cache and logs stay in the workspace under the prefix, so one cached transcript serves every folder.

Listing a profile only reads its file name. `fCheckProfiles` loads every profile, so `get_server_info` and `reload_configuration` report the ones that fail as `profile_errors`, with the wrong keys. A profile is usable only when it is absent from that list.

Persisted documents are read back with the same strict models that write them. Job records, cached transcripts, edit plans, the runtime configuration and profiles reject unknown keys, edit plans must carry the current schema version, and there is no migration code. A file written with a field that no longer exists fails validation: an unreadable cached transcript is discarded and transcribed again, an old job record is skipped in listings, and a removed configuration or profile key has to be deleted by hand.

Mastering normalises loudness with FFmpeg's `loudnorm`. When the true-peak ceiling forbids its linear gain, it falls back to dynamic normalisation and can land an LU short of the target, so `ExecutionEngine._mCorrectLoudness` measures the render and adds the gap as gain in front of the limiter, at most twice and only when a limiter protects the ceiling.

Everything a caller controls is bounded before it costs memory, disk or a process. FFmpeg and ffprobe open media only through `fInputArguments`: the `file` protocol and the demuxers of the accepted audio containers (`cInputFormats`), so a playlist or a concat script saved under an audio extension cannot make them open files or URLs the path guard never saw. Decoded samples are held to `cMaxDecodedBytes` of mono float32 (3 GiB), checked before libsndfile reads and enforced while FFmpeg streams, and long signals are reduced a chunk at a time. The copy decoded for a file libsndfile cannot read is named after the source's identity (`fSourceIdentity`: device, inode, size, modification and change times), sized from the stream before decoding, refused when it would leave less than 2 GiB free, decoded under a temporary name and renamed when complete; analysis scratch directories are kept under 16 GiB, least recently used first.

An edit plan holds at most 200,000 operations; a plan over `safety.max_operations` is rejected before any other check, contradictory filters are compared once, and diagnostics are returned 100 at a time with their full counts. Background jobs are admitted only while `jobs.max_concurrent` running plus `jobs.max_queued` waiting leave room, a finished job leaves the table at once, and job records older than `jobs.retention_days` are purged when the pipeline starts. A Whisper model named in a request must be a model name, not a path, and one other than the profile's is loaded from the cache only; at most two models stay loaded. Audacity's script pipes are used only when they are FIFOs owned by the user and closed to everybody else, and only the very ones inspected; what is read from them is bounded. A run holds its artifact directory (`PodcastPipeline.mReserveOutput`, an `flock` under `<work>/locks`) from its existence checks to its last write, so a second run with the same name is refused at once. Without `overwrite`, exports and JSON artifacts are published by hard link, or by a last check where the filesystem has none, and never replace a file that appeared in the meantime.

<a id="modules"></a>
## Modules

| Module | Path | Responsibility | Depends on | Used by |
| --- | --- | --- | --- | --- |
| Installer | `deploy/install.sh` | Local/remote source resolution, idempotent deployment | Bash, APT, pip; curl/tar for remote sources | Administrator |
| User service | `deploy/systemd/podcast-editor-mcp-server-audacity.service` | Optional headless Audacity startup | Installed CLI, systemd user manager | Installer, working user |
| Uninstaller | `deploy/uninstall.sh` | Application removal and optional data purge | Prefix layout, systemd | Administrator |
| Packaging | `pyproject.toml` | Map `backend/` to the installed namespace; dependencies, metadata and command entry points | setuptools, README | pip |
| Server / CLI / diagnostics | `backend/{server,cli,doctor}.py` | Protocol, command and diagnostic entry points | MCP SDK, configuration, pipeline | Users and MCP clients |
| MCP adapters | `backend/mcp/` | Tools, resources, prompts and server state | Pipeline, job store | Server |
| Configuration | `backend/config.py` | Validated runtime settings, profiles and workspace | Pydantic, YAML, path guard | CLI, server, pipeline |
| Identity, errors and logging | `backend/{version,errors,logging_setup}.py` | Product title, exception hierarchy and logger namespace | Python standard library | All runtime modules |
| Pipeline | `backend/pipeline.py` | Coordinate analysis, plans, jobs and rendering | Modules below | CLI and MCP adapters |
| Audio / transcription / analysis | `backend/{audio,transcription,analysis,vad}/` | Measurements, transcripts and edit candidates | FFmpeg, optional models, caches | Pipeline |
| Planning / validation / execution | `backend/{planning,validation,execution}/` | Build, validate and apply edit plans | Models, backends | Pipeline |
| Backends | `backend/backends/` | FFmpeg rendering and Audacity integration | External executables | Pipeline, execution |
| Jobs / security / models / utilities | `backend/{jobs,security,models,util}/` | Persistence, path restrictions and shared types/helpers | Filesystem, Pydantic | Pipeline and adapters |
<a id="symbols"></a>
## Key symbols

| Symbol | File:line | Responsibility |
| --- | --- | --- |
| `fLocateSource` | `deploy/install.sh:725` | Choose the adjacent local source tree or remote mode |
| `fValidateSource` | `deploy/install.sh:752` | Check package identity, `backend/server.py`, the built-in profile and the service template |
| `fDownloadSource` | `deploy/install.sh:768` | Download, extract and validate the GitHub archive |
| `fStageLocalSource` | `deploy/install.sh:808` | Copy local sources into the staging area so pip never builds inside them |
| `fInstallApplication` | `deploy/install.sh:830` | Reuse/create the virtual environment, install and record source origin |
| `fCleanup` | `deploy/install.sh:210` | Remove the temporary source copy on exit |
| `fResolveUser` | `deploy/install.sh:458` | Choose the working user when `--user` is absent |
| `fRunAsRoot` | `deploy/install.sh:307` | Run the installer again through sudo for an ordinary user |
| `fMain` (installer) | `deploy/install.sh:1421` | Coordinate deployment stages |
| `fInstallSystemdUnit` | `deploy/install.sh:1086` | Render and link the correctly named Audacity unit |
| `fRestartAudacityUnit` | `deploy/install.sh:1208` | Restart the working user's running Audacity unit on the new version |
| `fStopAudacityUnit` | `deploy/uninstall.sh:205` | Disable the user's unit through `--machine=<user>@` and remove its links |
| `ExecutionEngine._mCorrectLoudness` | `backend/execution/engine.py:331` | Add the loudness `loudnorm` left short as gain in front of the limiter |
| `fBuildServer` | `backend/server.py:80` | Build the MCP server and register adapters; publish the repository URL |
| `fLoadRuntimeConfig` | `backend/config.py:911` | Load the selected runtime configuration |
| `Workspace` | `backend/config.py:933` | Resolve the workspace roles, honouring the input and output overrides |
| `fBaseDirectory` | `backend/config.py:98` | `CLAUDE_PROJECT_DIR` when set, otherwise the working directory |
| `fEnvironmentLocation` | `backend/config.py:112` | Resolve a path from the environment; a relative one starts from `fBaseDirectory` |
| `fCheckProfiles` | `backend/config.py:782` | Load every discoverable profile; report failures, wrong keys and name mismatches |
| `PodcastPipeline.mProcess` | `backend/pipeline.py:591` | Run the processing pipeline and produce artifacts |
| `fRegisterTools` | `backend/mcp/tools.py:133` | Register public MCP tools |
| `PodcastEditorMcpServerError` | `backend/errors.py:13` | Base exception; generic code `podcast_editor_mcp_server_error` |
| `fRegisterPrompts` | `backend/mcp/prompts.py:16` | Register workflows, including `troubleshoot_podcast_editor_mcp_server` |
| `cLoggerRoot` | `backend/logging_setup.py:24` | Logger namespace `podcast_editor_mcp_server` |
| `fInputArguments` | `backend/backends/ffmpeg.py:79` | Restrict a media input to local files and the accepted audio demuxers |
| `FfmpegBackend.mExport` | `backend/backends/ffmpeg.py:611` | Encode a delivery format beside the target and publish it, replacing only with `overwrite` |
| `fDecodeMono` | `backend/audio/samples.py:158` | Decode through FFmpeg to mono float32, read in pieces under `cMaxDecodedBytes` |
| `fReadSamples` | `backend/audio/samples.py:123` | Read with libsndfile, mixing to mono block by block after checking the budget |
| `fEnsureReadableCopy` | `backend/audio/analysis.py:111` | Decoded copy named after the source's identity, sized against the free space, published when complete |
| `_fSweepScratchDirs` | `backend/mcp/tools.py:1419` | Age and size quota (least recently used first) of the analysis scratch directories |
| `fValidateEditPlan` | `backend/validation/validator.py:155` | Validate a plan; stop at the operation count; return capped diagnostics through `fIssueDicts` |
| `ServerState.mCheckBackgroundCapacity` | `backend/mcp/state.py:117` | Refuse a background job when running plus queued jobs fill every slot |
| `FasterWhisperBackend.mLoadModel` | `backend/transcription/faster_whisper_backend.py:240` | Load a Whisper model, keeping the two most recently used |
| `AudacityPipeClient.mConnect` | `backend/backends/audacity.py:308` | Open the script pipes once `_mCheckPipe` and `_mCheckOpened` accept them; bounded drain |
| `PodcastPipeline.mReserveOutput` | `backend/pipeline.py:215` | Hold an artifact directory or file for a whole run |
| `fAtomicPath` | `backend/util/atomic.py:32` | Temporary file published by rename, or without replacing an existing target |
| `fAcquireLock` | `backend/util/atomic.py:98` | Non-blocking `flock` that stays valid when its file is removed concurrently |

<a id="flows"></a>
## Call paths

1. **Local deployment:** script entry → `fMain` → argument/layout/root checks → `fOpenLog` → `fLocateSource` → `fValidateSource` → OS/user checks → packages → installation tree → `fStageLocalSource` → application/CUDA/launchers → settings/service/workspace → model/Audacity setup → `fRestartAudacityUnit` → existing installation diagnostics → EXIT trap → `fCleanup`.
2. **Remote deployment:** same flow, with `fLocateSource` selecting download mode → packages including curl/certificates/tar/gzip → `fDownloadSource` → curl archive → tar → `fValidateSource` → shared installation flow → EXIT trap → `fCleanup`.
3. **Check mode:** source selection and read-only preconditions → report planned package/source/tree/application steps → skip final installation diagnostics. No source download and no log mutation.
4. **MCP processing:** `server.fMain` → `ServerState` / `fBuildServer` → `fRegisterTools` → pipeline operation → job store, plan validation and rendering.
5. **CLI processing:** `cli.fMain` → command handler → `PodcastPipeline.mProcess` → analysis/transcription → plan/validation → execution/backends → output artifacts and job state. `master` goes through `PodcastPipeline.mMasterFile` instead: analysis → mastering-only plan → validation → execution → delivery formats.
6. **MCP client configured in a folder:** client started in the folder → its MCP configuration → `server.fMain` with folder-relative overrides → first tool call → `AppContext` → `Workspace` / `fEnvironmentLocation` → the folder's `input/` and `output/` → `get_server_info` → `fCheckProfiles` → `prepare_episode_edit` → `finish_episode_edit` (with `mastering=false`, only the cuts) → optionally `master_audio` → `PodcastPipeline.mMasterFile` → artifacts in the folder's `output/`.
7. **Background job:** `process_podcast` / `finish_episode_edit` with `wait=false` → `_fStartBackgroundJob` → `ServerState.admission_lock` → `ServerState.mCheckBackgroundCapacity` → queued `Job` saved → executor → `PodcastPipeline.mProcess` → done callback removes the future from the table.
8. **Artifact publication:** `PodcastPipeline.mProcess` / `mMasterFile` / `export_audio` → `PodcastPipeline.mReserveOutput` → `_mCheckOutputTargets` → render → `FfmpegBackend.mExport` and `fWriteJsonAtomic` through `fAtomicPath` (no replacement without `overwrite`) → job log copied → reservation released.

<a id="entry-points"></a>
## Entry points

| Command or interface | Handler | File |
| --- | --- | --- |
| `bash deploy/install.sh --user NAME` | `fMain` → local source selection | `deploy/install.sh` |
| `curl -sL https://raw.githubusercontent.com/nipegun/podcast-editor-mcp-server/refs/heads/main/deploy/install.sh \| bash` | `fMain` → remote source selection, `fResolveUser` | `deploy/install.sh` |
| `bash deploy/uninstall.sh` | `fMain` | `deploy/uninstall.sh` |
| `podcast-editor-mcp-server` | `fMain` | `backend/server.py` |
| `podcast-editor-mcp-server-cli` | `fMain` / command handlers | `backend/cli.py` |
| `podcast-editor-mcp-server-doctor` | `fMain` | `backend/doctor.py` |
| `_/temp/venv/bin/python3 -m podcast_editor_mcp_server.cli` after editable installation | `fMain` | `backend/cli.py` |
| MCP tools, including `prepare_episode_edit` / `finish_episode_edit` / `master_audio` | `fRegisterTools` | `backend/mcp/tools.py` |
<a id="impact"></a>
## Impact analysis

- Changes to source layout require matching updates to `fValidateSource`, the package directory mapping, development helper paths and the documentation. New module directories are discovered under `backend/`; refresh the editable installation to expose them.
- Repository or branch changes affect the installer's URL constants, Python metadata, the server website URL, the unit's documentation URL and installation examples.
- Changing the prefix layout affects configuration, launchers, model/workspace paths and the uninstaller.
- Python consumers import `podcast_editor_mcp_server`; environment overrides use `PODCAST_EDITOR_MCP_SERVER_*`. Keep packaging entry points, installer-generated launchers and client configuration aligned with those names. Development helpers resolve temporary paths under this repository's `_/temp/`.
- Changing planning or validation affects both CLI and MCP operations because they share the pipeline.
- Changing profile defaults affects new jobs; persisted custom profiles and system settings survive installer reruns.
- Removing or renaming a field of a persisted model (`Job`, `Transcript`, `EditPlan`, `RuntimeConfig`, `Profile`) makes the existing files that carry it unreadable; nothing migrates them.
- MCP clients depend on the tool names, the response fields (`review_items`, `joins`, `workspace`, `profile_errors`, `error_count`, `warning_count`) and the profile keys; renaming any of them breaks the clients that use them.
- Changing `fBaseDirectory`, `fEnvironmentLocation` or the input/output overrides changes where folder-configured clients read and write; keep the manuals aligned.
- `cInputFormats` must name a demuxer for every extension in `cAudioReadExtensions`; one left out makes those files unreadable everywhere, because every FFmpeg input goes through `fInputArguments`.
- `cMaxDecodedBytes` bounds the longest recording that can be analysed sample by sample, transcribed or scanned by Silero; `cScratchMaxBytes` and `cScratchFreeReserveBytes` bound the analysis copies on disk.
- `PodcastPipeline.mReserveOutput` is held by every run that writes into `output/`; a new writer of artifacts has to take it too, or it can overwrite a concurrent run.

<a id="extensions"></a>
## Extension points

Add installer options in `fParseArguments` and `fUsage`, initialize their state, and keep mutations behind `--check` guards. Check download/extraction failures explicitly: the conditional invocation of `fMain` means Bash's `set -e` is not sufficient inside functions. Temporary source acquisition belongs in `fDownloadSource`; never consume the installer's stdin.

For a new editing operation, implement shared behavior in the pipeline or a focused module, then expose it through the CLI and/or `mcp/tools.py`. Add packaged presets under `backend/data/profiles/`; use personal or shared profile directories for installation-specific overrides. Keep this symbol index and the translated manuals aligned with changed entry points and flows.

To accept a new audio extension, add it to `cAudioReadExtensions` and its demuxer to `cInputFormats`, once you have checked that the demuxer opens no secondary files or URLs. A new step that holds whole recordings in memory goes through `fCheckDecodedBudget`, and one that writes into `output/` runs inside `PodcastPipeline.mReserveOutput`.