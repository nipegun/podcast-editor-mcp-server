# podcast-editor-mcp-server

This README is also available in:\
[es-AR](doc/README.es-AR.md), [es-ES](doc/README.es-ES.md)

## What is it

A local MCP server and command line application for editing and mastering podcasts on Debian 13.

- Analyze recordings, silence, speech, loudness and clipping.
- Detect potential fillers, repetitions, false starts and retakes.
- Prepare and validate edit plans before rendering audio with FFmpeg.
- Optionally transcribe with faster-whisper and create editable Audacity projects.
- Keep original recordings intact and save edited audio, transcripts and reports separately.

Use it from the command line or an MCP client over stdio. Audio processing runs locally.

## Screenshots

No screenshots are available yet. The application provides a CLI and an MCP server.

## Deploy

Run it as `root`.

**Debian**

Use Debian 13 with `curl` and `ca-certificates` installed:

```bash
curl -sL https://raw.githubusercontent.com/nipegun/podcast-editor-mcp-server/refs/heads/main/deploy/install.sh | bash
```

The installer downloads the repository's `main` branch and sets up the ordinary account that will process the recordings: the one that called sudo or logged in, or the only one on the machine. Running it again updates the installation. Then follow the [quick start](doc/MANUAL.md#quick-start) of the manual to check the installation, connect your MCP client and edit your first recording. The manual also covers installation options, updates and local installation.

## Read the documentation

Explore:

- [doc/MANUAL.md](doc/MANUAL.md) for installation, configuration and usage.
- [doc/CODE.md](doc/CODE.md) for architecture, modules and development reference.

## Sponsor this project

- **Hire me:** commission new features at [nipegun@gmail.com](mailto:nipegun@gmail.com).
- **GitHub Sponsors:** support development at [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun).
