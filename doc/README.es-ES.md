# podcast-editor-mcp-server

Este README también está disponible en:\
[en-US](../README.md), [es-AR](README.es-AR.md)

## Qué es

Un servidor MCP local y una aplicación de línea de comandos para editar y masterizar pódcasts en Debian 13.

- Analiza grabaciones, silencios, voz, sonoridad y saturación.
- Detecta posibles muletillas, repeticiones, arranques fallidos y tomas repetidas.
- Prepara y valida planes de edición antes de generar el audio con FFmpeg.
- Permite transcribir con faster-whisper y crear proyectos editables de Audacity de forma opcional.
- Conserva las grabaciones originales y guarda por separado el audio editado, las transcripciones y los informes.

Puedes utilizarlo desde la línea de comandos o desde un cliente MCP mediante stdio. El procesamiento del audio se realiza localmente.

## Capturas de pantalla

Todavía no hay capturas disponibles. La aplicación ofrece una CLI y un servidor MCP.

## Desplegar

Ejecútalo como `root`.

**Debian**

Utiliza Debian 13 con `curl` y `ca-certificates` instalados:

```bash
curl -sL https://raw.githubusercontent.com/nipegun/podcast-editor-mcp-server/refs/heads/main/deploy/install.sh | bash
```

El instalador descarga la rama `main` del repositorio y prepara la cuenta normal que va a procesar las grabaciones: la que llamó a sudo o inició la sesión, o la única que hay en la máquina. Si vuelves a ejecutarlo, actualiza la instalación. Después sigue el [inicio rápido](MANUAL.es-ES.md#inicio-rápido) del manual para comprobar la instalación, conectar tu cliente MCP y editar tu primera grabación. El manual explica también las opciones de instalación, las actualizaciones y la instalación local.

## Leer la documentación

Explora:

- [doc/MANUAL.es-ES.md](MANUAL.es-ES.md) para consultar la instalación, la configuración y el uso.
- [doc/CODE.es-ES.md](CODE.es-ES.md) para conocer la arquitectura, los módulos y la referencia de desarrollo.

## Patrocinar este proyecto

- **Contrátame:** encarga nuevas funciones escribiendo a [nipegun@gmail.com](mailto:nipegun@gmail.com).
- **GitHub Sponsors:** apoya el desarrollo en [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun).
