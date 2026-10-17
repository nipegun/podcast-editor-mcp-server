# Manual de uso

## Índice

- [Inicio rápido](#inicio-rápido)
- [Instalación y actualizaciones](#instalación-y-actualizaciones)
- [Ejecutar desde el código fuente](#ejecutar-desde-el-código-fuente)
- [Archivos y configuración](#archivos-y-configuración)
- [Editar una grabación](#editar-una-grabación)
- [Conectar un cliente MCP](#conectar-un-cliente-mcp)
- [Audacity y resolución de problemas](#audacity-y-resolución-de-problemas)
- [Desinstalar](#desinstalar)

## Inicio rápido

Cinco pasos te llevan de una máquina con Debian 13 a un episodio editado. Cada uno enlaza con la sección que da los detalles.

1. **Instalá**, como root. `--with-audacity` instala también Audacity, que hace falta para el proyecto editable; sacalo si solo querés el audio:

   ```bash
   curl -sL https://raw.githubusercontent.com/nipegun/podcast-editor-mcp-server/refs/heads/main/deploy/install.sh | bash -s -- --with-audacity
   ```

   El instalador prepara la cuenta normal que va a procesar las grabaciones (consultá [Instalación y actualizaciones](#instalación-y-actualizaciones)) y termina con un resumen de dónde está todo. En una máquina con GPU NVIDIA instala también CUDA, así que un episodio largo se transcribe en minutos en lugar de en horas.

2. **Comprobá la instalación**, con esa cuenta normal, nunca como root:

   ```bash
   podcast-editor-mcp-server-doctor
   podcast-editor-mcp-server-cli workspace
   ```

   `doctor` enumera lo que funciona y lo que falta, y `workspace` muestra tus directorios `input/` y `output/`. Si el instalador acaba de agregar tu cuenta al grupo de la aplicación, volvé a iniciar sesión antes.

3. **Registrá el servidor en tu cliente MCP**, con la misma cuenta. Para Claude Code:

   ```bash
   claude mcp add-json --scope user podcast-editor-mcp-server '{"type":"stdio","command":"/usr/local/bin/podcast-editor-mcp-server","args":[],"timeout":14400000}'
   claude mcp list
   ```

   Codex y los demás clientes se explican en [Conectar un cliente MCP](#conectar-un-cliente-mcp).

4. **Abrí Audacity** en tu escritorio si querés el proyecto editable `.aup3`, o activá una vez su servicio (consultá [Audacity y resolución de problemas](#audacity-y-resolución-de-problemas)). Sin él igual obtenés el audio y las transcripciones.

5. **Editá**: copiá la grabación a `input/`, iniciá el cliente (`claude`) y pedile, por ejemplo, «Editá episodio-42.wav: sacá las tomas repetidas y las pausas largas, y después masterizalo». Un episodio largo tarda minutos con GPU y bastante más con CPU. Los resultados aparecen en `output/<nombre>/` (consultá [Resultados](#resultados)).

## Instalación y actualizaciones

Usá Debian 13 y una cuenta normal existente con directorio personal, la que va a procesar las grabaciones. Instalá `ca-certificates` y `curl` con APT antes de usar el método remoto si faltan. Como root:

```bash
curl -sL https://raw.githubusercontent.com/nipegun/podcast-editor-mcp-server/refs/heads/main/deploy/install.sh | bash
```

Las opciones van después de `bash -s --`. Los ejemplos usan `alice`; reemplazala por tu cuenta de trabajo. Con `-f` y `pipefail`, una descarga fallida detiene la tubería en lugar de pasarle a bash una página de error:

```bash
set -o pipefail
curl -fsSL https://raw.githubusercontent.com/nipegun/podcast-editor-mcp-server/refs/heads/main/deploy/install.sh | bash -s -- --user alice --yes
```

Sin `--user`, la cuenta de trabajo es la que llamó a sudo, la que inició la sesión (`su -` la conserva), la única que preparó una ejecución anterior o la única cuenta normal de la máquina. Si hay varias y una terminal, el instalador las enumera y pregunta; sin terminal, como en `ssh servidor 'curl ... | bash'`, se detiene y pide `--user`. Root nunca es la cuenta de trabajo, así que una máquina sin cuentas normales necesita una antes (`adduser alice`). El mismo comando también funciona desde una cuenta normal con sudo: el instalador pide la contraseña y se vuelve a ejecutar como root para esa cuenta. Sin terminal, todas las preguntas se responden que no, así que Audacity solo se instala con `--with-audacity`.

O, desde la raíz de un repositorio descargado y extraído:

```bash
bash deploy/install.sh --user alice --yes
```

Ambas formas aceptan los mismos argumentos después del nombre del script o de `bash -s --`. El método local instala el código contiguo al script. El remoto obtiene `main` y necesita acceso público a GitHub. Un instalador descargado de forma independiente también obtiene el repositorio. No se requiere Git, y sudo solo se usa cuando el instalador lo inicia una cuenta normal.

Repetí cualquiera de los comandos para actualizar; sin `--user`, se actualiza la cuenta que se preparó la primera vez. Descargá una versión más reciente del repositorio antes de una actualización local. Se conservan los ajustes, los perfiles, las grabaciones, los trabajos, los resultados y los modelos existentes. Una actualización no migra nada: una clave de ajustes o de perfil que la nueva versión ya no tiene impide cargar ese archivo hasta que borrés la clave (el error la indica), y los registros de trabajos o las transcripciones en caché escritos por una versión anterior se omiten o se vuelven a transcribir. Se reutiliza el entorno virtual cuando es válido. Si falla la descarga o la extracción del código, la instalación se detiene antes de reemplazar la aplicación; el código temporal se elimina al salir.

Usá `--with-audacity` para obtener proyectos editables, `--no-model` para aplazar la descarga inicial del modelo, `--no-transcription` para omitir la transcripción o `--no-cuda` para omitir las dependencias de GPU. `--whisper-model small` solo elige el modelo que se descarga previamente: cambiá también el perfil de edición si querés utilizarlo para transcribir. Con `--yes`, Audacity se omite si no está instalado, salvo que se solicite expresamente. `--no-apt` exige todas las dependencias del sistema, incluidos curl, certificados, tar y gzip para el código remoto.

Agregá `--check` para consultar las acciones previstas sin escribir archivos, descargar código ni instalar paquetes. No garantiza que un archivo remoto sea accesible o válido. `--help` muestra todas las opciones.

## Ejecutar desde el código fuente

El repositorio guarda los módulos Python directamente en `backend/`, incluidos `server.py`, `cli.py`, `config.py` y los subdirectorios de módulos. El paquete instalado sigue llamándose `podcast_editor_mcp_server` y los comandos instalados conservan sus nombres.

Para desarrollar, usá una cuenta normal con Python 3.13 o posterior y ejecutá estos comandos desde la raíz del repositorio:

```bash
python3 -m venv _/temp/venv
_/temp/venv/bin/python3 -m pip install --break-system-packages -e .
export PODCAST_EDITOR_MCP_SERVER_PREFIX="$PWD/_/temp/prefix"
_/temp/venv/bin/python3 -m podcast_editor_mcp_server.cli --help
```

La instalación editable asocia el nombre del paquete con `backend/`, por lo que este entorno utiliza las modificaciones posteriores de los archivos Python. Repetí la instalación editable si cambiás los metadatos del paquete o agregás directorios de módulos. Para desplegar, seguí usando el instalador anterior: instala una copia independiente que después no depende del árbol de código descargado.

## Archivos y configuración

Prefijo predeterminado: `/opt/podcast-editor-mcp-server`. `--prefix` permite cambiarlo; los lanzadores instalados pasan el prefijo correspondiente a la aplicación.

| Ubicación dentro del prefijo | Función |
| --- | --- |
| `venv/`, `bin/` | Aplicación, dependencias y lanzadores |
| `etc/config.yaml` | Configuración global, conservada al actualizar |
| `etc/users/alice/config.yaml` | Configuración personal opcional de ejecución |
| `etc/profiles/`, `etc/users/alice/profiles/` | Perfiles de edición compartidos y personales |
| `var/workspace/alice/input/` | Grabaciones originales |
| `var/workspace/alice/output/` | Audio editado, informes, transcripciones y proyectos opcionales |
| `var/workspace/alice/jobs/`, `work/`, `cache/`, `logs/` | Estado de trabajos, archivos de trabajo, cachés y registros |
| `var/models/` | Modelos Whisper compartidos |
| `var/log/install.log` | Registro de instalación, legible por root |
| `share/source-directory` o `share/source-url` | Ruta del código local o URL del archivo remoto |
| `_/temp/source.*/` | Copia temporal del código, descargado o local, eliminada al salir del instalador |

La configuración de ejecución utiliza la ruta explícita, si se indica; si no, el primer archivo existente entre `PODCAST_EDITOR_MCP_SERVER_CONFIG`, los ajustes personales y los globales. Los campos ausentes utilizan valores predeterminados internos. El archivo personal de ejecución no se mezcla con el global. Los perfiles admiten `extends` y personalizaciones; guardá tus perfiles fuera del entorno virtual.

La sección `jobs` limita el trabajo en segundo plano: se ejecutan a la vez `max_concurrent` trabajos (2 de forma predeterminada) y hasta `max_queued` más esperan a que quede libre un proceso de trabajo (8 de forma predeterminada). Una petición que supere ese número se rechaza con un mensaje que pide volver a intentarlo cuando termine un trabajo, y no se registra nada de ella. Cada vez que se inicia el servidor se borran los registros de trabajos terminados con más de `retention_days` días (90 de forma predeterminada), como hace `podcast-editor-mcp-server-cli jobs --purge`; sus resultados se conservan.

Usá estos nombres exactos para las variables de entorno de los clientes MCP y los scripts:

| Variable | Función |
| --- | --- |
| `PODCAST_EDITOR_MCP_SERVER_PREFIX` | Raíz de la instalación |
| `PODCAST_EDITOR_MCP_SERVER_CONFIG` | Archivo de configuración de ejecución |
| `PODCAST_EDITOR_MCP_SERVER_WORKSPACE` | Raíz del espacio de trabajo |
| `PODCAST_EDITOR_MCP_SERVER_INPUT` | Directorio del que se leen las grabaciones originales, en lugar del `input/` del espacio de trabajo |
| `PODCAST_EDITOR_MCP_SERVER_OUTPUT` | Directorio en el que se escriben los resultados, en lugar del `output/` del espacio de trabajo |
| `PODCAST_EDITOR_MCP_SERVER_PROFILES_DIR` | Directorio adicional de perfiles |
| `PODCAST_EDITOR_MCP_SERVER_PROFILE` | Perfil de edición activo |
| `PODCAST_EDITOR_MCP_SERVER_MODEL_DIR` | Directorio de modelos Whisper |
| `PODCAST_EDITOR_MCP_SERVER_FFMPEG` | Ejecutable de FFmpeg |
| `PODCAST_EDITOR_MCP_SERVER_FFPROBE` | Ejecutable de ffprobe |

Un valor relativo en `PODCAST_EDITOR_MCP_SERVER_WORKSPACE`, `PODCAST_EDITOR_MCP_SERVER_INPUT`, `PODCAST_EDITOR_MCP_SERVER_OUTPUT` o `PODCAST_EDITOR_MCP_SERVER_PROFILES_DIR` parte de la carpeta indicada en `CLAUDE_PROJECT_DIR`, que Claude Code define para los servidores que inicia, y, si no existe, del directorio en el que se inició el servidor.

## Editar una grabación

Ejecutá los comandos de procesamiento como usuario normal. Copiá una grabación a su carpeta `input/`; `podcast-editor-mcp-server-cli workspace` muestra las rutas exactas.

```bash
podcast-editor-mcp-server-cli probe episode.wav
podcast-editor-mcp-server-cli analyze episode.wav
podcast-editor-mcp-server-cli process episode.wav --dry-run
podcast-editor-mcp-server-cli process episode.wav
```

La simulación prepara y evalúa el plan sin generar audio. El procesamiento normal guarda los resultados en `output/`. Los resultados existentes se protegen salvo que indiques `--overwrite`. Usá `--no-transcribe` y `--no-audacity` al procesar si no querés esos componentes opcionales.

Dos ejecuciones que escriben los mismos resultados nunca se solapan: mientras una escribe en `output/<nombre>/`, otra con el mismo nombre se rechaza en el acto, incluso con `--overwrite`; esperá a que termine la primera o elegí otro nombre de salida. Sin `--overwrite`, tampoco se reemplaza un archivo que aparezca en la salida mientras trabaja una ejecución.

Las grabaciones solo se leen de archivos locales en los formatos de audio aceptados: WAV, FLAC, MP3, M4A/MP4, AAC, Ogg (Vorbis y Opus), WavPack, AIFF y CAF. Se rechaza un archivo cuyo contenido sea otra cosa, como una lista de reproducción o una lista de concatenación guardada con una extensión de audio. El análisis muestra a muestra, la transcripción y la detección de voz tienen como mucho 3 GiB de audio decodificado en memoria, más de cuatro horas a 48 kHz; esos pasos rechazan una grabación más larga y el análisis lo indica con un aviso. Una grabación comprimida se decodifica una sola vez en el directorio de trabajo para analizarla, y esa decodificación se rechaza si dejaría menos de 2 GiB libres en ese disco; esas copias se mantienen por debajo de 16 GiB en total y se borran primero las menos usadas.

```bash
podcast-editor-mcp-server-cli profiles
podcast-editor-mcp-server-cli profiles podcast-default
podcast-editor-mcp-server-cli process episode.wav --profile podcast-tight
```

La edición y la masterización también pueden ser dos pasos separados. `process --no-mastering` genera solo los cortes, sin filtros, normalización de sonoridad, limitador ni fundidos, y escribe `output/<nombre>/<nombre>.wav`. Después, `master` aplica a ese archivo la cadena de limpieza y masterización del perfil y escribe al lado `<nombre>.master.wav` y los demás formatos de entrega:

```bash
podcast-editor-mcp-server-cli process episode.wav --no-mastering
podcast-editor-mcp-server-cli master episode/episode.wav
```

Cuando el límite de pico real impide que la normalización de sonoridad de FFmpeg alcance el objetivo del perfil, se mide el máster y se agrega la ganancia que falta antes del limitador, como mucho dos veces, hasta quedar a 0,2 LU del objetivo.

`podcast-editor-mcp-server-cli jobs` enumera los trabajos. `plan`, `validate`, `simulate` y `execute` permiten controlar los planes guardados; consultá el `--help` de cada comando para ver sus argumentos.

### Resultados

Cada edición escribe sus archivos en `output/<nombre>/`, donde `<nombre>` es el nombre de salida que hayas elegido o, si no elegís ninguno, el nombre de la grabación sin su extensión. La grabación de `input/` nunca se modifica.

| Archivo | Qué es |
| --- | --- |
| `<nombre>.master.wav`, `<nombre>.mp3` | El episodio editado y masterizado, en los formatos de entrega del perfil |
| `<nombre>.wav` | El episodio editado sin masterizar, cuando la edición y la masterización son pasos separados; al masterizarlo después se agregan los dos archivos anteriores |
| `<nombre>.aup3` | Proyecto editable de Audacity, solo si Audacity estaba accesible |
| `<nombre>.transcript.json`, `<nombre>.transcript.txt`, `<nombre>.srt` | Transcripción del episodio editado |
| `<nombre>.labels.txt` | La misma transcripción como pista de etiquetas de Audacity |
| `<nombre>.edit-plan.json`, `<nombre>.edit-report.json` | Los cortes realizados y el informe de la edición |
| `<nombre>.analysis.json`, `<nombre>.candidates.json`, `<nombre>.job.json`, `<nombre>.job.log` | Mediciones, candidatos detectados, registro del trabajo y su log |

Los formatos de audio y de transcripción que aparecen dependen de la sección `output` del perfil; los perfiles predefinidos entregan WAV y MP3, y la transcripción en JSON, texto y SRT.

## Conectar un cliente MCP

Ejecutá el cliente MCP con la misma cuenta normal, no con sudo. El cliente inicia el servidor mediante stdio. No hay escucha HTTP ni puerto de red.

Transcribir, preparar o terminar un episodio largo lleva minutos con GPU y mucho más con CPU, más de lo que estos clientes esperan por defecto a una herramienta, así que cada configuración de las siguientes amplía ese límite a cuatro horas.

**Claude Code.** Registrá el servidor para toda la cuenta:

```bash
claude mcp add-json --scope user podcast-editor-mcp-server '{"type":"stdio","command":"/usr/local/bin/podcast-editor-mcp-server","args":[],"timeout":14400000}'
```

`timeout` va en milisegundos y solo se aplica a este servidor. Configurar `MCP_TOOL_TIMEOUT` en su lugar no alcanza: Claude Code sigue cortando una llamada a un servidor local después de 30 minutos sin mensajes de progreso, y este servidor no manda ninguno. Abrí una sesión nueva de `claude` para que cargue el servidor; `claude mcp list` muestra si conecta.

**Codex.** Agregá a `~/.codex/config.toml`:

```toml
[mcp_servers.podcast-editor-mcp-server]
command = "/usr/local/bin/podcast-editor-mcp-server"
args = []
startup_timeout_sec = 30
tool_timeout_sec = 14400
env_vars = ["DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR"]
```

`codex mcp add podcast-editor-mcp-server -- /usr/local/bin/podcast-editor-mcp-server` escribe las dos primeras líneas; agregá las demás debajo. Codex inicia el servidor con un entorno casi vacío, y `env_vars` le pasa las variables del escritorio que necesita para iniciar Audacity. `codex mcp list` muestra el servidor.

**Otros clientes.** En los clientes que utilizan una configuración JSON con `mcpServers`, agregá lo siguiente y ampliá su tiempo de espera de herramientas si lo tienen:

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

Copiá las grabaciones al directorio `input/` de la cuenta (`podcast-editor-mcp-server-cli workspace` lo muestra) y pedile al asistente algo como «Editá episodio-42.wav, sacá los silencios y las muletillas y masterizalo para pódcast». Los resultados aparecen en `output/`.

Empezá con la herramienta `get_server_info` para consultar las funciones y rutas disponibles. Las herramientas `prepare_episode_edit` y `finish_episode_edit` permiten editar episodios. Con `mastering=false`, `finish_episode_edit` entrega el audio editado sin masterizar (`<nombre>.wav`), y la herramienta `master_audio` lo limpia y lo masteriza después.

`transcribe_audio` acepta un `model` como `small` o `large-v3`, o un repositorio `organización/nombre`, nunca una ruta. Un modelo distinto del del perfil solo se usa si ya está descargado: una petición nunca hace que el servidor descargue uno. Como mucho quedan cargados dos modelos Whisper a la vez. Las respuestas de validación enumeran como mucho 100 errores y 100 avisos; `error_count` y `warning_count` dan los totales.

## Audacity y resolución de problemas

El proyecto editable (`<nombre>.aup3`) solo se crea si Audacity está en marcha y responde por sus tuberías de scripts en el momento en que se genera la edición. El instalador activa el módulo `mod-script-pipe` de Audacity para la cuenta de trabajo; `podcast-editor-mcp-server-cli audacity-status --enable-module` hace lo mismo a mano, con Audacity cerrado. Después, antes de editar, abrí Audacity en tu escritorio y dejalo abierto, o bien, si instalaste con `--with-audacity`, activá una vez el servicio opcional como usuario de trabajo, para que siempre haya un Audacity oculto en marcha:

```bash
systemctl --user enable --now podcast-editor-mcp-server-audacity.service
```

Si no se puede llegar a Audacity, la edición entrega igual el audio y las transcripciones e incluye el aviso `Audacity project not created`. La herramienta `create_audacity_project` crea el proyecto más tarde a partir del audio editado, cuando Audacity ya esté en marcha.

Al reinstalar o actualizar, esa unidad se reinicia si está en marcha, para que Audacity use la versión nueva. Al desinstalar, se detiene y se desactiva, y se borran los enlaces que la activaban.

Audacity guarda sus propias preferencias en el directorio personal del usuario. `podcast-editor-mcp-server-cli audacity-status` informa del estado de la integración. El procesamiento no debe ejecutarse como root.

El servidor maneja Audacity mediante sus tuberías de scripts de `/tmp` solo si son FIFO de tu cuenta y están cerradas a los demás usuarios, tal como las crea Audacity. Se rechazan las tuberías creadas por otra cuenta, y también una respuesta que no termina nunca.

Ejecutá `podcast-editor-mcp-server-doctor` para consultar las dependencias. Si falla la instalación, revisá `var/log/install.log` como root. Si el instalador acaba de agregar tu cuenta al grupo de la aplicación, volvé a iniciar sesión antes de descargar modelos desde una sesión existente. Un error 404 al descargar indica que el repositorio, la rama o el archivo no son públicos en la URL documentada.

## Desinstalar

Desde un repositorio descargado, como root:

```bash
bash deploy/uninstall.sh --user alice --yes
```

Se conservan las grabaciones, los resultados, los modelos y la configuración. Agregá `--purge-config` para eliminar la configuración compartida. Para borrar permanentemente también todos los espacios de trabajo, los ajustes personales, los modelos y la configuración compartida:

```bash
bash deploy/uninstall.sh --user alice --purge-data --yes
```

Se conservan los paquetes del sistema, como FFmpeg y Audacity.
