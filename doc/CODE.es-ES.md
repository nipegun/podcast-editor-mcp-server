# Mapa del código

## Índice

- [Arquitectura](#architecture)
- [Módulos](#modules)
- [Símbolos clave](#symbols)
- [Flujos](#flows)
- [Entradas](#entry-points)
- [Análisis de impacto](#impact)
- [Puntos de extensión](#extensions)

<a id="architecture"></a>
## Arquitectura

El paquete Python se llama `podcast_editor_mcp_server`; la distribución y los comandos instalados se llaman `podcast-editor-mcp-server`. La CLI y las herramientas MCP comparten `PodcastPipeline`, por lo que utilizan la misma edición, validación y generación de resultados. El transporte MCP es stdio, sin escucha HTTP. FFmpeg genera el audio; faster-whisper y Audacity son integraciones opcionales.

El código del paquete vive directamente en `backend/`: los módulos de entrada y los directorios `audio/`, `mcp/`, `models/` y los demás módulos comparten esa raíz. `pyproject.toml` asocia `podcast_editor_mcp_server` con `backend/` mediante `tool.setuptools.package-dir`; setuptools descubre los subpaquetes de espacio de nombres implícito dentro de esa ubicación. Los imports, los comandos y los perfiles incluidos en `data/profiles/` conservan sus nombres instalados. Así se elimina un nivel de directorios del código sin cambiar la estructura del paquete instalado ni introducir archivos `__init__.py`.

La ejecución desde el código utiliza una instalación editable en `_/temp/venv/`, descrita en el manual. Ese entorno virtual resuelve el nombre instalado del paquete hacia estos archivos fuente. Repite la instalación editable si cambias los metadatos o añades directorios de módulos. El despliegue sigue utilizando una instalación normal, por lo que se puede eliminar el código descargado después.

El instalador de Debian utiliza un prefijo que contiene el entorno virtual, los lanzadores, los ajustes, los modelos y los espacios de trabajo. Fuera de ese árbol solo se crean los enlaces de comandos y de systemd, además del grupo del sistema y las dependencias APT. Audacity administra sus propias preferencias de usuario.

El instalador captura `BASH_SOURCE[0]` antes de entrar en las funciones. Si el árbol contiguo al script tiene los metadatos del paquete, lo valida y utiliza. En caso contrario descarga el archivo público de la rama `main` de GitHub. No deduce el código local a partir de una carpeta de trabajo cualquiera. Un árbol local incompleto con `pyproject.toml` provoca un error de validación, sin sustituirlo silenciosamente por código remoto. Cuando se pasa a bash por una tubería (`curl -sL .../refs/heads/main/deploy/install.sh | bash`), `BASH_SOURCE[0]` está vacío, así que se elige el modo remoto.

Sin `--user`, `fResolveUser` toma `SUDO_USER`, `PKEXEC_UID` o `logname`; desde una terminal de root que no nombra a nadie, toma la única cuenta con espacio de trabajo bajo el prefijo (la de una ejecución anterior) o, si no hay, la única cuenta normal (`fListOrdinaryUsers`: el rango de UID de `/etc/login.defs`, un directorio personal y un intérprete de inicio de sesión). Si hay varias candidatas, `fChooseUser` las ofrece en la terminal, y las rechaza si no hay terminal o con `--yes`. Si lo inicia una cuenta normal, `fRunAsRoot` valida antes sudo y vuelve a ejecutar el instalador como root para esa cuenta: el mismo archivo o, si llegó por una tubería, la misma URL descargada de nuevo. `fHasTerminal` indica si `/dev/tty` se abre de verdad, para que una ejecución sin terminal responda que no en lugar de fallar.

El código remoto se prepara en `<prefix>/_/temp/source.*/`, después de instalar las dependencias de descarga. curl debe terminar correctamente antes de extraer con tar y validar el código. pip realiza una instalación normal, no editable, por lo que eliminar el árbol temporal al salir no afecta al paquete instalado. La procedencia se registra en `share/source-url` para el código remoto o `share/source-directory` para el local, eliminando el archivo alternativo obsoleto. El código local se copia en esa misma área temporal antes de que pip lo instale, porque pip compila dentro del árbol que recibe y dejaría `build/` y `*.egg-info` en el código fuente.

La reinstalación conserva los ajustes, los perfiles y los datos, reutiliza un entorno virtual válido y consulta los modelos almacenados. `--check` no descarga ni extrae código y no crea ni modifica el registro de instalación. La llamada a `fMain` recibe stdin desde `/dev/null`, para evitar que los subprocesos consuman el script enviado por la tubería; las confirmaciones se leen de `/dev/tty`. Si la unidad de Audacity del usuario de trabajo está en marcha, el instalador recarga el systemd de ese usuario y reinicia la unidad mediante `systemctl --user --machine=<usuario>@`, para que use la versión nueva; `runuser` no llega al bus del usuario. El desinstalador desactiva la unidad del mismo modo y elimina los enlaces que quedan en `~/.config/systemd/user`.

Un cliente MCP configurado dentro de una carpeta puede indicar al servidor los directorios de esa carpeta con rutas relativas en `PODCAST_EDITOR_MCP_SERVER_INPUT`, `PODCAST_EDITOR_MCP_SERVER_OUTPUT` y `PODCAST_EDITOR_MCP_SERVER_PROFILES_DIR`. `fEnvironmentLocation` las resuelve a partir de `CLAUDE_PROJECT_DIR` cuando Claude Code la define y, si no, a partir del directorio de trabajo del servidor, que es la carpeta en la que se inició el cliente. Solo los roles de entrada y salida pasan a la carpeta: los archivos de trabajo, los trabajos, la caché de transcripciones y los registros siguen en el espacio de trabajo bajo el prefijo, de modo que una misma transcripción en caché sirve para todas las carpetas.

Enumerar un perfil solo lee el nombre de su archivo. `fCheckProfiles` carga todos los perfiles, así que `get_server_info` y `reload_configuration` informan de los que fallan como `profile_errors`, con las claves incorrectas. Un perfil solo es utilizable cuando no aparece en esa lista.

Los documentos persistentes se leen con los mismos modelos estrictos que los escriben. Los registros de trabajos, las transcripciones en caché, los planes de edición, la configuración de ejecución y los perfiles rechazan las claves desconocidas, los planes de edición deben llevar la versión de esquema actual y no hay código de migración. Un archivo escrito con un campo que ya no existe no supera la validación: una transcripción en caché ilegible se descarta y se vuelve a transcribir, un registro de trabajo antiguo se omite en los listados y una clave de configuración o de perfil eliminada hay que borrarla a mano.

La masterización normaliza la sonoridad con el filtro `loudnorm` de FFmpeg. Cuando el límite de pico real impide su ganancia lineal, pasa a la normalización dinámica y puede quedarse 1 LU por debajo del objetivo, así que `ExecutionEngine._mCorrectLoudness` mide el resultado y añade la diferencia como ganancia antes del limitador, como mucho dos veces y solo si hay un limitador que proteja ese límite.

Todo lo que controla quien llama está acotado antes de que cueste memoria, disco o un proceso. FFmpeg y ffprobe abren el audio solo mediante `fInputArguments`: el protocolo `file` y los demultiplexores de los contenedores de audio aceptados (`cInputFormats`), de modo que una lista de reproducción o un guion de concatenación guardado con una extensión de audio no puede hacer que abran archivos o URL que el control de rutas nunca vio. Las muestras decodificadas se limitan a `cMaxDecodedBytes` de mono float32 (3 GiB), que se comprueba antes de que libsndfile lea y se hace cumplir mientras FFmpeg entrega datos, y las señales largas se reducen por bloques. La copia que se decodifica de un archivo que libsndfile no puede leer lleva el nombre de la identidad del origen (`fSourceIdentity`: dispositivo, inodo, tamaño y fechas de modificación y de cambio), se dimensiona a partir del flujo antes de decodificar, se rechaza si dejaría menos de 2 GiB libres, se decodifica con un nombre temporal y se renombra al terminar; los directorios de análisis se mantienen por debajo de 16 GiB, borrando primero los menos usados.

Un plan de edición contiene como mucho 200 000 operaciones; un plan que supera `safety.max_operations` se rechaza antes de cualquier otra comprobación, los filtros contradictorios se comparan una sola vez y los diagnósticos se devuelven de 100 en 100 con sus recuentos completos. Los trabajos en segundo plano solo se admiten mientras los `jobs.max_concurrent` en ejecución y los `jobs.max_queued` en espera dejan sitio, un trabajo terminado sale de la tabla en el acto y los registros de trabajos con más de `jobs.retention_days` días se purgan al arrancar el pipeline. Un modelo Whisper indicado en una petición debe ser un nombre de modelo, no una ruta, y uno distinto del del perfil solo se carga desde la caché; como mucho quedan cargados dos modelos. Las tuberías de scripts de Audacity solo se usan si son FIFO del usuario y cerradas a los demás, y solo las mismas que se inspeccionaron; lo que se lee de ellas está acotado. Una ejecución reserva su directorio de resultados (`PodcastPipeline.mReserveOutput`, un `flock` en `<work>/locks`) desde sus comprobaciones de existencia hasta su última escritura, así que una segunda ejecución con el mismo nombre se rechaza en el acto. Sin `overwrite`, las exportaciones y los artefactos JSON se publican con un enlace duro, o con una última comprobación si el sistema de archivos no los admite, y nunca reemplazan un archivo que haya aparecido entretanto.

<a id="modules"></a>
## Módulos

| Módulo | Ruta | Responsabilidad | Depende de | Usado por |
| --- | --- | --- | --- | --- |
| Instalador | `deploy/install.sh` | Resolver código local/remoto y desplegar de forma idempotente | Bash, APT, pip; curl/tar para código remoto | Administrador |
| Servicio de usuario | `deploy/systemd/podcast-editor-mcp-server-audacity.service` | Inicio opcional de Audacity sin pantalla | CLI instalada, gestor systemd de usuario | Instalador, usuario de trabajo |
| Desinstalador | `deploy/uninstall.sh` | Retirar la aplicación y borrar datos opcionalmente | Estructura del prefijo, systemd | Administrador |
| Empaquetado | `pyproject.toml` | Asociar `backend/` con el espacio de nombres instalado; dependencias, metadatos y comandos | setuptools, README | pip |
| Servidor / CLI / diagnóstico | `backend/{server,cli,doctor}.py` | Entradas de protocolo, comandos y diagnóstico | SDK MCP, configuración, pipeline | Usuarios y clientes MCP |
| Adaptadores MCP | `backend/mcp/` | Herramientas, recursos, prompts y estado | Pipeline, almacén de trabajos | Servidor |
| Configuración | `backend/config.py` | Ajustes validados, perfiles y espacio de trabajo | Pydantic, YAML, control de rutas | CLI, servidor, pipeline |
| Identidad, errores y registros | `backend/{version,errors,logging_setup}.py` | Nombre del producto, jerarquía de excepciones y espacio de nombres de registro | Biblioteca estándar de Python | Todos los módulos de ejecución |
| Pipeline | `backend/pipeline.py` | Coordinar análisis, planes, trabajos y generación | Módulos siguientes | CLI y adaptadores MCP |
| Audio / transcripción / análisis | `backend/{audio,transcription,analysis,vad}/` | Medidas, transcripciones y candidatos de edición | FFmpeg, modelos opcionales, cachés | Pipeline |
| Planificación / validación / ejecución | `backend/{planning,validation,execution}/` | Crear, validar y aplicar planes | Modelos, backends | Pipeline |
| Backends | `backend/backends/` | Generación con FFmpeg e integración con Audacity | Ejecutables externos | Pipeline, ejecución |
| Trabajos / seguridad / modelos / utilidades | `backend/{jobs,security,models,util}/` | Persistencia, control de rutas, tipos y funciones comunes | Sistema de archivos, Pydantic | Pipeline y adaptadores |

<a id="symbols"></a>
## Símbolos clave

| Símbolo | Archivo:línea | Responsabilidad |
| --- | --- | --- |
| `fLocateSource` | `deploy/install.sh:725` | Elegir el árbol local contiguo o el modo remoto |
| `fValidateSource` | `deploy/install.sh:752` | Comprobar la identidad del paquete, `backend/server.py`, el perfil incluido y la plantilla del servicio |
| `fDownloadSource` | `deploy/install.sh:768` | Descargar, extraer y validar el archivo de GitHub |
| `fStageLocalSource` | `deploy/install.sh:808` | Copiar el código local al área temporal para que pip nunca compile dentro de él |
| `fInstallApplication` | `deploy/install.sh:830` | Reutilizar/crear el entorno virtual, instalar y registrar la procedencia |
| `fCleanup` | `deploy/install.sh:210` | Eliminar la copia temporal del código al salir |
| `fResolveUser` | `deploy/install.sh:458` | Elige el usuario de trabajo cuando falta `--user` |
| `fRunAsRoot` | `deploy/install.sh:307` | Vuelve a ejecutar el instalador con sudo para una cuenta normal |
| `fMain` (instalador) | `deploy/install.sh:1421` | Coordinar las etapas del despliegue |
| `fInstallSystemdUnit` | `deploy/install.sh:1086` | Generar y enlazar la unidad de Audacity con su nombre correcto |
| `fRestartAudacityUnit` | `deploy/install.sh:1208` | Reiniciar con la versión nueva la unidad de Audacity del usuario si está en marcha |
| `fStopAudacityUnit` | `deploy/uninstall.sh:205` | Desactivar la unidad del usuario mediante `--machine=<usuario>@` y eliminar sus enlaces |
| `ExecutionEngine._mCorrectLoudness` | `backend/execution/engine.py:331` | Añadir como ganancia antes del limitador la sonoridad que `loudnorm` no alcanzó |
| `fBuildServer` | `backend/server.py:80` | Construir el servidor MCP y registrar adaptadores; anunciar la URL del repositorio |
| `fLoadRuntimeConfig` | `backend/config.py:911` | Cargar la configuración de ejecución seleccionada |
| `Workspace` | `backend/config.py:933` | Resolver los roles del espacio de trabajo, respetando las variables de entrada y salida |
| `fBaseDirectory` | `backend/config.py:98` | `CLAUDE_PROJECT_DIR` si está definida; si no, el directorio de trabajo |
| `fEnvironmentLocation` | `backend/config.py:112` | Resolver una ruta del entorno; una relativa parte de `fBaseDirectory` |
| `fCheckProfiles` | `backend/config.py:782` | Cargar todos los perfiles disponibles e informar de fallos, claves incorrectas y nombres que no coinciden |
| `PodcastPipeline.mProcess` | `backend/pipeline.py:591` | Ejecutar el procesamiento y generar resultados |
| `fRegisterTools` | `backend/mcp/tools.py:133` | Registrar las herramientas MCP públicas |
| `PodcastEditorMcpServerError` | `backend/errors.py:13` | Excepción base; código genérico `podcast_editor_mcp_server_error` |
| `fRegisterPrompts` | `backend/mcp/prompts.py:16` | Registrar flujos, incluido `troubleshoot_podcast_editor_mcp_server` |
| `cLoggerRoot` | `backend/logging_setup.py:24` | Espacio de nombres de registro `podcast_editor_mcp_server` |
| `fInputArguments` | `backend/backends/ffmpeg.py:79` | Limita una entrada de audio a archivos locales y a los demultiplexores de audio aceptados |
| `FfmpegBackend.mExport` | `backend/backends/ffmpeg.py:611` | Codifica un formato de entrega junto al destino y lo publica; solo reemplaza con `overwrite` |
| `fDecodeMono` | `backend/audio/samples.py:158` | Decodifica con FFmpeg a mono float32, leyendo por partes sin pasar de `cMaxDecodedBytes` |
| `fReadSamples` | `backend/audio/samples.py:123` | Lee con libsndfile y mezcla a mono bloque a bloque tras comprobar el presupuesto |
| `fEnsureReadableCopy` | `backend/audio/analysis.py:111` | Copia decodificada con el nombre de la identidad del origen, dimensionada según el espacio libre y publicada al terminar |
| `_fSweepScratchDirs` | `backend/mcp/tools.py:1419` | Caducidad y cuota de tamaño (primero los menos usados) de los directorios de análisis |
| `fValidateEditPlan` | `backend/validation/validator.py:155` | Valida un plan; se detiene en el recuento de operaciones; devuelve diagnósticos acotados con `fIssueDicts` |
| `ServerState.mCheckBackgroundCapacity` | `backend/mcp/state.py:117` | Rechaza un trabajo en segundo plano cuando los que se ejecutan y los que esperan ocupan todas las plazas |
| `FasterWhisperBackend.mLoadModel` | `backend/transcription/faster_whisper_backend.py:240` | Carga un modelo Whisper y mantiene los dos usados más recientemente |
| `AudacityPipeClient.mConnect` | `backend/backends/audacity.py:308` | Abre las tuberías de scripts cuando `_mCheckPipe` y `_mCheckOpened` las aceptan; vaciado acotado |
| `PodcastPipeline.mReserveOutput` | `backend/pipeline.py:215` | Reserva un directorio o archivo de resultados durante toda una ejecución |
| `fAtomicPath` | `backend/util/atomic.py:32` | Archivo temporal que se publica renombrándolo, o sin reemplazar un destino existente |
| `fAcquireLock` | `backend/util/atomic.py:98` | `flock` sin espera que sigue siendo válido aunque se borre su archivo a la vez |

<a id="flows"></a>
## Flujos

1. **Despliegue local:** entrada del script → `fMain` → argumentos/rutas/root → `fOpenLog` → `fLocateSource` → `fValidateSource` → comprobaciones de sistema/usuario → paquetes → árbol de instalación → `fStageLocalSource` → aplicación/CUDA/lanzadores → configuración/servicio/espacio de trabajo → modelo/Audacity → `fRestartAudacityUnit` → diagnósticos de instalación existentes → trampa EXIT → `fCleanup`.
2. **Despliegue remoto:** mismo flujo, con `fLocateSource` seleccionando la descarga → paquetes con curl/certificados/tar/gzip → `fDownloadSource` → archivo mediante curl → tar → `fValidateSource` → instalación común → trap EXIT → `fCleanup`.
3. **Modo de consulta:** selección del código y precondiciones de lectura → informar de las acciones sobre paquetes/código/árbol/aplicación → omitir diagnósticos finales. No se descarga código ni se modifica el registro.
4. **Procesamiento MCP:** `server.fMain` → `ServerState` / `fBuildServer` → `fRegisterTools` → operación del pipeline → almacén de trabajos, validación y generación.
5. **Procesamiento CLI:** `cli.fMain` → manejador del comando → `PodcastPipeline.mProcess` → análisis/transcripción → plan/validación → ejecución/backends → resultados y estado del trabajo. `master` pasa en cambio por `PodcastPipeline.mMasterFile`: análisis → plan solo de masterización → validación → ejecución → formatos de entrega.
6. **Cliente MCP configurado en una carpeta:** cliente iniciado en la carpeta → su configuración MCP → `server.fMain` con variables relativas a la carpeta → primera llamada a una herramienta → `AppContext` → `Workspace` / `fEnvironmentLocation` → `input/` y `output/` de la carpeta → `get_server_info` → `fCheckProfiles` → `prepare_episode_edit` → `finish_episode_edit` (con `mastering=false`, solo los cortes) → opcionalmente `master_audio` → `PodcastPipeline.mMasterFile` → resultados en el `output/` de la carpeta.
7. **Trabajo en segundo plano:** `process_podcast` / `finish_episode_edit` con `wait=false` → `_fStartBackgroundJob` → `ServerState.admission_lock` → `ServerState.mCheckBackgroundCapacity` → se guarda el `Job` en cola → ejecutor → `PodcastPipeline.mProcess` → la retrollamada de fin quita el future de la tabla.
8. **Publicación de resultados:** `PodcastPipeline.mProcess` / `mMasterFile` / `export_audio` → `PodcastPipeline.mReserveOutput` → `_mCheckOutputTargets` → generación → `FfmpegBackend.mExport` y `fWriteJsonAtomic` mediante `fAtomicPath` (sin reemplazar si no hay `overwrite`) → copia del registro del trabajo → se libera la reserva.

<a id="entry-points"></a>
## Entradas

| Comando o interfaz | Manejador | Archivo |
| --- | --- | --- |
| `bash deploy/install.sh --user NAME` | `fMain` → selección local | `deploy/install.sh` |
| `curl -sL https://raw.githubusercontent.com/nipegun/podcast-editor-mcp-server/refs/heads/main/deploy/install.sh \| bash` | `fMain` → selección remota | `deploy/install.sh` |
| `bash deploy/uninstall.sh` | `fMain` | `deploy/uninstall.sh` |
| `podcast-editor-mcp-server` | `fMain` | `backend/server.py` |
| `podcast-editor-mcp-server-cli` | `fMain` / manejadores de comandos | `backend/cli.py` |
| `podcast-editor-mcp-server-doctor` | `fMain` | `backend/doctor.py` |
| `_/temp/venv/bin/python3 -m podcast_editor_mcp_server.cli` tras la instalación editable | `fMain` | `backend/cli.py` |
| Herramientas MCP, incluidas `prepare_episode_edit` / `finish_episode_edit` / `master_audio` | `fRegisterTools` | `backend/mcp/tools.py` |

<a id="impact"></a>
## Análisis de impacto

- Cambiar la estructura del código exige actualizar `fValidateSource`, la asociación del directorio del paquete, las rutas de las herramientas de desarrollo y la documentación. Los nuevos directorios de módulos se descubren dentro de `backend/`; actualiza la instalación editable para utilizarlos.
- Cambiar el repositorio o la rama afecta a las constantes URL del instalador, los metadatos de Python, la web anunciada por el servidor, la documentación de la unidad y los ejemplos.
- Cambiar la estructura del prefijo afecta a la configuración, los lanzadores, los modelos, los espacios de trabajo y el desinstalador.
- Los consumidores Python importan `podcast_editor_mcp_server`; las variables de entorno utilizan `PODCAST_EDITOR_MCP_SERVER_*`. Los puntos de entrada del paquete, los lanzadores del instalador y la configuración de clientes deben mantener esos nombres. Las herramientas de desarrollo resuelven sus rutas temporales dentro de `_/temp/` de este repositorio.
- Cambiar la planificación o la validación afecta tanto a la CLI como a MCP, porque comparten el pipeline.
- Cambiar los valores predeterminados de los perfiles afecta a los trabajos nuevos; los perfiles personalizados y los ajustes globales persistentes sobreviven a la reinstalación.
- Quitar o renombrar un campo de un modelo persistente (`Job`, `Transcript`, `EditPlan`, `RuntimeConfig`, `Profile`) hace ilegibles los archivos existentes que lo contienen; nada los migra.
- Los clientes MCP dependen de los nombres de las herramientas, de los campos de respuesta (`review_items`, `joins`, `workspace`, `profile_errors`, `error_count`, `warning_count`) y de las claves de perfil; renombrar cualquiera de ellos rompe los clientes que los usan.
- Cambiar `fBaseDirectory`, `fEnvironmentLocation` o las variables de entrada y salida cambia dónde leen y escriben los clientes configurados en una carpeta; mantén alineados los manuales.
- `cInputFormats` debe nombrar un demultiplexor para cada extensión de `cAudioReadExtensions`; si falta uno, esos archivos dejan de poder leerse en todas partes, porque toda entrada de FFmpeg pasa por `fInputArguments`.
- `cMaxDecodedBytes` limita la grabación más larga que se puede analizar muestra a muestra, transcribir o recorrer con Silero; `cScratchMaxBytes` y `cScratchFreeReserveBytes` limitan las copias de análisis en disco.
- Toda ejecución que escribe en `output/` reserva `PodcastPipeline.mReserveOutput`; un nuevo escritor de resultados también tiene que reservarlo o podrá sobrescribir una ejecución simultánea.

<a id="extensions"></a>
## Puntos de extensión

Las nuevas opciones del instalador deben añadirse a `fParseArguments` y `fUsage`, inicializar su estado y proteger sus modificaciones con comprobaciones de `--check`. Los errores de descarga y extracción deben comprobarse expresamente: al invocar `fMain` en un condicional, `set -e` de Bash no es suficiente dentro de las funciones. La obtención temporal de código corresponde a `fDownloadSource`; no debe consumirse stdin del instalador.

Las operaciones de edición nuevas deben implementar su comportamiento compartido en el pipeline o en un módulo específico y exponerse mediante la CLI o `mcp/tools.py`. Los perfiles incluidos en el paquete están en `backend/data/profiles/`; las personalizaciones de cada instalación utilizan directorios personales o compartidos. Este índice de símbolos y los manuales traducidos deben reflejar los cambios en entradas y flujos.

Para aceptar una extensión de audio nueva, añádela a `cAudioReadExtensions` y su demultiplexor a `cInputFormats`, después de comprobar que ese demultiplexor no abre archivos ni URL secundarios. Un paso nuevo que tenga grabaciones enteras en memoria pasa por `fCheckDecodedBudget`, y uno que escriba en `output/` se ejecuta dentro de `PodcastPipeline.mReserveOutput`.
