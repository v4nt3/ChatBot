# DocuBot AI

DocuBot AI es un proyecto académico para consultar documentos PDF y TXT desde Telegram o una API REST. Usa TF-IDF y similitud coseno para buscar fragmentos, Gemini para responder en lenguaje natural y SQLite para guardar documentos y consultas. No entrena un modelo supervisado.

## Requisitos

- Python 3.11 o superior.
- Token de Telegram creado con BotFather para usar el bot.
- Clave de Google AI Studio y un identificador de modelo que esté disponible en esa cuenta para usar Gemini.

## Instalación

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Completa `.env`. No subas ese archivo al repositorio.

```env
TELEGRAM_BOT_TOKEN=token_entregado_por_BotFather
GEMINI_API_KEY=clave_de_Google_AI_Studio
GEMINI_MODEL=identificador_disponible_en_tu_cuenta
API_TOKEN=un_token_largo_y_aleatorio
API_USER_ID=usuario-academico
MAX_FILE_SIZE_MB=10
TELEGRAM_READ_TIMEOUT=30
TELEGRAM_GET_UPDATES_READ_TIMEOUT=40
SIMILITUD_MINIMA=0.05
DATABASE_PATH=storage/docubot.sqlite3
DOCUMENTS_DIR=storage/documents
MAX_CONTEXT_CHARS=100000
```

El nombre del modelo no viene preseleccionado: consúltalo en Google AI Studio y configúralo en `GEMINI_MODEL`. La API REST usa un token Bearer y una identidad fija (`API_USER_ID`), adecuada para una demostración académica de un solo usuario API. Telegram identifica a cada usuario por su ID.

## Ejecución

En una terminal, inicia la API:

```powershell
python -m uvicorn app.main:app --reload
```

La documentación interactiva queda en `http://127.0.0.1:8000/docs`. En otra terminal, inicia el bot:

```powershell
python -m app.bot
```

En Telegram usa `/start`, envía un PDF o TXT, elige un documento con `/documentos`, elige `/modo` y escribe una pregunta. `/estadisticas` muestra el resumen de actividad.

Las llamadas a Telegram usan timeouts ampliados configurables. `TELEGRAM_READ_TIMEOUT` aplica a las solicitudes normales y `TELEGRAM_GET_UPDATES_READ_TIMEOUT` al sondeo de mensajes. Si persisten los timeouts, revisa la conexión y el acceso de la red a `api.telegram.org`.

## Modos de consulta

- **TF-IDF** transforma los fragmentos y la pregunta con `TfidfVectorizer`, calcula `cosine_similarity`, ordena las puntuaciones y muestra hasta cinco resultados sobre el umbral. No llama a Gemini.
- **LLM** envía a Gemini el texto completo del documento. Si excede `MAX_CONTEXT_CHARS`, la consulta se rechaza sin truncar el texto.
- **Híbrido** recupera hasta cinco fragmentos con TF-IDF y envía solo esos fragmentos a Gemini.

El vectorizador y la matriz TF-IDF se generan al procesar el documento y se guardan como índice local. Los fragmentos mantienen la página de origen en PDF. No se aplica OCR.

## Estructura

```text
app/                  API, bot, persistencia y lógica compartida
storage/documents/    Archivos privados cargados
storage/indices/      Índices TF-IDF locales
tests/                Pruebas básicas
```

SQLite almacena documentos, fragmentos y conversaciones. El modo y el documento elegidos en Telegram se mantienen durante la ejecución del bot, por usuario.

## API REST

Excepto `/health`, los endpoints requieren `Authorization: Bearer <API_TOKEN>`.

| Método | Ruta | Uso |
|---|---|---|
| GET | `/health` | Estado del servicio |
| POST | `/documents/upload` | Carga multipart con campo `file` |
| GET | `/documents` | Lista documentos del usuario API |
| GET | `/documents/{document_id}` | Detalle del documento |
| DELETE | `/documents/{document_id}` | Elimina documento y archivos asociados |
| POST | `/chat/query` | Consulta `{ "document_id": 1, "mode": "hibrido", "question": "..." }` |
| GET | `/analytics/summary` | Estadísticas básicas |

## Pruebas

```powershell
python -m pytest
```

Las pruebas principales no necesitan credenciales de Telegram ni Gemini. Para probar Gemini en vivo, configura las variables de entorno. Los documentos sin texto extraíble no se procesan.

## Evaluación experimental y gráficas

El módulo `app.evaluacion` ejecuta comparaciones sobre documentos reales ya cargados. No incluye preguntas, puntuaciones ni respuestas inventadas; los CSV iniciales contienen solo encabezados y las gráficas aparecen cuando hay observaciones suficientes.

1. Exporta los fragmentos de un documento para revisar sus IDs y contenido:

   ```powershell
   python -m app.evaluacion --exportar-fragmentos 1 --usuario 123456
   ```

   Sustituye esos valores por el ID real del documento y su propietario. El CSV se guarda en `results/fragmentos_documento_1.csv`.

2. Completa `results/preguntas_prueba.csv` con al menos dos preguntas reales. Usa una fila por pregunta y las mismas preguntas/documentos serán ejecutados en las tres modalidades. En `relevant_chunk_ids`, escribe los IDs de fragmentos juzgados manualmente como relevantes, separados por punto y coma (`12;15`). Si no identificas fragmentos relevantes, deja el campo vacío; Recall de esa pregunta se omite explícitamente. `expected_answer` es la referencia para la evaluación humana.

3. Ejecuta el experimento:

   ```powershell
   python -m app.evaluacion
   ```

   Se ejecuta una consulta de calentamiento por modalidad usando el primer caso. Las filas quedan marcadas `is_warmup=True` en tiempos y tokens y se excluyen de las figuras. Después se ejecutan los casos medidos. Los modos rotan de orden entre preguntas para reducir el sesgo de orden. Los calentamientos de LLM e híbrido sí hacen llamadas reales a Gemini y pueden consumir cuota.

4. Revisa las respuestas en `results/calidad_respuestas.csv` y asigna manualmente `grade` con exactamente una de estas categorías: `Correcta`, `Parcial`, `Incorrecta` o `Sin información`. Contrasta cada respuesta con el documento. No se asignan categorías automáticamente. Regenera las figuras después de completar la evaluación:

   ```powershell
   python -m app.evaluacion --solo-graficas
   ```

El experimento guarda datos medidos en `results/tiempos_respuesta.csv`, `metricas_retrieval.csv`, `evaluacion_umbral.csv`, `calidad_respuestas.csv` y `consumo_tokens.csv`. Genera PNG de 300 DPI en `graficas/` cuando cada conjunto contiene datos válidos. La figura de tokens usa solo `prompt_token_count` reportado por Gemini; si falta, no se estima ni se genera esa figura. También se registran tokens de salida cuando están disponibles.

La medición de latencia abarca el procesamiento local y la respuesta de Gemini, pero no el tiempo de red de Telegram. Precision@K usa los fragmentos realmente recuperados hasta K. Si una pregunta no tiene IDs relevantes anotados, su precisión se calcula considerando que ningún fragmento recuperado coincide y su Recall se omite. Para umbrales, el fragmento se considera recuperado cuando su similitud es estrictamente mayor que el umbral; Precision y Recall se promedian por pregunta, y preguntas sin juicios positivos se omiten del promedio de Recall. Las filas de cada ejecución llevan un `run_id`, de modo que se pueden distinguir y analizar ejecuciones repetidas.
