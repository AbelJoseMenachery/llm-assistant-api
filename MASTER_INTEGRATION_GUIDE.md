# LLM Assistant API — integrated portfolio project

This repository keeps the upstream Open Assistant API as the main application and adds three connected portfolio capabilities:

1. **Gemini-compatible LLM backend** — configure the upstream OpenAI-compatible backend with Google's OpenAI-compatible endpoint. The selected model name is supplied by the assistant configuration; verify current model availability for your account.
2. **Aegis RAG microservice gateway** — Aegis remains an independently deployable service (separate process and dependency set). The main API now proxies health checks, document uploads, and grounded chat queries to it.
3. **Langfuse observability** — when the Langfuse SDK is installed and both credentials are configured, the main OpenAI-compatible LLM client is created through Langfuse's OpenAI instrumentation wrapper.
4. **Promptfoo evaluations** — a starter model-level smoke evaluation lives in `evaluations/promptfoo`.

## Important architecture

The assistant API and Aegis RAG are separate services. The gateway routes are:

- `GET /api/v1/rag/health`
- `POST /api/v1/rag/query` with `{"message":"...", "max_iterations":5}`
- `POST /api/v1/rag/documents` with a PDF or TXT multipart upload

The Aegis RAG service uses its own local embedding model/vector memory and Gemini generation. Its first startup may download model artifacts and consume CPU/RAM. Do not start it alongside every other Docker service on a low-memory machine. The Aegis API dependencies are separate from the main API dependencies.

## Main API setup (Windows PowerShell)

From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
py -3.12 -m poetry install --no-root
Copy-Item .env.example .env
```

Edit `.env` locally. Never commit it. For Gemini-compatible chat completions, set:

```ini
OPENAI_API_BASE=https://generativelanguage.googleapis.com/v1beta/openai/
OPENAI_API_KEY=YOUR_GEMINI_API_KEY
LLM_REQUEST_TIMEOUT=90
LLM_MAX_RETRIES=1
```

Choose a model currently enabled for your Google AI Studio/API account when creating an assistant. The fact that `/models` returns a list does not guarantee generation availability; test a small generation request before claiming live-provider success.

Run the main API using the repository's normal database/Redis/object-storage configuration and startup instructions in `README.md`. API docs are exposed at `/docs` when the service is running.

## Start Aegis RAG separately

```powershell
cd integrations\aegis-rag
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.api.txt
Copy-Item .env.example .env
```

Set `GEMINI_API_KEY` in `integrations\aegis-rag\.env` if you want the Aegis UI or its own API to use the key. The main API gateway instead reads `GEMINI_API_KEY` from the main project's `.env` and forwards it to Aegis for `/rag/query`; set `AEGIS_RAG_URL=http://127.0.0.1:8000` in the main `.env`.

Start Aegis:

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Then use the main API gateway routes listed above. For UI usage, install `requirements.ui.txt` as well and start Streamlit with `python -m streamlit run src/app.py` in a separate terminal.

## Langfuse tracing

From the main repository root, install the optional SDK:

```powershell
python -m pip install -r requirements-observability.txt
```

Set these values in the main `.env` only if you have a Langfuse project:

```ini
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

With both keys present and the SDK installed, OpenAI-compatible calls from the main LLM backend are instrumented by Langfuse. Without keys, the project uses the standard OpenAI SDK client. Traces are only delivered when the configured Langfuse endpoint and credentials are valid.

## Promptfoo evaluation

Install Node.js, then from `evaluations\promptfoo` run:

```powershell
$env:PROMPTFOO_MODEL = "YOUR_ENABLED_MODEL_ID"
$env:OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
$env:OPENAI_API_KEY = "YOUR_GEMINI_API_KEY"
npx promptfoo eval
npx promptfoo view
```

This is a **model-level smoke evaluation**, not an end-to-end evaluation of the full assistant or RAG pipeline. Keep credentials in the shell environment; do not put them in YAML or commit them.

## Tests and evidence

Run deterministic gateway tests:

```powershell
python -m pytest tests/integration -q
```

The included integration tests mock Aegis responses; they verify gateway routing and missing-key handling without requiring a live Gemini API or a running Aegis server. They do not certify live Gemini generation, model downloads, Langfuse delivery, Promptfoo provider evaluation, or the full database-backed upstream API.

See `INTEGRATION_STATUS.md` for the exact verification boundary. Never describe a live provider or end-to-end test as passed unless you have run it successfully with your own credentials and services.
