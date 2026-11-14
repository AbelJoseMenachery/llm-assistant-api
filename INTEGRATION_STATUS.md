# Integration status and verification boundaries

## Implemented in this bundle
- Original Open Assistant API source retained.
- OpenAI-compatible LLM backend supports configurable timeout/retry settings.
- Optional Langfuse OpenAI SDK instrumentation activates when both Langfuse keys are set and `langfuse` is installed.
- Prompt/completion/tool payloads are no longer written to application logs by the LLM backend.
- Aegis RAG is included as a separate FastAPI microservice. The main API exposes `/api/v1/rag/health`, `/api/v1/rag/query`, and `/api/v1/rag/documents` as a gateway to it.
- Promptfoo smoke evaluation configuration is included under `evaluations/promptfoo`.

## Not claimed as verified here
- Live Gemini generation: the supplied machine logs show model-list HTTP 200, but generation timed out and streaming returned HTTP 503.
- Live Aegis model bootstrap, embedding downloads, and document ingestion.
- Delivery of traces to a real Langfuse project.
- Promptfoo live-provider evaluation.
- Full upstream database/Redis/MinIO application end-to-end test.

Run deterministic local tests with `python -m pytest tests/integration -q`. Live tests require your own services and credentials.
