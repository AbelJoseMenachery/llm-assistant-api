# LLM Assistant API — portfolio project

A Python/FastAPI assistant API with a configurable OpenAI-compatible LLM backend, an Aegis RAG microservice gateway, optional Langfuse OpenAI-call tracing, and Promptfoo model-level smoke evaluations.

- Architecture and Windows setup: [MASTER_INTEGRATION_GUIDE.md](MASTER_INTEGRATION_GUIDE.md)
- Verification boundaries: [INTEGRATION_STATUS.md](INTEGRATION_STATUS.md)
- Promptfoo starter: [evaluations/promptfoo](evaluations/promptfoo)
- Aegis RAG service: [integrations/aegis-rag](integrations/aegis-rag)

The RAG engine is a separate service, not an in-process vector database inside the main API. Langfuse and live-provider tests require user-supplied credentials. See the integration status before describing the project as fully live-tested.
