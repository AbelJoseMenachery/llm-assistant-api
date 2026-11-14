# Langfuse observability integration

The main project already has an OpenAI-compatible LLM backend. This optional example shows
how to instrument an OpenAI-compatible request with the current Langfuse Python SDK v4.

Install from the project root:
`pip install langfuse`

Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL` in your local
`.env` file. Never commit real credentials. See `example.py` for a minimal integration
pattern. This is an opt-in example; the upstream application is not silently patched.
