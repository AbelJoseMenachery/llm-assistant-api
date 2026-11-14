# Promptfoo evaluation starter

Install Promptfoo separately (Node.js required), then from this directory run:

```powershell
$env:PROMPTFOO_MODEL = "gpt-4o-mini"
$env:OPENAI_API_KEY = "your-provider-key"
npx promptfoo eval
npx promptfoo view
```

For Gemini, use Google's OpenAI-compatible endpoint by setting `OPENAI_BASE_URL` to
`https://generativelanguage.googleapis.com/v1beta/openai/` and `OPENAI_API_KEY`
to your Gemini API key; set `PROMPTFOO_MODEL` to a model supported by your account.
Check Google's current model/API compatibility before running. Do not commit keys.
The included tests are smoke checks, not a comprehensive quality benchmark.
