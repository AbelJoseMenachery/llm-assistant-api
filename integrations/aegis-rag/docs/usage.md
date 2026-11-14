# 🚀 Usage Guide

## Getting a Gemini API Key

Aegis relies on Google's Gemini models for its Agentic reasoning.

1. Go to [Google AI Studio](https://aistudio.google.com/).
2. Sign in with your Google Account.
3. Click **"Get API key"** and create a new key.
4. Google provides a highly generous free tier (15 requests/minute, 1M tokens/minute) which is more than enough for
   Aegis's ReAct loops.

## Using the UI (Streamlit)

1. **Authenticate:** On the left sidebar, paste your Gemini API Key into the password field. Your browser (e.g., Chrome,
   Edge) will prompt you to save this password. We highly recommend doing so for a seamless, 1-click experience in the
   future.
2. **Upload Documents:** Use the "Add Knowledge" uploader in the sidebar to ingest `.txt` or `.pdf` files.
3. **Chat:** Ask Aegis a question. For example:
    * *"What is the main topic of the report I just uploaded?"*
    * *"Calculate 234 * 85 for me."* (Aegis will intelligently switch to the calculator tool instead of searching the
      documents).

## Using the API (Headless Mode)

Aegis exposes a REST API via FastAPI, allowing you to connect any custom frontend (e.g., React, Vue, iOS).

### 1. Upload a Document (`POST /upload`)

Uploads a document, chunks it, embeds it locally, and stores it in memory.

**Request (multipart/form-data):**

* `file`: The `.txt` or `.pdf` file.

**Response:**

```json
{
  "status": "success",
  "message": "Successfully processed 'manual.pdf' and added 42 knowledge chunks to memory."
}
```

### 2. Chat with the Agent (`POST /chat`)

Triggers the ReAct agent loop.

**Request (application/json):**

```json
{
  "message": "What does the manual say about error 404?",
  "api_key": "YOUR_GEMINI_API_KEY",
  "max_iterations": 5
}
```

**Response:**

```json
{
  "answer": "According to [Source: manual.pdf], error 404 indicates a loss of connection to the mainframe."
}
```

---

## ⚙️ MLOps: Zero-Downtime Artifact Updates

Aegis supports enterprise-grade hot swapping of inference artifacts (ONNX models, tokenizers) in production without ever
taking down the API. It also supports **Self-Bootstrapping**, meaning you can deploy a completely empty Docker
container, and it will automatically download its required models before starting.

### 1. Configure the Environment

Ensure your `.env` on the production server contains:

```env
GITHUB_WEBHOOK_SECRET=your_super_secret_string
GITHUB_TOKEN=your_github_personal_access_token
GITHUB_REPO=yourusername/aegis
```

*Note: `GITHUB_REPO` is crucial. It tells a completely fresh, empty deployment exactly where to download its initial
models from on its very first boot.*

### 2. Configure GitHub Webhook

1. Go to your GitHub Repository -> **Settings** -> **Webhooks** -> **Add webhook**.
2. **Payload URL:** `https://your-api.com/webhook/update-artifacts`
3. **Content type:** `application/json`
4. **Secret:** Match the `GITHUB_WEBHOOK_SECRET` from your `.env`.
5. **Which events?** Select *Let me select individual events*, and ONLY check **Releases**.

### 3. Trigger a Release

Aegis uses GitHub Actions to automate the export of the ONNX embedder and the creation of release artifacts. You do not
need to push artifacts manually.

*Note: During inference and deployment, you may see a warning from the `transformers` library indicating that PyTorch is
missing. This is completely expected and safe to ignore, as PyTorch is only used during the GitHub Action workflow to
export the model, and is not required for ONNX deployment.*

**Option 1: Trigger from the Command Line (GitHub CLI)**
Ensure you have the GitHub CLI (`gh`) installed and authenticated.

Run the GitHub Actions release workflow:

```bash
gh workflow run release.yml -f version=v1.0.0
```

**Option 2: Trigger from the GitHub Web UI**

1. Go to your repository on GitHub.
2. Click the **Actions** tab.
3. Select the **Release Artifacts** workflow from the left sidebar.
4. Click **Run workflow**.
5. Enter the release version (for example, `v1.0.0`).
6. Click **Run workflow**.

**What happens next?**

1. GitHub Actions will spin up a fresh runner, install dependencies, export the embedder to ONNX, and create a GitHub
   Release with the artifacts.
2. The webhook fires and hits your live production API.
3. The API downloads the files asynchronously into a `temp/` folder.
4. It loads the new models into memory.
5. It performs an atomic pointer swap, replacing the old models with the new ones.
6. Your production app is now running the new models with **0 seconds of downtime.**

---
