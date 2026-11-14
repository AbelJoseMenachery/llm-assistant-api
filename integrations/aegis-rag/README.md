# 🛡️ Aegis Omnisearch Agent

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![ONNX](https://img.shields.io/badge/ONNX_Runtime-Local_Embeddings-blueviolet.svg)](https://onnxruntime.ai/)
[![Gemini](https://img.shields.io/badge/Gemini_3.6_Flash-API-orange.svg)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Aegis** is a lightweight, high-performance Retrieval-Augmented Generation (RAG) AI Agent built **from scratch**
without relying on bloated frameworks like LangChain.

It uses local ONNX models for lightning-fast text embeddings, a custom ReAct (Reason + Act) loop for autonomous tool
usage, and Google's Gemini API for generation. It features a stateless "Bring Your Own Key" (BYOK) architecture,
ensuring maximum user privacy.

---

## 📖 Documentation

Dive deeper into how Aegis works under the hood:

- [🧠 System Architecture](docs/architecture.md) - Learn about the ReAct loop, Vector Store, Local ONNX embeddings, and
  the Zero-Downtime MLOps pipeline.
- [🚀 Usage Guide](docs/usage.md) - Instructions for running locally, via Docker, and managing production artifact
  releases.

---

## ✨ Features

- **Built from Scratch:** Custom RAG pipeline and Agent orchestration for deep system understanding.
- **Local Embeddings:** Uses `onnxruntime` and `all-MiniLM-L6-v2` to generate vector embeddings entirely locally on CPU,
  saving API costs and avoiding PyTorch bloat.
- **Self-Bootstrapping:** Fresh container deployments automatically download the latest inference artifacts from GitHub
  on cold start before accepting requests.
- **Zero-Downtime MLOps:** GitHub Webhook integration for atomic, hot-swappable artifact updates (ONNX models,
  tokenizers) without restarting the server or dropping active requests.
- **Render 512 MB Free-Tier Optimized:** Designed to prevent Out-Of-Memory (OOM) crashes via page-level PDF streaming,
  ultra-low batch sizes ($N=2$), disabled ONNX memory arenas, and proactive Python garbage collection.
- **Autonomous ReAct Agent:** Cycles through Thoughts, Actions, and Observations to solve complex multi-step queries.
- **Grounded Citations:** Automatically cites source documents (e.g., `[Source: manual.pdf]`) to prevent hallucinations.
- **Sliding Window Chunking:** Intelligently chunks uploaded PDFs and TXTs with overlap to preserve semantic context.
- **Stateless BYOK Auth:** Users provide their own Gemini API key dynamically via the UI (fully compatible with browser
  password managers).

---

## 🛠️ Tech Stack

- **Backend:** FastAPI, Python 3.12, Uvicorn
- **AI / LLM:** Google GenAI SDK (`gemini-3.6-flash`)
- **Vector Database:** FAISS (Facebook AI Similarity Search)
- **Embeddings Engine:** ONNX Runtime, HuggingFace Tokenizers
- **Document Processing:** PyPDF2
- **UI:** Streamlit (can be easily replaced with any frontend)

---

## ⚡ Low-Memory & Production Architecture

Aegis is specifically engineered to run within harsh resource constraints (e.g., **Render's 512 MB RAM Free Tier**):

- **ONNX Memory Pools:** Execution uses `enable_cpu_mem_arena = False` and single-threaded execution (
  `intra_op_num_threads = 1`) to eliminate model allocation spikes.
- **Incremental PDF Ingestion:** Text extraction streams page-by-page directly into vector storage rather than
  accumulating full documents into RAM buffers.
- **Micro-Batching:** Embeddings are generated in micro-batches of $N=2$ with immediate garbage collection (
  `gc.collect()`) after processing each page.

---

## ⚡ Quick Start (Local Setup)

1. **Install dependencies:**
   Ensure you are in the `aegis-api` directory.
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   pip install -r requirements.api.txt -r requirements.ui.txt

```

2. **Start the FastAPI Backend:**
```bash
python -m uvicorn src.api:app --reload

```

3. **Start the UI (in a new terminal):**

```bash
python -m streamlit run src/app.py

```

4. **Interact:**
   Open `http://localhost:8501`. Grab a free API key from [Google AI Studio](https://aistudio.google.com/), paste it
   into the sidebar, and upload a document!

---

## 🐳 Docker Deployment

Aegis is fully containerized for easy deployment.

```bash
docker-compose up --build -d

```

* The API will be available at `http://localhost:8000`
* The UI will be available at `http://localhost:8501`

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📝 License

This project is [MIT](https://www.google.com/search?q=LICENSE) licensed.
