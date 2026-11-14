import gc
import hashlib
import hmac
import os
import shutil
import sys
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

# Ensure Python can find modules in the src directory when running via uvicorn from the root
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

from agent import AegisAgent
from artifact_manager import ArtifactManager

print("Booting Aegis Agent Engine...")
agent = AegisAgent()
artifact_manager = ArtifactManager(agent.memory)


# --- Startup Synchronization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup, bootstrap or sync if we missed an update while offline.
    github_repo = os.getenv("GITHUB_REPO")
    if github_repo:
        await artifact_manager.sync_on_startup(repo=github_repo)
    else:
        print("⚠️ GITHUB_REPO not set in .env. Skipping startup bootstrap sync.")

    try:
        agent.bootstrap_dummy_data()
    except Exception as e:
        print(f"⚠️ Failed to bootstrap dummy data (likely due to missing models): {e}")

    yield


app = FastAPI(title="Aegis Agent API", description="Omnisearch Agent REST API", lifespan=lifespan)

cors_origins_env = os.getenv("CORS_ORIGINS", "*")
allow_origins = [origin.strip() for origin in cors_origins_env.split(",") if
                 origin.strip()] if cors_origins_env != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# MLOps: Zero-Downtime Hot Update Webhook
# =====================================================================
@app.post("/webhook/update-artifacts", status_code=202)
async def webhook_update_artifacts(
        request: Request,
        background_tasks: BackgroundTasks,
        x_hub_signature_256: str = Header(None)
):
    """Listens for GitHub Release events to trigger zero-downtime model updates."""
    secret = os.getenv("GITHUB_WEBHOOK_SECRET")
    if not secret:
        raise HTTPException(status_code=500, detail="Webhook secret not configured on server.")

    if not x_hub_signature_256:
        raise HTTPException(status_code=401, detail="Missing signature header")

    body = await request.body()
    expected_mac = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    expected_sig = f"sha256={expected_mac}"

    if not hmac.compare_digest(expected_sig, x_hub_signature_256):
        raise HTTPException(status_code=401, detail="Invalid signature")

    payload = await request.json()
    action = payload.get("action")

    if action != "published":
        return {"msg": "Event ignored (not a publish event)"}

    release = payload.get("release", {})
    version = release.get("tag_name")
    assets = release.get("assets", [])
    repo_name = payload.get("repository", {}).get("full_name")

    if not version or not assets:
        raise HTTPException(status_code=400, detail="Invalid release payload structure.")

    background_tasks.add_task(artifact_manager.process_update, version, assets, repo_name)

    return {"status": "accepted", "msg": f"Artifact update for {version} queued."}


# =====================================================================
# Standard API Endpoints
# =====================================================================

class ChatRequest(BaseModel):
    message: str
    max_iterations: int = 5
    api_key: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    try:
        final_answer = agent.run(
            req.message,
            max_iterations=req.max_iterations,
            api_key=req.api_key
        )
        return ChatResponse(answer=final_answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a text or PDF file, streams it to disk, chunks it per page, and indexes in tiny batches to stay under 512MB RAM."""
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported.")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    chunk_size = 150
    overlap = 30
    total_chunks = 0

    try:
        if file.filename.lower().endswith(".pdf"):
            if PyPDF2 is None:
                raise HTTPException(status_code=500, detail="PyPDF2 is not installed. Run: pip install PyPDF2")

            with open(tmp_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)

                # Process strictly page-by-page to keep RAM flat
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if not page_text:
                        continue

                    words = page_text.split()
                    page_chunks = []
                    page_metadata = []

                    # Generate chunks for the current page
                    i = 0
                    while i < len(words):
                        chunk = " ".join(words[i:i + chunk_size])
                        if chunk.strip():
                            page_chunks.append(chunk)
                            page_metadata.append({"filename": file.filename})
                        i += (chunk_size - overlap)

                    # Ingest using tiny batch_size=2 to prevent ONNX tensor memory spikes
                    if page_chunks:
                        agent.memory.add_texts(page_chunks, page_metadata, batch_size=2)
                        total_chunks += len(page_chunks)

                    # Destroy local page variables and collect garbage immediately
                    del page_text, words, page_chunks, page_metadata
                    gc.collect()

        elif file.filename.lower().endswith(".txt"):
            buffer_words = []
            chunks = []
            metadata = []

            def flush_txt_chunks():
                nonlocal chunks, metadata, total_chunks
                if chunks:
                    agent.memory.add_texts(chunks, metadata, batch_size=2)
                    total_chunks += len(chunks)
                    chunks.clear()
                    metadata.clear()
                    gc.collect()

            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    buffer_words.extend(line.split())
                    while len(buffer_words) >= chunk_size:
                        chunk = " ".join(buffer_words[:chunk_size])
                        chunks.append(chunk)
                        metadata.append({"filename": file.filename})
                        buffer_words = buffer_words[chunk_size - overlap:]

                        if len(chunks) >= 10:
                            flush_txt_chunks()

            if buffer_words:
                chunk = " ".join(buffer_words)
                if chunk.strip():
                    chunks.append(chunk)
                    metadata.append({"filename": file.filename})

            flush_txt_chunks()

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        gc.collect()

    if total_chunks == 0:
        raise HTTPException(status_code=400, detail="No extractable text found in the file.")

    return {
        "status": "success",
        "message": f"Successfully processed '{file.filename}' and added {total_chunks} knowledge chunks to memory."
    }


@app.get("/health")
def health_check():
    return {"status": "online", "system": "Aegis Omnisearch Agent"}
