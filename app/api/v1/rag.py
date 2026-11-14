"""Optional gateway to the Aegis RAG microservice."""
import os
from typing import Optional

import httpx
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

router = APIRouter()


class RAGQuery(BaseModel):
    message: str = Field(..., min_length=1, max_length=12000)
    max_iterations: int = Field(5, ge=1, le=10)


class RAGAnswer(BaseModel):
    answer: str


def _settings():
    base = os.getenv("AEGIS_RAG_URL", "http://127.0.0.1:8000").rstrip("/")
    api_key = os.getenv("GEMINI_API_KEY", "")
    timeout = float(os.getenv("RAG_REQUEST_TIMEOUT", "120"))
    return base, api_key, timeout


def _raise_upstream(response: httpx.Response):
    if response.is_error:
        detail = response.text[:1000]
        raise HTTPException(status_code=502, detail=f"Aegis RAG service returned {response.status_code}: {detail}")


@router.get("/health")
def rag_health():
    base, _, timeout = _settings()
    try:
        response = httpx.get(f"{base}/health", timeout=min(timeout, 5.0))
        _raise_upstream(response)
        return {"status": "connected", "service": "aegis-rag", "upstream": response.json()}
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"Aegis RAG unavailable: {type(exc).__name__}") from exc


@router.post("/query", response_model=RAGAnswer)
def rag_query(request: RAGQuery):
    base, api_key, timeout = _settings()
    if not api_key:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not configured for the RAG service")
    try:
        response = httpx.post(
            f"{base}/chat",
            json={"message": request.message, "max_iterations": request.max_iterations, "api_key": api_key},
            timeout=timeout,
        )
        _raise_upstream(response)
        return response.json()
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Aegis RAG request timed out") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Aegis RAG request failed: {type(exc).__name__}") from exc


@router.post("/documents")
def rag_upload(file: UploadFile = File(...)):
    base, _, timeout = _settings()
    try:
        content = file.file.read()
        response = httpx.post(
            f"{base}/upload",
            files={"file": (file.filename or "upload.txt", content, file.content_type or "application/octet-stream")},
            timeout=timeout,
        )
        _raise_upstream(response)
        return response.json()
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Aegis RAG upload timed out") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Aegis RAG upload failed: {type(exc).__name__}") from exc
