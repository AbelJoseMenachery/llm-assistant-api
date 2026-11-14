from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch
import httpx

from app.api.v1 import rag

app = FastAPI()
app.include_router(rag.router, prefix="/api/v1/rag")
client = TestClient(app)


def test_rag_health_gateway_success(monkeypatch):
    monkeypatch.setenv("AEGIS_RAG_URL", "http://aegis.test")
    response = httpx.Response(200, json={"status": "online"}, request=httpx.Request("GET", "http://aegis.test/health"))
    with patch("app.api.v1.rag.httpx.get", return_value=response):
        result = client.get("/api/v1/rag/health")
    assert result.status_code == 200
    assert result.json()["status"] == "connected"


def test_rag_query_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = client.post("/api/v1/rag/query", json={"message": "Find the warranty period"})
    assert result.status_code == 503
    assert "GEMINI_API_KEY" in result.json()["detail"]


def test_rag_query_forwards_to_aegis(monkeypatch):
    monkeypatch.setenv("AEGIS_RAG_URL", "http://aegis.test")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    response = httpx.Response(200, json={"answer": "The warranty is 2 years."}, request=httpx.Request("POST", "http://aegis.test/chat"))
    with patch("app.api.v1.rag.httpx.post", return_value=response) as post:
        result = client.post("/api/v1/rag/query", json={"message": "What is the warranty?"})
    assert result.status_code == 200
    assert result.json()["answer"] == "The warranty is 2 years."
    assert post.call_args.kwargs["json"]["api_key"] == "test-key"
