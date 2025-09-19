from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_chat():
    r = client.post("/chat", json={"question":"What is RAG?"})
    assert r.status_code == 200
    assert "answer" in r.json()
    assert isinstance(r.json()["answer"], str)
