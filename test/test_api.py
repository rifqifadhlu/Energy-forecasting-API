from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200

def test_prediction_endpoint():
sample_input = {
        "features": [1, 4, 10, 282, 1556, 1984, 1857, 1670.895, 241.555]
    }
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    assert "prediction" in response.json()
