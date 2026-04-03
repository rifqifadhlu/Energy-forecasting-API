from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200

def test_prediction_endpoint():
    sample_input = {
        "features": [100, 102, 98, 101, 99]
    }

    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    assert "prediction" in response.json()