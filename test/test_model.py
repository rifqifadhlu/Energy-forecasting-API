import joblib
import numpy as np

def test_model_load():
    model = joblib.load("model.pkl")
    assert model is not None

def test_model_prediction_shape():
    model = joblib.load("model.pkl")
    sample = np.random.rand(1, 5)
    prediction = model.predict(sample)
    assert prediction.shape[0] == 1