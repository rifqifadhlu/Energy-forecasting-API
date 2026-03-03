from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from project.inference import predict_next_24_hours

app = FastAPI(title='Energy Load Forecast API')

class HistoryInput(BaseModel):
    datetime: List[str]
    load: List[float]

@app.get("/")
def root():
    return {"message": "Energy forecast API is running"}

@app.post("/predict")
def predict(data: HistoryInput):
    
    df = pd.DataFrame({
        "Datetime": pd.to_datetime(data.datetime),
        "DAYTON_MW": data.load
    })

    df = df.set_index("Datetime")
    df = df.sort_index()

    if len(df) <168:
        return {"error": "Minimum 168 hours of history required"}

    forecast = predict_next_24_hours(df)

    return {
        "forecast_next_24_hours": forecast
    }