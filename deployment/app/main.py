# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List
from model_utils import predict

app = FastAPI(title="Predictive Maintenance API")

class Instance(BaseModel):
    data: Dict[str, Any]       # For single prediction

class Batch(BaseModel):
    items: List[Dict[str, Any]]  # For batch prediction

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict_single(req: Instance):
    return predict(req.data)

@app.post("/predict_batch")
def predict_batch(req: Batch):
    return predict(req.items)
