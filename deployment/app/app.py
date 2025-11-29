from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List
from model_utils import predict

app = FastAPI(title="Predictive Maintenance API")

# =============================
# CORS MUST COME IMMEDIATELY
# =============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# DATA MODELS
# =============================
class Instance(BaseModel):
    data: Dict[str, Any]

class Batch(BaseModel):
    items: List[Dict[str, Any]]

# =============================
# ROUTES
# =============================
@app.get("/health")
def health():
    return {"status": "ok"}

@app.options("/predict")
def options_predict():
    return {"message": "OK"}   # <--- Handles OPTIONS request

@app.post("/predict")
def predict_single(req: Instance):
    return predict(req.data)

@app.options("/predict_batch")
def options_predict_batch():
    return {"message": "OK"}

@app.post("/predict_batch")
def predict_batch(req: Batch):
    return predict(req.items)
