from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

try:
    from model.predict import make_prediction
except Exception:
    make_prediction = None

def simple_fallback_scoring(contract, tenure, charges):
    base = 0.2
    if contract == "Month-to-month":
        base += 0.25
    elif contract == "One year":
        base += 0.10
    else:
        base += 0.05
    base += max(0, (70 - (tenure or 0))) / 140.0
    base += max(0, (charges or 0) - 60) / 200.0
    return float(np.clip(base, 0.01, 0.99))

class Record(BaseModel):
    Contract: str
    Tenure: float
    MonthlyCharges: float

class BatchIn(BaseModel):
    records: List[Record]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def _predict_batch(df):
    if make_prediction is not None:
        try:
            out = make_prediction(input_data=df)
            if isinstance(out, dict):
                if "probabilities" in out:
                    probs = out["probabilities"]
                    if isinstance(probs[0], (list, tuple)):
                        return [float(p[1]) for p in probs]
                    return [float(p) for p in probs]
                if "probas" in out:
                    return [float(p) for p in out["probas"]]
            if hasattr(out, "columns"):
                for col in ["proba_yes", "prob_yes", "p_yes", "churn_proba"]:
                    if col in out.columns:
                        return [float(p) for p in out[col].to_list()]
        except Exception:
            pass
    return [simple_fallback_scoring(r.Contract, r.Tenure, r.MonthlyCharges)
            for r in df.itertuples(index=False)]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict_batch")
def predict_batch(req: BatchIn):
    if not req.records:
        raise HTTPException(status_code=400, detail="no records")
    df = pd.DataFrame([r.model_dump() for r in req.records])
    probas = _predict_batch(df)
    return {"probas": probas}

@app.post("/predict")
def predict_one(rec: Record):
    df = pd.DataFrame([rec.model_dump()])
    proba = _predict_batch(df)[0]
    return {"proba": proba}
