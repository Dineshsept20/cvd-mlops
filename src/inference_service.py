import numpy as np
import onnxruntime as ort
import joblib
import json
import math
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from prometheus_client import make_asgi_app, Counter, Histogram
import os
import traceback
from evidently.report import Report
from evidently.metrics import DataDriftTable
import pandas as pd

app = FastAPI()
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Disable CoreML optimization warnings
os.environ['ORT_ENABLE_ORT_FORMAT'] = '0'

# Prometheus metrics
REQUEST_COUNT = Counter('request_count', 'Total API Request Count')
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request Latency')
PREDICTION_VALUE = Histogram('prediction_value', 'Prediction Value')

# Load artifacts
ecg_session = ort.InferenceSession("artifacts/ecg_model.onnx")
fusion_session = ort.InferenceSession("artifacts/fusion_model.onnx")
ehr_scaler = joblib.load("artifacts/ehr_scaler.joblib")

class EHRInput(BaseModel):
    age: float
    sex: int
    height: float
    weight: float

def safe_float(value):
    """Convert value to float with NaN/Inf handling"""
    try:
        fval = float(value)
        if math.isnan(fval) or math.isinf(fval):
            return 0.0
        return fval
    except (TypeError, ValueError):
        return 0.0

def predict_cvd(ecg_signal: np.ndarray, ehr_data: EHRInput) -> float:
    """Run end-to-end prediction with numerical stability"""
    try:
        # Ensure proper ECG shape
        if ecg_signal.ndim == 1:
            ecg_signal = ecg_signal.reshape(1, 1, -1)
        
        # Preprocess EHR with safe float conversion
        ehr_features = np.array([
            safe_float(ehr_data.age),
            safe_float(ehr_data.sex),
            safe_float(ehr_data.height),
            safe_float(ehr_data.weight)
        ]).reshape(1, -1)
        
        # Check for NaN/inf in EHR
        if np.isnan(ehr_features).any() or np.isinf(ehr_features).any():
            raise ValueError("EHR features contain invalid values after sanitization")
        
        scaled_ehr = ehr_scaler.transform(ehr_features).astype(np.float32)
        
        # Sanitize ECG signal
        ecg_signal = np.nan_to_num(ecg_signal, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # ECG inference
        ecg_input = {"ecg_input": ecg_signal.astype(np.float32)}
        ecg_embedding = ecg_session.run(["ecg_embedding"], ecg_input)[0]
        
        # Fusion inference
        fusion_input = {
            "ecg_embedding": ecg_embedding,
            "ehr_input": scaled_ehr
        }
        logit = fusion_session.run(["risk_logit"], fusion_input)[0][0][0]
        
        # Numerically stable sigmoid with clamping
        if logit > 100:  # Extreme positive
            probability = 1.0
        elif logit < -100:  # Extreme negative
            probability = 0.0
        else:
            probability = 1 / (1 + math.exp(-logit))
        
        # Final clamp to [0,1]
        return max(0.0, min(1.0, probability))
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": f"Internal server error: {str(exc)}"}
    )

# Load reference data
REFERENCE_DATA = pd.read_csv("reference_data.csv")

# Initialize drift report (runs hourly)
last_drift_check = time.time()
drift_data = []

@app.post("/predict", response_class=JSONResponse)
async def predict(
    ehr: str = Form(...),
    ecg_file: UploadFile = File(...)
):
    try:
        REQUEST_COUNT.inc()
        with REQUEST_LATENCY.time():
            # Parse EHR JSON
            try:
                ehr_dict = json.loads(ehr)
                ehr_data = EHRInput(**ehr_dict)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid EHR format: {str(e)}")
            
            # Read and process ECG file
            content = await ecg_file.read()
            
            try:
                ecg_signal = np.frombuffer(content, dtype=np.float32)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"ECG processing error: {str(e)}")
            
            if len(ecg_signal) != 5000:
                raise HTTPException(
                    status_code=400,
                    detail=f"ECG must contain exactly 5000 samples (got {len(ecg_signal)})"
                )
            
            # Run prediction
            probability = predict_cvd(ecg_signal, ehr_data)
            
            # Convert to JSON-safe float
            safe_probability = safe_float(probability)
            
            PREDICTION_VALUE.observe(safe_probability)
            
            return {
                "cvd_risk_probability": safe_probability,
                "risk_category": "High" if safe_probability > 0.4 else "Low"
            }
            
    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_ready": True}