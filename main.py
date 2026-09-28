import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Corporate Fraud Inference API",
    description="High-sensitivity filter for evaluating financial statements.",
    version="1.0.0"
)

# 1. Load both artifacts into memory on startup
try:
    lr_model = joblib.load(BASE_DIR / "lr_fraud_model.joblib")
    scaler = joblib.load(BASE_DIR / "scaler.joblib")
except (FileNotFoundError, OSError, ValueError) as e:
    raise RuntimeError(f"Startup failed. Missing joblib artifacts: {e}")

# Hardcode the optimal business threshold
FRAUD_THRESHOLD = 0.60

# 2. Strict Input Schema
class FinancialStatementInput(BaseModel):
    Receivables_Net_t_minus_1: float = Field(..., alias="Receivables-Net(t-1)")
    Cash_Gen_t: float = Field(..., alias="Cash-Gen(t)")
    SalesGenAdmExpen_R_Dexpense_t_minus_1: float = Field(..., alias="SalesGenAdmExpen - R&Dexpense(t-1)")
    Sales_t: float = Field(..., alias="Sales(t)")
    AQI: float
    DEPI: float
    SGI: float
    DSRI: float
    TATA: float
    GMI: float
    SGAI: float
    LVGI: float

class FraudPredictionResponse(BaseModel):
    fraud_probability: float
    is_fraudulent: bool
    inference_latency_ms: float

# 3. Inference Endpoint
@app.post("/predict", response_model=FraudPredictionResponse)
def predict_fraud(data: FinancialStatementInput):
    start_time = time.perf_counter()
    
    try:
        # Convert to dictionary using original aliases
        input_dict = data.model_dump(by_alias=True)
        
        # Enforce exact column order for the StandardScaler
        expected_columns = [
            'Receivables-Net(t-1)', 'Cash-Gen(t)', 'SalesGenAdmExpen - R&Dexpense(t-1)', 'Sales(t)',
            'AQI', 'DEPI', 'SGI', 'DSRI', 'TATA', 'GMI', 'SGAI', 'LVGI'
        ]
        input_df = pd.DataFrame([input_dict], columns=expected_columns)
        
        # Scale the inputs
        scaled_features = scaler.transform(input_df)
        
        # Generate probability
        probability = float(lr_model.predict_proba(scaled_features)[0, 1])
        is_fraud = probability >= FRAUD_THRESHOLD
        
        latency = (time.perf_counter() - start_time) * 1000
        
        return FraudPredictionResponse(
            fraud_probability=round(probability, 4),
            is_fraudulent=is_fraud,
            inference_latency_ms=round(latency, 2)
        )
        
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)