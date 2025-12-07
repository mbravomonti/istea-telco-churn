import pandas as pd
import joblib
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Define paths
MODEL_PATH = os.path.join('models', 'model.pkl')
ENCODERS_PATH = os.path.join('models', 'encoders.pkl')

# Initialize FastAPI app
app = FastAPI(title="Telco Churn Prediction API", version="1.0")

# Load model and encoders
try:
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    print("Model and encoders loaded successfully.")
except Exception as e:
    print(f"Error loading model/encoders: {e}")
    model = None
    encoders = None

# Define input data schema
class CustomerData(BaseModel):
    age: int
    gender: str
    region: str
    contract_type: str
    tenure_months: int
    monthly_charges: float
    total_charges: float
    internet_service: str
    phone_service: str
    multiple_lines: str
    payment_method: str

@app.get("/")
def read_root():
    return {"message": "Welcome to Telco Churn Prediction API"}

@app.post("/predict")
def predict_churn(data: CustomerData):
    if model is None or encoders is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input data to DataFrame
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])
        
        # Normalize text columns (same as data_prep.py)
        # Note: In a real scenario, this logic should be shared/imported
        # Here we assume input comes clean or we apply minimal cleaning
        
        # Apply encoding
        for column, le in encoders.items():
            if column in df.columns:
                # Handle unseen labels if necessary, for now assume valid input
                # Using map/apply to handle potential errors gracefully or just transform
                try:
                    df[column] = le.transform(df[column])
                except ValueError:
                     raise HTTPException(status_code=400, detail=f"Invalid value for {column}")

        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        return {
            "churn_prediction": int(prediction),
            "churn_probability": float(probability),
            "message": "Customer is likely to churn" if prediction == 1 else "Customer is likely to stay"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
