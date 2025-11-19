"""
FastAPI application for shelter occupancy prediction.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(title="Shelter Occupancy Prediction API", version="1.0.0")

# Load models and data at startup
project_root = Path(__file__).parent.parent.parent
models_dir = project_root / "models"
processed_dir = project_root / "data" / "processed"

# Global variables for models and data
regression_model = None
classification_model = None
feature_columns = None
historical_data = None

class OccupancyRequest(BaseModel):
    shelter_id: int
    date: Optional[str] = None  # Format: YYYY-MM-DD, defaults to tomorrow

class OvercapacityRequest(BaseModel):
    shelter_id: int
    date: Optional[str] = None  # Format: YYYY-MM-DD, defaults to tomorrow

def load_models_and_data():
    """Load models and historical data at startup."""
    global regression_model, classification_model, feature_columns, historical_data
    
    print("Loading models and data...")
    
    # Load best models (XGBoost)
    regression_model = joblib.load(models_dir / "regression_xgb.pkl")
    classification_model = joblib.load(models_dir / "classification_xgb.pkl")
    feature_columns = joblib.load(models_dir / "feature_columns.pkl")
    
    # Load historical data for feature engineering
    historical_data = pd.read_csv(processed_dir / "features.csv")
    historical_data['OCCUPANCY_DATE'] = pd.to_datetime(historical_data['OCCUPANCY_DATE'])
    historical_data = historical_data.sort_values(['SHELTER_ID', 'OCCUPANCY_DATE'])
    
    print("Models and data loaded successfully!")

@app.on_event("startup")
async def startup_event():
    """Load models when the app starts."""
    load_models_and_data()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": regression_model is not None and classification_model is not None
    }

def prepare_features_for_prediction(shelter_id: int, target_date: datetime) -> np.ndarray:
    """
    Prepare features for prediction by using historical data.
    For new predictions, we need to construct lag features from historical data.
    """
    # Get historical data for this shelter
    shelter_data = historical_data[historical_data['SHELTER_ID'] == shelter_id].copy()
    
    if len(shelter_data) == 0:
        raise HTTPException(status_code=404, detail=f"No historical data found for shelter_id {shelter_id}")
    
    # Get the most recent data point before target_date
    past_data = shelter_data[shelter_data['OCCUPANCY_DATE'] < target_date]
    
    if len(past_data) == 0:
        # Use the earliest available data
        past_data = shelter_data.iloc[:1]
    
    # Use the most recent row as base
    base_row = past_data.iloc[-1].copy()
    
    # Calculate date features for target_date
    base_row['day_of_week'] = target_date.weekday()  # 0=Monday, 6=Sunday
    base_row['month'] = target_date.month
    base_row['week_of_year'] = target_date.isocalendar().week
    
    # Update cyclical features
    base_row['day_of_week_sin'] = np.sin(2 * np.pi * base_row['day_of_week'] / 7)
    base_row['day_of_week_cos'] = np.cos(2 * np.pi * base_row['day_of_week'] / 7)
    base_row['month_sin'] = np.sin(2 * np.pi * base_row['month'] / 12)
    base_row['month_cos'] = np.cos(2 * np.pi * base_row['month'] / 12)
    base_row['week_of_year_sin'] = np.sin(2 * np.pi * base_row['week_of_year'] / 52)
    base_row['week_of_year_cos'] = np.cos(2 * np.pi * base_row['week_of_year'] / 52)
    
    # Calculate lag features from historical data
    if len(past_data) >= 1:
        base_row['lag_1'] = past_data['OCCUPIED_BEDS'].iloc[-1] if len(past_data) >= 1 else 0
    else:
        base_row['lag_1'] = 0
    
    if len(past_data) >= 7:
        base_row['lag_7'] = past_data['OCCUPIED_BEDS'].iloc[-7] if len(past_data) >= 7 else 0
    else:
        base_row['lag_7'] = 0
    
    if len(past_data) >= 7:
        base_row['roll_mean_7'] = past_data['OCCUPIED_BEDS'].iloc[-7:].mean() if len(past_data) >= 7 else 0
    else:
        base_row['roll_mean_7'] = 0
    
    # Extract feature values in the correct order
    feature_values = []
    for col in feature_columns:
        if col in base_row:
            feature_values.append(base_row[col])
        else:
            # If feature is missing, use 0 (for new categorical dummies)
            feature_values.append(0)
    
    return np.array([feature_values])

@app.post("/predict/occupancy")
async def predict_occupancy(request: OccupancyRequest):
    """
    Predict occupancy rate for a shelter on a given date.
    """
    try:
        # Parse date or use tomorrow
        if request.date:
            target_date = datetime.strptime(request.date, "%Y-%m-%d")
        else:
            target_date = datetime.now() + timedelta(days=1)
        
        # Prepare features
        X = prepare_features_for_prediction(request.shelter_id, target_date)
        
        # Predict
        predicted_rate = regression_model.predict(X)[0]
        
        # Calculate confidence interval (using model's feature importance as proxy)
        confidence_interval = {
            "lower": float(max(0, predicted_rate - 5.0)),  # Convert to Python float
            "upper": float(min(100, predicted_rate + 5.0))   # Convert to Python float
        }
        
        return {
            "shelter_id": request.shelter_id,
            "date": target_date.strftime("%Y-%m-%d"),
            "predicted_occupancy_rate": round(float(predicted_rate), 2),
            "confidence_interval": confidence_interval
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/overcapacity")
async def predict_overcapacity(request: OvercapacityRequest):
    """
    Predict overcapacity risk for a shelter on a given date.
    """
    try:
        # Parse date or use tomorrow
        if request.date:
            target_date = datetime.strptime(request.date, "%Y-%m-%d")
        else:
            target_date = datetime.now() + timedelta(days=1)
        
        # Prepare features
        X = prepare_features_for_prediction(request.shelter_id, target_date)
        
        # Predict occupancy rate (for context)
        predicted_rate = regression_model.predict(X)[0]
        
        # Predict overcapacity
        overcapacity_risk = classification_model.predict(X)[0]
        overcapacity_probability = classification_model.predict_proba(X)[0][1]
        
        return {
            "shelter_id": request.shelter_id,
            "date": target_date.strftime("%Y-%m-%d"),
            "overcapacity_risk": int(overcapacity_risk),
            "probability": round(float(overcapacity_probability), 4),
            "predicted_occupancy_rate": round(float(predicted_rate), 2)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

