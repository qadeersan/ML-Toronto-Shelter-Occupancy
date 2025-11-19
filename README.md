# ML - Toronto Shelter Occupancy & Overcapacity Predictor

Machine Learning pipeline to forecast daily shelter occupancy and predict overcapacity events in Toronto, helping shelters plan resources efficiently.
Dataset provided by Toronto Open Data for 107 shelters.

## Project Overview

This project implements an end-to-end ML pipeline that:
- **Predicts occupancy rates** for individual shelters (regression)
- **Classifies overcapacity risk** (>95% occupancy) for early warning (classification)
- **Deploys a FastAPI service** for real-time predictions

## Project Structure

```
ML-Toronto-Shelter-Occupancy/
├── data/
│   ├── sheltersdata.csv          # Raw data
│   └── processed/
│       ├── features.csv          # Processed features
│       ├── train.csv             # Training set
│       └── test.csv              # Test set
├── models/                       # Trained models (.pkl files)
├── notebooks/
│   ├── 01_EDA.ipynb              # Exploratory data analysis
|   ├── 02_data_preparation.ipynb # Clean and organize data
|   ├── 03_train_test_split.ipynb # Split Data
|   ├── 04_model_training.ipynb   # Train models & save
|   └── 05_model_evaluation.ipynb # Score models for best use
├── reports/
│   └── evaluation_results.json   # Model evaluation metrics
├── src/
│   ├── prepare_data.py           # Data preparation
│   ├── train_test_split.py       # Time-series split
│   ├── train_models.py           # Model training
│   ├── evaluate_models.py        # Model evaluation
│   └── api/
│       └── main.py               # FastAPI application
├── Dockerfile
├── fly.toml
├── requirements.txt
└── run_pipeline.py               # Main pipeline script
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline (Notebook-based)

Execute the notebooks in order:

1. **`notebooks/01_EDA.ipynb`** - *(Optional)* Exploratory Data Analysis - only needed if you need to download data
2. **`notebooks/02_data_preparation.ipynb`** - Prepare features for ML
3. **`notebooks/03_train_test_split.ipynb`** - Split data chronologically
4. **`notebooks/04_model_training.ipynb`** - Train 6 models (3 regression + 3 classification)
5. **`notebooks/05_model_evaluation.ipynb`** - Evaluate all models

Each notebook saves intermediate results that the next notebook uses.

**See `WORKFLOW.md` for detailed step-by-step instructions including deployment.**

### 3. Test the API Locally

```bash
uvicorn src.api.main:app --reload
```

Then visit:
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

```
Unfortunately most shelters in Toronto are more often than not overcapacity (95%), the ones with lower averge occupancy rates are easier
to test these models with.

Top 20 Shelters with Lowest Average Occupancy Rates:
================================================================================
 SHELTER_ID      mean   min   max       std  count
         82 63.385821  2.27 100.0 21.367927    201
        104 64.333727 36.36  78.0 13.942934    161
        100 66.299412  8.33 100.0 27.572852     68
        102 68.017423 23.08 100.0 15.144154     97
         12 76.290254 33.33 100.0 14.836759   2714
          2 80.945289 15.00 100.0 18.910124    450
        101 84.370977 33.33 100.0 13.763914    266
         95 84.865632  2.00 100.0 23.937745    277
        103 86.212368  6.67 100.0 27.092714    380
         60 89.473303 20.00 100.0 12.027366   1850
          6 89.755512  6.67 100.0 13.209352   1945
         41 90.113355 45.45 100.0 12.638065   3562
         85 91.041322 33.33 100.0 13.285284   3541
         26 91.482231  7.14 100.0 13.337116    771
          1 91.632717 23.53 100.0 11.390886   3913
         20 91.798935  6.67 100.0 17.029647   6582
         53 92.192437 16.67 100.0 14.979324    279
         52 93.226443 50.00 100.0 10.225878   3562
         17 93.415759 58.82 100.0  7.808175    540
         40 93.757829 12.50 100.0 15.533092   2395

================================================================================
```

### 4. Deploy to Fly.io

```bash
# Install flyctl if needed
# https://fly.io/docs/getting-started/installing-flyctl/

# Login to Fly.io
flyctl auth login

# Launch the app (first time)
flyctl launch

# Deploy
flyctl deploy
```
My deployment loom:  https://www.loom.com/share/d692500050394f51a2c3854bb9f0a35e

## API Endpoints

### Predict Occupancy Rate

```bash
curl -X POST "http://localhost:8000/predict/occupancy" \
  -H "Content-Type: application/json" \
  -d '{"shelter_id": 40, "date": "2025-01-15"}'
```

Response:
```json
{
  "shelter_id": 40,
  "date": "2025-01-15",
  "predicted_occupancy_rate": 96.5,
  "confidence_interval": {
    "lower": 91.5,
    "upper": 101.5
  }
}
```

### Predict Overcapacity Risk

```bash
curl -X POST "http://localhost:8000/predict/overcapacity" \
  -H "Content-Type: application/json" \
  -d '{"shelter_id": 40, "date": "2025-01-15"}'
```

Response:
```json
{
  "shelter_id": 40,
  "date": "2025-01-15",
  "overcapacity_risk": 1,
  "probability": 0.85,
  "predicted_occupancy_rate": 96.5
}
```

## Models

### Regression Models (Predict Occupancy Rate)
- **Linear Regression** (baseline)
- **Random Forest Regressor** (production-ready)
- **XGBoost Regressor** (best performance)

### Classification Models (Predict Overcapacity)
- **Logistic Regression** (baseline)
- **XGBoost Classifier** (best performance)
- **Random Forest Classifier** (backup)

## Features

The models use:
- **Temporal features**: Cyclical encoding (sin/cos) for day of week, month, week of year
- **Lag features**: Previous day (lag_1), 7 days ago (lag_7)
- **Rolling features**: 7-day rolling average
- **Categorical features**: One-hot encoded SECTOR, PROGRAM_MODEL, OVERNIGHT_SERVICE_TYPE, PROGRAM_AREA
- **Numeric features**: Capacity, service user count, etc.

## Evaluation Metrics

- **Regression**: MAE, RMSE, R²
- **Classification**: Precision, Recall, F1, ROC-AUC

Results are saved in `reports/evaluation_results.json`.

## Development

### Notebook Workflow

The pipeline is designed to run as Jupyter notebooks for:
- **Interactive exploration** - See outputs and visualizations at each step
- **Reproducibility** - Clear documentation of each transformation
- **Debugging** - Easy to inspect intermediate results
- **Consistency** - Matches the EDA workflow


