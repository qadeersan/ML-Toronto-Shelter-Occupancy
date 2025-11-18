# ML-Toronto-Shelter-Occupancy

Machine Learning pipeline to forecast daily shelter occupancy and predict overcapacity events in Toronto, helping shelters plan resources efficiently.

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
│       └── test.csv               # Test set
├── models/                        # Trained models (.pkl files)
├── notebooks/
│   └── 01_EDA.ipynb              # Exploratory data analysis
├── reports/
│   └── evaluation_results.json   # Model evaluation metrics
├── src/
│   ├── prepare_data.py           # Data preparation
│   ├── train_test_split.py       # Time-series split
│   ├── train_models.py           # Model training
│   ├── evaluate_models.py       # Model evaluation
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

1. **`notebooks/01_EDA.ipynb`** - Exploratory Data Analysis (already done)
2. **`notebooks/02_data_preparation.ipynb`** - Prepare features for ML
3. **`notebooks/03_train_test_split.ipynb`** - Split data chronologically
4. **`notebooks/04_model_training.ipynb`** - Train 6 models (3 regression + 3 classification)
5. **`notebooks/05_model_evaluation.ipynb`** - Evaluate all models

Each notebook saves intermediate results that the next notebook uses.

### 3. Test the API Locally

```bash
uvicorn src.api.main:app --reload
```

Then visit:
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

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

### Alternative: Python Scripts

If you prefer running scripts, the equivalent Python scripts are available in `src/`:
- `src/prepare_data.py`
- `src/train_test_split.py`
- `src/train_models.py`
- `src/evaluate_models.py`

These can be run directly or used as modules.

## License

MIT
