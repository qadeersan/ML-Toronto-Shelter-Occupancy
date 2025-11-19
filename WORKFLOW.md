# Complete Workflow: From Data to Deployment

## Quick Start Order

### Step 1: Optional - Explore Data (if needed)
**Notebook:** `notebooks/01_EDA.ipynb`
- Downloads data from Toronto OpenData (if not already downloaded)
- Explores data structure and patterns
- Visualizes trends and seasonality
- **Skip if you already have `data/sheltersdata.csv`**

### Step 2: Prepare Data
**Notebook:** `notebooks/02_data_preparation.ipynb`
- Loads raw data
- Creates all features (temporal, lag, rolling, cyclical encoding)
- Selects features for ML
- Saves to `data/processed/features.csv`

### Step 3: Split Data
**Notebook:** `notebooks/03_train_test_split.ipynb`
- Chronological 80/20 train/test split
- Saves to `data/processed/train.csv` and `data/processed/test.csv`

### Step 4: Train Models
**Notebook:** `notebooks/04_model_training.ipynb`
- Trains 3 regression models (Linear, RandomForest, XGBoost)
- Trains 3 classification models (Logistic, XGBoost, RandomForest)
- Saves all models to `models/` directory

### Step 5: Evaluate Models
**Notebook:** `notebooks/05_model_evaluation.ipynb`
- Evaluates all models on test set
- Saves metrics to `reports/evaluation_results.json`
- Identifies best models

### Step 6: Test API Locally
```bash
uvicorn src.api.main:app --reload
```
- Visit http://localhost:8000/docs for API documentation
- Test endpoints:
  - `POST /predict/occupancy` - Predict occupancy rate
  - `POST /predict/overcapacity` - Predict overcapacity risk
  - `GET /health` - Health check

### Step 7: Deploy to Fly.io
```bash
# Install flyctl if needed
# https://fly.io/docs/getting-started/installing-flyctl/

# Login
flyctl auth login

# Launch (first time only - will prompt for app name)
flyctl launch

# Deploy
flyctl deploy
```

## File Dependencies

```
01_EDA.ipynb (optional)
    ↓
02_data_preparation.ipynb
    ↓ creates: data/processed/features.csv
    ↓
03_train_test_split.ipynb
    ↓ creates: data/processed/train.csv, test.csv
    ↓
04_model_training.ipynb
    ↓ creates: models/*.pkl
    ↓
05_model_evaluation.ipynb
    ↓ creates: reports/evaluation_results.json
    ↓
src/api/main.py (uses models and data)
    ↓
Dockerfile + fly.toml
    ↓
fly.io deployment
```

## Prerequisites

1. **Data file:** `data/sheltersdata.csv` (download via 01_EDA.ipynb or manually)
2. **Dependencies:** `pip install -r requirements.txt`
3. **Fly.io account:** Sign up at https://fly.io

## Time Estimate

- Data preparation: ~5 minutes
- Model training: ~10-30 minutes (depending on hardware)
- Evaluation: ~2 minutes
- API testing: ~5 minutes
- Deployment: ~10 minutes

**Total: ~30-60 minutes** (excluding model training time)

## Troubleshooting

### Models not found
- Make sure you've run notebooks 02-04 in order
- Check that `models/` directory contains `.pkl` files

### API errors
- Ensure models are trained and saved
- Check that `data/processed/features.csv` exists
- Verify `models/feature_columns.pkl` exists

### Deployment issues
- Check Dockerfile paths are correct
- Ensure all model files are included in Docker image
- Verify fly.toml configuration

