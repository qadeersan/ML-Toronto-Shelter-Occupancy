"""
Train regression and classification models for shelter occupancy prediction.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

def train_models():
    """Train all regression and classification models."""
    
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    print("Loading train data...")
    train_df = pd.read_csv(processed_dir / "train.csv")
    
    # Load feature list
    with open(processed_dir / "feature_list.txt", 'r') as f:
        feature_cols = [line.strip() for line in f.readlines()]
    
    # Prepare features and targets
    X_train = train_df[feature_cols].values
    y_train_reg = train_df['OCCUPANCY_RATE_BEDS'].values
    y_train_clf = train_df['overcapacity'].values
    
    print(f"Training on {len(X_train)} samples with {len(feature_cols)} features")
    print(f"Regression target range: {y_train_reg.min():.2f} to {y_train_reg.max():.2f}")
    print(f"Classification target distribution: {np.bincount(y_train_clf)}")
    
    # ===== REGRESSION MODELS =====
    print("\n" + "="*50)
    print("Training Regression Models")
    print("="*50)
    
    # Baseline: Linear Regression
    print("\n1. Training Linear Regression...")
    lr_reg = LinearRegression()
    lr_reg.fit(X_train, y_train_reg)
    joblib.dump(lr_reg, models_dir / "regression_lr.pkl")
    print("   ✓ Saved to models/regression_lr.pkl")
    
    # Production: Random Forest Regressor
    print("\n2. Training Random Forest Regressor...")
    rf_reg = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf_reg.fit(X_train, y_train_reg)
    joblib.dump(rf_reg, models_dir / "regression_rf.pkl")
    print("   ✓ Saved to models/regression_rf.pkl")
    
    # Advanced: XGBoost Regressor
    print("\n3. Training XGBoost Regressor...")
    xgb_reg = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1
    )
    xgb_reg.fit(X_train, y_train_reg)
    joblib.dump(xgb_reg, models_dir / "regression_xgb.pkl")
    print("   ✓ Saved to models/regression_xgb.pkl")
    
    # ===== CLASSIFICATION MODELS =====
    print("\n" + "="*50)
    print("Training Classification Models")
    print("="*50)
    
    # Baseline: Logistic Regression
    print("\n1. Training Logistic Regression...")
    log_clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, n_jobs=-1)
    log_clf.fit(X_train, y_train_clf)
    joblib.dump(log_clf, models_dir / "classification_lr.pkl")
    print("   ✓ Saved to models/classification_lr.pkl")
    
    # Production: XGBoost Classifier
    print("\n2. Training XGBoost Classifier...")
    xgb_clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=3,
        random_state=42,
        n_jobs=-1
    )
    xgb_clf.fit(X_train, y_train_clf)
    joblib.dump(xgb_clf, models_dir / "classification_xgb.pkl")
    print("   ✓ Saved to models/classification_xgb.pkl")
    
    # Backup: Random Forest Classifier
    print("\n3. Training Random Forest Classifier...")
    rf_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_clf.fit(X_train, y_train_clf)
    joblib.dump(rf_clf, models_dir / "classification_rf.pkl")
    print("   ✓ Saved to models/classification_rf.pkl")
    
    # Save feature columns for inference
    joblib.dump(feature_cols, models_dir / "feature_columns.pkl")
    print("\n✓ Saved feature columns to models/feature_columns.pkl")
    
    print("\n" + "="*50)
    print("All models trained successfully!")
    print("="*50)
    
    return {
        'regression': {
            'lr': lr_reg,
            'rf': rf_reg,
            'xgb': xgb_reg
        },
        'classification': {
            'lr': log_clf,
            'xgb': xgb_clf,
            'rf': rf_clf
        }
    }

if __name__ == "__main__":
    train_models()

