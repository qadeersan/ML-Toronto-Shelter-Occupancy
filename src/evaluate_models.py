"""
Evaluate all trained models on test set.
"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

def evaluate_models():
    """Evaluate all regression and classification models."""
    
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    print("Loading test data...")
    test_df = pd.read_csv(processed_dir / "test.csv")
    
    # Load feature columns
    feature_cols = joblib.load(models_dir / "feature_columns.pkl")
    
    # Prepare features and targets
    X_test = test_df[feature_cols].values
    y_test_reg = test_df['OCCUPANCY_RATE_BEDS'].values
    y_test_clf = test_df['overcapacity'].values
    
    print(f"Evaluating on {len(X_test)} test samples")
    print(f"Regression target range: {y_test_reg.min():.2f} to {y_test_reg.max():.2f}")
    print(f"Classification target distribution: {np.bincount(y_test_clf)}")
    
    results = {}
    
    # ===== REGRESSION EVALUATION =====
    print("\n" + "="*50)
    print("Regression Model Evaluation")
    print("="*50)
    
    regression_models = {
        'lr': 'regression_lr.pkl',
        'rf': 'regression_rf.pkl',
        'xgb': 'regression_xgb.pkl'
    }
    
    results['regression'] = {}
    
    for name, model_file in regression_models.items():
        print(f"\nEvaluating {name.upper()}...")
        model = joblib.load(models_dir / model_file)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test_reg, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred))
        r2 = r2_score(y_test_reg, y_pred)
        
        results['regression'][name] = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R2': float(r2)
        }
        
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²:   {r2:.4f}")
    
    # ===== CLASSIFICATION EVALUATION =====
    print("\n" + "="*50)
    print("Classification Model Evaluation")
    print("="*50)
    
    classification_models = {
        'lr': 'classification_lr.pkl',
        'xgb': 'classification_xgb.pkl',
        'rf': 'classification_rf.pkl'
    }
    
    results['classification'] = {}
    
    for name, model_file in classification_models.items():
        print(f"\nEvaluating {name.upper()}...")
        model = joblib.load(models_dir / model_file)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        precision = precision_score(y_test_clf, y_pred, zero_division=0)
        recall = recall_score(y_test_clf, y_pred, zero_division=0)
        f1 = f1_score(y_test_clf, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test_clf, y_pred_proba)
        cm = confusion_matrix(y_test_clf, y_pred).tolist()
        
        results['classification'][name] = {
            'Precision': float(precision),
            'Recall': float(recall),
            'F1': float(f1),
            'ROC-AUC': float(roc_auc),
            'Confusion_Matrix': cm
        }
        
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f} (most important for overcapacity detection)")
        print(f"  F1:        {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    {cm}")
    
    # Save results
    results_path = reports_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Evaluation results saved to {results_path}")
    
    # Print best models
    print("\n" + "="*50)
    print("Best Models")
    print("="*50)
    
    # Best regression (lowest RMSE)
    best_reg = min(results['regression'].items(), key=lambda x: x[1]['RMSE'])
    print(f"Best Regression Model: {best_reg[0].upper()} (RMSE: {best_reg[1]['RMSE']:.4f})")
    
    # Best classification (highest recall, then F1)
    best_clf = max(results['classification'].items(), 
                   key=lambda x: (x[1]['Recall'], x[1]['F1']))
    print(f"Best Classification Model: {best_clf[0].upper()} (Recall: {best_clf[1]['Recall']:.4f}, F1: {best_clf[1]['F1']:.4f})")
    
    return results

if __name__ == "__main__":
    evaluate_models()

