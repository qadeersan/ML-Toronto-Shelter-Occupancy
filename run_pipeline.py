"""
Main pipeline script to run the entire ML pipeline end-to-end.
"""
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def run_pipeline():
    """Run the complete ML pipeline."""
    print("="*60)
    print("ML Pipeline: Shelter Occupancy Prediction")
    print("="*60)
    
    # Step 1: Data Preparation
    print("\n[1/5] Data Preparation...")
    from prepare_data import prepare_data
    prepare_data()
    
    # Step 2: Train/Test Split
    print("\n[2/5] Train/Test Split...")
    from train_test_split import train_test_split
    train_test_split()
    
    # Step 3: Train Models
    print("\n[3/5] Training Models...")
    from train_models import train_models
    train_models()
    
    # Step 4: Evaluate Models
    print("\n[4/5] Evaluating Models...")
    from evaluate_models import evaluate_models
    evaluate_models()
    
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review evaluation results in reports/evaluation_results.json")
    print("2. Test the API locally: uvicorn src.api.main:app")
    print("3. Deploy to fly.io: flyctl deploy")

if __name__ == "__main__":
    run_pipeline()

