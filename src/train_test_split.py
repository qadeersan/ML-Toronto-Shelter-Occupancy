"""
Time-series aware train/test split.
Splits data chronologically (80% train, 20% test) while maintaining per-shelter grouping.
"""
import pandas as pd
from pathlib import Path

def train_test_split():
    """Split data chronologically for time-series modeling."""
    
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / "data" / "processed"
    
    print("Loading processed features...")
    df = pd.read_csv(processed_dir / "features.csv")
    df['OCCUPANCY_DATE'] = pd.to_datetime(df['OCCUPANCY_DATE'])
    
    # Sort by SHELTER_ID and OCCUPANCY_DATE
    df = df.sort_values(['SHELTER_ID', 'OCCUPANCY_DATE']).reset_index(drop=True)
    
    # Chronological split: 80% earliest dates, 20% latest dates
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"\nTrain set: {len(train_df)} rows ({len(train_df) / len(df) * 100:.1f}%)")
    print(f"Test set: {len(test_df)} rows ({len(test_df) / len(df) * 100:.1f}%)")
    print(f"\nTrain date range: {train_df['OCCUPANCY_DATE'].min()} to {train_df['OCCUPANCY_DATE'].max()}")
    print(f"Test date range: {test_df['OCCUPANCY_DATE'].min()} to {test_df['OCCUPANCY_DATE'].max()}")
    
    # Save splits
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    
    print(f"\nSaving train set to {train_path}...")
    train_df.to_csv(train_path, index=False)
    
    print(f"Saving test set to {test_path}...")
    test_df.to_csv(test_path, index=False)
    
    print("Train/test split complete!")
    return train_df, test_df

if __name__ == "__main__":
    train_test_split()

