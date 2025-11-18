"""
Data preparation script that replicates EDA notebook feature engineering
and prepares data for ML modeling.
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

def prepare_data():
    """Prepare data for ML modeling by replicating EDA feature engineering."""
    
    # Set up paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(exist_ok=True)
    
    print("Loading raw data...")
    df = pd.read_csv(data_dir / "sheltersdata.csv")
    
    # Convert date
    df['OCCUPANCY_DATE'] = pd.to_datetime(df['OCCUPANCY_DATE'], format='mixed')
    
    # Drop rows missing the regression target
    print("Dropping rows with missing OCCUPIED_BEDS...")
    df = df.dropna(subset=['OCCUPIED_BEDS'])
    
    # Classification target
    df['overcapacity'] = (df['OCCUPANCY_RATE_BEDS'] > 95).astype(int)
    
    # Temporal Features
    print("Creating temporal features...")
    df['day_of_week'] = df['OCCUPANCY_DATE'].dt.dayofweek
    df['month'] = df['OCCUPANCY_DATE'].dt.month
    df['week_of_year'] = df['OCCUPANCY_DATE'].dt.isocalendar().week
    
    # Lag & Rolling Features
    print("Creating lag and rolling features...")
    df = df.sort_values(['SHELTER_ID', 'OCCUPANCY_DATE'])
    df['lag_1'] = df.groupby('SHELTER_ID')['OCCUPIED_BEDS'].shift(1)
    df['lag_7'] = df.groupby('SHELTER_ID')['OCCUPIED_BEDS'].shift(7)
    df['roll_mean_7'] = df.groupby('SHELTER_ID')['OCCUPIED_BEDS'].rolling(7).mean().shift(1).reset_index(0, drop=True)
    
    # Encode categorical features as bools
    print("Encoding categorical features...")
    cat_cols = ['SECTOR', 'PROGRAM_MODEL', 'OVERNIGHT_SERVICE_TYPE', 'PROGRAM_AREA']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Cyclical encoding for temporal features
    print("Creating cyclical features...")
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['week_of_year_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_of_year_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    # Handle missing values for lag features
    print("Handling missing values...")
    df.fillna(0, inplace=True)
    
    # Select features for ML
    print("Selecting features for ML...")
    
    # Features to drop (metadata and raw temporal features)
    drop_cols = [
        '_id', 'OCCUPANCY_DATE',
        'ORGANIZATION_NAME', 'SHELTER_GROUP', 'LOCATION_NAME',
        'LOCATION_ADDRESS', 'LOCATION_POSTAL_CODE', 'LOCATION_CITY', 'LOCATION_PROVINCE',
        'PROGRAM_NAME', 'CAPACITY_TYPE',
        'day_of_week', 'month', 'week_of_year',  # Use cyclical versions instead
        'UNOCCUPIED_BEDS', 'UNAVAILABLE_BEDS',  # These are components of OCCUPANCY_RATE_BEDS
        'CAPACITY_ACTUAL_ROOM', 'CAPACITY_FUNDING_ROOM', 'OCCUPIED_ROOMS',
        'UNOCCUPIED_ROOMS', 'UNAVAILABLE_ROOMS', 'OCCUPANCY_RATE_ROOMS'  # Room-based, not bed-based
    ]
    # Note: Keep OCCUPIED_BEDS for lag feature calculation in API
    
    # Keep all other columns as features (exclude OCCUPIED_BEDS as it's highly correlated with target)
    feature_cols = [col for col in df.columns if col not in drop_cols and col not in ['OCCUPANCY_RATE_BEDS', 'overcapacity', 'OCCUPIED_BEDS']]
    
    # Create feature dataset (include OCCUPIED_BEDS for API lag feature calculation)
    features_df = df[['SHELTER_ID', 'OCCUPANCY_DATE', 'OCCUPIED_BEDS'] + feature_cols + ['OCCUPANCY_RATE_BEDS', 'overcapacity']].copy()
    
    print(f"\nDataset shape: {features_df.shape}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols[:10]}..." if len(feature_cols) > 10 else f"Feature columns: {feature_cols}")
    
    # Save processed dataset
    output_path = processed_dir / "features.csv"
    print(f"\nSaving processed data to {output_path}...")
    features_df.to_csv(output_path, index=False)
    
    # Save feature list for later use
    feature_list_path = processed_dir / "feature_list.txt"
    with open(feature_list_path, 'w') as f:
        f.write('\n'.join(feature_cols))
    
    print("Data preparation complete!")
    return features_df, feature_cols

if __name__ == "__main__":
    prepare_data()

