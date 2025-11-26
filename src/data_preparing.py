# data_preparing.py
# ============================================================
# DATA LOADING, PREPROCESSING & FEATURE ENGINEERING
# ============================================================

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def engineer_features_advanced(df):
    """Advanced feature engineering with interaction terms"""
    df = df.copy()

    # Time-based features
    if 'Installation_Year' in df.columns:
        df['Machine_Age'] = pd.Timestamp.now().year - df['Installation_Year']
        df['Machine_Age_Squared'] = df['Machine_Age'] ** 2

    if 'Operational_Hours' in df.columns:
        df['Log_Operational_Hours'] = np.log1p(df['Operational_Hours'])
        df['Sqrt_Operational_Hours'] = np.sqrt(df['Operational_Hours'])
        df['Operational_Hours_Squared'] = df['Operational_Hours'] ** 2

    # Critical ratios
    if 'Oil_Level_pct' in df.columns and 'Coolant_Level_pct' in df.columns:
        df['Oil_to_Coolant_Ratio'] = df['Oil_Level_pct'] / (df['Coolant_Level_pct'] + 1e-5)
        df['Fluid_Level_Product'] = df['Oil_Level_pct'] * df['Coolant_Level_pct']
        df['Fluid_Level_Difference'] = df['Oil_Level_pct'] - df['Coolant_Level_pct']

    if 'Power_Consumption_kW' in df.columns and 'Operational_Hours' in df.columns:
        df['Power_per_Hour'] = df['Power_Consumption_kW'] / (df['Operational_Hours'] + 1e-5)
        df['Cumulative_Power'] = df['Power_Consumption_kW'] * df['Operational_Hours']

    if 'Maintenance_History_Count' in df.columns and 'Operational_Hours' in df.columns:
        df['Maintenance_Rate'] = df['Maintenance_History_Count'] / (df['Operational_Hours'] + 1e-5)
        df['Maintenance_Intensity'] = df['Maintenance_History_Count'] / (df['Machine_Age'] + 1e-5) if 'Machine_Age' in df.columns else 0

    if 'Failure_History_Count' in df.columns and 'Operational_Hours' in df.columns:
        df['Failure_Rate'] = df['Failure_History_Count'] / (df['Operational_Hours'] + 1e-5)
        df['Failure_Frequency'] = df['Failure_History_Count'] / (df['Machine_Age'] + 1e-5) if 'Machine_Age' in df.columns else 0

    # Multi-sensor stress index with weights
    sensors = ['Temperature_C', 'Vibration_mms', 'Sound_dB']
    if all(col in df.columns for col in sensors):
        df['Stress_Index'] = (
            0.4 * df['Temperature_C'] / (df['Temperature_C'].max() + 1e-5) +
            0.35 * df['Vibration_mms'] / (df['Vibration_mms'].max() + 1e-5) +
            0.25 * df['Sound_dB'] / (df['Sound_dB'].max() + 1e-5)
        )
        df['Sensor_Product'] = df['Temperature_C'] * df['Vibration_mms'] * df['Sound_dB']
        df['Temp_Vibration_Interaction'] = df['Temperature_C'] * df['Vibration_mms']

    # Maintenance urgency
    if 'Last_Maintenance_Days_Ago' in df.columns:
        df['Maintenance_Recency'] = 1 / (1 + df['Last_Maintenance_Days_Ago'])
        df['Maintenance_Overdue'] = (df['Last_Maintenance_Days_Ago'] > 90).astype(int)
        df['Critical_Maintenance_Window'] = ((df['Last_Maintenance_Days_Ago'] > 60) &
                                             (df['Last_Maintenance_Days_Ago'] <= 90)).astype(int)

    # Error patterns
    if 'Error_Codes_Last_30_Days' in df.columns:
        df['Has_Errors'] = (df['Error_Codes_Last_30_Days'] > 0).astype(int)
        df['High_Error_Rate'] = (df['Error_Codes_Last_30_Days'] > df['Error_Codes_Last_30_Days'].median()).astype(int)
        if 'Operational_Hours' in df.columns:
            df['Error_Density'] = df['Error_Codes_Last_30_Days'] / (df['Operational_Hours'] + 1e-5)

    # Critical thresholds (more granular)
    if 'Temperature_C' in df.columns:
        temp_p75 = df['Temperature_C'].quantile(0.75)
        temp_p90 = df['Temperature_C'].quantile(0.90)
        df['High_Temperature_Flag'] = (df['Temperature_C'] > temp_p75).astype(int)
        df['Critical_Temperature_Flag'] = (df['Temperature_C'] > temp_p90).astype(int)
        df['Temp_Deviation'] = np.abs(df['Temperature_C'] - df['Temperature_C'].mean())

    if 'Vibration_mms' in df.columns:
        vib_p75 = df['Vibration_mms'].quantile(0.75)
        vib_p90 = df['Vibration_mms'].quantile(0.90)
        df['High_Vibration_Flag'] = (df['Vibration_mms'] > vib_p75).astype(int)
        df['Critical_Vibration_Flag'] = (df['Vibration_mms'] > vib_p90).astype(int)

    if 'Oil_Level_pct' in df.columns:
        oil_p25 = df['Oil_Level_pct'].quantile(0.25)
        oil_p10 = df['Oil_Level_pct'].quantile(0.10)
        df['Low_Oil_Flag'] = (df['Oil_Level_pct'] < oil_p25).astype(int)
        df['Critical_Oil_Flag'] = (df['Oil_Level_pct'] < oil_p10).astype(int)

    # AI override patterns
    if 'AI_Override_Events' in df.columns:
        df['Has_AI_Override'] = (df['AI_Override_Events'] > 0).astype(int)
        df['High_AI_Override'] = (df['AI_Override_Events'] > df['AI_Override_Events'].median()).astype(int)

    # Combined risk score
    risk_factors = ['High_Temperature_Flag', 'High_Vibration_Flag', 'Low_Oil_Flag',
                   'Has_Errors', 'Maintenance_Overdue']
    if all(col in df.columns for col in risk_factors):
        df['Risk_Score'] = df[risk_factors].sum(axis=1)
        df['High_Risk'] = (df['Risk_Score'] >= 3).astype(int)

    # Machine Type encoding
    if 'Machine_Type' in df.columns:
        df = pd.get_dummies(df, columns=['Machine_Type'], prefix='Machine', drop_first=False)

    return df


def prepare_data(data_path, target_col='Failure_Within_7_Days'):
    """
    Load and prepare train, validation, and test datasets
    
    Args:
        data_path: Path to data directory
        target_col: Name of target column
        
    Returns:
        Dictionary containing processed data and metadata
    """
    print("=" * 70)
    print("DATA PREPARATION PIPELINE")
    print("=" * 70)
    
    # Load data
    train = pd.read_csv(os.path.join(data_path, "train.csv"))
    val = pd.read_csv(os.path.join(data_path, "val.csv"))
    test = pd.read_csv(os.path.join(data_path, "test_with_target.csv"))
    
    print(f"\nTrain shape: {train.shape}")
    print(f"Val shape: {val.shape}")
    print(f"Test shape: {test.shape}")
    
    # Identify columns to drop
    drop_cols = ['Machine_ID', 'Remaining_Useful_Life_days'] if 'Remaining_Useful_Life_days' in train.columns else ['Machine_ID']
    
    # Check class distribution
    print(f"\nTarget Distribution (Train):\n{train[target_col].value_counts()}")
    class_ratio = train[target_col].value_counts()[0] / train[target_col].value_counts()[1]
    print(f"Class Imbalance Ratio: {class_ratio:.2f}:1")
    
    # Feature engineering
    print("\nApplying advanced feature engineering...")
    train_fe = engineer_features_advanced(train)
    val_fe = engineer_features_advanced(val)
    test_fe = engineer_features_advanced(test)
    
    # Separate target
    y_train = train_fe[target_col]
    y_val = val_fe[target_col]
    y_test = test_fe[target_col]
    
    # Drop non-feature columns
    X_train = train_fe.drop(columns=[col for col in drop_cols + [target_col] if col in train_fe.columns])
    X_val = val_fe.drop(columns=[col for col in drop_cols + [target_col] if col in val_fe.columns])
    X_test = test_fe.drop(columns=[col for col in drop_cols + [target_col] if col in test_fe.columns])
    
    # Align columns
    common_cols = X_train.columns.intersection(X_val.columns).intersection(X_test.columns)
    X_train = X_train[common_cols]
    X_val = X_val[common_cols]
    X_test = X_test[common_cols]
    
    print(f"\nFeature shape after engineering: {X_train.shape}")
    print(f"Number of features: {len(common_cols)}")
    
    # Handle missing values
    X_train = X_train.fillna(X_train.median())
    X_val = X_val.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(" Data preparation completed!")
    
    return {
        'X_train': X_train_scaled,
        'y_train': y_train,
        'X_val': X_val_scaled,
        'y_val': y_val,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': common_cols.tolist(),
        'drop_cols': drop_cols,
        'target_col': target_col,
        'class_ratio': class_ratio,
        'original_test': test
    }


if __name__ == "__main__":
    # Example usage
    from google.colab import drive
    drive.mount('/content/drive')
    
    PATH = "/content/drive/MyDrive/predictive_maintenance/"
    data = prepare_data(PATH)
    
    print("\n" + "=" * 70)
    print("Data prepared and ready for training!")
    print("=" * 70)