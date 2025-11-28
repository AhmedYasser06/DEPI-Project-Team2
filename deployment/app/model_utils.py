import joblib
import json
import pandas as pd
import numpy as np

# ============================================================
# LOAD ARTIFACTS
# ============================================================

MODEL_PATH = "artifacts/BEST_CatBoost_Optimized.pkl"
SCALER_PATH = "artifacts/scaler.pkl"
THRESHOLD_PATH = "artifacts/model_thresholds.json"
FEATURE_INFO_PATH = "artifacts/feature_names.json"

# Load CatBoost model
model = joblib.load(MODEL_PATH)

# Load scaler
scaler = joblib.load(SCALER_PATH)

# Load thresholds (your JSON is a dict of model_name → threshold)
with open(THRESHOLD_PATH, "r") as f:
    threshold_info = json.load(f)

# Use threshold of the best model
optimal_threshold = threshold_info["CatBoost_Optimized"]

# Load feature names (your JSON is a LIST, not a dict)
with open(FEATURE_INFO_PATH, "r") as f:
    FEATURE_NAMES = json.load(f)    # <-- FIXED


# ============================================================
# FEATURE ENGINEERING (must match training pipeline exactly)
# ============================================================

def engineer_features_advanced(df):
    df = df.copy()

    if 'Installation_Year' in df.columns:
        df['Machine_Age'] = pd.Timestamp.now().year - df['Installation_Year']
        df['Machine_Age_Squared'] = df['Machine_Age'] ** 2

    if 'Operational_Hours' in df.columns:
        df['Log_Operational_Hours'] = np.log1p(df['Operational_Hours'])
        df['Sqrt_Operational_Hours'] = np.sqrt(df['Operational_Hours'])
        df['Operational_Hours_Squared'] = df['Operational_Hours'] ** 2

    if 'Oil_Level_pct' in df.columns and 'Coolant_Level_pct' in df.columns:
        df['Oil_to_Coolant_Ratio'] = df['Oil_Level_pct'] / (df['Coolant_Level_pct'] + 1e-5)
        df['Fluid_Level_Product'] = df['Oil_Level_pct'] * df['Coolant_Level_pct']
        df['Fluid_Level_Difference'] = df['Oil_Level_pct'] - df['Coolant_Level_pct']

    if 'Power_Consumption_kW' in df.columns and 'Operational_Hours' in df.columns:
        df['Power_per_Hour'] = df['Power_Consumption_kW'] / (df['Operational_Hours'] + 1e-5)
        df['Cumulative_Power'] = df['Power_Consumption_kW'] * df['Operational_Hours']

    if 'Maintenance_History_Count' in df.columns:
        df['Maintenance_Rate'] = df['Maintenance_History_Count'] / (df['Operational_Hours'] + 1e-5)
        df['Maintenance_Intensity'] = df['Maintenance_History_Count'] / (df['Machine_Age'] + 1e-5) if 'Machine_Age' in df.columns else 0

    if 'Failure_History_Count' in df.columns:
        df['Failure_Rate'] = df['Failure_History_Count'] / (df['Operational_Hours'] + 1e-5)
        df['Failure_Frequency'] = df['Failure_History_Count'] / (df['Machine_Age'] + 1e-5) if 'Machine_Age' in df.columns else 0

    sensors = ['Temperature_C', 'Vibration_mms', 'Sound_dB']
    if all(s in df.columns for s in sensors):
        df['Stress_Index'] = (
            0.4 * df['Temperature_C'] / (df['Temperature_C'].max() + 1e-5) +
            0.35 * df['Vibration_mms'] / (df['Vibration_mms'].max() + 1e-5) +
            0.25 * df['Sound_dB'] / (df['Sound_dB'].max() + 1e-5)
        )
        df['Sensor_Product'] = df['Temperature_C'] * df['Vibration_mms'] * df['Sound_dB']
        df['Temp_Vibration_Interaction'] = df['Temperature_C'] * df['Vibration_mms']

    if 'Last_Maintenance_Days_Ago' in df.columns:
        df['Maintenance_Recency'] = 1 / (1 + df['Last_Maintenance_Days_Ago'])
        df['Maintenance_Overdue'] = (df['Last_Maintenance_Days_Ago'] > 90).astype(int)
        df['Critical_Maintenance_Window'] = (
            (df['Last_Maintenance_Days_Ago'] > 60) &
            (df['Last_Maintenance_Days_Ago'] <= 90)
        ).astype(int)

    if 'Error_Codes_Last_30_Days' in df.columns:
        df['Has_Errors'] = (df['Error_Codes_Last_30_Days'] > 0).astype(int)
        df['High_Error_Rate'] = (df['Error_Codes_Last_30_Days'] > df['Error_Codes_Last_30_Days'].median()).astype(int)
        df['Error_Density'] = df['Error_Codes_Last_30_Days'] / (df['Operational_Hours'] + 1e-5)

    if 'Temperature_C' in df.columns:
        p75 = df['Temperature_C'].quantile(0.75)
        p90 = df['Temperature_C'].quantile(0.9)
        df['High_Temperature_Flag'] = (df['Temperature_C'] > p75).astype(int)
        df['Critical_Temperature_Flag'] = (df['Temperature_C'] > p90).astype(int)
        df['Temp_Deviation'] = np.abs(df['Temperature_C'] - df['Temperature_C'].mean())

    if 'Vibration_mms' in df.columns:
        p75 = df['Vibration_mms'].quantile(0.75)
        p90 = df['Vibration_mms'].quantile(0.9)
        df['High_Vibration_Flag'] = (df['Vibration_mms'] > p75).astype(int)
        df['Critical_Vibration_Flag'] = (df['Vibration_mms'] > p90).astype(int)

    if 'Oil_Level_pct' in df.columns:
        p25 = df['Oil_Level_pct'].quantile(0.25)
        p10 = df['Oil_Level_pct'].quantile(0.10)
        df['Low_Oil_Flag'] = (df['Oil_Level_pct'] < p25).astype(int)
        df['Critical_Oil_Flag'] = (df['Oil_Level_pct'] < p10).astype(int)

    if 'AI_Override_Events' in df.columns:
        df['Has_AI_Override'] = (df['AI_Override_Events'] > 0).astype(int)
        df['High_AI_Override'] = (df['AI_Override_Events'] > df['AI_Override_Events'].median()).astype(int)

    risk_cols = ['High_Temperature_Flag', 'High_Vibration_Flag', 'Low_Oil_Flag', 'Has_Errors', 'Maintenance_Overdue']
    if all(c in df.columns for c in risk_cols):
        df['Risk_Score'] = df[risk_cols].sum(axis=1)
        df['High_Risk'] = (df['Risk_Score'] >= 3).astype(int)

    if "Machine_Type" in df.columns:
        df = pd.get_dummies(df, columns=["Machine_Type"], prefix="Machine", drop_first=False)

    return df


# ============================================================
# PREPARE FEATURES FOR INFERENCE
# ============================================================

def prepare_features(payload):
    if isinstance(payload, dict):
        df = pd.DataFrame([payload])
    else:
        df = pd.DataFrame(payload)

    df = engineer_features_advanced(df)

    # Add missing columns with 0
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0

    df = df[FEATURE_NAMES].fillna(0)
    X_scaled = scaler.transform(df)

    return X_scaled


# ============================================================
# PREDICT FUNCTION
# ============================================================

def predict(payload):
    X = prepare_features(payload)
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= optimal_threshold).astype(int)

    return {
        "predictions": pred.tolist(),
        "probabilities": prob.tolist(),
        "threshold_used": optimal_threshold
    }
