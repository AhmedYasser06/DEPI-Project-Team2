import gradio as gr
import joblib
import json
import pandas as pd
import numpy as np


# ============================
# Load model + scaler + config
# ============================

MODEL_PATH = "BEST_CatBoost_Optimized.pkl"
SCALER_PATH = "scaler.pkl"
FEATURE_PATH = "feature_names.json"
THRESHOLD_PATH = "model_thresholds.json"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(FEATURE_PATH, "r") as f:
    FEATURE_NAMES = json.load(f)

with open(THRESHOLD_PATH, "r") as f:
    threshold_info = json.load(f)

optimal_threshold = threshold_info["CatBoost_Optimized"]



# ============================================================
# Advanced Feature Engineering
# (matches your training pipeline)
# ============================================================

def engineer_features_advanced(df):
    df = df.copy()

    # Machine Age
    df["Machine_Age"] = 2025 - df["Installation_Year"]
    df["Machine_Age_Squared"] = df["Machine_Age"] ** 2

    # Operational hours
    df["Log_Operational_Hours"] = np.log1p(df["Operational_Hours"])
    df["Sqrt_Operational_Hours"] = np.sqrt(df["Operational_Hours"])
    df["Operational_Hours_Squared"] = df["Operational_Hours"] ** 2

    # Fluids
    df["Oil_to_Coolant_Ratio"] = df["Oil_Level_pct"] / (df["Coolant_Level_pct"] + 1e-5)
    df["Fluid_Level_Product"] = df["Oil_Level_pct"] * df["Coolant_Level_pct"]
    df["Fluid_Level_Difference"] = df["Oil_Level_pct"] - df["Coolant_Level_pct"]

    # Power
    df["Power_per_Hour"] = df["Power_Consumption_kW"] / (df["Operational_Hours"] + 1e-5)
    df["Cumulative_Power"] = df["Power_Consumption_kW"] * df["Operational_Hours"]

    # Maintenance / Failures
    df["Maintenance_Rate"] = df["Maintenance_History_Count"] / (df["Operational_Hours"] + 1e-5)
    df["Failure_Rate"] = df["Failure_History_Count"] / (df["Operational_Hours"] + 1e-5)

    # Error indicators
    df["Has_Errors"] = (df["Error_Codes_Last_30_Days"] > 0).astype(int)
    df["Error_Density"] = df["Error_Codes_Last_30_Days"] / (df["Operational_Hours"] + 1e-5)

    # Sensor interactions
    df["Stress_Index"] = (
        0.4 * df["Temperature_C"] +
        0.35 * df["Vibration_mms"] +
        0.25 * df["Sound_dB"]
    )
    df["Sensor_Product"] = df["Temperature_C"] * df["Vibration_mms"] * df["Sound_dB"]
    df["Temp_Vibration_Interaction"] = df["Temperature_C"] * df["Vibration_mms"]

    # Risk flags
    df["High_Temperature_Flag"] = (df["Temperature_C"] > 70).astype(int)
    df["Critical_Temperature_Flag"] = (df["Temperature_C"] > 85).astype(int)

    df["High_Vibration_Flag"] = (df["Vibration_mms"] > 10).astype(int)
    df["Critical_Vibration_Flag"] = (df["Vibration_mms"] > 18).astype(int)

    df["Low_Oil_Flag"] = (df["Oil_Level_pct"] < 25).astype(int)
    df["Critical_Oil_Flag"] = (df["Oil_Level_pct"] < 10).astype(int)

    df["Maintenance_Overdue"] = (df["Last_Maintenance_Days_Ago"] > 180).astype(int)
    df["Critical_Maintenance_Window"] = (
        (df["Last_Maintenance_Days_Ago"] >= 100) &
        (df["Last_Maintenance_Days_Ago"] <= 180)
    ).astype(int)

    df["Risk_Score"] = (
        df["High_Temperature_Flag"] +
        df["High_Vibration_Flag"] +
        df["Low_Oil_Flag"] +
        df["Has_Errors"] +
        df["Maintenance_Overdue"]
    )

    df["High_Risk"] = (df["Risk_Score"] >= 3).astype(int)

    return df



# ============================================================
# Predict Function (no machine type, AI supervision = boolean)
# ============================================================

def predict_failure(
    Installation_Year,
    Operational_Hours,
    Temperature_C,
    Vibration_mms,
    Sound_dB,
    Oil_Level_pct,
    Coolant_Level_pct,
    Power_Consumption_kW,
    Last_Maintenance_Days_Ago,
    Maintenance_History_Count,
    Failure_History_Count,
    AI_Supervision_bool,
    Error_Codes_Last_30_Days
):

    payload = {
        "Installation_Year": Installation_Year,
        "Operational_Hours": Operational_Hours,
        "Temperature_C": Temperature_C,
        "Vibration_mms": Vibration_mms,
        "Sound_dB": Sound_dB,
        "Oil_Level_pct": Oil_Level_pct,
        "Coolant_Level_pct": Coolant_Level_pct,
        "Power_Consumption_kW": Power_Consumption_kW,
        "Last_Maintenance_Days_Ago": Last_Maintenance_Days_Ago,
        "Maintenance_History_Count": Maintenance_History_Count,
        "Failure_History_Count": Failure_History_Count,
        "AI_Supervision": 1 if AI_Supervision_bool else 0,
        "Error_Codes_Last_30_Days": Error_Codes_Last_30_Days
    }

    df = pd.DataFrame([payload])

    df = engineer_features_advanced(df)

    # Align with model
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0

    df = df[FEATURE_NAMES]
    X_scaled = scaler.transform(df)

    prob = model.predict_proba(X_scaled)[:, 1][0]
    pred = int(prob >= optimal_threshold)

    # Return Yes / No only
    if pred == 1:
        return "Yes"
    else:
        return "No"



# ============================================================
# NEW CLEAN UI (No machine type, AI supervision boolean)
# ============================================================

inputs = [
    gr.Number(label="Installation Year"),
    gr.Number(label="Operational Hours"),
    gr.Number(label="Temperature (°C)"),
    gr.Number(label="Vibration (mm/s)"),
    gr.Number(label="Sound (dB)"),
    gr.Number(label="Oil Level (%)"),
    gr.Number(label="Coolant Level (%)"),
    gr.Number(label="Power Consumption (kW)"),
    gr.Number(label="Last Maintenance Days Ago"),
    gr.Number(label="Maintenance History Count"),
    gr.Number(label="Failure History Count"),

    gr.Checkbox(label="AI Supervision Enabled?"),

    gr.Number(label="Error Codes Last 30 Days"),
]

gr.Interface(
    fn=predict_failure,
    inputs=inputs,
    outputs="text",
    title="Predictive Maintenance Model",
    description="Enter machine sensor readings to predict whether maintenance is needed."
).launch()
