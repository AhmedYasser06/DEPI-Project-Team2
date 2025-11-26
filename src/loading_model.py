# loading_model.py
# ============================================================
# MODEL LOADING & INFERENCE
# ============================================================

import numpy as np
import pandas as pd
import joblib
import json
import os
from sklearn.preprocessing import StandardScaler


class PredictiveMaintenanceModel:
    """Wrapper class for loading and using trained predictive maintenance model"""
    
    def __init__(self, model_path, threshold_path, scaler_path, feature_info_path):
        """
        Initialize the model with all necessary components
        
        Args:
            model_path: Path to saved model (.pkl)
            threshold_path: Path to threshold JSON file
            scaler_path: Path to saved scaler (.pkl)
            feature_info_path: Path to feature info JSON
        """
        print("=" * 70)
        print("Loading Predictive Maintenance Model")
        print("=" * 70)
        
        # Load model
        print(f"\nLoading model from: {model_path}")
        self.model = joblib.load(model_path)
        print(" Model loaded")
        
        # Load threshold info
        print(f"Loading threshold from: {threshold_path}")
        with open(threshold_path, 'r') as f:
            threshold_info = json.load(f)
        self.threshold = threshold_info['optimal_threshold']
        self.model_name = threshold_info['model_name']
        self.metrics = threshold_info['metrics']
        print(f" Threshold loaded: {self.threshold:.3f}")
        
        # Load scaler
        print(f"Loading scaler from: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        print(" Scaler loaded")
        
        # Load feature info
        print(f"Loading feature info from: {feature_info_path}")
        with open(feature_info_path, 'r') as f:
            feature_info = json.load(f)
        self.feature_names = feature_info['feature_names']
        self.n_features = feature_info['n_features']
        print(f" Feature info loaded ({self.n_features} features)")
        
        print("\n" + "=" * 70)
        print(f"Model: {self.model_name}")
        print(f"Threshold: {self.threshold:.3f}")
        print(f"Performance Metrics:")
        print(f"  F1-Score:  {self.metrics['f1_score']:.4f}")
        print(f"  Precision: {self.metrics['precision']:.4f}")
        print(f"  Recall:    {self.metrics['recall']:.4f}")
        print(f"  ROC-AUC:   {self.metrics['roc_auc']:.4f}")
        print("=" * 70)
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X: Feature matrix (numpy array or pandas DataFrame)
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Ensure X is numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
        else:
            probabilities = self.model.decision_function(X_scaled)
        
        # Apply threshold
        predictions = (probabilities >= self.threshold).astype(int)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'failure_risk': predictions.astype(bool)
        }
    
    def predict_single(self, sample):
        """
        Make prediction for a single sample
        
        Args:
            sample: Single sample (1D array, Series, or dict)
            
        Returns:
            Dictionary with prediction details
        """
        # Convert to DataFrame if dict
        if isinstance(sample, dict):
            sample = pd.DataFrame([sample])
        elif not isinstance(sample, pd.DataFrame):
            sample = np.array(sample).reshape(1, -1)
        
        result = self.predict(sample)
        
        return {
            'will_fail': bool(result['predictions'][0]),
            'failure_probability': float(result['probabilities'][0]),
            'confidence': 'High' if abs(result['probabilities'][0] - self.threshold) > 0.2 else 'Medium'
        }
    
    def batch_predict(self, X, return_dataframe=True):
        """
        Make predictions on batch of data with detailed output
        
        Args:
            X: Feature matrix
            return_dataframe: Whether to return as DataFrame
            
        Returns:
            DataFrame or dict with predictions
        """
        result = self.predict(X)
        
        if return_dataframe:
            df = pd.DataFrame({
                'Predicted_Failure': result['predictions'],
                'Failure_Probability': result['probabilities'],
                'Risk_Level': ['High' if p > 0.7 else 'Medium' if p > 0.4 else 'Low' 
                              for p in result['probabilities']]
            })
            return df
        
        return result
    
    def get_feature_names(self):
        """Return list of expected feature names"""
        return self.feature_names
    
    def get_model_info(self):
        """Return model information"""
        return {
            'model_name': self.model_name,
            'threshold': self.threshold,
            'n_features': self.n_features,
            'metrics': self.metrics
        }


def load_model(model_dir='./'):
    """
    Convenience function to load model with default file names
    
    Args:
        model_dir: Directory containing model files
        
    Returns:
        PredictiveMaintenanceModel instance
    """
    # Find model file (looks for best_model_*.pkl)
    model_files = [f for f in os.listdir(model_dir) if f.startswith('best_model_') and f.endswith('.pkl')]
    
    if not model_files:
        raise FileNotFoundError(f"No model file found in {model_dir}")
    
    model_path = os.path.join(model_dir, model_files[0])
    threshold_path = os.path.join(model_dir, 'model_threshold.json')
    scaler_path = os.path.join(model_dir, 'scaler_optimized.pkl')
    feature_info_path = os.path.join(model_dir, 'feature_info.json')
    
    return PredictiveMaintenanceModel(model_path, threshold_path, scaler_path, feature_info_path)


if __name__ == "__main__":
    # Example usage
    print("\n" + "=" * 70)
    print("EXAMPLE USAGE")
    print("=" * 70)
    
    print("""
# Load the model
from loading_model import load_model

model = load_model('./models/')

# Make prediction on single sample
sample = {
    'Temperature_C': 85.5,
    'Vibration_mms': 12.3,
    'Oil_Level_pct': 45.2,
    # ... other features
}

prediction = model.predict_single(sample)
print(prediction)
# Output: {'will_fail': True, 'failure_probability': 0.85, 'confidence': 'High'}

# Make batch predictions
import pandas as pd
data = pd.read_csv('new_data.csv')
predictions = model.batch_predict(data)
print(predictions)

# Get model info
info = model.get_model_info()
print(info)
    """)