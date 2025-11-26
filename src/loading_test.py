# loading_test.py
# ============================================================
# TEST SCRIPT FOR LOADING AND USING TRAINED MODEL
# ============================================================

import numpy as np
import pandas as pd
from loading_model import load_model
import warnings
warnings.filterwarnings('ignore')


def test_model_loading(model_dir='./'):
    """Test model loading functionality"""
    print("=" * 70)
    print("TEST 1: Model Loading")
    print("=" * 70)
    
    try:
        model = load_model(model_dir)
        print("\n  Model loaded successfully!")
        
        # Display model info
        info = model.get_model_info()
        print("\nModel Information:")
        print(f"  Name: {info['model_name']}")
        print(f"  Threshold: {info['threshold']:.3f}")
        print(f"  Features: {info['n_features']}")
        print(f"  F1-Score: {info['metrics']['f1_score']:.4f}")
        
        return model
    except Exception as e:
        print(f"\n  Error loading model: {str(e)}")
        return None


def test_single_prediction(model):
    """Test prediction on a single sample"""
    print("\n" + "=" * 70)
    print("TEST 2: Single Sample Prediction")
    print("=" * 70)
    
    # Create a synthetic sample (you'll need actual feature values)
    print("\nCreating synthetic test sample...")
    
    # Get feature names
    feature_names = model.get_feature_names()
    
    # Create random sample (in practice, use real data)
    sample_data = {name: np.random.randn() for name in feature_names}
    
    print(f"Sample has {len(sample_data)} features")
    
    try:
        prediction = model.predict_single(sample_data)
        
        print("\n  Prediction successful!")
        print(f"\nResults:")
        print(f"  Will Fail: {prediction['will_fail']}")
        print(f"  Probability: {prediction['failure_probability']:.4f}")
        print(f"  Confidence: {prediction['confidence']}")
        
        return True
    except Exception as e:
        print(f"\n  Error making prediction: {str(e)}")
        return False


def test_batch_prediction(model, test_data_path=None):
    """Test batch predictions"""
    print("\n" + "=" * 70)
    print("TEST 3: Batch Prediction")
    print("=" * 70)
    
    try:
        if test_data_path and pd.io.common.file_exists(test_data_path):
            print(f"\nLoading test data from: {test_data_path}")
            test_data = pd.read_csv(test_data_path)
            
            # Remove non-feature columns if present
            cols_to_drop = ['Machine_ID', 'Failure_Within_7_Days', 'Remaining_Useful_Life_days']
            test_features = test_data.drop(columns=[col for col in cols_to_drop if col in test_data.columns])
            
            # Align with model features
            feature_names = model.get_feature_names()
            
            # For this test, create synthetic data with correct features
            print("\nNote: Using synthetic data for testing")
            n_samples = min(100, len(test_data))
            test_features = pd.DataFrame(
                np.random.randn(n_samples, len(feature_names)),
                columns=feature_names
            )
        else:
            print("\nCreating synthetic batch data...")
            feature_names = model.get_feature_names()
            n_samples = 100
            test_features = pd.DataFrame(
                np.random.randn(n_samples, len(feature_names)),
                columns=feature_names
            )
        
        print(f"Test batch size: {len(test_features)} samples")
        
        # Make predictions
        predictions = model.batch_predict(test_features)
        
        print("\n  Batch prediction successful!")
        print(f"\nPrediction Summary:")
        print(f"  Total samples: {len(predictions)}")
        print(f"  Predicted failures: {predictions['Predicted_Failure'].sum()}")
        print(f"  Failure rate: {100 * predictions['Predicted_Failure'].mean():.2f}%")
        print(f"\nRisk Level Distribution:")
        print(predictions['Risk_Level'].value_counts())
        
        print(f"\nSample predictions (first 5):")
        print(predictions.head())
        
        return predictions
    except Exception as e:
        print(f"\n  Error in batch prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_feature_engineering_pipeline(model, raw_data_path=None):
    """Test the complete pipeline from raw data to prediction"""
    print("\n" + "=" * 70)
    print("TEST 4: Complete Pipeline (with Feature Engineering)")
    print("=" * 70)
    
    try:
        # Import feature engineering from data_preparing
        from data_preparing import engineer_features_advanced
        
        if raw_data_path and pd.io.common.file_exists(raw_data_path):
            print(f"\nLoading raw data from: {raw_data_path}")
            raw_data = pd.read_csv(raw_data_path)
        else:
            print("\nCreating synthetic raw data...")
            # Create synthetic raw data with typical features
            n_samples = 50
            raw_data = pd.DataFrame({
                'Machine_ID': range(n_samples),
                'Installation_Year': np.random.randint(2015, 2023, n_samples),
                'Operational_Hours': np.random.randint(1000, 50000, n_samples),
                'Temperature_C': np.random.uniform(60, 100, n_samples),
                'Vibration_mms': np.random.uniform(5, 20, n_samples),
                'Sound_dB': np.random.uniform(70, 95, n_samples),
                'Oil_Level_pct': np.random.uniform(30, 100, n_samples),
                'Coolant_Level_pct': np.random.uniform(40, 100, n_samples),
                'Power_Consumption_kW': np.random.uniform(50, 150, n_samples),
                'Maintenance_History_Count': np.random.randint(0, 20, n_samples),
                'Failure_History_Count': np.random.randint(0, 5, n_samples),
                'Last_Maintenance_Days_Ago': np.random.randint(1, 180, n_samples),
                'Error_Codes_Last_30_Days': np.random.randint(0, 10, n_samples),
                'AI_Override_Events': np.random.randint(0, 5, n_samples),
            })
        
        print(f"Raw data shape: {raw_data.shape}")
        
        # Apply feature engineering
        print("\nApplying feature engineering...")
        engineered_data = engineer_features_advanced(raw_data)
        
        # Remove non-feature columns
        cols_to_drop = ['Machine_ID', 'Failure_Within_7_Days', 'Remaining_Useful_Life_days']
        features = engineered_data.drop(columns=[col for col in cols_to_drop if col in engineered_data.columns])
        
        print(f"Engineered data shape: {features.shape}")
        
        # Align with model features
        model_features = model.get_feature_names()
        missing_features = set(model_features) - set(features.columns)
        extra_features = set(features.columns) - set(model_features)
        
        print(f"\nFeature alignment:")
        print(f"  Model expects: {len(model_features)} features")
        print(f"  Data has: {len(features.columns)} features")
        print(f"  Missing features: {len(missing_features)}")
        print(f"  Extra features: {len(extra_features)}")
        
        # Add missing features with zeros
        for feat in missing_features:
            features[feat] = 0
        
        # Keep only model features in correct order
        features = features[model_features]
        
        # Make predictions
        print("\nMaking predictions...")
        predictions = model.batch_predict(features)
        
        print("\n  Complete pipeline successful!")
        print(f"\nResults:")
        print(f"  Samples processed: {len(predictions)}")
        print(f"  Predicted failures: {predictions['Predicted_Failure'].sum()}")
        print(f"  Average probability: {predictions['Failure_Probability'].mean():.4f}")
        
        # Create final output
        output = pd.DataFrame({
            'Machine_ID': raw_data['Machine_ID'] if 'Machine_ID' in raw_data.columns else range(len(predictions)),
            'Predicted_Failure': predictions['Predicted_Failure'],
            'Failure_Probability': predictions['Failure_Probability'],
            'Risk_Level': predictions['Risk_Level']
        })
        
        print(f"\nSample output (first 10):")
        print(output.head(10))
        
        return output
    except Exception as e:
        print(f"\n  Error in complete pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_all_tests(model_dir='./', test_data_path=None):
    """Run all tests"""
    print("\n" + "=" * 70)
    print("RUNNING ALL MODEL TESTS")
    print("=" * 70)
    
    results = {
        'loading': False,
        'single_prediction': False,
        'batch_prediction': False,
        'complete_pipeline': False
    }
    
    # Test 1: Loading
    model = test_model_loading(model_dir)
    if model:
        results['loading'] = True
        
        # Test 2: Single prediction
        results['single_prediction'] = test_single_prediction(model)
        
        # Test 3: Batch prediction
        batch_result = test_batch_prediction(model, test_data_path)
        if batch_result is not None:
            results['batch_prediction'] = True
        
        # Test 4: Complete pipeline
        pipeline_result = test_feature_engineering_pipeline(model, test_data_path)
        if pipeline_result is not None:
            results['complete_pipeline'] = True
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n  All tests passed! Model is ready for deployment.")
    else:
        print("\n  Some tests failed. Please check the errors above.")
    
    return results


if __name__ == "__main__":
    # Run tests
    # You can specify the model directory and test data path
    
    import sys
    
    model_dir = sys.argv[1] if len(sys.argv) > 1 else './'
    test_data_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("=" * 70)
    print("PREDICTIVE MAINTENANCE MODEL - TESTING SUITE")
    print("=" * 70)
    print(f"\nModel directory: {model_dir}")
    if test_data_path:
        print(f"Test data path: {test_data_path}")
    else:
        print("Test data path: None (using synthetic data)")
    
    results = run_all_tests(model_dir, test_data_path)