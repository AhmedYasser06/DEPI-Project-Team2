# main_pipeline.py
# ============================================================
# COMPLETE PREDICTIVE MAINTENANCE PIPELINE
# Orchestrates all modules: preparing, training, evaluation
# ============================================================

import sys
import argparse
from datetime import datetime

# Import all modules
from data_preparing import prepare_data
from data_training import train_pipeline
from data_evaluation import evaluation_pipeline
from loading_model import load_model


def run_complete_pipeline(data_path, save_path='./', n_trials=20):
    """
    Run the complete end-to-end pipeline
    
    Args:
        data_path: Path to data directory containing train.csv, val.csv, test_with_target.csv
        save_path: Path to save models and results
        n_trials: Number of Optuna optimization trials
    """
    
    print("\n" + "=" * 70)
    print("PREDICTIVE MAINTENANCE - COMPLETE PIPELINE")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data path: {data_path}")
    print(f"Save path: {save_path}")
    print(f"Optimization trials: {n_trials}")
    print("=" * 70)
    
    try:
        # STEP 1: Data Preparation
        print("\n" + "=>" * 35)
        print("STEP 1: DATA PREPARATION")
        print("=>" * 35)
        
        data = prepare_data(data_path)
        
        # STEP 2: Model Training
        print("\n" + "=>" * 35)
        print("STEP 2: MODEL TRAINING")
        print("=>" * 35)
        
        training_results = train_pipeline(data, n_trials=n_trials, save_path=save_path)
        
        # STEP 3: Model Evaluation
        print("\n" + "=>" * 35)
        print("STEP 3: MODEL EVALUATION")
        print("=>" * 35)
        
        evaluation_results = evaluation_pipeline(training_results, data, save_path=save_path)
        
        # STEP 4: Summary
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        print(f"\n Best Model: {evaluation_results['best_model_name']}")
        print(f"   Threshold: {evaluation_results['best_threshold']:.3f}")
        
        best_metrics = evaluation_results['eval_df'].iloc[0]
        print(f"\n Performance Metrics:")
        print(f"   F1-Score:  {best_metrics['F1-Score']:.4f}")
        print(f"   Precision: {best_metrics['Precision']:.4f}")
        print(f"   Recall:    {best_metrics['Recall']:.4f}")
        print(f"   ROC-AUC:   {best_metrics['ROC-AUC']:.4f}")
        print(f"   Accuracy:  {best_metrics['Accuracy']:.4f}")
        
        
        print(f"\n End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        return {
            'data': data,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'success': True
        }
        
    except Exception as e:
        print(f"\n ERROR: Pipeline failed!")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'error': str(e)
        }


def run_inference_only(model_dir, input_data_path, output_path='predictions.csv'):
    """
    Run inference on new data using a trained model
    
    Args:
        model_dir: Directory containing trained model files
        input_data_path: Path to CSV file with new data
        output_path: Path to save predictions
    """
    
    print("\n" + "=" * 70)
    print("INFERENCE MODE - MAKING PREDICTIONS")
    print("=" * 70)
    
    try:
        # Import feature engineering
        from data_preparing import engineer_features_advanced
        import pandas as pd
        
        # Load model
        print("\nLoading trained model...")
        model = load_model(model_dir)
        
        # Load data
        print(f"\nLoading data from: {input_data_path}")
        raw_data = pd.read_csv(input_data_path)
        print(f"Data shape: {raw_data.shape}")
        
        # Apply feature engineering
        print("\nApplying feature engineering...")
        engineered_data = engineer_features_advanced(raw_data)
        
        # Extract features
        cols_to_drop = ['Machine_ID', 'Failure_Within_7_Days', 'Remaining_Useful_Life_days']
        features = engineered_data.drop(columns=[col for col in cols_to_drop if col in engineered_data.columns])
        
        # Align features with model
        model_features = model.get_feature_names()
        missing_features = set(model_features) - set(features.columns)
        
        # Add missing features
        for feat in missing_features:
            features[feat] = 0
        
        # Keep only model features in correct order
        features = features[model_features]
        
        # Make predictions
        print("\nMaking predictions...")
        predictions = model.batch_predict(features)
        
        # Create output
        output = pd.DataFrame({
            'Machine_ID': raw_data['Machine_ID'] if 'Machine_ID' in raw_data.columns else range(len(predictions)),
        })
        output = pd.concat([output, predictions], axis=1)
        
        # Save predictions
        output.to_csv(output_path, index=False)
        
        print(f"\n Predictions saved to: {output_path}")
        print(f"\nPrediction Summary:")
        print(f"   Total machines: {len(output)}")
        print(f"   Predicted failures: {output['Predicted_Failure'].sum()}")
        print(f"   Failure rate: {100 * output['Predicted_Failure'].mean():.2f}%")
        print(f"\nRisk Level Distribution:")
        print(output['Risk_Level'].value_counts())
        
        print("\n" + "=" * 70)
        print("INFERENCE COMPLETED!")
        print("=" * 70)
        
        return output
        
    except Exception as e:
        print(f"\n ERROR: Inference failed!")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point with command-line arguments"""
    
    parser = argparse.ArgumentParser(
        description='Predictive Maintenance Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete training pipeline
  python main_pipeline.py --mode train --data_path /path/to/data --save_path ./models --trials 20
  
  # Run inference on new data
  python main_pipeline.py --mode inference --model_dir ./models --input_data new_machines.csv --output predictions.csv
        """
    )
    
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference'],
                       help='Mode: train (full pipeline) or inference (predictions only)')
    
    # Training mode arguments
    parser.add_argument('--data_path', type=str,
                       help='Path to data directory (for training mode)')
    parser.add_argument('--save_path', type=str, default='./',
                       help='Path to save models and results (default: ./)')
    parser.add_argument('--trials', type=int, default=20,
                       help='Number of Optuna optimization trials (default: 20)')
    
    # Inference mode arguments
    parser.add_argument('--model_dir', type=str,
                       help='Path to directory containing trained model (for inference mode)')
    parser.add_argument('--input_data', type=str,
                       help='Path to input CSV file (for inference mode)')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Path to save predictions (default: predictions.csv)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        if not args.data_path:
            print("ERROR: --data_path is required for training mode")
            sys.exit(1)
        
        result = run_complete_pipeline(
            data_path=args.data_path,
            save_path=args.save_path,
            n_trials=args.trials
        )
        
        if result['success']:
            sys.exit(0)
        else:
            sys.exit(1)
    
    elif args.mode == 'inference':
        if not args.model_dir or not args.input_data:
            print("ERROR: --model_dir and --input_data are required for inference mode")
            sys.exit(1)
        
        result = run_inference_only(
            model_dir=args.model_dir,
            input_data_path=args.input_data,
            output_path=args.output
        )
        
        if result is not None:
            sys.exit(0)
        else:
            sys.exit(1)


if __name__ == "__main__":
    # Check if running in Colab
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False
    
    if IN_COLAB:
        # Colab usage - run with default paths
        print("Running in Google Colab environment")
        from google.colab import drive
        drive.mount('/content/drive')
        
        DATA_PATH = "/content/drive/MyDrive/predictive_maintenance/"
        SAVE_PATH = "/content/drive/MyDrive/predictive_maintenance/models/"
        
        result = run_complete_pipeline(
            data_path=DATA_PATH,
            save_path=SAVE_PATH,
            n_trials=20
        )
    else:
        # Command-line usage
        main()