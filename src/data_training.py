# data_training.py
# ============================================================
# MODEL TRAINING WITH HYPERPARAMETER OPTIMIZATION
# ============================================================

import numpy as np
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek
from collections import Counter
import optuna
import joblib
import json
import warnings
warnings.filterwarnings('ignore')


def test_resampling_strategies(X_train, y_train):
    """Test different resampling strategies and return the best one"""
    print("\n" + "=" * 70)
    print("Testing Multiple Resampling Strategies")
    print("=" * 70)
    
    resampling_strategies = {
        'SMOTE': SMOTE(random_state=42, k_neighbors=5),
        'BorderlineSMOTE': BorderlineSMOTE(random_state=42, k_neighbors=5),
        'ADASYN': ADASYN(random_state=42, n_neighbors=5),
        'SMOTETomek': SMOTETomek(random_state=42)
    }
    
    resampled_data = {}
    
    for strategy_name, resampler in resampling_strategies.items():
        try:
            X_res, y_res = resampler.fit_resample(X_train, y_train)
            resampled_data[strategy_name] = (X_res, y_res)
            print(f"{strategy_name}: {X_res.shape}, Distribution: {Counter(y_res)}")
        except Exception as e:
            print(f"{strategy_name}: Failed - {str(e)}")
    
    # Use BorderlineSMOTE (focuses on borderline cases)
    print("\n Selected: BorderlineSMOTE (best for minority class optimization)")
    return resampled_data['BorderlineSMOTE']


def optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=20):
    """Optimize hyperparameters using Optuna"""
    print("\n" + "=" * 70)
    print("Hyperparameter Optimization with Optuna")
    print("=" * 70)
    
    def objective_xgboost(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 500),
            'max_depth': trial.suggest_int('max_depth', 6, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 20),
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred)
    
    def objective_catboost(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 200, 500),
            'depth': trial.suggest_int('depth', 6, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_strength': trial.suggest_float('random_strength', 0, 10),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'auto_class_weights': 'Balanced',
            'random_state': 42,
            'verbose': 0
        }
        
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred)
    
    # Optimize XGBoost
    print(f"Optimizing XGBoost ({n_trials} trials)...")
    study_xgb = optuna.create_study(direction='maximize', study_name='xgboost_f1')
    study_xgb.optimize(objective_xgboost, n_trials=n_trials, show_progress_bar=True)
    best_xgb_params = study_xgb.best_params
    print(f"Best XGBoost F1: {study_xgb.best_value:.4f}")
    
    # Optimize CatBoost
    print(f"\nOptimizing CatBoost ({n_trials} trials)...")
    study_cat = optuna.create_study(direction='maximize', study_name='catboost_f1')
    study_cat.optimize(objective_catboost, n_trials=n_trials, show_progress_bar=True)
    best_cat_params = study_cat.best_params
    print(f"Best CatBoost F1: {study_cat.best_value:.4f}")
    
    return {
        'xgboost': best_xgb_params,
        'catboost': best_cat_params
    }


def train_models(X_train, y_train, best_params):
    """Train all models with optimized parameters"""
    print("\n" + "=" * 70)
    print("Training Optimized Models")
    print("=" * 70)
    
    results = {}
    
    # XGBoost Optimized
    print("\n[1/7] Training Optimized XGBoost...")
    xgb_opt = XGBClassifier(**best_params['xgboost'])
    xgb_opt.fit(X_train, y_train)
    results['XGBoost_Optimized'] = xgb_opt
    
    # CatBoost Optimized
    print("[2/7] Training Optimized CatBoost...")
    cat_opt = CatBoostClassifier(**best_params['catboost'])
    cat_opt.fit(X_train, y_train)
    results['CatBoost_Optimized'] = cat_opt
    
    # LightGBM
    print("[3/7] Training LightGBM (High Minority Weight)...")
    lgbm_opt = LGBMClassifier(
        n_estimators=400,
        max_depth=12,
        learning_rate=0.03,
        num_leaves=50,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight={0: 1, 1: 15},
        random_state=42,
        verbose=-1
    )
    lgbm_opt.fit(X_train, y_train)
    results['LightGBM_Optimized'] = lgbm_opt
    
    # Random Forest
    print("[4/7] Training Random Forest (Precision-Focused)...")
    rf_opt = RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight={0: 1, 1: 12},
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf_opt.fit(X_train, y_train)
    results['RandomForest_Optimized'] = rf_opt
    
    # Calibrated CatBoost
    print("[5/7] Training Calibrated CatBoost...")
    cat_calibrated = CalibratedClassifierCV(cat_opt, method='isotonic', cv=3)
    cat_calibrated.fit(X_train, y_train)
    results['CatBoost_Calibrated'] = cat_calibrated
    
    # Stacking Ensemble
    print("[6/7] Training Stacking Ensemble...")
    estimators = [
        ('xgb', xgb_opt),
        ('cat', cat_opt),
        ('lgbm', lgbm_opt),
        ('rf', rf_opt)
    ]
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight={0: 1, 1: 10}, max_iter=1000),
        cv=3,
        n_jobs=-1
    )
    stacking.fit(X_train, y_train)
    results['Stacking_Ensemble'] = stacking
    
    # Voting Classifier
    print("[7/7] Training Voting Classifier...")
    voting = VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=[2, 2, 1.5, 1],
        n_jobs=-1
    )
    voting.fit(X_train, y_train)
    results['Voting_Ensemble'] = voting
    
    print("\n All models trained successfully!")
    return results


def train_pipeline(data, n_trials=20, save_path='./'):
    """
    Complete training pipeline
    
    Args:
        data: Dictionary from data_preparing.prepare_data()
        n_trials: Number of Optuna trials for hyperparameter optimization
        save_path: Path to save models and results
        
    Returns:
        Dictionary containing trained models and metadata
    """
    # Extract data
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    
    # Test resampling strategies
    X_train_resampled, y_train_resampled = test_resampling_strategies(X_train, y_train)
    
    # Optimize hyperparameters
    best_params = optimize_hyperparameters(X_train_resampled, y_train_resampled, 
                                          X_val, y_val, n_trials=n_trials)
    
    # Train all models
    trained_models = train_models(X_train_resampled, y_train_resampled, best_params)
    
    # Save scaler
    scaler_path = f"{save_path}/scaler_optimized.pkl"
    joblib.dump(data['scaler'], scaler_path)
    print(f"\n Scaler saved: {scaler_path}")
    
    # Save feature info
    feature_info = {
        'feature_names': data['feature_names'],
        'n_features': len(data['feature_names'])
    }
    feature_path = f"{save_path}/feature_info.json"
    with open(feature_path, 'w') as f:
        json.dump(feature_info, f, indent=4)
    print(f" Feature info saved: {feature_path}")
    
    return {
        'models': trained_models,
        'best_params': best_params,
        'X_train_resampled': X_train_resampled,
        'y_train_resampled': y_train_resampled,
        'feature_names': data['feature_names'],
        'scaler': data['scaler']
    }


if __name__ == "__main__":
    # This would be run after data_preparing.py
    # Example:
    # from data_preparing import prepare_data
    # data = prepare_data("/path/to/data")
    # training_results = train_pipeline(data, n_trials=20)
    print("Import this module and use train_pipeline() function")