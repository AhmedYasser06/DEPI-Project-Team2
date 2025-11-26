# data_evaluation.py
# ============================================================
# MODEL EVALUATION, THRESHOLD OPTIMIZATION & VISUALIZATION
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, precision_recall_curve, f1_score, accuracy_score,
                             precision_score, recall_score)
import joblib
import json
import warnings
warnings.filterwarnings('ignore')


def find_optimal_threshold(y_true, y_proba, metric='f1'):
    """Find threshold that maximizes given metric"""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_score = 0
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def optimize_thresholds(models, X_val, y_val, X_test, y_test):
    """Optimize decision thresholds for all models"""
    print("\n" + "=" * 70)
    print("Optimizing Decision Thresholds")
    print("=" * 70)
    
    optimized_results = {}
    
    for model_name, model in models.items():
        # Get probabilities on validation set
        if hasattr(model, 'predict_proba'):
            y_val_proba = model.predict_proba(X_val)[:, 1]
        else:
            y_val_proba = model.decision_function(X_val)
        
        # Find optimal threshold
        opt_threshold, val_f1 = find_optimal_threshold(y_val, y_val_proba, metric='f1')
        
        # Apply to test set
        y_test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
        y_test_pred = (y_test_proba >= opt_threshold).astype(int)
        
        optimized_results[model_name] = {
            'predictions': y_test_pred,
            'probabilities': y_test_proba,
            'threshold': opt_threshold,
            'model': model
        }
        
        print(f"{model_name}: Threshold={opt_threshold:.3f}, Val F1={val_f1:.4f}")
    
    return optimized_results


def evaluate_models(optimized_results, y_test):
    """Comprehensive evaluation of all models"""
    print("\n" + "=" * 70)
    print("Model Evaluation")
    print("=" * 70)
    
    evaluation_df = []
    
    for model_name, result in optimized_results.items():
        pred = result['predictions']
        proba = result['probabilities']
        threshold = result['threshold']
        
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        auc = roc_auc_score(y_test, proba)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        specificity = tn / (tn + fp)
        
        evaluation_df.append({
            'Model': model_name,
            'Threshold': threshold,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc,
            'Specificity': specificity,
            'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn
        })
        
        print(f"\n{model_name} (Threshold={threshold:.3f}):")
        print(f"  Accuracy:    {acc:.4f}")
        print(f"  Precision:   {prec:.4f}")
        print(f"  Recall:      {rec:.4f}")
        print(f"  F1-Score:    {f1:.4f}")
        print(f"  ROC-AUC:     {auc:.4f}")
        print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    
    eval_df = pd.DataFrame(evaluation_df)
    eval_df = eval_df.sort_values('F1-Score', ascending=False)
    
    print("\n" + "=" * 70)
    print("Model Ranking by F1-Score:")
    print("=" * 70)
    print(eval_df[['Model', 'Threshold', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].to_string(index=False))
    
    return eval_df


def generate_visualizations(optimized_results, y_test, eval_df, save_path='./'):
    """Generate all evaluation visualizations"""
    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70)
    
    top_models = eval_df.head(6)['Model'].tolist()
    
    # Confusion Matrices
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Confusion Matrices - Top 6 Models', fontsize=16, fontweight='bold')
    
    for idx, model_name in enumerate(top_models):
        row, col = idx // 3, idx % 3
        cm = confusion_matrix(y_test, optimized_results[model_name]['predictions'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=axes[row, col],
                    cbar_kws={'label': 'Count'})
        
        f1 = eval_df[eval_df['Model'] == model_name]['F1-Score'].values[0]
        prec = eval_df[eval_df['Model'] == model_name]['Precision'].values[0]
        rec = eval_df[eval_df['Model'] == model_name]['Recall'].values[0]
        
        axes[row, col].set_title(f'{model_name}\nF1={f1:.3f}, Prec={prec:.3f}, Rec={rec:.3f}')
        axes[row, col].set_ylabel('True')
        axes[row, col].set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/confusion_matrices_optimized.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ROC Curves
    plt.figure(figsize=(12, 8))
    for model_name in top_models:
        fpr, tpr, _ = roc_curve(y_test, optimized_results[model_name]['probabilities'])
        auc_score = roc_auc_score(y_test, optimized_results[model_name]['probabilities'])
        plt.plot(fpr, tpr, label=f'{model_name} (AUC={auc_score:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('ROC Curves - Optimized Models', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig(f'{save_path}/roc_curves_optimized.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Precision-Recall Curves
    plt.figure(figsize=(12, 8))
    for model_name in top_models:
        precision, recall, _ = precision_recall_curve(y_test, optimized_results[model_name]['probabilities'])
        plt.plot(recall, precision, label=f'{model_name}', linewidth=2)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - Optimized Models', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.savefig(f'{save_path}/precision_recall_optimized.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Metrics Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metrics = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple']
    
    for idx, metric in enumerate(metrics):
        row, col = idx // 2, idx % 2
        data = eval_df.sort_values(metric, ascending=True).tail(6)
        axes[row, col].barh(data['Model'], data[metric], color=colors[idx])
        axes[row, col].set_xlabel(metric, fontsize=12)
        axes[row, col].set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
        axes[row, col].grid(axis='x', alpha=0.3)
        
        for i, v in enumerate(data[metric]):
            axes[row, col].text(v + 0.01, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("All visualizations saved!")


def analyze_errors(optimized_results, y_test, X_test, feature_names, best_model_name, save_path='./'):
    """Perform detailed error analysis"""
    print("\n" + "=" * 70)
    print("Error Analysis")
    print("=" * 70)
    
    best_predictions = optimized_results[best_model_name]['predictions']
    best_probabilities = optimized_results[best_model_name]['probabilities']
    
    # Create error analysis dataframe
    error_df = pd.DataFrame({
        'True_Label': y_test,
        'Predicted_Label': best_predictions,
        'Probability': best_probabilities
    })
    
    # Add original features
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    error_df = pd.concat([error_df, X_test_df.reset_index(drop=True)], axis=1)
    
    # Analyze errors
    false_positives = error_df[(error_df['True_Label'] == 0) & (error_df['Predicted_Label'] == 1)]
    false_negatives = error_df[(error_df['True_Label'] == 1) & (error_df['Predicted_Label'] == 0)]
    true_positives = error_df[(error_df['True_Label'] == 1) & (error_df['Predicted_Label'] == 1)]
    
    print(f"\nFalse Positives: {len(false_positives)}")
    if len(false_positives) > 0:
        print(f"  Avg probability: {false_positives['Probability'].mean():.4f}")
    
    print(f"\nFalse Negatives: {len(false_negatives)}")
    if len(false_negatives) > 0:
        print(f"  Avg probability: {false_negatives['Probability'].mean():.4f}")
    
    print(f"\nTrue Positives: {len(true_positives)}")
    if len(true_positives) > 0:
        print(f"  Avg probability: {true_positives['Probability'].mean():.4f}")
    
    # Save error analysis
    error_df.to_csv(f'{save_path}/error_analysis.csv', index=False)
    print(f"\n✓ Error analysis saved: {save_path}/error_analysis.csv")
    
    return error_df


def save_best_model(eval_df, optimized_results, save_path='./'):
    """Save the best performing model and its metadata"""
    print("\n" + "=" * 70)
    print("Saving Best Model")
    print("=" * 70)
    
    best_model_name = eval_df.iloc[0]['Model']
    best_model = optimized_results[best_model_name]['model']
    best_threshold = optimized_results[best_model_name]['threshold']
    
    print(f"\n BEST MODEL: {best_model_name}")
    print(f"   Optimal Threshold: {best_threshold:.3f}")
    print(f"   F1-Score:  {eval_df.iloc[0]['F1-Score']:.4f}")
    print(f"   Precision: {eval_df.iloc[0]['Precision']:.4f}")
    print(f"   Recall:    {eval_df.iloc[0]['Recall']:.4f}")
    print(f"   ROC-AUC:   {eval_df.iloc[0]['ROC-AUC']:.4f}")
    
    # Save model
    model_filename = f'{save_path}/best_model_{best_model_name.replace(" ", "_").lower()}.pkl'
    joblib.dump(best_model, model_filename)
    print(f"\n  Model saved: {model_filename}")
    
    # Save threshold and metrics
    threshold_info = {
        'model_name': best_model_name,
        'optimal_threshold': float(best_threshold),
        'metrics': {
            'f1_score': float(eval_df.iloc[0]['F1-Score']),
            'precision': float(eval_df.iloc[0]['Precision']),
            'recall': float(eval_df.iloc[0]['Recall']),
            'roc_auc': float(eval_df.iloc[0]['ROC-AUC'])
        }
    }
    
    with open(f'{save_path}/model_threshold.json', 'w') as f:
        json.dump(threshold_info, f, indent=4)
    print(f"  Threshold saved: {save_path}/model_threshold.json")
    
    # Save evaluation results
    eval_df.to_csv(f'{save_path}/evaluation_results_optimized.csv', index=False)
    print(f"  Evaluation results saved: {save_path}/evaluation_results_optimized.csv")
    
    return best_model_name, best_model, best_threshold


def evaluation_pipeline(training_results, data, save_path='./'):
    """
    Complete evaluation pipeline
    
    Args:
        training_results: Dictionary from data_training.train_pipeline()
        data: Dictionary from data_preparing.prepare_data()
        save_path: Path to save results
        
    Returns:
        Dictionary containing evaluation results
    """
    models = training_results['models']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Optimize thresholds
    optimized_results = optimize_thresholds(models, X_val, y_val, X_test, y_test)
    
    # Evaluate all models
    eval_df = evaluate_models(optimized_results, y_test)
    
    # Generate visualizations
    generate_visualizations(optimized_results, y_test, eval_df, save_path)
    
    # Save best model
    best_model_name, best_model, best_threshold = save_best_model(eval_df, optimized_results, save_path)
    
    # Error analysis
    error_df = analyze_errors(optimized_results, y_test, X_test, 
                              training_results['feature_names'], 
                              best_model_name, save_path)
    
    # Classification report
    print("\n" + "=" * 70)
    print("Detailed Classification Report - Best Model")
    print("=" * 70)
    best_predictions = optimized_results[best_model_name]['predictions']
    print(classification_report(y_test, best_predictions,
                              target_names=['No Failure', 'Failure'],
                              digits=4))
    
    print("\n" + "=" * 70)
    print(" EVALUATION PIPELINE COMPLETED!")
    print("=" * 70)
    
    return {
        'eval_df': eval_df,
        'best_model_name': best_model_name,
        'best_model': best_model,
        'best_threshold': best_threshold,
        'optimized_results': optimized_results,
        'error_df': error_df
    }


if __name__ == "__main__":
    print("Import this module and use evaluation_pipeline() function")