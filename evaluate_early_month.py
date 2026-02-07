# Evaluation Script for Early Month Predictions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def evaluate_early_month_performance(predictions_csv):
    """
    Evaluate predictions specifically for early month (days 1-10) vs rest of month.
    
    Args:
        predictions_csv: Path to predictions CSV file from mvp_train.py
    """
    if not os.path.exists(predictions_csv):
        print(f"ERROR: File not found: {predictions_csv}")
        print("Please run training first: python mvp_train.py --config config/config_DRY.yaml --category DRY")
        return None, None
    
    df = pd.read_csv(predictions_csv)
    
    # Check required columns
    required_cols = ['date', 'day_of_month', 'actual', 'predicted']
    if not all(col in df.columns for col in required_cols):
        print(f"ERROR: CSV missing required columns. Found: {df.columns.tolist()}")
        return None, None
    
    print("=" * 80)
    print("EARLY MONTH OVER-PREDICTION EVALUATION")
    print("=" * 80)
    print(f"Data file: {predictions_csv}")
    print(f"Total predictions: {len(df)}")
    print()
    
    # Split into early month (1-10) and rest of month (11-31)
    early_month = df[df['day_of_month'] <= 10].copy()
    rest_of_month = df[df['day_of_month'] > 10].copy()
    
    # Days 1-3 (critical period with 20x penalty)
    days_1_3 = df[df['day_of_month'] <= 3].copy()
    
    # Days 4-10 (transition period with linear decay)
    days_4_10 = df[(df['day_of_month'] >= 4) & (df['day_of_month'] <= 10)].copy()
    
    # Helper function to compute metrics
    def compute_metrics(data, name):
        if len(data) == 0:
            print(f"[{name}]: No data")
            return None
        
        # Handle zero actuals for MAPE
        non_zero_data = data[data['actual'] > 0].copy()
        
        mae = data['abs_error'].mean()
        rmse = np.sqrt((data['error'] ** 2).mean())
        mape = (non_zero_data['abs_error'] / non_zero_data['actual']).mean() * 100 if len(non_zero_data) > 0 else np.nan
        over_pred_rate = ((data['predicted'] > data['actual']).sum() / len(data)) * 100
        avg_error = data['error'].mean()
        avg_actual = data['actual'].mean()
        avg_predicted = data['predicted'].mean()
        
        print(f"\n[{name}]")
        print(f"  Samples: {len(data)}")
        print(f"  MAE: {mae:.2f} CBM")
        print(f"  RMSE: {rmse:.2f} CBM")
        if not np.isnan(mape):
            print(f"  MAPE: {mape:.2f}%")
        print(f"  Over-prediction rate: {over_pred_rate:.1f}%")
        print(f"  Avg error (pred - actual): {avg_error:+.2f} CBM {'⚠ OVER-PREDICTING' if avg_error > 50 else '✓ OK' if abs(avg_error) < 50 else '⚠ UNDER-PREDICTING'}")
        print(f"  Avg actual: {avg_actual:.2f} CBM")
        print(f"  Avg predicted: {avg_predicted:.2f} CBM")
        
        return {
            'samples': len(data),
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'over_pred_rate': over_pred_rate,
            'avg_error': avg_error,
            'avg_actual': avg_actual,
            'avg_predicted': avg_predicted
        }
    
    # Compute metrics for each period
    metrics_days_1_3 = compute_metrics(days_1_3, "DAYS 1-3 (20x Penalty)")
    metrics_days_4_10 = compute_metrics(days_4_10, "DAYS 4-10 (Linear Decay Penalty)")
    metrics_early = compute_metrics(early_month, "EARLY MONTH (Days 1-10)")
    metrics_rest = compute_metrics(rest_of_month, "REST OF MONTH (Days 11-31)")
    
    # Comparison and success criteria
    if metrics_early and metrics_rest:
        print("\n" + "=" * 80)
        print("COMPARISON & ANALYSIS")
        print("=" * 80)
        print(f"  Early month MAE vs Rest: {metrics_early['mae']:.2f} vs {metrics_rest['mae']:.2f}")
        print(f"  Early month MAPE vs Rest: {metrics_early['mape']:.2f}% vs {metrics_rest['mape']:.2f}%")
        print(f"  Early month over-pred rate vs Rest: {metrics_early['over_pred_rate']:.1f}% vs {metrics_rest['over_pred_rate']:.1f}%")
        
        print("\n" + "=" * 80)
        print("SUCCESS CRITERIA")
        print("=" * 80)
        
        # Check if early month over-prediction is fixed
        success_checks = []
        
        # 1. Early month MAPE should be reasonable (<30%)
        early_mape_ok = metrics_early['mape'] < 30
        success_checks.append(early_mape_ok)
        print(f"  {'✓' if early_mape_ok else '✗'} Early month MAPE < 30%: {metrics_early['mape']:.2f}% {'(PASS)' if early_mape_ok else '(FAIL)'}")
        
        # 2. Over-prediction rate should not be excessive (ideally <70%)
        over_pred_ok = metrics_early['over_pred_rate'] < 70
        success_checks.append(over_pred_ok)
        print(f"  {'✓' if over_pred_ok else '✗'} Over-prediction rate < 70%: {metrics_early['over_pred_rate']:.1f}% {'(PASS)' if over_pred_ok else '(FAIL)'}")
        
        # 3. Avg error should not be heavily positive (not systematically over-predicting by >100 CBM)
        avg_error_ok = metrics_early['avg_error'] < 100
        success_checks.append(avg_error_ok)
        print(f"  {'✓' if avg_error_ok else '✗'} Avg error < 100 CBM: {metrics_early['avg_error']:+.2f} CBM {'(PASS)' if avg_error_ok else '(FAIL)'}")
        
        # 4. Days 1-3 should have low error (critical period)
        if metrics_days_1_3:
            days_1_3_ok = metrics_days_1_3['mae'] < 300
            success_checks.append(days_1_3_ok)
            print(f"  {'✓' if days_1_3_ok else '✗'} Days 1-3 MAE < 300 CBM: {metrics_days_1_3['mae']:.2f} CBM {'(PASS)' if days_1_3_ok else '(FAIL)'}")
        
        print()
        if all(success_checks):
            print("  🎉 SUCCESS: Early month over-prediction issue appears to be FIXED!")
        else:
            print("  ⚠ WARNING: Early month still has issues. Consider:")
            print("    1. Training for more epochs (50-100)")
            print("    2. Adjusting penalty weights (reduce from 20x to 15x)")
            print("    3. Checking if post_peak_signal feature is being used effectively")
    
    return early_month, rest_of_month


if __name__ == "__main__":
    # Default path
    default_path = "outputs/DRY/test_predictions.csv"
    
    if len(sys.argv) > 1:
        predictions_csv = sys.argv[1]
    else:
        predictions_csv = default_path
    
    print(f"Evaluating predictions from: {predictions_csv}\n")
    
    early, rest = evaluate_early_month_performance(predictions_csv)
    
    if early is not None and rest is not None:
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
        print("\nTo fix remaining issues, see EARLY_MONTH_OVERPREDICTION_SOLUTION.md")

