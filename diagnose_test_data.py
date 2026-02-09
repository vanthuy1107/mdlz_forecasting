"""
Diagnostic script to analyze test set composition and identify data issues.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load predictions
df = pd.read_csv('output/dow-anchored/DRY/test_predictions_dow-anchored.csv')

print("=" * 80)
print("TEST SET DATA QUALITY ANALYSIS")
print("=" * 80)

# Overall statistics
print(f"\nTotal predictions: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Actual values analysis
print("\n" + "=" * 80)
print("ACTUAL VALUES DISTRIBUTION")
print("=" * 80)
print(f"Mean: {df['actual'].mean():.2f} CBM")
print(f"Median: {df['actual'].median():.2f} CBM")
print(f"Min: {df['actual'].min():.2f} CBM")
print(f"Max: {df['actual'].max():.2f} CBM")
print(f"Std: {df['actual'].std():.2f} CBM")

# Zero and low value analysis
zeros = (df['actual'] == 0).sum()
very_low = (df['actual'] < 50).sum()
low = (df['actual'] < 100).sum()

print(f"\n🔍 LOW VALUE ANALYSIS:")
print(f"Zeros (0 CBM): {zeros} ({zeros/len(df)*100:.1f}%)")
print(f"Very low (<50 CBM): {very_low} ({very_low/len(df)*100:.1f}%)")
print(f"Low (<100 CBM): {low} ({low/len(df)*100:.1f}%)")

# Early month vs rest
early = df[df['day_of_month'] <= 10]
rest = df[df['day_of_month'] > 10]

print("\n" + "=" * 80)
print("EARLY MONTH vs REST OF MONTH")
print("=" * 80)

print(f"\n📅 EARLY MONTH (Days 1-10): {len(early)} samples")
print(f"Mean actual: {early['actual'].mean():.2f} CBM")
print(f"Median actual: {early['actual'].median():.2f} CBM")
print(f"Zeros: {(early['actual'] == 0).sum()} ({(early['actual'] == 0).sum()/len(early)*100:.1f}%)")
print(f"Very low (<50): {(early['actual'] < 50).sum()} ({(early['actual'] < 50).sum()/len(early)*100:.1f}%)")

print(f"\n📅 REST OF MONTH (Days 11-31): {len(rest)} samples")
print(f"Mean actual: {rest['actual'].mean():.2f} CBM")
print(f"Median actual: {rest['actual'].median():.2f} CBM")
print(f"Zeros: {(rest['actual'] == 0).sum()} ({(rest['actual'] == 0).sum()/len(rest)*100:.1f}%)")
print(f"Very low (<50): {(rest['actual'] < 50).sum()} ({(rest['actual'] < 50).sum()/len(rest)*100:.1f}%)")

# Prediction bias analysis
print("\n" + "=" * 80)
print("MODEL BIAS ANALYSIS")
print("=" * 80)

bias = df['predicted'].mean() - df['actual'].mean()
print(f"\nOverall bias: {bias:+.2f} CBM")
print(f"Model predicts {bias:+.2f} CBM higher on average")

# When actual is zero, what does model predict?
zeros_df = df[df['actual'] == 0]
if len(zeros_df) > 0:
    print(f"\n⚠️ ZERO DAYS ANALYSIS:")
    print(f"Number of zero days: {len(zeros_df)}")
    print(f"Model prediction on zero days (avg): {zeros_df['predicted'].mean():.2f} CBM")
    print(f"Model prediction on zero days (median): {zeros_df['predicted'].median():.2f} CBM")
    print(f"This explains {(zeros_df['abs_error'].sum() / df['abs_error'].sum() * 100):.1f}% of total error")

# MAPE breakdown by actual value ranges
print("\n" + "=" * 80)
print("MAPE BY ACTUAL VALUE RANGE")
print("=" * 80)

ranges = [
    (0, 100, "Very Low (0-100)"),
    (100, 300, "Low (100-300)"),
    (300, 500, "Medium (300-500)"),
    (500, 700, "High (500-700)"),
    (700, 2000, "Very High (700+)")
]

for low, high, label in ranges:
    mask = (df['actual'] >= low) & (df['actual'] < high)
    subset = df[mask]
    if len(subset) > 0:
        mape = (subset['abs_error'] / (subset['actual'] + 1e-8)).mean() * 100
        print(f"{label:20s}: {len(subset):3d} samples, MAPE = {mape:6.1f}%")

# Recommendations
print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

if zeros / len(df) > 0.05:  # More than 5% zeros
    print("\n⚠️ HIGH ZERO RATE DETECTED")
    print("  Issue: Model cannot predict zeros (always predicts positive)")
    print("  Solutions:")
    print("    1. Use zero-inflated model (classification + regression)")
    print("    2. Accept that MAPE will be high for zero days")
    print("    3. Focus on MAE instead of MAPE")

if bias > 100:
    print("\n⚠️ SYSTEMATIC OVER-PREDICTION DETECTED")
    print(f"  Issue: Model predicts {bias:.0f} CBM too high on average")
    print("  Solutions:")
    print("    1. Check if training data has EOM spikes biasing predictions")
    print("    2. Consider removing EOM days from training")
    print("    3. Add more aggressive scaling/normalization")

if early['actual'].mean() < rest['actual'].mean():
    diff = rest['actual'].mean() - early['actual'].mean()
    print("\n✓ EARLY MONTH IS GENUINELY LOWER")
    print(f"  Early month avg: {early['actual'].mean():.0f} CBM")
    print(f"  Rest of month avg: {rest['actual'].mean():.0f} CBM")
    print(f"  Difference: {diff:.0f} CBM ({diff/rest['actual'].mean()*100:.1f}%)")
    print("  → This confirms the early month over-prediction problem is real")
else:
    print("\n⚠️ WARNING: Early month not significantly lower")
    print("  The 'early month over-prediction' might not be as critical as thought")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
