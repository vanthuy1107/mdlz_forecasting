"""Metrics calculation utilities."""
import numpy as np
import pandas as pd
import os
from typing import Union, Tuple


def calculate_sfa_accuracy(
    actual: Union[np.ndarray, pd.Series],
    forecast: Union[np.ndarray, pd.Series]
) -> Union[float, np.ndarray]:
    """
    Calculate Shipping Forecast Accuracy (SFA) for individual items or arrays.
    
    Formula: 
    - If abs(actual - forecast) > actual, then accuracy = 0
    - Otherwise, accuracy = 1 - abs(actual - forecast) / actual
    
    Special case:
    - If actual = 0 and forecast = 0, then accuracy = 1
    - If actual = 0 and forecast != 0, then accuracy = 0
    
    Args:
        actual: Actual values (numpy array or pandas Series)
        forecast: Forecasted values (same shape as actual)
    
    Returns:
        Accuracy value(s) between 0 and 1. Returns single float for scalar input,
        numpy array for array input.
    """
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)
    
    error = np.abs(actual - forecast)
    
    # Calculate accuracy where actual > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        accuracy = 1.0 - error / actual
    
    # Rule 1: If error > actual, set accuracy = 0
    accuracy = np.where(error > actual, 0.0, accuracy)
    
    # Rule 2: Handle actual = 0 cases
    # If actual = 0 and forecast = 0, accuracy = 1
    # If actual = 0 and forecast != 0, accuracy = 0
    accuracy = np.where(actual == 0,
                        np.where(forecast == 0, 1.0, 0.0),
                        accuracy)
    
    return accuracy.item() if accuracy.shape == () else accuracy


def calculate_sfa_by_category(
    df: pd.DataFrame,
    actual_col: str = 'actual',
    forecast_col: str = 'forecast',
    category_col: str = 'category'
) -> pd.DataFrame:
    """
    Calculate SFA accuracy grouped by category.
    
    Args:
        df: DataFrame with columns [actual_col, forecast_col, category_col]
        actual_col: Column name for actual values
        forecast_col: Column name for forecast values
        category_col: Column name for categories
    
    Returns:
        DataFrame with columns ['category', 'accuracy', 'count']
    """
    results = []
    
    for category in df[category_col].unique():
        cat_data = df[df[category_col] == category]
        accuracies = calculate_sfa_accuracy(cat_data[actual_col].values, cat_data[forecast_col].values)
        mean_accuracy = np.mean(accuracies)
        
        results.append({
            'category': category,
            'accuracy': mean_accuracy,
            'count': len(cat_data)
        })
    
    return pd.DataFrame(results)


def calculate_sfa_by_date(
    df: pd.DataFrame,
    actual_col: str = 'actual',
    forecast_col: str = 'forecast',
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Calculate SFA accuracy grouped by date (daily).
    
    Args:
        df: DataFrame with columns [actual_col, forecast_col, date_col]
        actual_col: Column name for actual values
        forecast_col: Column name for forecast values
        date_col: Column name for dates
    
    Returns:
        DataFrame with columns ['date', 'accuracy', 'count']
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    results = []
    
    for date in sorted(df[date_col].unique()):
        date_data = df[df[date_col] == date]
        accuracies = calculate_sfa_accuracy(date_data[actual_col].values, date_data[forecast_col].values)
        mean_accuracy = np.mean(accuracies)
        
        results.append({
            'date': date,
            'accuracy': mean_accuracy,
            'count': len(date_data)
        })
    
    return pd.DataFrame(results)


def calculate_sfa_by_month(
    df: pd.DataFrame,
    actual_col: str = 'actual',
    forecast_col: str = 'forecast',
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Calculate SFA accuracy grouped by month (monthly).
    
    Args:
        df: DataFrame with columns [actual_col, forecast_col, date_col]
        actual_col: Column name for actual values
        forecast_col: Column name for forecast values
        date_col: Column name for dates
    
    Returns:
        DataFrame with columns ['month', 'accuracy', 'count']
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['month'] = df[date_col].dt.to_period('M').astype(str)
    
    results = []
    
    for month in sorted(df['month'].unique()):
        month_data = df[df['month'] == month]
        accuracies = calculate_sfa_accuracy(month_data[actual_col].values, month_data[forecast_col].values)
        mean_accuracy = np.mean(accuracies)
        
        results.append({
            'month': month,
            'accuracy': mean_accuracy,
            'count': len(month_data)
        })
    
    return pd.DataFrame(results)


def calculate_sfa_by_category_and_date(
    df: pd.DataFrame,
    actual_col: str = 'actual',
    forecast_col: str = 'forecast',
    category_col: str = 'category',
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Calculate SFA accuracy grouped by both category and date (daily per category).
    
    Args:
        df: DataFrame with columns [actual_col, forecast_col, category_col, date_col]
        actual_col: Column name for actual values
        forecast_col: Column name for forecast values
        category_col: Column name for categories
        date_col: Column name for dates
    
    Returns:
        DataFrame with columns ['category', 'date', 'accuracy', 'actual', 'forecast']
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    results = []
    
    for category in sorted(df[category_col].unique()):
        cat_data = df[df[category_col] == category]
        
        for date in sorted(cat_data[date_col].unique()):
            date_data = cat_data[cat_data[date_col] == date]
            accuracies = calculate_sfa_accuracy(date_data[actual_col].values, date_data[forecast_col].values)
            mean_accuracy = np.mean(accuracies)
            total_actual = date_data[actual_col].sum()
            total_forecast = date_data[forecast_col].sum()
            
            results.append({
                'category': category,
                'date': date,
                'accuracy': mean_accuracy,
                'actual': total_actual,
                'forecast': total_forecast
            })
    
    return pd.DataFrame(results)


def calculate_sfa_by_category_and_month(
    df: pd.DataFrame,
    actual_col: str = 'actual',
    forecast_col: str = 'forecast',
    category_col: str = 'category',
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Calculate SFA accuracy grouped by both category and month (monthly per category).
    
    Args:
        df: DataFrame with columns [actual_col, forecast_col, category_col, date_col]
        actual_col: Column name for actual values
        forecast_col: Column name for forecast values
        category_col: Column name for categories
        date_col: Column name for dates
    
    Returns:
        DataFrame with columns ['category', 'month', 'accuracy', 'actual', 'forecast']
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['month'] = df[date_col].dt.to_period('M').astype(str)
    
    results = []
    
    for category in sorted(df[category_col].unique()):
        cat_data = df[df[category_col] == category]
        
        for month in sorted(cat_data['month'].unique()):
            month_data = cat_data[cat_data['month'] == month]
            accuracies = calculate_sfa_accuracy(month_data[actual_col].values, month_data[forecast_col].values)
            mean_accuracy = np.mean(accuracies)
            total_actual = month_data[actual_col].sum()
            total_forecast = month_data[forecast_col].sum()
            
            results.append({
                'category': category,
                'month': month,
                'accuracy': mean_accuracy,
                'actual': total_actual,
                'forecast': total_forecast
            })
    
    return pd.DataFrame(results)


def generate_accuracy_report(
    df: pd.DataFrame,
    actual_col: str = 'actual',
    forecast_col: str = 'predicted',
    category_col: str = 'category',
    date_col: str = 'date',
    category_name_map: dict = None,
    output_path: str = None
) -> str:
    """
    Generate a comprehensive accuracy report and optionally save to file.
    
    Args:
        df: DataFrame with prediction results
        actual_col: Column name for actual values
        forecast_col: Column name for forecast values
        category_col: Column name for categories
        date_col: Column name for dates
        category_name_map: Dictionary mapping category IDs to names (optional)
        output_path: Path to save report (optional, returns string if not provided)
    
    Returns:
        Report string
    """
    report = []
    report.append("=" * 80)
    report.append("SHIPPING FORECAST ACCURACY (SFA) REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Overall accuracy
    overall_acc = calculate_sfa_accuracy(df[actual_col].values, df[forecast_col].values)
    overall_accuracy = np.mean(overall_acc)
    report.append(f"OVERALL ACCURACY: {overall_accuracy:.2%}")
    report.append("")
    
    # By category
    report.append("-" * 80)
    report.append("ACCURACY BY CATEGORY")
    report.append("-" * 80)
    cat_acc = calculate_sfa_by_category(df, actual_col, forecast_col, category_col)
    cat_acc = cat_acc.sort_values('accuracy', ascending=False)
    
    for _, row in cat_acc.iterrows():
        cat_id = row['category']
        cat_name = category_name_map.get(cat_id, f"Category_{cat_id}") if category_name_map else f"Category_{cat_id}"
        report.append(f"  {cat_name:30s} | Accuracy: {row['accuracy']:6.2%} | Count: {int(row['count']):5d}")
    report.append("")
    
    # By category and month
    report.append("-" * 80)
    report.append("ACCURACY BY CATEGORY AND MONTH")
    report.append("-" * 80)
    
    cat_month_acc = calculate_sfa_by_category_and_month(df, actual_col, forecast_col, category_col, date_col)
    cat_month_acc = cat_month_acc.sort_values(['category', 'month'])
    
    current_category = None
    for _, row in cat_month_acc.iterrows():
        cat_id = row['category']
        cat_name = category_name_map.get(cat_id, f"Category_{cat_id}") if category_name_map else f"Category_{cat_id}"
        
        if current_category != cat_id:
            if current_category is not None:
                report.append("")
            report.append(f"\n{cat_name}:")
            current_category = cat_id
        
        month = row['month']
        accuracy = row['accuracy']
        actual_sum = row['actual']
        forecast_sum = row['forecast']
        
        report.append(f"  {month} | Accuracy: {accuracy:6.2%} | Actual: {actual_sum:10.0f} | Forecast: {forecast_sum:10.0f}")
    
    report.append("")
    report.append("=" * 80)
    
    report_str = "\n".join(report)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or "outputs", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_str)
        print(f"\n✓ Accuracy report saved to: {output_path}")
    
    return report_str


__all__ = [
    'calculate_sfa_accuracy',
    'calculate_sfa_by_category',
    'calculate_sfa_by_date',
    'calculate_sfa_by_month',
    'calculate_sfa_by_category_and_date',
    'calculate_sfa_by_category_and_month',
    'generate_accuracy_report',
]
