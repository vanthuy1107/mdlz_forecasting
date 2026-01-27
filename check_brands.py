"""Helper script to check available brands in data.

Usage:
    python check_brands.py
    python check_brands.py --category DRY
    python check_brands.py --data-file dataset/data_cat/data_2024.csv
"""
import pandas as pd
import argparse
from pathlib import Path
from config import load_config
from src.data import DataReader


def main():
    parser = argparse.ArgumentParser(
        description="Check available brands in data"
    )
    parser.add_argument(
        '--category',
        type=str,
        help='Filter to specific category (e.g., DRY, FRESH)'
    )
    parser.add_argument(
        '--data-file',
        type=str,
        help='Path to data file. If not specified, loads from config'
    )
    parser.add_argument(
        '--brand-col',
        type=str,
        default='BRAND',
        help='Name of BRAND column (default: BRAND)'
    )
    parser.add_argument(
        '--cat-col',
        type=str,
        default='CATEGORY',
        help='Name of CATEGORY column (default: CATEGORY)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Brand Availability Checker")
    print("=" * 80)
    
    # Load data
    print("\n[1/2] Loading data...")
    if args.data_file:
        print(f"  - Loading from: {args.data_file}")
        data = pd.read_csv(args.data_file)
    else:
        # Load from config
        config = load_config()
        data_config = config.data
        data_reader = DataReader(
            data_dir=data_config['data_dir'],
            file_pattern=data_config['file_pattern']
        )
        
        years = data_config['years']
        print(f"  - Loading years: {years}")
        try:
            data = data_reader.load(years=years)
        except FileNotFoundError:
            print("[WARNING] Combined year files not found, trying pattern-based loading...")
            file_prefix = data_config.get('file_prefix', 'Outboundreports')
            data = data_reader.load_by_file_pattern(
                years=years,
                file_prefix=file_prefix
            )
    
    print(f"  - Loaded {len(data):,} samples")
    
    # Check columns
    if args.brand_col not in data.columns:
        print(f"\n[ERROR] BRAND column '{args.brand_col}' not found in data")
        print(f"        Available columns: {list(data.columns)}")
        return 1
    
    if args.cat_col not in data.columns:
        print(f"\n[ERROR] CATEGORY column '{args.cat_col}' not found in data")
        print(f"        Available columns: {list(data.columns)}")
        return 1
    
    # Filter by category if specified
    if args.category:
        print(f"\n[2/2] Filtering to category: {args.category}")
        data = data[data[args.cat_col] == args.category].copy()
        print(f"  - Filtered to {len(data):,} samples")
        
        if len(data) == 0:
            print(f"\n[ERROR] No data found for category '{args.category}'")
            available_cats = sorted(data[args.cat_col].unique())
            print(f"        Available categories: {available_cats}")
            return 1
    
    # Get brands
    print(f"\n{'=' * 80}")
    print("BRAND SUMMARY")
    print(f"{'=' * 80}")
    
    if args.category:
        print(f"Category: {args.category}")
    else:
        print("All Categories")
    
    # Get unique brands (excluding NaN)
    brands = data[args.brand_col].dropna().unique()
    brands = [b for b in brands if str(b).strip() and str(b).upper() != 'NAN']
    brands = sorted(brands)
    
    print(f"\nFound {len(brands)} brand(s):")
    print()
    
    # Calculate statistics for each brand
    brand_stats = []
    for brand in brands:
        brand_data = data[data[args.brand_col] == brand]
        
        # Get date range
        time_col = 'ACTUALSHIPDATE'
        if time_col in brand_data.columns:
            brand_data[time_col] = pd.to_datetime(brand_data[time_col])
            min_date = brand_data[time_col].min()
            max_date = brand_data[time_col].max()
            date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
        else:
            date_range = "N/A"
        
        # Get total volume
        target_col = 'Total CBM'
        if target_col in brand_data.columns:
            total_volume = brand_data[target_col].sum()
        else:
            total_volume = None
        
        brand_stats.append({
            'brand': brand,
            'samples': len(brand_data),
            'date_range': date_range,
            'total_volume': total_volume
        })
    
    # Print results
    for i, stats in enumerate(brand_stats, 1):
        print(f"{i:2d}. {stats['brand']}")
        print(f"    Samples: {stats['samples']:,}")
        print(f"    Date range: {stats['date_range']}")
        if stats['total_volume'] is not None:
            print(f"    Total volume: {stats['total_volume']:,.2f} CBM")
        print()
    
    # Check for trained models
    print(f"{'=' * 80}")
    print("TRAINED MODELS")
    print(f"{'=' * 80}")
    
    if args.category:
        config = load_config()
        base_output_dir = Path(config.output['output_dir'])
        
        print(f"\nChecking for trained models in: {base_output_dir}")
        print()
        
        brands_with_models = []
        brands_without_models = []
        
        for brand in brands:
            brand_output_name = f"{args.category}_{brand.replace(' ', '_').replace('/', '_')}"
            brand_model_path = base_output_dir / brand_output_name / 'models' / 'best_model.pth'
            
            if brand_model_path.exists():
                brands_with_models.append(brand)
                print(f"✓ {brand}")
                print(f"  Model: {brand_model_path}")
            else:
                brands_without_models.append(brand)
                print(f"✗ {brand}")
                print(f"  No model found at: {brand_model_path}")
            print()
        
        print(f"{'=' * 80}")
        print(f"Models found: {len(brands_with_models)}/{len(brands)}")
        
        if brands_without_models:
            print(f"\nTo train missing models:")
            brand_list = ' '.join(brands_without_models)
            print(f"  python train_by_brand.py --category {args.category} --brands {brand_list}")
    
    print(f"\n{'=' * 80}")
    
    return 0


if __name__ == "__main__":
    exit(main())
