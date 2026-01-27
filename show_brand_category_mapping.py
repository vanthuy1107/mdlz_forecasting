"""Show BRAND to CATEGORY mapping from masterdata file.

This script reads the masterdata Excel file and displays the mapping
of which BRAND belongs to which CATEGORY.

Usage:
    python show_brand_category_mapping.py
"""
import pandas as pd
from pathlib import Path

# Path to masterdata file
script_dir = Path(__file__).resolve().parent
masterdata_path = script_dir / "masterdata" / "masterdata category-brand.xlsx"

print("=" * 80)
print("BRAND to CATEGORY Mapping")
print("=" * 80)
print(f"\nReading from: {masterdata_path}")

if not masterdata_path.exists():
    print(f"\n[ERROR] Masterdata file not found at: {masterdata_path}")
    exit(1)

# Read the Excel file
try:
    df = pd.read_excel(masterdata_path)
    print(f"\nLoaded {len(df):,} rows from masterdata file")
    
    # Show column names
    print(f"\nColumns in file: {list(df.columns)}")
    
    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()
    
    # Find CATEGORY and BRAND columns (case-insensitive)
    cols_upper = {c.upper(): c for c in df.columns}
    
    # Find GROUP OF CAGO column (category column)
    cat_col = None
    for key, orig in cols_upper.items():
        if 'CATEGORY' in key or 'GROUP' in key or 'CAGO' in key:
            cat_col = orig
            break
    
    # Find BRAND column
    brand_col = None
    for key, orig in cols_upper.items():
        if 'BRAND' in key:
            brand_col = orig
            break
    
    if cat_col is None:
        print(f"\n[ERROR] Could not find CATEGORY column")
        print(f"        Available columns: {list(df.columns)}")
        exit(1)
    
    if brand_col is None:
        print(f"\n[ERROR] Could not find BRAND column")
        print(f"        Available columns: {list(df.columns)}")
        exit(1)
    
    print(f"\nUsing columns:")
    print(f"  - Category column: '{cat_col}'")
    print(f"  - Brand column: '{brand_col}'")
    
    # Clean the data
    df[cat_col] = df[cat_col].astype(str).str.strip()
    df[brand_col] = df[brand_col].astype(str).str.strip()
    
    # Remove NaN values
    df = df[(df[cat_col] != 'nan') & (df[brand_col] != 'nan')]
    df = df[(df[cat_col] != '') & (df[brand_col] != '')]
    
    # Get unique category-brand pairs
    mapping = df[[cat_col, brand_col]].drop_duplicates()
    mapping = mapping.sort_values([cat_col, brand_col])
    
    print(f"\n{'=' * 80}")
    print("BRAND to CATEGORY Mapping")
    print(f"{'=' * 80}\n")
    
    # Group by category
    for category in sorted(mapping[cat_col].unique()):
        brands = mapping[mapping[cat_col] == category][brand_col].tolist()
        print(f"{category}:")
        for i, brand in enumerate(brands, 1):
            print(f"  {i:2d}. {brand}")
        print()
    
    # Summary by category
    print(f"{'=' * 80}")
    print("SUMMARY BY CATEGORY")
    print(f"{'=' * 80}\n")
    
    summary = mapping.groupby(cat_col)[brand_col].count().sort_values(ascending=False)
    for category, count in summary.items():
        print(f"{category:20s}: {count:3d} brand(s)")
    
    print(f"\nTotal categories: {mapping[cat_col].nunique()}")
    print(f"Total brands: {mapping[brand_col].nunique()}")
    
    print(f"\n{'=' * 80}")
    
except Exception as e:
    print(f"\n[ERROR] Failed to read masterdata file: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
