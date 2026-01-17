"""Script to combine data files for each year separately (2022-2025)."""
import os
import pandas as pd
from pathlib import Path
import argparse
import re
from typing import List, Optional, Dict
from collections import defaultdict


def extract_year_from_filename(filename: str) -> Optional[int]:
    """
    Extract year from filename patterns like:
    - Outboundreports_YYYYMMDD_YYYYMMDD.csv
    - data_YYYY.csv
    
    Args:
        filename: Name of the file.
    
    Returns:
        Year as integer, or None if not found.
    """
    # Pattern 1: Outboundreports_YYYYMMDD_YYYYMMDD.csv
    match = re.search(r'Outboundreports_(\d{4})\d{4}_\d{8}', filename)
    if match:
        return int(match.group(1))
    
    # Pattern 2: data_YYYY.csv
    match = re.search(r'data_(\d{4})\.csv', filename)
    if match:
        return int(match.group(1))
    
    # Pattern 3: Any 4-digit year in filename
    match = re.search(r'(\d{4})', filename)
    if match:
        year = int(match.group(1))
        # Validate year is reasonable (2000-2100)
        if 2000 <= year <= 2100:
            return year
    
    return None


def find_files_by_year(data_dir: Path, years: List[int], file_prefix: str = "Outboundreports") -> Dict[int, List[Path]]:
    """
    Find all CSV files in directory and group them by year.
    
    Args:
        data_dir: Directory to search.
        years: List of years to filter by.
        file_prefix: Prefix to filter files (e.g., "Outboundreports").
    
    Returns:
        Dictionary mapping year to list of file paths.
    """
    files_by_year = defaultdict(list)
    years_set = set(years)
    
    if not data_dir.exists():
        return files_by_year
    
    # Find all CSV files
    csv_files = list(data_dir.glob("*.csv"))
    
    print(f"Found {len(csv_files)} CSV files in directory")
    
    for filepath in csv_files:
        filename = filepath.name
        
        # Filter by prefix if specified
        if file_prefix and not filename.startswith(file_prefix):
            continue
        
        year = extract_year_from_filename(filename)
        
        if year and year in years_set:
            files_by_year[year].append(filepath)
    
    # Sort files within each year
    for year in files_by_year:
        files_by_year[year].sort()
    
    return files_by_year


def combine_yearly_data(
    data_dir: str,
    output_dir: str,
    years: List[int] = [2022, 2023, 2024, 2025],
    file_prefix: str = "Outboundreports",
    output_pattern: str = "data_{year}.csv",
    time_col: str = "ACTUALSHIPDATE",
    sort_by_time: bool = True
) -> Dict[int, Path]:
    """
    Combine data files for each year separately.
    
    Handles files named like: Outboundreports_YYYYMMDD_YYYYMMDD.csv
    Creates separate combined files for each year: data_2022.csv, data_2023.csv, etc.
    
    Args:
        data_dir: Directory containing yearly data files.
        output_dir: Directory to save combined year files.
        years: List of years to combine.
        file_prefix: Prefix to filter files (e.g., "Outboundreports").
        output_pattern: Pattern for output file names (must contain {year}).
        time_col: Name of time column for sorting.
        sort_by_time: Whether to sort combined data by time column.
    
    Returns:
        Dictionary mapping year to output file path.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Combining data files for years: {years}")
    print(f"Source directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"File prefix: {file_prefix}\n")
    
    # Find files by year
    files_by_year = find_files_by_year(data_dir, years, file_prefix)
    
    if not files_by_year:
        raise FileNotFoundError(
            f"No data files found for years {years} in directory: {data_dir}\n"
            f"Expected files matching pattern: {file_prefix}_YYYYMMDD_YYYYMMDD.csv"
        )
    
    saved_files = {}
    total_files_loaded = 0
    failed_files = []
    
    # Process each year separately
    for year in sorted(years):
        if year not in files_by_year:
            print(f"[⚠] No files found for year {year}")
            continue
        
        year_files = files_by_year[year]
        print(f"\n[Year {year}] Found {len(year_files)} file(s):")
        
        year_data = []
        for filepath in year_files:
            try:
                print(f"  Loading: {filepath.name}...", end=" ")
                data = pd.read_csv(filepath)
                print(f"✓ ({len(data)} rows)")
                
                year_data.append(data)
                total_files_loaded += 1
            except Exception as e:
                print(f"✗ ERROR: {e}")
                failed_files.append((filepath, str(e)))
        
        if not year_data:
            print(f"  [SKIP] No data loaded for year {year}")
            continue
        
        # Combine all files for this year
        year_combined = pd.concat(year_data, ignore_index=True)
        print(f"  Combined: {len(year_combined)} total rows")
        
        # Ensure time column is datetime and sort if requested
        if time_col in year_combined.columns:
            if not pd.api.types.is_datetime64_any_dtype(year_combined[time_col]):
                print(f"  Converting {time_col} to datetime...")
                year_combined[time_col] = pd.to_datetime(year_combined[time_col])
            
            if sort_by_time:
                print(f"  Sorting by {time_col}...")
                year_combined = year_combined.sort_values(time_col).reset_index(drop=True)
            
            date_range = f"{year_combined[time_col].min()} to {year_combined[time_col].max()}"
            print(f"  Date range: {date_range}")
        else:
            print(f"  [WARNING] Time column '{time_col}' not found. Available columns: {list(year_combined.columns)}")
        
        # Save combined file for this year
        output_file = output_dir / output_pattern.format(year=year)
        print(f"  Saving to: {output_file}")
        year_combined.to_csv(output_file, index=False)
        saved_files[year] = output_file
        print(f"  ✓ Saved {len(year_combined):,} rows to {output_file.name}")
    
    # Print summary
    print("\n" + "="*60)
    print("COMBINATION SUMMARY")
    print("="*60)
    print(f"Years processed: {len(saved_files)}/{len(years)}")
    print(f"Total source files loaded: {total_files_loaded}")
    print(f"Output files created: {len(saved_files)}")
    
    if saved_files:
        print(f"\nCreated files:")
        total_rows = 0
        for year in sorted(saved_files.keys()):
            filepath = saved_files[year]
            # Count rows in saved file
            df = pd.read_csv(filepath)
            total_rows += len(df)
            print(f"  [{year}] {filepath.name} - {len(df):,} rows")
        print(f"\nTotal rows across all years: {total_rows:,}")
    
    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for f in failed_files[:10]:  # Show first 10
            print(f"  - {f[0].name}: {f[1]}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
    
    print(f"\nOutput directory: {output_dir}")
    print("="*60)
    
    if not saved_files:
        raise ValueError("No data files could be loaded and saved!")
    
    return saved_files


def main():
    """Main entry point for data combination script."""
    parser = argparse.ArgumentParser(
        description="Combine data files for each year separately (2022-2025). Creates data_YYYY.csv files."
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default="G:/My Drive/26. MDLZ/WMS_history",
        help='Directory containing source data files (Outboundreports_*.csv)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="./dataset/data_cat",
        help='Output directory for combined year files (will create data_YYYY.csv files)'
    )
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        default=[2022, 2023, 2024, 2025],
        help='Years to combine (default: 2022 2023 2024 2025)'
    )
    parser.add_argument(
        '--file-prefix',
        type=str,
        default="Outboundreports",
        help='Prefix to filter source files (e.g., "Outboundreports" for Outboundreports_YYYYMMDD_YYYYMMDD.csv)'
    )
    parser.add_argument(
        '--output-pattern',
        type=str,
        default="data_{year}.csv",
        help='Pattern for output file names (must contain {year}, default: data_{year}.csv)'
    )
    parser.add_argument(
        '--time-col',
        type=str,
        default="ACTUALSHIPDATE",
        help='Name of time column for sorting'
    )
    parser.add_argument(
        '--no-sort',
        action='store_true',
        help='Do not sort by time column'
    )
    
    args = parser.parse_args()
    
    try:
        saved_files = combine_yearly_data(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            years=args.years,
            file_prefix=args.file_prefix,
            output_pattern=args.output_pattern,
            time_col=args.time_col,
            sort_by_time=not args.no_sort
        )
        print("\n✓ Data combination completed successfully!")
        print(f"✓ Created {len(saved_files)} year file(s) in {args.output_dir}")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

