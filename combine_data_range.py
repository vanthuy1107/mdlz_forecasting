"""Script to combine data files by date range without grouping."""
import os
import pandas as pd
from pathlib import Path
import argparse
import re
from typing import List, Optional
from datetime import datetime


def extract_dates_from_filename(filename: str) -> Optional[tuple]:
    """
    Extract date range from filename patterns like:
    - Outboundreports_YYYYMMDD_YYYYMMDD.csv
    
    Args:
        filename: Name of the file.
    
    Returns:
        Tuple of (start_date, end_date) as datetime objects, or None if not found.
    """
    # Pattern: Outboundreports_YYYYMMDD_YYYYMMDD.csv
    match = re.search(r'Outboundreports_(\d{8})_(\d{8})', filename)
    if match:
        try:
            start_date = datetime.strptime(match.group(1), '%Y%m%d')
            end_date = datetime.strptime(match.group(2), '%Y%m%d')
            return (start_date, end_date)
        except ValueError:
            return None
    
    return None


def find_files_for_date_range(
    data_dir: Path,
    start_date: datetime,
    end_date: datetime,
    file_prefix: str = "Outboundreports"
) -> List[Path]:
    """
    Find CSV files that might contain data within the specified date range.
    
    Args:
        data_dir: Directory to search.
        start_date: Start date for filtering.
        end_date: End date for filtering.
        file_prefix: Prefix to filter files (e.g., "Outboundreports").
    
    Returns:
        List of file paths that might contain data within the date range.
    """
    if not data_dir.exists():
        return []
    
    # Find all CSV files
    csv_files = list(data_dir.glob("*.csv"))
    
    print(f"Found {len(csv_files)} CSV files in directory")
    
    matching_files = []
    year_str = str(start_date.year)
    
    for filepath in csv_files:
        filename = filepath.name
        
        # Filter by prefix if specified
        if file_prefix and not filename.startswith(file_prefix):
            continue
        
        # Check if file date range overlaps with target date range
        date_range = extract_dates_from_filename(filename)
        if date_range:
            file_start_date, file_end_date = date_range
            # Check if date range overlaps with target date range
            if not (file_end_date < start_date or file_start_date > end_date):
                matching_files.append(filepath)
        else:
            # If we can't extract dates, include files from target year (will filter by data later)
            if year_str in filename:
                matching_files.append(filepath)
    
    # Sort files
    matching_files.sort()
    
    return matching_files


def combine_data_by_date_range(
    data_dir: str,
    output_file: str,
    start_date: datetime,
    end_date: datetime,
    file_prefix: str = "Outboundreports",
    time_col: str = "ACTUALSHIPDATE",
    sort_by_time: bool = True
) -> Path:
    """
    Combine data files within a date range without grouping.
    
    Args:
        data_dir: Directory containing source data files.
        output_file: Path to output file.
        start_date: Start date for filtering data.
        end_date: End date for filtering data.
        file_prefix: Prefix to filter files (e.g., "Outboundreports").
        time_col: Name of time column for filtering and sorting.
        sort_by_time: Whether to sort combined data by time column.
    
    Returns:
        Path to output file.
    """
    data_dir = Path(data_dir)
    output_path = Path(output_file)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to date objects for comparison
    start_date_obj = start_date.date()
    end_date_obj = end_date.date()
    
    date_range_str = f"{start_date_obj} to {end_date_obj}"
    
    print("="*60)
    print(f"COMBINING DATA FOR DATE RANGE: {date_range_str} (NO GROUPING)")
    print("="*60)
    print(f"Source directory: {data_dir}")
    print(f"Output file: {output_path}")
    print(f"File prefix: {file_prefix}")
    print(f"Date range: {date_range_str}\n")
    
    # Find files that might contain data within the date range
    candidate_files = find_files_for_date_range(data_dir, start_date, end_date, file_prefix)
    
    if not candidate_files:
        raise FileNotFoundError(
            f"No data files found for date range {date_range_str} in directory: {data_dir}\n"
            f"Expected files matching pattern: {file_prefix}_YYYYMMDD_YYYYMMDD.csv"
        )
    
    print(f"Found {len(candidate_files)} candidate file(s) for date range {date_range_str}:")
    for f in candidate_files:
        print(f"  - {f.name}")
    print()
    
    all_data = []
    total_files_loaded = 0
    failed_files = []
    total_rows_before_filter = 0
    total_rows_after_filter = 0
    
    # Load and filter files
    for filepath in candidate_files:
        try:
            print(f"Loading: {filepath.name}...", end=" ")
            data = pd.read_csv(filepath)
            total_rows_before_filter += len(data)
            print(f"✓ ({len(data)} rows)")
            
            # Filter for data within date range
            if time_col in data.columns:
                # Convert to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
                    data[time_col] = pd.to_datetime(data[time_col])
                
                # Convert to date (remove time component)
                data[time_col] = pd.to_datetime(data[time_col]).dt.date
                
                # Filter for date range
                mask = (data[time_col] >= start_date_obj) & (data[time_col] <= end_date_obj)
                data_filtered = data[mask].copy()
                
                if len(data_filtered) > 0:
                    all_data.append(data_filtered)
                    total_rows_after_filter += len(data_filtered)
                    print(f"  → {len(data_filtered)} rows in date range")
                else:
                    print(f"  → 0 rows in date range (skipped)")
            else:
                # If time column not found, include all data (user should verify)
                print(f"  [WARNING] Time column '{time_col}' not found. Including all rows.")
                print(f"  Available columns: {list(data.columns)}")
                all_data.append(data)
                total_rows_after_filter += len(data)
            
            total_files_loaded += 1
            
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed_files.append((filepath, str(e)))
    
    if not all_data:
        raise ValueError(
            f"No data found in date range {date_range_str} in any files!\n"
            f"Total rows loaded before filtering: {total_rows_before_filter:,}"
        )
    
    # Combine all data (no grouping)
    print(f"\nCombining {len(all_data)} file(s) without grouping...")
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Combined: {len(combined_data):,} total rows")
    
    # Show date range if time column exists
    if time_col in combined_data.columns:
        actual_date_range = f"{combined_data[time_col].min()} to {combined_data[time_col].max()}"
        print(f"Actual date range: {actual_date_range}")
    
    # Sort by time if requested
    if sort_by_time and time_col in combined_data.columns:
        print(f"Sorting by {time_col}...")
        combined_data = combined_data.sort_values(time_col).reset_index(drop=True)
    
    # Save combined file
    print(f"\nSaving to: {output_path}")
    combined_data.to_csv(output_path, index=False)
    print(f"✓ Saved {len(combined_data):,} rows to {output_path.name}")
    
    # Print summary
    print("\n" + "="*60)
    print("COMBINATION SUMMARY")
    print("="*60)
    print(f"Date range: {date_range_str}")
    print(f"Files processed: {total_files_loaded}/{len(candidate_files)}")
    print(f"Total rows before filtering: {total_rows_before_filter:,}")
    print(f"Total rows after filtering: {total_rows_after_filter:,}")
    print(f"Final combined rows: {len(combined_data):,}")
    
    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for f in failed_files:
            print(f"  - {f[0].name}: {f[1]}")
    
    print(f"\nOutput file: {output_path}")
    print("="*60)
    
    return output_path


def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")


def main():
    """Main entry point for data combination script."""
    
    # ============================================================================
    # CONFIGURATION: Adjust these values directly in code
    # ============================================================================
    DEFAULT_START_DATE = datetime(2025, 8, 1)  # Change this date as needed (year, month, day)
    DEFAULT_END_DATE = datetime(2025, 8, 30)    # Change this date as needed (year, month, day)
    DEFAULT_DATA_DIR = "G:/My Drive/26. MDLZ/WMS_history"
    DEFAULT_OUTPUT_FILE = "./dataset/data_combined.csv"
    DEFAULT_FILE_PREFIX = "Outboundreports"
    DEFAULT_TIME_COL = "ACTUALSHIPDATE"
    DEFAULT_SORT_BY_TIME = True
    # ============================================================================
    
    parser = argparse.ArgumentParser(
        description="Combine data files by date range without grouping."
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=DEFAULT_DATA_DIR,
        help='Directory containing source data files (Outboundreports_*.csv)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help='Output file path for combined data'
    )
    parser.add_argument(
        '--start-date',
        type=parse_date,
        default=None,
        help='Start date for filtering data (format: YYYY-MM-DD). If not provided, uses default from code.'
    )
    parser.add_argument(
        '--end-date',
        type=parse_date,
        default=None,
        help='End date for filtering data (format: YYYY-MM-DD). If not provided, uses default from code.'
    )
    parser.add_argument(
        '--file-prefix',
        type=str,
        default=DEFAULT_FILE_PREFIX,
        help='Prefix to filter source files (e.g., "Outboundreports" for Outboundreports_YYYYMMDD_YYYYMMDD.csv)'
    )
    parser.add_argument(
        '--time-col',
        type=str,
        default=DEFAULT_TIME_COL,
        help='Name of time column for filtering and sorting'
    )
    parser.add_argument(
        '--no-sort',
        action='store_true',
        help='Do not sort by time column'
    )
    
    args = parser.parse_args()
    
    # Use command-line arguments if provided, otherwise use defaults from code
    start_date = args.start_date if args.start_date is not None else DEFAULT_START_DATE
    end_date = args.end_date if args.end_date is not None else DEFAULT_END_DATE
    
    # Validate date range
    if start_date > end_date:
        print(f"✗ Error: Start date ({start_date.date()}) must be before or equal to end date ({end_date.date()})")
        return 1
    
    try:
        output_path = combine_data_by_date_range(
            data_dir=args.data_dir,
            output_file=args.output_file,
            start_date=start_date,
            end_date=end_date,
            file_prefix=args.file_prefix,
            time_col=args.time_col,
            sort_by_time=not args.no_sort
        )
        print("\n✓ Data combination completed successfully!")
        print(f"✓ Created file: {output_path}")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

