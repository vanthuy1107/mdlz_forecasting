"""Script to combine data files for each year separately (2022-2025)."""
import os
import pandas as pd
from pathlib import Path
import argparse
import re
import time
from typing import List, Optional, Dict
from collections import defaultdict
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False


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
        
        # Ensure time column is datetime first (for date extraction)
        if time_col in year_combined.columns:
            if not pd.api.types.is_datetime64_any_dtype(year_combined[time_col]):
                print(f"  Converting {time_col} to datetime...")
                year_combined[time_col] = pd.to_datetime(year_combined[time_col])
            
            # Convert datetime to date only (remove time component)
            print(f"  Converting {time_col} to date (removing time component)...")
            year_combined[time_col] = year_combined[time_col].dt.date
            
            date_range = f"{year_combined[time_col].min()} to {year_combined[time_col].max()}"
            print(f"  Date range: {date_range}")
        else:
            print(f"  [WARNING] Time column '{time_col}' not found. Available columns: {list(year_combined.columns)}")
        
        # Filter: keep only rows where CATEGORY == "TEST"
        if 'CATEGORY' in year_combined.columns:
            rows_before = len(year_combined)
            year_combined = year_combined[year_combined['CATEGORY'] != "TEST"]
            rows_after = len(year_combined)
            print(f"  Filtered by CATEGORY != 'TEST': {rows_before:,} -> {rows_after:,} rows")
        else:
            print(f"  [WARNING] CATEGORY column not found. Cannot filter by CATEGORY.")
        
        # Group by columns: ACTUALSHIPDATE, TYPENAME, WHSEID, CATEGORY
        groupby_cols = [time_col, 'TYPENAME', 'WHSEID', 'CATEGORY']
        available_cols = [col for col in groupby_cols if col in year_combined.columns]
        
        if len(available_cols) < len(groupby_cols):
            missing = set(groupby_cols) - set(available_cols)
            print(f"  [WARNING] Some groupby columns not found: {missing}")
            print(f"  Available columns: {list(year_combined.columns)}")
        
        if available_cols:
            # Identify source columns to aggregate: QTY -> Total QTY, CUBE_OUT -> Total CBM
            agg_dict = {}
            if 'QTY' in year_combined.columns:
                agg_dict['Total QTY'] = 'QTY'
            if 'CUBE_OUT' in year_combined.columns:
                agg_dict['Total CBM'] = 'CUBE_OUT'
                # First priority: If SKU = "4305254", then CUBE_OUT = QTY/16*0.027
                sku_mask = None
                if 'SKU' in year_combined.columns and 'QTY' in year_combined.columns:
                    sku_mask = year_combined['SKU'] == "4305254"
                    if sku_mask.any():
                        count = sku_mask.sum()
                        print(f"  Processing CUBE_OUT: {count} row(s) with SKU='4305254' will use CUBE_OUT = QTY/16*0.027")
                        year_combined.loc[sku_mask, 'CUBE_OUT'] = year_combined.loc[sku_mask, 'QTY'] / 16 * 0.027
                    
                    # If SKU = "ZW840244", then QTY = QTY/120*0.019
                    sku_mask_zw = year_combined['SKU'] == "ZW840244"
                    if sku_mask_zw.any():
                        count = sku_mask_zw.sum()
                        print(f"  Processing QTY: {count} row(s) with SKU='ZW840244' will use QTY = QTY/120*0.019")
                        year_combined.loc[sku_mask_zw, 'QTY'] = year_combined.loc[sku_mask_zw, 'QTY'] / 120 * 0.019
                
                # Process CUBE_OUT: if >= 2.7, divide by 1000, else keep as is (exclude SKU="4305254" rows)
                print(f"  Processing CUBE_OUT: values >= 2.7 will be divided by 1000")
                if sku_mask is not None:
                    # Apply rule only to rows that are NOT SKU="4305254"
                    other_mask = ~sku_mask
                    year_combined.loc[other_mask, 'CUBE_OUT'] = year_combined.loc[other_mask, 'CUBE_OUT'].apply(
                        lambda x: x / 1000 if x >= 2.7 else x
                    )
                else:
                    # No SKU column or no special SKU rows, apply to all
                    year_combined['CUBE_OUT'] = year_combined['CUBE_OUT'].apply(
                        lambda x: x / 1000 if x >= 2.7 else x
                    )
            
            if agg_dict:
                print(f"  Grouping by: {available_cols}")
                print(f"  Aggregating (sum): {list(agg_dict.values())} -> {list(agg_dict.keys())}")
                print(f"  Rows before grouping: {len(year_combined):,}")
                
                # Group and aggregate - sum the source columns
                source_cols = list(agg_dict.values())
                grouped = year_combined.groupby(available_cols, as_index=False)[source_cols].sum()
                
                # Rename aggregated columns to output names
                rename_dict = {old_name: new_name for new_name, old_name in agg_dict.items()}
                grouped = grouped.rename(columns=rename_dict)
                
                year_combined = grouped
                
                print(f"  Rows after grouping: {len(year_combined):,}")
                
                # Sort by time if requested
                if sort_by_time and time_col in available_cols:
                    print(f"  Sorting by {time_col}...")
                    year_combined = year_combined.sort_values(time_col).reset_index(drop=True)
            else:
                print(f"  [WARNING] No aggregation columns found (QTY, CUBE_OUT). Available columns: {list(year_combined.columns)}")
        
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


def create_monthly_summary(saved_files: Dict[int, Path], time_col: str = "ACTUALSHIPDATE") -> pd.DataFrame:
    """
    Create monthly summary by aggregating data by Year, CATEGORY, and Month.
    
    Args:
        saved_files: Dictionary mapping year to file path.
        time_col: Name of time column.
    
    Returns:
        DataFrame with columns: Year, CATEGORY, Total QTY, Total CBM, Month
    """
    print(f"\n[Monthly Summary] Creating monthly summary from {len(saved_files)} year file(s)...")
    all_data = []
    
    for year in sorted(saved_files.keys()):
        filepath = saved_files[year]
        df = pd.read_csv(filepath)
        all_data.append(df)
    
    if not all_data:
        print("[Monthly Summary] No data to summarize")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"[Monthly Summary] Total rows: {len(combined_df):,}")
    
    # Convert time column to datetime if needed
    if time_col in combined_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(combined_df[time_col]):
            combined_df[time_col] = pd.to_datetime(combined_df[time_col])
        
        # Extract Year and Month
        combined_df['Year'] = combined_df[time_col].dt.year
        combined_df['Month'] = combined_df[time_col].dt.month
    else:
        print(f"[Monthly Summary] WARNING: Time column '{time_col}' not found. Using year from filename.")
        # Fallback: try to extract year from data if available
        if 'Year' not in combined_df.columns:
            print("[Monthly Summary] ERROR: Cannot create monthly summary without time column")
            return pd.DataFrame()
    
    # Check required columns
    required_cols = ['Year', 'CATEGORY', 'Month']
    agg_cols = {}
    
    if 'Total QTY' in combined_df.columns:
        agg_cols['Total QTY'] = 'sum'
    if 'Total CBM' in combined_df.columns:
        agg_cols['Total CBM'] = 'sum'
    
    if not agg_cols:
        print("[Monthly Summary] WARNING: No aggregation columns found (Total QTY, Total CBM)")
        return pd.DataFrame()
    
    # Group by Year, CATEGORY, and Month
    groupby_cols = [col for col in required_cols if col in combined_df.columns]
    
    if len(groupby_cols) < len(required_cols):
        missing = set(required_cols) - set(groupby_cols)
        print(f"[Monthly Summary] ERROR: Missing required columns: {missing}")
        return pd.DataFrame()
    
    print(f"[Monthly Summary] Grouping by: {groupby_cols}")
    print(f"[Monthly Summary] Aggregating: {list(agg_cols.keys())}")
    
    # Group and aggregate
    summary = combined_df.groupby(groupby_cols, as_index=False)[list(agg_cols.keys())].sum()
    
    # Reorder columns: Year, CATEGORY, Total QTY, Total CBM, Month
    column_order = ['Year', 'CATEGORY'] + list(agg_cols.keys()) + ['Month']
    summary = summary[[col for col in column_order if col in summary.columns]]
    
    # Sort by Year, Month, CATEGORY
    summary = summary.sort_values(['Year', 'Month', 'CATEGORY']).reset_index(drop=True)
    
    print(f"[Monthly Summary] Created summary with {len(summary):,} rows")
    return summary


def retry_with_backoff(func, max_retries: int = 5, initial_delay: float = 1.0, max_delay: float = 60.0, backoff_factor: float = 2.0):
    """
    Retry a function with exponential backoff, specifically handling Google Sheets API quota errors.
    
    Args:
        func: Function to retry (should be a callable that takes no arguments).
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay in seconds between retries.
        backoff_factor: Factor to multiply delay by after each retry.
    
    Returns:
        Result of the function call.
    
    Raises:
        Exception: If all retries are exhausted.
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except gspread.exceptions.APIError as e:
            last_exception = e
            # Check if it's a quota/rate limit error (429)
            # Check both status_code and error message for robustness
            is_quota_error = False
            if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                is_quota_error = e.response.status_code == 429
            elif '429' in str(e) or 'quota' in str(e).lower() or 'rate limit' in str(e).lower():
                is_quota_error = True
            
            if is_quota_error:
                if attempt < max_retries:
                    wait_time = min(delay, max_delay)
                    print(f"  [Retry {attempt + 1}/{max_retries}] Quota exceeded. Waiting {wait_time:.1f} seconds before retry...")
                    time.sleep(wait_time)
                    delay *= backoff_factor
                else:
                    print(f"  [ERROR] Max retries ({max_retries}) exceeded for quota error")
                    raise
            else:
                # Not a quota error, re-raise immediately
                raise
        except Exception as e:
            # For other exceptions, re-raise immediately
            raise
    
    # Should never reach here, but just in case
    if last_exception:
        raise last_exception


def upload_to_google_sheets(
    saved_files: Dict[int, Path],
    spreadsheet_id: str,
    sheet_name: str = "History",
    credentials_path: str = "key.json",
    data_df: Optional[pd.DataFrame] = None
) -> bool:
    """
    Upload combined data to Google Sheets.
    
    Args:
        saved_files: Dictionary mapping year to file path (used if data_df is None).
        spreadsheet_id: Google Sheets spreadsheet ID.
        sheet_name: Name of the sheet to upload to.
        credentials_path: Path to service account credentials JSON file.
        data_df: Optional DataFrame to upload directly (if provided, saved_files is ignored).
    
    Returns:
        True if successful, False otherwise.
    """
    if not GSPREAD_AVAILABLE:
        print("\n[WARNING] gspread not available. Install with: pip install gspread google-auth")
        return False
    
    try:
        # Authenticate with service account
        credentials_path = Path(credentials_path)
        if not credentials_path.exists():
            print(f"\n[ERROR] Credentials file not found: {credentials_path}")
            return False
        
        print(f"\n[Google Sheets] Authenticating with {credentials_path}...")
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        creds = Credentials.from_service_account_file(str(credentials_path), scopes=scopes)
        client = gspread.authorize(creds)
        
        # Open the spreadsheet
        print(f"[Google Sheets] Opening spreadsheet: {spreadsheet_id}...")
        spreadsheet = client.open_by_key(spreadsheet_id)
        
        # Get or create the sheet
        try:
            worksheet = spreadsheet.worksheet(sheet_name)
            print(f"[Google Sheets] Found existing sheet: {sheet_name}")
        except gspread.exceptions.WorksheetNotFound:
            print(f"[Google Sheets] Creating new sheet: {sheet_name}...")
            worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=20)
        
        # Get data to upload
        if data_df is not None:
            # Use provided DataFrame
            combined_df = data_df.copy()
            print(f"[Google Sheets] Using provided DataFrame with {len(combined_df):,} rows")
        else:
            # Combine all data from all years
            print(f"[Google Sheets] Combining data from {len(saved_files)} year file(s)...")
            all_data = []
            for year in sorted(saved_files.keys()):
                filepath = saved_files[year]
                df = pd.read_csv(filepath)
                all_data.append(df)
            
            if not all_data:
                print("[Google Sheets] No data to upload")
                return False
            
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"[Google Sheets] Total rows to upload: {len(combined_df):,}")
        
        # Convert DataFrame to list of lists
        # First row is headers
        values = [combined_df.columns.tolist()]
        # Add data rows
        for _, row in combined_df.iterrows():
            # Convert row to list, handling dates and NaN values
            row_values = []
            for val in row:
                if pd.isna(val):
                    row_values.append("")
                else:
                    # Convert dates/datetimes to strings
                    from datetime import date, datetime
                    if isinstance(val, (date, datetime, pd.Timestamp)):
                        row_values.append(str(val))
                    else:
                        row_values.append(val)
            values.append(row_values)
        
        # Clear existing data and upload new data with retry logic
        print(f"[Google Sheets] Clearing existing data in sheet '{sheet_name}'...")
        retry_with_backoff(
            lambda: worksheet.clear(),
            max_retries=5,
            initial_delay=2.0,
            max_delay=60.0
        )
        
        # Add a small delay between clear and update to avoid hitting rate limits
        time.sleep(1.0)
        
        print(f"[Google Sheets] Uploading {len(values):,} rows (including header)...")
        # Batch updates in chunks to avoid quota issues
        # Google Sheets API limit is typically 10000 cells per request
        # Using smaller batches to be safe and add delays between requests
        batch_size = 500  # Conservative batch size
        if len(values) > batch_size:
            # Upload in batches
            for i in range(0, len(values), batch_size):
                batch = values[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = ((len(values) - 1) // batch_size) + 1
                start_row = i + 1  # Google Sheets is 1-indexed
                
                print(f"  Uploading batch {batch_num}/{total_batches} (starting at row {start_row}, {len(batch)} rows)...")
                
                # Create a closure-safe function for retry with default arguments
                def upload_batch(b=batch, sr=start_row):
                    # Use A1 notation for the starting cell
                    start_cell = f'A{sr}'
                    return worksheet.update(start_cell, b, value_input_option='USER_ENTERED')
                
                retry_with_backoff(
                    upload_batch,
                    max_retries=5,
                    initial_delay=2.0,
                    max_delay=60.0
                )
                # Add delay between batches to avoid rate limiting
                if i + batch_size < len(values):
                    time.sleep(2.0)  # Increased delay between batches
        else:
            # Small dataset, upload all at once
            retry_with_backoff(
                lambda: worksheet.update(values, value_input_option='USER_ENTERED'),
                max_retries=5,
                initial_delay=2.0,
                max_delay=60.0
            )
        
        print(f"[Google Sheets] ✓ Successfully uploaded {len(combined_df):,} rows to sheet '{sheet_name}'")
        return True
        
    except Exception as e:
        print(f"\n[Google Sheets] ✗ Error uploading to Google Sheets: {e}")
        import traceback
        traceback.print_exc()
        return False


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
        default=[2023, 2024, 2025],
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
    parser.add_argument(
        '--no-upload',
        action='store_true',
        help='Disable uploading to Google Sheets (by default, uploads to History and SummaryByMonth sheets)'
    )
    parser.add_argument(
        '--spreadsheet-id',
        type=str,
        default="1I8JEqZbWGZNOsebzOBfeHKJ7Z7jA1zcfX8JhjqGSowE",
        help='Google Sheets spreadsheet ID'
    )
    parser.add_argument(
        '--sheet-name',
        type=str,
        default="History",
        help='Name of the sheet to upload to (default: History)'
    )
    parser.add_argument(
        '--credentials',
        type=str,
        default="key.json",
        help='Path to service account credentials JSON file (default: key.json)'
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
        
        # Upload to Google Sheets (default behavior)
        if not args.no_upload:
            # Upload History sheet
            print("\n[Upload] Starting Google Sheets upload...")
            upload_to_google_sheets(
                saved_files=saved_files,
                spreadsheet_id=args.spreadsheet_id,
                sheet_name=args.sheet_name,
                credentials_path=args.credentials
            )
            
            # Create and upload monthly summary to SummaryByMonth sheet
            monthly_summary = create_monthly_summary(saved_files, time_col=args.time_col)
            if not monthly_summary.empty:
                upload_to_google_sheets(
                    saved_files={},  # Not used for summary
                    spreadsheet_id=args.spreadsheet_id,
                    sheet_name="SummaryByMonth",
                    credentials_path=args.credentials,
                    data_df=monthly_summary  # Pass the summary DataFrame directly
                )
            print("\n✓ Google Sheets upload completed!")
        else:
            print("\n[Skip] Google Sheets upload skipped (--no-upload flag set)")
        
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

