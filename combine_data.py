"""Script to combine data files for each year separately (2022-2025)."""
import os
import pandas as pd
from pathlib import Path
import argparse
import re
import time
from typing import List, Optional, Dict
from collections import defaultdict
from src.utils.google_sheets import upload_to_google_sheets, GSPREAD_AVAILABLE


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
            
            # Convert UTC to Vietnam time (UTC+7): add 7 hours
            print(f"  Converting {time_col} from UTC to Vietnam time (UTC+7): adding 7 hours...")
            year_combined[time_col] = year_combined[time_col] + pd.Timedelta(hours=7)
            
            # Apply time-based date adjustment: if time >= 17:00 (5 PM), add 1 day
            print(f"  Processing {time_col}: if time >= 17:00, date will be adjusted to next day...")
            hour_mask = year_combined[time_col].dt.hour >= 17
            if hour_mask.any():
                count = hour_mask.sum()
                print(f"  Adjusting {count} row(s) with time >= 17:00 to next day")
                year_combined.loc[hour_mask, time_col] = year_combined.loc[hour_mask, time_col] + pd.Timedelta(days=1)
            
            # Convert datetime to date only (remove time component)
            print(f"  Converting {time_col} to date (removing time component)...")
            year_combined[time_col] = year_combined[time_col].dt.date
            
            # Add Week column with day names (1.Monday, 2.Tuesday, etc.)
            print(f"  Adding Week column with day names (1.Monday, 2.Tuesday, etc.)...")
            date_dt = pd.to_datetime(year_combined[time_col])
            year_combined['Week'] = (date_dt.dt.dayofweek + 1).astype(str) + '.' + date_dt.dt.day_name()
            
            # Add day of month column (1, 2, 3, ..., 31)
            print(f"  Adding day of month column...")
            year_combined['Day'] = date_dt.dt.day
            
            date_range = f"{year_combined[time_col].min()} to {year_combined[time_col].max()}"
            print(f"  Date range: {date_range}")
        else:
            print(f"  [WARNING] Time column '{time_col}' not found. Available columns: {list(year_combined.columns)}")
        
        # Filter: keep only rows where CATEGORY != "TEST" and CATEGORY != "OFFBOM"
        if 'CATEGORY' in year_combined.columns:
            rows_before = len(year_combined)
            year_combined = year_combined[(year_combined['CATEGORY'] != "TEST") & (year_combined['CATEGORY'] != "OFFBOM")]
            rows_after = len(year_combined)
            print(f"  Filtered by CATEGORY != 'TEST' and CATEGORY != 'OFFBOM': {rows_before:,} -> {rows_after:,} rows")
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
                
                # Add Week column back after grouping (derived from date)
                if time_col in grouped.columns:
                    date_dt = pd.to_datetime(grouped[time_col])
                    grouped['Week'] = (date_dt.dt.dayofweek + 1).astype(str) + '.' + date_dt.dt.day_name()
                    grouped['Day'] = date_dt.dt.day
                
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


def create_pivot_summary(saved_files: Dict[int, Path], time_col: str = "ACTUALSHIPDATE", value_col: str = "Total CBM") -> pd.DataFrame:
    """
    Create pivot-style monthly summary with months as rows and years as columns.
    
    Args:
        saved_files: Dictionary mapping year to file path.
        time_col: Name of time column.
        value_col: Column to aggregate (default: "Total QTY", can be "Total CBM").
    
    Returns:
        DataFrame with columns: Month, 2023, 2024, 2025, ... (years as columns)
    """
    print(f"\n[Pivot Summary] Creating pivot summary from {len(saved_files)} year file(s)...")
    all_data = []
    
    for year in sorted(saved_files.keys()):
        filepath = saved_files[year]
        df = pd.read_csv(filepath)
        all_data.append(df)
    
    if not all_data:
        print("[Pivot Summary] No data to summarize")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"[Pivot Summary] Total rows: {len(combined_df):,}")
    
    # Convert time column to datetime if needed
    if time_col in combined_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(combined_df[time_col]):
            combined_df[time_col] = pd.to_datetime(combined_df[time_col])
        
        # Extract Year and Month
        combined_df['Year'] = combined_df[time_col].dt.year
        combined_df['Month'] = combined_df[time_col].dt.month
    else:
        print(f"[Pivot Summary] WARNING: Time column '{time_col}' not found. Using year from filename.")
        if 'Year' not in combined_df.columns:
            print("[Pivot Summary] ERROR: Cannot create pivot summary without time column")
            return pd.DataFrame()
    
    # Check if value column exists
    if value_col not in combined_df.columns:
        print(f"[Pivot Summary] WARNING: Column '{value_col}' not found. Available columns: {list(combined_df.columns)}")
        # Try alternative
        if value_col == "Total QTY" and "Total CBM" in combined_df.columns:
            value_col = "Total CBM"
            print(f"[Pivot Summary] Using '{value_col}' instead")
        else:
            return pd.DataFrame()
    
    # Group by Year and Month (aggregating across all categories)
    print(f"[Pivot Summary] Grouping by Year and Month, aggregating '{value_col}'")
    monthly_agg = combined_df.groupby(['Year', 'Month'], as_index=False)[value_col].sum()
    
    # Create pivot table: Month as rows, Year as columns
    print(f"[Pivot Summary] Creating pivot table...")
    pivot_df = monthly_agg.pivot_table(
        index='Month',
        columns='Year',
        values=value_col,
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    
    # Ensure all months 1-12 are present
    all_months = pd.DataFrame({'Month': range(1, 13)})
    pivot_df = all_months.merge(pivot_df, on='Month', how='left').fillna(0)
    
    # Rename columns: keep 'Month' as is, convert year columns to strings
    pivot_df.columns = [str(col) if col != 'Month' else col for col in pivot_df.columns]
    
    # Sort by Month
    pivot_df = pivot_df.sort_values('Month').reset_index(drop=True)
    
    # Ensure integer Month values
    pivot_df['Month'] = pivot_df['Month'].astype(int)
    
    print(f"[Pivot Summary] Created pivot summary with {len(pivot_df)} rows (months 1-12)")
    return pivot_df


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
        default=[2023, 2024, 2025,2026],
        help='Years to combine (default: 2022 2023 2024 2025 2026)'
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
            
            # Create and upload pivot summary to Summary sheet
            pivot_summary = create_pivot_summary(saved_files, time_col=args.time_col, value_col="Total CBM")
            if not pivot_summary.empty:
                upload_to_google_sheets(
                    saved_files={},  # Not used for summary
                    spreadsheet_id=args.spreadsheet_id,
                    sheet_name="Summary",
                    credentials_path=args.credentials,
                    data_df=pivot_summary  # Pass the pivot summary DataFrame directly
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

