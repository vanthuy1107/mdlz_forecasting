"""Google Sheets upload utilities."""
import time
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False
    gspread = None
    Credentials = None


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
    if not GSPREAD_AVAILABLE:
        raise ImportError("gspread is not available. Install with: pip install gspread google-auth")
    
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
    data_df: Optional[pd.DataFrame] = None,
    update_mode: bool = False,
    merge_keys: Optional[list] = None
) -> bool:
    """
    Upload combined data to Google Sheets.
    
    Args:
        saved_files: Dictionary mapping year to file path (used if data_df is None).
        spreadsheet_id: Google Sheets spreadsheet ID.
        sheet_name: Name of the sheet to upload to.
        credentials_path: Path to service account credentials JSON file.
        data_df: Optional DataFrame to upload directly (if provided, saved_files is ignored).
        update_mode: If True, merge with existing data instead of overwriting. Requires merge_keys.
        merge_keys: List of column names to use as merge keys when update_mode=True.
                    Defaults to ['Year', 'CATEGORY', 'Month'] for ResultByMonth sheet.
    
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
            new_df = data_df.copy()
            print(f"[Google Sheets] Using provided DataFrame with {len(new_df):,} rows")
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
            
            new_df = pd.concat(all_data, ignore_index=True)
            print(f"[Google Sheets] Total rows to upload: {len(new_df):,}")
        
        # Handle update mode: merge with existing data
        if update_mode:
            print(f"[Google Sheets] Update mode enabled for sheet '{sheet_name}'")
            
            # Determine merge keys
            if merge_keys is None:
                # Default merge keys for known sheets
                if sheet_name == "ResultByMonth":
                    merge_keys = ['Year', 'CATEGORY', 'Month']
                elif sheet_name == "SummaryByMonth":
                    # Summary-by-month rows: one per (CATEGORY, Month)
                    merge_keys = ['CATEGORY', 'Month']
                elif sheet_name == "Result":
                    # Daily results: one row per (date, CATEGORY)
                    merge_keys = ['date', 'CATEGORY']
                else:
                    print(f"[Google Sheets] Warning: update_mode=True but no merge_keys provided and no default for sheet '{sheet_name}'")
                    print(f"[Google Sheets] Falling back to overwrite mode")
                    update_mode = False
            
            if update_mode and merge_keys:
                # Read existing data from sheet
                try:
                    existing_data = worksheet.get_all_values()
                    if existing_data and len(existing_data) > 1:
                        # Convert to DataFrame
                        existing_df = pd.DataFrame(existing_data[1:], columns=existing_data[0])
                        print(f"[Google Sheets] Found {len(existing_df):,} existing rows in sheet")

                        # For ResultByMonth sheet, drop any legacy/helper rows where Year is blank/null
                        if sheet_name == "ResultByMonth" and "Year" in existing_df.columns:
                            before_drop = len(existing_df)
                            existing_df = existing_df[
                                existing_df["Year"].notna() & (existing_df["Year"] != "")
                            ]
                            after_drop = len(existing_df)
                            if after_drop != before_drop:
                                print(f"[Google Sheets] Removed {before_drop - after_drop} rows with empty Year from existing sheet data")
                        
                        # For SummaryByMonth we want simple APPEND behavior (no merge on keys),
                        # and we must preserve the existing column format.
                        if sheet_name == "SummaryByMonth":
                            # Align columns between existing and new data
                            for col in existing_df.columns:
                                if col not in new_df.columns:
                                    new_df[col] = np.nan
                            for col in new_df.columns:
                                if col not in existing_df.columns:
                                    existing_df[col] = np.nan
                            # Reorder new_df columns to match existing sheet
                            new_df = new_df[existing_df.columns]
                            # Append new rows
                            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                            print(f"[Google Sheets] Appended {len(new_df):,} rows to 'SummaryByMonth' (total {len(combined_df):,})")
                            # Disable update_mode-based merge for this case
                            update_mode = False
                        else:
                            # Ensure merge keys exist in both DataFrames
                            missing_keys_new = [k for k in merge_keys if k not in new_df.columns]
                            missing_keys_existing = [k for k in merge_keys if k not in existing_df.columns]
                            
                            if missing_keys_new:
                                print(f"[Google Sheets] Warning: Merge keys {missing_keys_new} not found in new data. Falling back to overwrite mode.")
                                update_mode = False
                            elif missing_keys_existing:
                                print(f"[Google Sheets] Warning: Merge keys {missing_keys_existing} not found in existing data. Appending new data only.")
                                combined_df = new_df
                            else:
                                # Convert merge key columns to same types for proper merging
                                for key in merge_keys:
                                    if key in existing_df.columns and key in new_df.columns:
                                        # Try to convert to numeric if possible
                                        try:
                                            existing_df[key] = pd.to_numeric(existing_df[key], errors='ignore')
                                            new_df[key] = pd.to_numeric(new_df[key], errors='ignore')
                                        except:
                                            pass
                                
                                # Set merge keys as index for easier merging
                                existing_df_indexed = existing_df.set_index(merge_keys)
                                new_df_indexed = new_df.set_index(merge_keys)
                                
                                # Update existing rows with new data, append new rows
                                # combine_first will use new_df values where available, existing_df otherwise
                                combined_df_indexed = new_df_indexed.combine_first(existing_df_indexed)
                                
                                # Reset index to get merge keys back as columns
                                combined_df = combined_df_indexed.reset_index()
                                
                                # Ensure column order matches new_df (merge keys first, then other columns)
                                col_order = merge_keys + [col for col in new_df.columns if col not in merge_keys]
                                # Add any columns that only exist in existing_df
                                for col in existing_df.columns:
                                    if col not in col_order:
                                        col_order.append(col)
                                
                                combined_df = combined_df[[col for col in col_order if col in combined_df.columns]]
                                
                                print(f"[Google Sheets] Merged data: {len(existing_df):,} existing + {len(new_df):,} new = {len(combined_df):,} total rows")
                    else:
                        # Sheet is empty or only has headers
                        print(f"[Google Sheets] Sheet is empty, using new data only")
                        combined_df = new_df
                except Exception as e:
                    print(f"[Google Sheets] Error reading existing data: {e}. Falling back to overwrite mode.")
                    combined_df = new_df
                    update_mode = False
            else:
                combined_df = new_df
        else:
            combined_df = new_df
        
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
        if not update_mode:
            print(f"[Google Sheets] Clearing existing data in sheet '{sheet_name}'...")
            retry_with_backoff(
                lambda: worksheet.clear(),
                max_retries=5,
                initial_delay=2.0,
                max_delay=60.0
            )
        else:
            print(f"[Google Sheets] Updating sheet '{sheet_name}' with merged data...")
            # Clear only if we're doing a full update (not incremental)
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
        batch_size = 1000  # Conservative batch size
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
