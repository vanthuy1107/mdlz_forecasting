"""Google Sheets upload utilities."""
import time
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import pandas as pd
import numpy as np

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
            worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=10000, cols=30)
        
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
                    # Summary-by-month rows: one per (Year, CATEGORY, BRAND, Month)
                    merge_keys = ['Year', 'CATEGORY', 'BRAND', 'Month']
                elif sheet_name == "Result":
                    # Daily results: one row per (date, CATEGORY)
                    merge_keys = ['date', 'CATEGORY']
                elif sheet_name == "History_prediction":
                    # History_prediction: simple append mode (no merge keys)
                    merge_keys = None  # Will trigger append-only behavior
                else:
                    print(f"[Google Sheets] Warning: update_mode=True but no merge_keys provided and no default for sheet '{sheet_name}'")
                    print(f"[Google Sheets] Falling back to overwrite mode")
                    update_mode = False
            
            if update_mode:
                # Read existing data from sheet
                try:
                    existing_data = worksheet.get_all_values()
                    
                    # For History_prediction and SummaryByMonth, always append (even if sheet is empty)
                    if sheet_name == "History_prediction" or sheet_name == "SummaryByMonth":
                        if existing_data and len(existing_data) > 1:
                            # Convert to DataFrame
                            existing_df = pd.DataFrame(existing_data[1:], columns=existing_data[0])
                            print(f"[Google Sheets] Found {len(existing_df):,} existing rows in sheet")
                            
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
                            sheet_display_name = "History_prediction" if sheet_name == "History_prediction" else "SummaryByMonth"
                            print(f"[Google Sheets] Appended {len(new_df):,} rows to '{sheet_display_name}' (total {len(combined_df):,})")
                        else:
                            # Sheet is empty or only has headers - use new data only (will create header row)
                            print(f"[Google Sheets] Sheet is empty or only has headers, using new data")
                            combined_df = new_df
                        # Disable update_mode-based merge for this case (we've already handled append)
                        update_mode = False
                    elif existing_data and len(existing_data) > 1:
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
                        
                        if merge_keys:
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
                            # No merge keys specified and not a special sheet - use new data only
                            print(f"[Google Sheets] No merge keys specified, using new data only")
                            combined_df = new_df
                            update_mode = False
                    else:
                        # Sheet is empty or only has headers (for non-append sheets)
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
        
        # Ensure worksheet has enough rows and columns
        required_rows = len(values)
        required_cols = len(values[0]) if values else 20
        current_rows = worksheet.row_count
        current_cols = worksheet.col_count
        
        if required_rows > current_rows or required_cols > current_cols:
            new_rows = max(required_rows, current_rows, 1000)
            new_cols = max(required_cols, current_cols, 30)
            print(f"[Google Sheets] Resizing sheet from {current_rows}x{current_cols} to {new_rows}x{new_cols}...")
            worksheet.resize(rows=new_rows, cols=new_cols)
            time.sleep(1.0)  # Give API time to process resize
        
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
        retry_with_backoff(
            lambda: worksheet.update(values, value_input_option='USER_ENTERED'),
            max_retries=5,
            initial_delay=2.0,
            max_delay=60.0
        )
        
        print(f"[Google Sheets] Successfully uploaded {len(combined_df):,} rows to sheet '{sheet_name}'")
        return True
        
    except Exception as e:
        print(f"\n[Google Sheets] Error uploading to Google Sheets: {e}")
        import traceback
        traceback.print_exc()
        return False


def upload_history_prediction(
    spreadsheet_id: str,
    category: str,
    prediction_year: int,
    num_months: int,
    tf_mse: Optional[float] = None,
    tf_mae: Optional[float] = None,
    tf_rmse: Optional[float] = None,
    tf_accuracy_abs: Optional[float] = None,
    tf_accuracy_sum: Optional[float] = None,
    rec_mse: Optional[float] = None,
    rec_mae: Optional[float] = None,
    rec_rmse: Optional[float] = None,
    rec_accuracy_abs: Optional[float] = None,
    rec_accuracy_sum: Optional[float] = None,
    mae_error_increase_pct: Optional[float] = None,
    rmse_error_increase_pct: Optional[float] = None,
    credentials_path: str = "key.json",
    sheet_name: str = "History_prediction"
) -> bool:
    """
    Upload history prediction results to Google Sheets.
    
    Args:
        spreadsheet_id: Google Sheets spreadsheet ID.
        category: Category name (e.g., "ALL", "DRY", "FRESH").
        prediction_year: Year for prediction (e.g., 2025).
        num_months: Number of months for prediction.
        tf_mse: Teacher Forcing MSE.
        tf_mae: Teacher Forcing MAE.
        tf_rmse: Teacher Forcing RMSE.
        tf_accuracy_abs: Teacher Forcing Accuracy (Σ|err|).
        tf_accuracy_sum: Teacher Forcing Accuracy (|Σ err|).
        rec_mse: Recursive MSE.
        rec_mae: Recursive MAE.
        rec_rmse: Recursive RMSE.
        rec_accuracy_abs: Recursive Accuracy (Σ|err|).
        rec_accuracy_sum: Recursive Accuracy (|Σ err|).
        mae_error_increase_pct: Error increase percentage for MAE.
        rmse_error_increase_pct: Error increase percentage for RMSE.
        credentials_path: Path to service account credentials JSON file.
        sheet_name: Name of the sheet to upload to.
    
    Returns:
        True if successful, False otherwise.
    """
    if not GSPREAD_AVAILABLE:
        print("\n[WARNING] gspread not available. Install with: pip install gspread google-auth")
        return False
    
    try:
        # Create DataFrame with the prediction results
        data = {
            'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Category': [category],
            'Year': [prediction_year],
            'Months': [num_months],
            'TF_MSE': [tf_mse if tf_mse is not None and not np.isnan(tf_mse) else None],
            'TF_MAE': [tf_mae if tf_mae is not None and not np.isnan(tf_mae) else None],
            'TF_RMSE': [tf_rmse if tf_rmse is not None and not np.isnan(tf_rmse) else None],
            'TF_Accuracy_SumAbs': [tf_accuracy_abs if tf_accuracy_abs is not None and not np.isnan(tf_accuracy_abs) else None],
            'TF_Accuracy_SumBeforeAbs': [tf_accuracy_sum if tf_accuracy_sum is not None and not np.isnan(tf_accuracy_sum) else None],
            'REC_MSE': [rec_mse if rec_mse is not None and not np.isnan(rec_mse) else None],
            'REC_MAE': [rec_mae if rec_mae is not None and not np.isnan(rec_mae) else None],
            'REC_RMSE': [rec_rmse if rec_rmse is not None and not np.isnan(rec_rmse) else None],
            'REC_Accuracy_SumAbs': [rec_accuracy_abs if rec_accuracy_abs is not None and not np.isnan(rec_accuracy_abs) else None],
            'REC_Accuracy_SumBeforeAbs': [rec_accuracy_sum if rec_accuracy_sum is not None and not np.isnan(rec_accuracy_sum) else None],
            'Error_Increase_MAE_Pct': [mae_error_increase_pct if mae_error_increase_pct is not None and not np.isnan(mae_error_increase_pct) else None],
            'Error_Increase_RMSE_Pct': [rmse_error_increase_pct if rmse_error_increase_pct is not None and not np.isnan(rmse_error_increase_pct) else None],
        }
        
        df = pd.DataFrame(data)
        
        # Use the existing upload function with update_mode=True to append
        print(f"\n[Google Sheets] Uploading history prediction results to '{sheet_name}'...")
        print(f"  - Category: {category}")
        print(f"  - Year: {prediction_year}")
        print(f"  - Months: {num_months}")
        
        # Upload using the existing function with update_mode to append
        return upload_to_google_sheets(
            saved_files={},
            spreadsheet_id=spreadsheet_id,
            sheet_name=sheet_name,
            credentials_path=credentials_path,
            data_df=df,
            update_mode=True,
            merge_keys=None  # Append mode - no merging, just append
        )
        
    except Exception as e:
        print(f"\n[Google Sheets] ✗ Error uploading history prediction: {e}")
        import traceback
        traceback.print_exc()
        return False
