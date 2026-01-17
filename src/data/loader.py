"""Data loading utilities."""
import os
import pandas as pd
import re
from pathlib import Path
from typing import List, Optional
from collections import defaultdict


class DataReader:
    """Class for loading data from CSV files."""
    
    def __init__(self, data_dir: str = "../dataset/data_cat", file_pattern: str = "data_{year}.csv"):
        """
        Initialize DataReader.
        
        Args:
            data_dir: Directory containing data files.
            file_pattern: Pattern for data file names. Must contain {year} placeholder.
        """
        self.data_dir = data_dir
        self.file_pattern = file_pattern
    
    def load(self, years: List[int]) -> pd.DataFrame:
        """
        Load data from CSV files for specified years.
        
        Args:
            years: List of years to load.
        
        Returns:
            Combined DataFrame from all years.
        
        Raises:
            FileNotFoundError: If any data file is not found.
        """
        result = []
        missing_files = []
        
        print(f"[DataReader] Loading data for years: {years}")
        print(f"[DataReader] Data directory: {self.data_dir}")
        
        for year in years:
            filepath = os.path.join(self.data_dir, self.file_pattern.format(year=year))
            if not os.path.exists(filepath):
                missing_files.append(filepath)
                print(f"[DataReader] [WARNING] File not found: {filepath}")
                continue
            
            try:
                data = pd.read_csv(filepath)
                print(f"[DataReader] Successfully loaded {len(data)} samples for year {year}")
                result.append(data)
            except Exception as e:
                print(f"[DataReader] [ERROR] Failed to load {filepath}: {e}")
                missing_files.append(filepath)
        
        if not result:
            error_msg = "No data files found for specified years."
            if missing_files:
                error_msg += f"\nMissing files:\n" + "\n".join([f"  - {f}" for f in missing_files])
            raise FileNotFoundError(error_msg)
        
        if missing_files:
            print(f"[DataReader] [WARNING] {len(missing_files)} file(s) not found, continuing with available data")
        
        combined_data = pd.concat(result, ignore_index=True)
        print(f"[DataReader] Combined data: {len(combined_data)} total samples")
        return combined_data
    
    def load_year(self, year: int) -> pd.DataFrame:
        """
        Load data for a single year.
        
        Args:
            year: Year to load.
        
        Returns:
            DataFrame for the specified year.
        """
        return self.load([year])
    
    def load_by_file_pattern(
        self, 
        years: List[int], 
        file_prefix: str = "Outboundreports"
    ) -> pd.DataFrame:
        """
        Load data by finding files matching pattern like Outboundreports_YYYYMMDD_YYYYMMDD.csv.
        
        This method automatically finds all files for each year and combines them.
        
        Args:
            years: List of years to load.
            file_prefix: Prefix to filter files (e.g., "Outboundreports").
        
        Returns:
            Combined DataFrame from all years.
        """
        data_dir = Path(self.data_dir)
        result = []
        
        def extract_year_from_filename(filename: str) -> Optional[int]:
            """Extract year from filename."""
            # Pattern: Outboundreports_YYYYMMDD_YYYYMMDD.csv
            match = re.search(r'Outboundreports_(\d{4})\d{4}_\d{8}', filename)
            if match:
                return int(match.group(1))
            # Pattern: data_YYYY.csv
            match = re.search(r'data_(\d{4})\.csv', filename)
            if match:
                return int(match.group(1))
            return None
        
        print(f"[DataReader] Loading data by file pattern for years: {years}")
        print(f"[DataReader] Data directory: {data_dir}")
        print(f"[DataReader] File prefix: {file_prefix}")
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Find all CSV files matching the prefix
        csv_files = list(data_dir.glob("*.csv"))
        files_by_year = defaultdict(list)
        years_set = set(years)
        
        for filepath in csv_files:
            if file_prefix and not filepath.name.startswith(file_prefix):
                continue
            
            year = extract_year_from_filename(filepath.name)
            if year and year in years_set:
                files_by_year[year].append(filepath)
        
        # Load files for each year
        for year in sorted(years):
            if year not in files_by_year:
                print(f"[DataReader] [WARNING] No files found for year {year}")
                continue
            
            year_files = sorted(files_by_year[year])
            print(f"[DataReader] Found {len(year_files)} file(s) for year {year}")
            
            year_data = []
            for filepath in year_files:
                try:
                    data = pd.read_csv(filepath)
                    print(f"[DataReader]   Loaded {filepath.name}: {len(data)} rows")
                    year_data.append(data)
                except Exception as e:
                    print(f"[DataReader] [ERROR] Failed to load {filepath}: {e}")
            
            if year_data:
                year_combined = pd.concat(year_data, ignore_index=True)
                print(f"[DataReader] Year {year} total: {len(year_combined)} rows")
                result.append(year_combined)
        
        if not result:
            raise FileNotFoundError(
                f"No data files found for years {years} in directory: {data_dir}\n"
                f"Expected files matching pattern: {file_prefix}_YYYYMMDD_YYYYMMDD.csv"
            )
        
        combined_data = pd.concat(result, ignore_index=True)
        print(f"[DataReader] Combined data: {len(combined_data)} total samples")
        return combined_data

