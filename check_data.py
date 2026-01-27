import pandas as pd
import sys

# Set UTF-8 encoding for output
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('dataset/test/data_2025.csv', encoding='utf-8', low_memory=False)
print(f'Total rows: {len(df)}')
print(f'Columns: {df.columns.tolist()}')

# Parse dates
df['ACTUALSHIPDATE'] = pd.to_datetime(df['ACTUALSHIPDATE'], format='mixed', dayfirst=True)
print(f'\nDate range: {df["ACTUALSHIPDATE"].min()} to {df["ACTUALSHIPDATE"].max()}')

# Check for 2026 dates
df_2026 = df[(df['ACTUALSHIPDATE'] >= '2026-01-01') & (df['ACTUALSHIPDATE'] < '2026-02-01')]
print(f'\nRows in 2026-01: {len(df_2026)}')

# Check categories
if len(df_2026) > 0:
    print(f'Categories in 2026-01: {df_2026["CATEGORY"].unique()}')
else:
    print('No data in 2026-01 range')
    # Check what dates we do have
    print(f'\nSample dates in file:')
    print(df['ACTUALSHIPDATE'].dt.date.value_counts().head(10))
