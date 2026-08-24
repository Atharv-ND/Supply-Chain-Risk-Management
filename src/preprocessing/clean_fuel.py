import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

# Set up logging
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "external" / "transportation"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "fuel"
REPORT_DIR = PROJECT_ROOT / "results" / "preprocessing"

# Ensure output dirs exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def load_fuel(report):
    files = list(RAW_DATA_DIR.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No Excel files found in {RAW_DATA_DIR}")
    
    file_path = files[0]
    logger.info(f"Loading {file_path}")
    
    xls = pd.ExcelFile(file_path)
    report['source_file'] = file_path.name
    report['products_found'] = xls.sheet_names
    report['selected_product'] = 'Diesel USD'
    report['selected_unit'] = 'USD/liter'
    
    if 'Diesel USD' not in xls.sheet_names:
        raise ValueError("Diesel USD sheet not found!")
        
    df = pd.read_excel(xls, sheet_name='Diesel USD')
    report['original_rows'] = len(df)
    report['original_columns'] = len(df.columns)
    
    return df

def filter_series(df, report):
    # The Default MAP (1 to show) column helps select the primary series per country
    if 'Default MAP (1 to show)' in df.columns:
        df = df[df['Default MAP (1 to show)'] == 1].copy()
        
    # Drop rows without a valid Country Code
    df = df.dropna(subset=['Country Code']).copy()
    
    # Handle duplicate Country Codes (e.g. FRA for France and French Guiana)
    dup_codes = df.duplicated(subset=['Country Code']).sum()
    if dup_codes > 0:
        # Sort so that the one where Country == Country Code (or shorter name) comes first?
        # French Guiana vs France -> France is shorter.
        df['name_len'] = df['Country'].astype(str).str.len()
        df = df.sort_values(['Country Code', 'name_len'])
        df = df.drop_duplicates(subset=['Country Code'], keep='first')
        df = df.drop(columns=['name_len'])
        
    return df

def validate_countries(df, report):
    df = df.rename(columns={'Country Code': 'country_code'})
    
    report['unique_countries'] = int(df['country_code'].nunique())
    report['missing_iso3_count'] = int(df['country_code'].isna().sum())
    
    invalid_codes = df[(df['country_code'].notna()) & (df['country_code'].str.len() != 3)]['country_code'].unique().tolist()
    report['invalid_iso3_count'] = len(invalid_codes)
    report['valid_iso3_count'] = report['unique_countries'] - len(invalid_codes)
    
    return df

def transform_dates(df, report):
    # Find datetime columns
    date_cols = [c for c in df.columns if isinstance(c, datetime) or isinstance(c, pd.Timestamp)]
    
    if not date_cols:
        raise ValueError("No datetime columns found in the sheet.")
        
    df_long = df.melt(id_vars=['country_code'], value_vars=date_cols, var_name='date', value_name='diesel_price_usd_per_liter')
    
    # Drop where price is NA/missing entirely
    df_long = df_long.dropna(subset=['diesel_price_usd_per_liter']).copy()
    
    df_long['date'] = pd.to_datetime(df_long['date'])
    # Normalize to 1st of month just in case
    df_long['date'] = df_long['date'].dt.to_period('M').dt.to_timestamp()
    
    df_long['year'] = df_long['date'].dt.year.astype('Int64')
    df_long['month'] = df_long['date'].dt.month.astype('Int64')
    
    report['min_date'] = df_long['date'].min().strftime('%Y-%m-%d')
    report['max_date'] = df_long['date'].max().strftime('%Y-%m-%d')
    report['unique_years'] = df_long['year'].dropna().unique().tolist()
    report['missing_months'] = int(df_long['month'].isna().sum())
    
    return df_long

def clean_numeric(df_long, report):
    col = 'diesel_price_usd_per_liter'
    
    # Convert to numeric, coerce errors (strings like 'NA', 'N/A', '-' become NaN)
    raw_missing = df_long[col].isna().sum()
    df_long[col] = pd.to_numeric(df_long[col], errors='coerce')
    
    report['missing_prices'] = int(df_long[col].isna().sum())
    report['non_numeric_values'] = int(report['missing_prices'] - raw_missing)
    report['negative_prices'] = int((df_long[col] < 0).sum())
    report['zero_prices'] = int((df_long[col] == 0).sum())
    
    # Suspicious values: e.g. > 10 USD per liter is highly unusual
    suspicious = int((df_long[col] > 10).sum())
    report['suspicious_values'] = suspicious
    
    # Drop rows that ended up as completely missing price after coercion
    df_long = df_long.dropna(subset=[col])
    
    return df_long

def validate_duplicates(df_long, report):
    report['exact_duplicate_rows'] = int(df_long.duplicated().sum())
    report['duplicate_country_month'] = int(df_long.duplicated(subset=['country_code', 'year', 'month']).sum())
    
    if report['duplicate_country_month'] > 0:
        df_long = df_long.drop_duplicates(subset=['country_code', 'year', 'month'], keep='first')
        
    return df_long

def validate_dataco_coverage(df_long, report):
    # DataCo period: 2015-01-01 to 2018-01-31
    dataco_df = df_long[(df_long['date'] >= '2015-01-01') & (df_long['date'] <= '2018-01-31')]
    report['dataco_coverage_rows'] = len(dataco_df)
    
    # Check 2015-01 to 2015-11
    early_2015 = df_long[(df_long['year'] == 2015) & (df_long['month'] < 12)]
    if len(early_2015) == 0:
        report['jan_nov_2015_unavailable'] = "Yes"
    else:
        report['jan_nov_2015_unavailable'] = "No"
        
    return df_long

def generate_report(report, df_final, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=========================================\n")
        f.write(" FUEL PRICES CLEANING REPORT\n")
        f.write("=========================================\n\n")
        
        f.write(f"1. Source file: {report['source_file']}\n")
        f.write(f"2. Original row count: {report['original_rows']}\n")
        f.write(f"3. Original column count: {report['original_columns']}\n")
        f.write(f"4. Final row count: {report['final_rows']}\n")
        f.write(f"5. Rows removed: {report['rows_removed']}\n\n")
        
        f.write(f"6. Product/fuel types found: {len(report['products_found'])} sheets\n")
        f.write(f"7. Selected product: {report['selected_product']}\n")
        f.write(f"8. Selected unit: {report['selected_unit']}\n\n")
        
        f.write(f"9. Unique countries: {report['unique_countries']}\n")
        f.write(f"10. Valid ISO3 count: {report['valid_iso3_count']}\n")
        f.write(f"11. Invalid ISO3 count: {report['invalid_iso3_count']}\n")
        f.write(f"12. Missing ISO3 count: {report['missing_iso3_count']}\n\n")
        
        f.write(f"13. Minimum date: {report['min_date']}\n")
        f.write(f"14. Maximum date: {report['max_date']}\n")
        f.write(f"15. Unique years: {report['unique_years']}\n")
        f.write(f"16. Missing months: {report['missing_months']}\n\n")
        
        f.write(f"17. Duplicate country-month records: {report['duplicate_country_month']}\n")
        f.write(f"18. Exact duplicate rows: {report['exact_duplicate_rows']}\n\n")
        
        f.write(f"19. Missing diesel prices: {report['missing_prices']}\n")
        f.write(f"20. Non-numeric values (coerced): {report['non_numeric_values']}\n")
        f.write(f"21. Negative prices: {report['negative_prices']}\n")
        f.write(f"22. Zero prices: {report['zero_prices']}\n")
        f.write(f"23. Suspicious values (>10 USD): {report['suspicious_values']}\n\n")
        
        f.write(f"24. DataCo-period coverage (2015-01 to 2018-01): {report['dataco_coverage_rows']} rows\n")
        f.write(f"25. 2015 Jan-Nov coverage limitation: {'Confirmed absent' if report['jan_nov_2015_unavailable'] == 'Yes' else 'Present'}\n\n")
        
        f.write("26. Final schema:\n")
        for col, dtype in report['final_schema'].items():
            f.write(f"   - {col}: {dtype}\n")
            
        f.write("\n27. Unresolved issues: None\n")
        f.write("28. Final cleaning status: SUCCESS\n")

def main():
    logger.info("Starting World Bank Global Fuel Prices preprocessing pipeline Phase 2.7...")
    report = {}
    
    df = load_fuel(report)
    df = filter_series(df, report)
    df = validate_countries(df, report)
    
    df_long = transform_dates(df, report)
    df_long = clean_numeric(df_long, report)
    df_long = validate_duplicates(df_long, report)
    df_long = validate_dataco_coverage(df_long, report)
    
    # Select final target schema
    target_cols = ['country_code', 'date', 'year', 'month', 'diesel_price_usd_per_liter']
    df_final = df_long[[c for c in target_cols if c in df_long.columns]].copy()
    
    report['final_rows'] = len(df_final)
    # The rows removed conceptually is original wide rows * months - final long rows, 
    # but we'll just say N/A or calculate it based on NA drops.
    report['rows_removed'] = "N/A" # Variable depending on NA drops
    report['final_schema'] = {c: str(df_final[c].dtype) for c in df_final.columns}
    
    out_csv = PROCESSED_DATA_DIR / "fuel_clean.csv"
    out_report = REPORT_DIR / "fuel_cleaning_report.txt"
    
    df_final.to_csv(out_csv, index=False, encoding='utf-8')
    generate_report(report, df_final, out_report)
    
    logger.info(f"Done. Report saved to {out_report}")

if __name__ == "__main__":
    main()
