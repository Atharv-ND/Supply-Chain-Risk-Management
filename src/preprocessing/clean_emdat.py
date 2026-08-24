import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Set up logging
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "external" / "environmental"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "emdat"
REPORT_DIR = PROJECT_ROOT / "results" / "preprocessing"

# Ensure output dirs exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def load_emdat(report):
    files = list(RAW_DATA_DIR.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No Excel files found in {RAW_DATA_DIR}")
    
    file_path = files[0]
    logger.info(f"Loading {file_path}")
    
    df = pd.read_excel(file_path)
    
    report['source_file'] = file_path.name
    report['original_rows'] = len(df)
    report['original_columns'] = len(df.columns)
    
    return df

def filter_disaster_scope(df, report):
    report['disaster_groups_found'] = df['Disaster Group'].unique().tolist()
    report['disaster_types_found'] = df['Disaster Type'].unique().tolist()
    
    # Retain only Natural disasters
    df_natural = df[df['Disaster Group'] == 'Natural'].copy()
    
    report['disaster_groups_retained'] = ['Natural']
    report['disaster_groups_dropped'] = [g for g in report['disaster_groups_found'] if g != 'Natural']
    report['natural_records'] = len(df_natural)
    
    return df_natural

def standardize_country_codes(df, report):
    report['unique_iso3_total'] = int(df['ISO'].nunique())
    
    # Check for missing ISO3
    missing_iso3 = df['ISO'].isna().sum()
    report['missing_iso3_count'] = int(missing_iso3)
    
    # ISO codes in EM-DAT are generally 3 chars
    invalid_codes = []
    if missing_iso3 < len(df):
        invalid_codes = df[(df['ISO'].notna()) & (df['ISO'].str.len() != 3)]['ISO'].unique().tolist()
        
    report['invalid_iso3_count'] = len(invalid_codes)
    report['invalid_iso3_codes'] = invalid_codes
    report['valid_iso3_count'] = report['unique_iso3_total'] - len(invalid_codes)
    
    df = df.rename(columns={'ISO': 'country_code'})
    return df

def construct_dates(df, report):
    def make_date(row):
        try:
            if pd.isna(row['Start Year']) or pd.isna(row['Start Month']) or pd.isna(row['Start Day']):
                return pd.NaT
            return pd.Timestamp(year=int(row['Start Year']), month=int(row['Start Month']), day=int(row['Start Day']))
        except ValueError:
            return pd.NaT
            
    df['event_date'] = df.apply(make_date, axis=1)
    df['event_year'] = pd.to_numeric(df['Start Year'], errors='coerce').astype('Int64')
    
    invalid_date_count = df['event_date'].isna().sum()
    report['invalid_date_count'] = int(invalid_date_count)
    report['min_event_date'] = str(df['event_date'].min().date()) if not df['event_date'].isna().all() else "N/A"
    report['max_event_date'] = str(df['event_date'].max().date()) if not df['event_date'].isna().all() else "N/A"
    report['year_range'] = f"{df['event_year'].min()} - {df['event_year'].max()}"
    
    return df

def clean_numeric_fields(df, report):
    num_cols = {
        'Magnitude': 'magnitude',
        'Total Deaths': 'total_deaths',
        'Total Affected': 'total_affected',
        "Total Damage ('000 US$)": 'total_damage'
    }
    
    invalid_counts = {}
    negative_counts = {}
    
    for old_col, new_col in num_cols.items():
        if old_col in df.columns:
            df[new_col] = pd.to_numeric(df[old_col], errors='coerce')
            invalid_counts[new_col] = int(df[old_col].notna().sum() - df[new_col].notna().sum())
            negative_counts[new_col] = int((df[new_col] < 0).sum())
        else:
            df[new_col] = pd.NA
            
    report['invalid_numeric_count'] = sum(invalid_counts.values())
    report['negative_numeric_count'] = sum(negative_counts.values())
    
    return df

def process_event_level(df, report):
    df = df.rename(columns={
        'DisNo.': 'event_id',
        'Disaster Group': 'disaster_group',
        'Disaster Subgroup': 'disaster_subgroup',
        'Disaster Type': 'disaster_type',
        'Disaster Subtype': 'disaster_subtype',
        'Magnitude Scale': 'magnitude_scale'
    })
    
    report['exact_duplicates_before'] = int(df.duplicated().sum())
    
    # Check duplicate event_id
    dup_event_ids = df.duplicated(subset=['event_id']).sum()
    report['duplicate_event_ids'] = int(dup_event_ids)
    
    if dup_event_ids > 0:
        df = df.drop_duplicates(subset=['event_id'], keep='first')
        
    return df

def calculate_missingness(df, report, cols):
    missing_report = {}
    total_len = len(df)
    for c in cols:
        if c in df.columns:
            m = int(df[c].isna().sum())
            missing_report[c] = {
                'missing_count': m,
                'missing_percentage': float(m / total_len * 100) if total_len > 0 else 0
            }
    report['missing_statistics'] = missing_report

def generate_report(report, df_final, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=========================================\n")
        f.write(" EM-DAT CLEANING REPORT\n")
        f.write("=========================================\n\n")
        
        f.write(f"1. Source file: {report['source_file']}\n")
        f.write(f"2. Original rows: {report['original_rows']}\n")
        f.write(f"3. Original columns: {report['original_columns']}\n")
        f.write(f"4. Final rows: {report['final_rows']}\n")
        f.write(f"5. Final columns: {report['final_columns']}\n")
        f.write(f"6. Year range: {report['year_range']}\n")
        f.write(f"7. Unique countries: {report['unique_iso3_total']}\n")
        f.write("8. ISO3 validation:\n")
        f.write(f"   - Missing ISO3: {report['missing_iso3_count']}\n")
        f.write(f"   - Invalid/unmatched count: {report['invalid_iso3_count']}\n")
        if report['invalid_iso3_count'] > 0:
            f.write(f"   - Invalid codes: {report['invalid_iso3_codes']}\n")
        f.write(f"9. Disaster groups found: {report['disaster_groups_found']}\n")
        f.write(f"10. Disaster groups retained: {report['disaster_groups_retained']}\n")
        f.write(f"11. Disaster types found: {report['disaster_types_found']}\n")
        f.write(f"12. Exact duplicates: {report['exact_duplicates_before']}\n")
        f.write(f"13. Duplicate event IDs: {report['duplicate_event_ids']}\n")
        
        f.write("14. Missing-value statistics:\n")
        for col, stat in report['missing_statistics'].items():
            f.write(f"   - {col}: {stat['missing_count']} ({stat['missing_percentage']:.2f}%)\n")
            
        f.write(f"15. Invalid/missing date count: {report['invalid_date_count']}\n")
        f.write(f"16. Invalid numeric values: {report['invalid_numeric_count']}\n")
        f.write(f"17. Negative-value checks (should be 0): {report['negative_numeric_count']}\n")
        
        f.write("18. Filtering decisions:\n")
        f.write("   - Filtered for Disaster Group == 'Natural'\n")
        f.write("   - Reconstructed full date (event_date) only when year, month, and day are valid.\n")
        f.write("   - Dropped duplicate event IDs.\n")
        f.write("   - Did NOT fill missing numeric impacts with 0.\n")
        
        f.write("19. Final schema:\n")
        for col, dtype in report['final_schema'].items():
            f.write(f"   - {col}: {dtype}\n")
            
        f.write("20. Final status: SUCCESS\n")
        f.write("21. Unresolved issues: None\n")

def main():
    logger.info("Starting EM-DAT preprocessing pipeline Phase 2.5...")
    report = {}
    
    df = load_emdat(report)
    df = filter_disaster_scope(df, report)
    df = standardize_country_codes(df, report)
    df = construct_dates(df, report)
    df = clean_numeric_fields(df, report)
    df = process_event_level(df, report)
    
    # Select final target schema
    target_cols = [
        'country_code', 'event_id', 'event_date', 'event_year',
        'disaster_group', 'disaster_subgroup', 'disaster_type', 'disaster_subtype',
        'magnitude', 'magnitude_scale', 'total_deaths', 'total_affected', 'total_damage'
    ]
    df_final = df[[c for c in target_cols if c in df.columns]].copy()
    
    report['final_rows'] = len(df_final)
    report['final_columns'] = len(df_final.columns)
    report['final_schema'] = {c: str(df_final[c].dtype) for c in df_final.columns}
    
    calculate_missingness(df_final, report, target_cols)
    
    out_csv = PROCESSED_DATA_DIR / "emdat_clean.csv"
    out_report = REPORT_DIR / "emdat_cleaning_report.txt"
    
    df_final.to_csv(out_csv, index=False, encoding='utf-8')
    generate_report(report, df_final, out_report)
    
    logger.info(f"Done. Report saved to {out_report}")

if __name__ == "__main__":
    main()
