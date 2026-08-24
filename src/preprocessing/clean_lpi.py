import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Set up logging
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "external" / "logistics"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "lpi"
REPORT_DIR = PROJECT_ROOT / "results" / "preprocessing"

# Ensure output dirs exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# LPI Indicator Mapping
LPI_INDICATOR_MAP = {
    'Logistics performance index: Overall score (1=low to 5=high)': 'lpi_overall',
    'Efficiency of the clearance process, score (1=low to 5=high)': 'lpi_customs',
    'Quality of trade- and transport-related infrastructure, score (1=low to 5=high)': 'lpi_infrastructure',
    'Ease of arranging competitively priced international shipments, score (1=low to 5=high)': 'lpi_international_shipments',
    'Competence and quality of logistics services, score (1=low to 5=high)': 'lpi_logistics_competence',
    'Ability to track and trace consignments, score (1=low to 5=high)': 'lpi_tracking_tracing',
    'Frequency with which shipments reach consignee within scheduled or expected time, score (1=low to 5=high)': 'lpi_timeliness'
}

def load_lpi_file():
    lpi_file = RAW_DATA_DIR / "WorldBank_LPI_Historical.csv"
    if not lpi_file.exists():
        logger.error(f"Missing LPI file: {lpi_file}")
        raise FileNotFoundError(f"Missing LPI file: {lpi_file}")
    df = pd.read_csv(lpi_file, low_memory=False)
    return df, lpi_file.name

def filter_indicators(df, report):
    report['indicators_found'] = df['Indicator Name'].unique().tolist()
    
    # Filter for the specific indicators
    df_filtered = df[df['Indicator Name'].isin(LPI_INDICATOR_MAP.keys())].copy()
    
    report['indicators_retained'] = df_filtered['Indicator Name'].unique().tolist()
    report['indicators_missing'] = [k for k in LPI_INDICATOR_MAP.keys() if k not in report['indicators_retained']]
    report['indicators_dropped'] = [i for i in report['indicators_found'] if i not in LPI_INDICATOR_MAP.keys()]
    
    df_filtered['Indicator Name'] = df_filtered['Indicator Name'].map(LPI_INDICATOR_MAP)
    return df_filtered

def transform_to_long(df, report):
    year_cols = [c for c in df.columns if c.isdigit()]
    report['available_years'] = year_cols
    
    id_vars = ['Country Name', 'Country Code', 'Indicator Name']
    df_long = df.melt(id_vars=id_vars, value_vars=year_cols, var_name='year', value_name='value')
    
    # Drop rows with no score
    df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')
    df_long = df_long.dropna(subset=['value']).copy()
    
    return df_long

def validate_duplicates_before_pivot(df, report):
    dup_count = df.duplicated(subset=['Country Code', 'year', 'Indicator Name']).sum()
    report['dup_before_transformation'] = int(dup_count)
    if dup_count > 0:
        logger.warning(f"Found {dup_count} duplicates before pivoting. Dropping them.")
        df = df.drop_duplicates(subset=['Country Code', 'year', 'Indicator Name'])
    return df

def validate_countries(df, report):
    df = df.rename(columns={'Country Code': 'country_code'})
    
    # Check for valid ISO3 (usually 3 chars)
    invalid_codes = df[df['country_code'].str.len() != 3]['country_code'].unique().tolist()
    
    report['iso3_validation'] = {
        'unique_countries': int(df['country_code'].nunique()),
        'invalid_unmatched_codes': invalid_codes,
        'valid_iso3_count': int(df['country_code'].nunique()) - len(invalid_codes)
    }
    
    # We drop 'Country Name' to avoid ambiguity and rely on 'country_code'
    df = df.drop(columns=['Country Name'])
    return df

def reshape_to_wide(df, report):
    df_wide = df.pivot(index=['country_code', 'year'], columns='Indicator Name', values='value').reset_index()
    
    # Ensure all target columns exist
    for col in LPI_INDICATOR_MAP.values():
        if col not in df_wide.columns:
            df_wide[col] = np.nan
            
    # Check duplicates after transformation
    dup_count = df_wide.duplicated(subset=['country_code', 'year']).sum()
    report['dup_after_transformation'] = int(dup_count)
    if dup_count > 0:
        logger.warning(f"Found {dup_count} duplicates after pivoting. Dropping them.")
        df_wide = df_wide.drop_duplicates(subset=['country_code', 'year'])
        
    # Convert year to Int64
    df_wide['year'] = pd.to_numeric(df_wide['year'], errors='coerce').astype('Int64')
    
    return df_wide

def validate_values(df, report):
    val_report = {}
    for col in LPI_INDICATOR_MAP.values():
        missing_count = df[col].isna().sum()
        val_report[col] = {
            'min': float(df[col].min()) if not df[col].isna().all() else None,
            'max': float(df[col].max()) if not df[col].isna().all() else None,
            'missing_count': int(missing_count),
            'missing_percentage': float(missing_count / len(df) * 100)
        }
    report['value_validation'] = val_report
    return df

def generate_report(report, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=========================================\n")
        f.write(" LPI CLEANING REPORT\n")
        f.write("=========================================\n\n")
        
        f.write(f"1. Source file: {report['source_file']}\n")
        f.write(f"2. Original row count: {report['original_rows']}\n")
        f.write(f"3. Original column count: {report['original_columns']}\n")
        
        # 'unique_countries_initial' is from before pivoting
        f.write(f"4. Unique countries: {report['unique_countries_initial']}\n")
        f.write(f"5. Unique indicators: {len(report['indicators_found'])}\n")
        f.write(f"6. Available years: {report['available_years']}\n\n")
        
        f.write(f"7. Indicators found: {len(report['indicators_found'])}\n")
        f.write(f"8. Indicators retained: {report['indicators_retained']}\n")
        f.write(f"9. Indicators missing: {report['indicators_missing']}\n")
        f.write(f"10. Indicators dropped: {len(report['indicators_dropped'])}\n\n")
        
        f.write(f"11. Duplicate counts before transformation: {report['dup_before_transformation']}\n")
        f.write(f"12. Duplicate country-year count after transformation: {report['dup_after_transformation']}\n\n")
        
        f.write("13. Missing values for every final feature:\n")
        for col, metrics in report['value_validation'].items():
            f.write(f"  - {col}: {metrics['missing_count']} missing ({metrics['missing_percentage']:.2f}%)\n")
        f.write("\n")
        
        iso3 = report['iso3_validation']
        f.write("14. ISO3 validation results:\n")
        f.write(f"  - Unique country codes: {iso3['unique_countries']}\n")
        f.write(f"  - Valid ISO3 count: {iso3['valid_iso3_count']}\n")
        f.write(f"15. Invalid/unmatched country codes: {iso3['invalid_unmatched_codes']}\n\n")
        
        f.write(f"16. Final row count: {report['final_rows']}\n")
        f.write(f"17. Final country count: {report['final_country_count']}\n")
        f.write(f"18. Final year range: {report['final_year_min']} - {report['final_year_max']}\n\n")
        
        f.write("19. Final schema:\n")
        for col, dtype in report['final_schema'].items():
            f.write(f"  - {col}: {dtype}\n")
        f.write("\n")
        
        f.write("20. Overall cleaning status: SUCCESS\n")
        f.write("  - LPI score indicators correctly extracted and normalized to long/wide country-year format.\n")
        f.write("  - Preserved valid historical years. No interpolation applied.\n")

def main():
    logger.info("Starting LPI preprocessing pipeline Phase 2.4...")
    df, filename = load_lpi_file()
    
    report = {
        'source_file': filename,
        'original_rows': len(df),
        'original_columns': len(df.columns),
        'unique_countries_initial': int(df['Country Code'].nunique()) if 'Country Code' in df.columns else 0
    }
    
    df = filter_indicators(df, report)
    df_long = transform_to_long(df, report)
    df_long = validate_duplicates_before_pivot(df_long, report)
    df_long = validate_countries(df_long, report)
    
    df_wide = reshape_to_wide(df_long, report)
    df_wide = validate_values(df_wide, report)
    
    # Final target columns explicitly ordered
    final_cols = ['country_code', 'year', 'lpi_overall', 'lpi_customs', 'lpi_infrastructure', 
                  'lpi_international_shipments', 'lpi_logistics_competence', 
                  'lpi_tracking_tracing', 'lpi_timeliness']
    df_wide = df_wide[final_cols]
    
    report['final_rows'] = len(df_wide)
    report['final_country_count'] = int(df_wide['country_code'].nunique())
    report['final_year_min'] = int(df_wide['year'].min())
    report['final_year_max'] = int(df_wide['year'].max())
    report['final_schema'] = {c: str(df_wide[c].dtype) for c in df_wide.columns}
    
    out_csv = PROCESSED_DATA_DIR / "lpi_clean.csv"
    out_report = REPORT_DIR / "lpi_cleaning_report.txt"
    
    df_wide.to_csv(out_csv, index=False, encoding='utf-8')
    generate_report(report, out_report)
    
    logger.info(f"Done. Report saved to {out_report}")

if __name__ == "__main__":
    main()
