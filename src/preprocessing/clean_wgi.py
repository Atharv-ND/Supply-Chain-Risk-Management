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
RAW_DATA_DIR = PROJECT_ROOT / "data" / "external" / "governance"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "wgi"
REPORT_DIR = PROJECT_ROOT / "results" / "preprocessing"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

INDICATOR_MAP = {
    "GOV_WGI_CC": "wgi_control_corruption",
    "GOV_WGI_PV": "wgi_political_stability",
    "GOV_WGI_GE": "wgi_government_effectiveness",
    "GOV_WGI_VA": "wgi_voice_accountability",
    "GOV_WGI_RL": "wgi_rule_of_law",
    "GOV_WGI_RQ": "wgi_regulatory_quality",
}

def load_wgi():
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DATA_DIR}")
    
    # Identify the actual file, likely WB_WGI.csv
    file_path = next((f for f in csv_files if 'WGI' in f.name.upper()), csv_files[0])
    logger.info(f"Loading {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    return df, file_path.name

def normalize_columns(df):
    df.columns = [c.lower() for c in df.columns]
    return df

def filter_indicators_and_metric(df, report):
    report['source_indicators'] = df['indicator'].unique().tolist()
    report['source_metrics'] = df['comp_breakdown_1'].unique().tolist()
    
    # Filter for the 6 indicators
    initial_rows = len(df)
    df = df[df['indicator'].isin(INDICATOR_MAP.keys())]
    
    # Filter for the specific metric (Governance Score: WGI_SC)
    df = df[df['comp_breakdown_1'] == 'WGI_SC']
    
    report['indicator_selection'] = {
        'retained_indicators': list(INDICATOR_MAP.keys()),
        'excluded_indicators': [ind for ind in report['source_indicators'] if ind not in INDICATOR_MAP.keys()]
    }
    
    report['metric_selection'] = {
        'selected_metric': 'WGI_SC',
        'rows_removed': initial_rows - len(df)
    }
    return df

def standardize_country_codes(df, report):
    # World Bank REF_AREA is already ISO3
    df = df.rename(columns={'ref_area': 'country_code'})
    
    invalid_codes = df[df['country_code'].str.len() != 3]['country_code'].unique().tolist()
    
    report['country_validation'] = {
        'unique_countries': int(df['country_code'].nunique()),
        'iso3_valid_count': int(df['country_code'].nunique()) - len(invalid_codes),
        'iso3_invalid_count': len(invalid_codes),
        'unmatched_codes': invalid_codes
    }
    return df

def validate_years(df, report):
    df = df.rename(columns={'time_period': 'year'})
    df['year'] = df['year'].astype(int, errors='ignore')
    
    try:
        df['year'] = df['year'].astype(int)
        invalid_years = 0
    except ValueError:
        invalid_years = pd.to_numeric(df['year'], errors='coerce').isna().sum()
    
    report['year_validation'] = {
        'min_year': int(df['year'].min()) if invalid_years == 0 else "N/A",
        'max_year': int(df['year'].max()) if invalid_years == 0 else "N/A",
        'invalid_year_count': int(invalid_years)
    }
    return df

def check_duplicates(df, report):
    dup_mask = df.duplicated(subset=['country_code', 'year', 'indicator'])
    report['duplicate_validation'] = {
        'duplicate_long_rows': int(dup_mask.sum())
    }
    if dup_mask.sum() > 0:
        logger.warning(f"Found {dup_mask.sum()} duplicate rows for country/year/indicator. Dropping duplicates.")
        df = df.drop_duplicates(subset=['country_code', 'year', 'indicator'])
    return df

def reshape_wgi(df):
    df_pivot = df.pivot(index=['country_code', 'year'], columns='indicator', values='obs_value').reset_index()
    df_pivot = df_pivot.rename(columns=INDICATOR_MAP)
    # Ensure all 6 columns exist even if some are completely missing
    for col in INDICATOR_MAP.values():
        if col not in df_pivot.columns:
            df_pivot[col] = np.nan
    return df_pivot

def validate_values(df, report):
    val_report = {}
    for col in INDICATOR_MAP.values():
        if col in df.columns:
            # Ensure numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            missing = df[col].isna().sum()
            out_of_range = ((df[col] < 0) | (df[col] > 100)).sum()
            
            val_report[col] = {
                'min': float(df[col].min()) if not df[col].isna().all() else None,
                'max': float(df[col].max()) if not df[col].isna().all() else None,
                'missing_count': int(missing),
                'missing_percentage': float(missing / len(df) * 100),
                'out_of_range_count': int(out_of_range)
            }
    report['value_validation'] = val_report
    return df

def generate_report(report_data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=========================================\n")
        f.write(" WGI CLEANING REPORT\n")
        f.write("=========================================\n\n")
        
        f.write("A. Dataset information\n")
        f.write(f"- Source file: {report_data['source_file']}\n")
        f.write(f"- Original rows: {report_data['original_rows']}\n")
        f.write(f"- Original columns: {report_data['original_columns']}\n\n")
        
        f.write("B. Source structure\n")
        f.write(f"- Unique indicators: {len(report_data['source_indicators'])}\n")
        f.write(f"- Unique metric/breakdown values: {len(report_data['source_metrics'])}\n\n")
        
        f.write("C. Indicator selection\n")
        f.write(f"- Indicators found: {len(report_data['source_indicators'])}\n")
        f.write(f"- Six indicators retained: {report_data['indicator_selection']['retained_indicators']}\n")
        f.write(f"- Indicators excluded: {report_data['indicator_selection']['excluded_indicators']}\n\n")
        
        f.write("D. Metric selection\n")
        f.write(f"- Metrics found: {report_data['source_metrics']}\n")
        f.write(f"- Selected metric: {report_data['metric_selection']['selected_metric']}\n")
        f.write(f"- Rows removed by metric filtering: {report_data['metric_selection']['rows_removed']}\n\n")
        
        cv = report_data['country_validation']
        f.write("E. Country validation\n")
        f.write(f"- Unique countries: {cv['unique_countries']}\n")
        f.write(f"- ISO3 valid count: {cv['iso3_valid_count']}\n")
        f.write(f"- ISO3 invalid/unmatched count: {cv['iso3_invalid_count']}\n")
        if cv['iso3_invalid_count'] > 0:
            f.write(f"- Unmatched codes/names: {cv['unmatched_codes']}\n")
        f.write("\n")
        
        yv = report_data['year_validation']
        f.write("F. Year validation\n")
        f.write(f"- Minimum year: {yv['min_year']}\n")
        f.write(f"- Maximum year: {yv['max_year']}\n")
        f.write(f"- Invalid year count: {yv['invalid_year_count']}\n\n")
        
        f.write("G. Value validation\n")
        for col, metrics in report_data['value_validation'].items():
            f.write(f"- {col}:\n")
            f.write(f"  * min: {metrics['min']}\n")
            f.write(f"  * max: {metrics['max']}\n")
            f.write(f"  * missing: {metrics['missing_count']} ({metrics['missing_percentage']:.2f}%)\n")
            f.write(f"  * out of range (not 0-100): {metrics['out_of_range_count']}\n")
        f.write("\n")
        
        f.write("H. Duplicate validation\n")
        f.write(f"- Duplicate country/year/indicator rows: {report_data['duplicate_validation']['duplicate_long_rows']}\n")
        f.write(f"- Duplicate final country/year rows: {report_data['duplicate_validation_wide']}\n\n")
        
        f.write("I. Final schema\n")
        for col, dtype in report_data['final_schema'].items():
            f.write(f"- {col}: {dtype}\n")
        f.write("\n")
        
        f.write("J. Final dataset\n")
        f.write(f"- Final rows: {report_data['final_rows']}\n")
        f.write(f"- Final columns: {report_data['final_columns']}\n\n")
        
        f.write("K. Processing decisions\n")
        f.write("- Filtered for the 6 WGI indicators related to governance.\n")
        f.write("- Filtered for WGI_SC metric to obtain Governance Score (0-100).\n")
        f.write("- Retained original REF_AREA as country_code (ISO3 format expected).\n")
        f.write("- Pivoted the dataset to create a wide format with one row per country-year.\n")
        f.write("- No ML imputation or scaling applied at this stage.\n")

def main():
    report_data = {}
    
    logger.info("Starting WGI cleaning pipeline...")
    df, filename = load_wgi()
    report_data['source_file'] = filename
    report_data['original_rows'] = len(df)
    report_data['original_columns'] = len(df.columns)
    
    df = normalize_columns(df)
    df = filter_indicators_and_metric(df, report_data)
    df = standardize_country_codes(df, report_data)
    df = validate_years(df, report_data)
    df = check_duplicates(df, report_data)
    
    df_wide = reshape_wgi(df)
    
    df_wide = validate_values(df_wide, report_data)
    
    dup_wide = df_wide.duplicated(subset=['country_code', 'year']).sum()
    report_data['duplicate_validation_wide'] = int(dup_wide)
    
    report_data['final_rows'] = len(df_wide)
    report_data['final_columns'] = len(df_wide.columns)
    report_data['final_schema'] = {col: str(df_wide[col].dtype) for col in df_wide.columns}
    
    out_csv = PROCESSED_DATA_DIR / "wgi_clean.csv"
    out_report = REPORT_DIR / "wgi_cleaning_report.txt"
    
    df_wide.to_csv(out_csv, index=False, encoding='utf-8')
    generate_report(report_data, out_report)
    
    logger.info("Cleaning complete.")
    logger.info(f"Cleaned CSV saved to {out_csv}")
    logger.info(f"Report saved to {out_report}")

if __name__ == "__main__":
    main()
