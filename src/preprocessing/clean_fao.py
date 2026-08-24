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
RAW_DATA_DIR = PROJECT_ROOT / "data" / "external" / "economics"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "fao"
REPORT_DIR = PROJECT_ROOT / "results" / "preprocessing"

# Ensure output dirs exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# We will try to use country_mapping if needed, but FAO has ISO3
sys.path.append(str(PROJECT_ROOT / "src"))
from preprocessing.country_mapping import DATACO_TO_ISO3

def load_fao_files():
    cpi_file = RAW_DATA_DIR / "FAO_CP.csv"
    gdp_file = RAW_DATA_DIR / "FAO_MK.csv"
    
    if not cpi_file.exists():
        logger.error(f"Missing CPI file: {cpi_file}")
        raise FileNotFoundError(f"Missing CPI file: {cpi_file}")
    if not gdp_file.exists():
        logger.error(f"Missing GDP file: {gdp_file}")
        raise FileNotFoundError(f"Missing GDP file: {gdp_file}")
        
    df_cpi = pd.read_csv(cpi_file, low_memory=False)
    df_gdp = pd.read_csv(gdp_file, low_memory=False)
    
    return df_cpi, df_gdp, cpi_file.name, gdp_file.name

def normalize_columns(df):
    df.columns = [c.lower() for c in df.columns]
    return df

def standardize_country_codes(df, report, prefix):
    # Rename ref_area to country_code
    if 'ref_area' in df.columns:
        df = df.rename(columns={'ref_area': 'country_code'})
    
    invalid_codes = df[df['country_code'].str.len() != 3]['country_code'].unique().tolist()
    
    report[f'{prefix}_country_validation'] = {
        'unique_countries': int(df['country_code'].nunique()),
        'iso3_valid_count': int(df['country_code'].nunique()) - len(invalid_codes),
        'unmatched_count': len(invalid_codes),
        'unmatched_codes': invalid_codes
    }
    return df

def validate_years(df, report, prefix):
    if 'time_period' in df.columns:
        df = df.rename(columns={'time_period': 'year'})
    
    # Extract just the year part (first 4 characters) if it's a string like '2015-12'
    df['year'] = df['year'].astype(str).str[:4]
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    invalid_years = df['year'].isna().sum()
    
    report[f'{prefix}_year_validation'] = {
        'min_year': int(df['year'].min()) if invalid_years < len(df) else "N/A",
        'max_year': int(df['year'].max()) if invalid_years < len(df) else "N/A",
        'invalid_year_count': int(invalid_years)
    }
    return df

def process_cpi(df, report):
    report['cpi_indicators_found'] = df['indicator'].unique().tolist()
    
    target_ind = 'FAO_CP_23012'
    df_cpi = df[df['indicator'] == target_ind].copy()
    
    # The source is monthly (FREQ='M'). Filter to December ('-12') for annual observation to avoid duplicates.
    df_cpi = df_cpi[df_cpi['time_period'].astype(str).str.endswith('-12')].copy()
    
    report['cpi_indicator_selection'] = {
        'selected_indicator': target_ind,
        'dropped_indicators': [i for i in report['cpi_indicators_found'] if i != target_ind],
        'selected_name': 'Consumer Prices, General Indices (2015 = 100)'
    }
    
    df_cpi = standardize_country_codes(df_cpi, report, 'cpi')
    df_cpi = validate_years(df_cpi, report, 'cpi')
    
    # Check duplicates
    dup_count = df_cpi.duplicated(subset=['country_code', 'year']).sum()
    report['cpi_duplicate_count'] = int(dup_count)
    if dup_count > 0:
        logger.warning(f"CPI Duplicates found: {dup_count}. Dropping duplicates.")
        df_cpi = df_cpi.drop_duplicates(subset=['country_code', 'year'])
    
    # Value validation
    df_cpi['obs_value'] = pd.to_numeric(df_cpi['obs_value'], errors='coerce')
    missing_count = df_cpi['obs_value'].isna().sum()
    non_numeric = df_cpi['obs_value'].isna().sum() # Same as missing here since coerced
    
    report['cpi_value_validation'] = {
        'min': float(df_cpi['obs_value'].min()),
        'max': float(df_cpi['obs_value'].max()),
        'missing_count': int(missing_count),
        'missing_percentage': float(missing_count / len(df_cpi) * 100),
        'non_numeric_count': int(non_numeric),
        'suspicious_values': int((df_cpi['obs_value'] < 0).sum()) # Index shouldn't be negative
    }
    
    df_cpi = df_cpi.rename(columns={'obs_value': 'cpi_general'})
    final_cols = ['country_code', 'year', 'cpi_general']
    df_cpi = df_cpi[final_cols]
    
    report['cpi_final_rows'] = len(df_cpi)
    report['cpi_schema'] = {c: str(df_cpi[c].dtype) for c in df_cpi.columns}
    
    return df_cpi

def process_gdp(df, report):
    report['gdp_indicators_found'] = df['indicator'].unique().tolist()
    
    target_ind = 'FAO_MK_22008'
    df_gdp = df[df['indicator'] == target_ind].copy()
    
    report['gdp_indicator_selection'] = {
        'selected_indicator': target_ind,
        'dropped_indicators': [i for i in report['gdp_indicators_found'] if i != target_ind],
        'selected_name': 'Gross Domestic Product'
    }
    
    report['gdp_units_found'] = df_gdp['unit_measure'].unique().tolist()
    
    target_unit = 'USD_K_2015'
    initial_gdp_rows = len(df_gdp)
    df_gdp = df_gdp[df_gdp['unit_measure'] == target_unit].copy()
    
    report['gdp_unit_selection'] = {
        'selected_unit': target_unit,
        'rows_removed': initial_gdp_rows - len(df_gdp)
    }
    
    df_gdp = standardize_country_codes(df_gdp, report, 'gdp')
    df_gdp = validate_years(df_gdp, report, 'gdp')
    
    # Check duplicates
    dup_count = df_gdp.duplicated(subset=['country_code', 'year']).sum()
    report['gdp_duplicate_count'] = int(dup_count)
    if dup_count > 0:
        logger.warning(f"GDP Duplicates found: {dup_count}. Dropping duplicates.")
        df_gdp = df_gdp.drop_duplicates(subset=['country_code', 'year'])
        
    # Value validation
    df_gdp['obs_value'] = pd.to_numeric(df_gdp['obs_value'], errors='coerce')
    missing_count = df_gdp['obs_value'].isna().sum()
    
    report['gdp_value_validation'] = {
        'min': float(df_gdp['obs_value'].min()),
        'max': float(df_gdp['obs_value'].max()),
        'missing_count': int(missing_count),
        'missing_percentage': float(missing_count / len(df_gdp) * 100),
        'non_numeric_count': int(missing_count),
        'suspicious_values': int((df_gdp['obs_value'] < 0).sum()) # GDP shouldn't be negative
    }
    
    df_gdp = df_gdp.rename(columns={'obs_value': 'gdp'})
    final_cols = ['country_code', 'year', 'gdp']
    df_gdp = df_gdp[final_cols]
    
    report['gdp_final_rows'] = len(df_gdp)
    report['gdp_schema'] = {c: str(df_gdp[c].dtype) for c in df_gdp.columns}
    
    return df_gdp

def generate_report(report, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=========================================\n")
        f.write(" FAO CLEANING REPORT\n")
        f.write("=========================================\n\n")
        
        f.write("A. Source files\n")
        f.write(f"- CPI source file: {report['cpi_file']}\n")
        f.write(f"- GDP source file: {report['gdp_file']}\n")
        f.write(f"- CPI original rows/cols: {report['cpi_orig_rows']} / {report['cpi_orig_cols']}\n")
        f.write(f"- GDP original rows/cols: {report['gdp_orig_rows']} / {report['gdp_orig_cols']}\n\n")
        
        f.write("B. CPI indicator selection\n")
        f.write(f"- Indicators found: {report['cpi_indicators_found']}\n")
        f.write(f"- Selected indicator code: {report['cpi_indicator_selection']['selected_indicator']}\n")
        f.write(f"- Selected indicator name: {report['cpi_indicator_selection']['selected_name']}\n")
        f.write(f"- Dropped indicators: {report['cpi_indicator_selection']['dropped_indicators']}\n\n")
        
        f.write("C. GDP indicator selection\n")
        f.write(f"- Indicators found: {report['gdp_indicators_found']}\n")
        f.write(f"- Selected indicator code: {report['gdp_indicator_selection']['selected_indicator']}\n")
        f.write(f"- Selected indicator name: {report['gdp_indicator_selection']['selected_name']}\n")
        f.write(f"- Dropped indicators: {report['gdp_indicator_selection']['dropped_indicators']}\n\n")
        
        f.write("D. GDP unit selection\n")
        f.write(f"- Units found: {report['gdp_units_found']}\n")
        f.write(f"- Selected unit: {report['gdp_unit_selection']['selected_unit']}\n")
        f.write(f"- Rows removed because of unit filtering: {report['gdp_unit_selection']['rows_removed']}\n\n")
        
        f.write("E. Country validation\n")
        f.write("For CPI:\n")
        cv = report['cpi_country_validation']
        f.write(f"- unique countries: {cv['unique_countries']}\n")
        f.write(f"- valid ISO3: {cv['iso3_valid_count']}\n")
        f.write(f"- unmatched: {cv['unmatched_count']}\n")
        if cv['unmatched_count'] > 0: f.write(f"- unmatched codes: {cv['unmatched_codes']}\n")
        
        f.write("For GDP:\n")
        gv = report['gdp_country_validation']
        f.write(f"- unique countries: {gv['unique_countries']}\n")
        f.write(f"- valid ISO3: {gv['iso3_valid_count']}\n")
        f.write(f"- unmatched: {gv['unmatched_count']}\n")
        if gv['unmatched_count'] > 0: f.write(f"- unmatched codes: {gv['unmatched_codes']}\n")
        f.write("\n")
        
        f.write("F. Year validation\n")
        cy = report['cpi_year_validation']
        f.write("For CPI:\n")
        f.write(f"- min year: {cy['min_year']}\n")
        f.write(f"- max year: {cy['max_year']}\n")
        f.write(f"- invalid year count: {cy['invalid_year_count']}\n")
        
        gy = report['gdp_year_validation']
        f.write("For GDP:\n")
        f.write(f"- min year: {gy['min_year']}\n")
        f.write(f"- max year: {gy['max_year']}\n")
        f.write(f"- invalid year count: {gy['invalid_year_count']}\n\n")
        
        f.write("G. CPI value validation\n")
        cvv = report['cpi_value_validation']
        f.write(f"- minimum: {cvv['min']}\n")
        f.write(f"- maximum: {cvv['max']}\n")
        f.write(f"- missing count: {cvv['missing_count']}\n")
        f.write(f"- missing percentage: {cvv['missing_percentage']:.2f}%\n")
        f.write(f"- non-numeric count: {cvv['non_numeric_count']}\n")
        f.write(f"- suspicious/invalid (negative) values: {cvv['suspicious_values']}\n\n")
        
        f.write("H. GDP value validation\n")
        gvv = report['gdp_value_validation']
        f.write(f"- minimum: {gvv['min']}\n")
        f.write(f"- maximum: {gvv['max']}\n")
        f.write(f"- missing count: {gvv['missing_count']}\n")
        f.write(f"- missing percentage: {gvv['missing_percentage']:.2f}%\n")
        f.write(f"- non-numeric count: {gvv['non_numeric_count']}\n")
        f.write(f"- suspicious/invalid (negative) values: {gvv['suspicious_values']}\n\n")
        
        f.write("I. Duplicate validation\n")
        f.write(f"CPI: {report['cpi_duplicate_count']}\n")
        f.write(f"GDP: {report['gdp_duplicate_count']}\n")
        f.write(f"Combined: {report['combined_duplicate_count']}\n\n")
        
        f.write("J. Coverage comparison\n")
        f.write(f"- CPI country-year rows: {report['cpi_final_rows']}\n")
        f.write(f"- GDP country-year rows: {report['gdp_final_rows']}\n")
        f.write(f"- complete CPI+GDP rows: {report['combined_both']}\n")
        f.write(f"- CPI-only rows: {report['combined_cpi_only']}\n")
        f.write(f"- GDP-only rows: {report['combined_gdp_only']}\n\n")
        
        f.write("K. Final schema\n")
        f.write("CPI schema:\n")
        for k, v in report['cpi_schema'].items(): f.write(f"- {k}: {v}\n")
        f.write(f"- row count: {report['cpi_final_rows']}\n")
        
        f.write("GDP schema:\n")
        for k, v in report['gdp_schema'].items(): f.write(f"- {k}: {v}\n")
        f.write(f"- row count: {report['gdp_final_rows']}\n")
        
        f.write("Combined schema:\n")
        for k, v in report['combined_schema'].items(): f.write(f"- {k}: {v}\n")
        f.write(f"- row count: {report['combined_rows']}\n\n")
        
        f.write("L. Processing decisions\n")
        f.write("- Retained FAO_CP_23012 for CPI as a pure 2015=100 index.\n")
        f.write("- Retained FAO_MK_22008 for GDP, explicitly filtering unit to USD_K_2015.\n")
        f.write("- Performed outer join to preserve distinct coverage gaps in CPI and GDP.\n")
        f.write("- Left missing values un-imputed.\n")

def main():
    logger.info("Starting FAO preprocessing pipeline...")
    df_cpi, df_gdp, cpi_fname, gdp_fname = load_fao_files()
    
    report = {
        'cpi_file': cpi_fname, 'gdp_file': gdp_fname,
        'cpi_orig_rows': len(df_cpi), 'cpi_orig_cols': len(df_cpi.columns),
        'gdp_orig_rows': len(df_gdp), 'gdp_orig_cols': len(df_gdp.columns)
    }
    
    df_cpi = normalize_columns(df_cpi)
    df_gdp = normalize_columns(df_gdp)
    
    df_cpi_clean = process_cpi(df_cpi, report)
    df_gdp_clean = process_gdp(df_gdp, report)
    
    # Create combined
    df_combined = pd.merge(df_cpi_clean, df_gdp_clean, on=['country_code', 'year'], how='outer', indicator=True)
    report['combined_both'] = (df_combined['_merge'] == 'both').sum()
    report['combined_cpi_only'] = (df_combined['_merge'] == 'left_only').sum()
    report['combined_gdp_only'] = (df_combined['_merge'] == 'right_only').sum()
    
    df_combined = df_combined.drop(columns=['_merge'])
    dup_combined = df_combined.duplicated(subset=['country_code', 'year']).sum()
    report['combined_duplicate_count'] = int(dup_combined)
    
    report['combined_rows'] = len(df_combined)
    report['combined_schema'] = {c: str(df_combined[c].dtype) for c in df_combined.columns}
    
    # Outputs
    out_cpi = PROCESSED_DATA_DIR / "fao_cpi_clean.csv"
    out_gdp = PROCESSED_DATA_DIR / "fao_gdp_clean.csv"
    out_combined = PROCESSED_DATA_DIR / "fao_economic_clean.csv"
    out_report = REPORT_DIR / "fao_cleaning_report.txt"
    
    df_cpi_clean.to_csv(out_cpi, index=False, encoding='utf-8')
    df_gdp_clean.to_csv(out_gdp, index=False, encoding='utf-8')
    df_combined.to_csv(out_combined, index=False, encoding='utf-8')
    
    generate_report(report, out_report)
    
    logger.info(f"Done. Report saved to {out_report}")

if __name__ == "__main__":
    main()
