import pandas as pd
import numpy as np
import logging
import re
import sys
from pathlib import Path

# Set up logging
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "dataco"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dataco"
REPORT_DIR = PROJECT_ROOT / "results" / "preprocessing"

# Add src to sys path for imports
sys.path.append(str(PROJECT_ROOT / "src"))
from preprocessing.country_mapping import DATACO_TO_ISO3

# Create output dirs
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def load_dataco():
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {RAW_DATA_DIR}")
        raise FileNotFoundError(f"No CSV files found in {RAW_DATA_DIR}")
    
    file_path = csv_files[0]
    logger.info(f"Loading {file_path}")
    df = pd.read_csv(file_path, encoding='latin1', low_memory=False)
    return df, file_path.name

def normalize_columns(df):
    mapping = {}
    for col in df.columns:
        if col == "Days for shipping (real)":
            clean_col = "days_shipping_real"
        elif col == "Days for shipment (scheduled)":
            clean_col = "days_shipping_scheduled"
        elif col == "order date (DateOrders)":
            clean_col = "order_date"
        elif col == "shipping date (DateOrders)":
            clean_col = "shipping_date"
        else:
            clean_col = col.replace('(', '').replace(')', '')
            clean_col = re.sub(r'[^a-zA-Z0-9]', '_', clean_col)
            clean_col = re.sub(r'_+', '_', clean_col).strip('_').lower()
        mapping[col] = clean_col
    
    df = df.rename(columns=mapping)
    return df, mapping

def parse_dates(df, report):
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    df['shipping_date'] = pd.to_datetime(df['shipping_date'], errors='coerce')

    null_order = df['order_date'].isnull().sum()
    null_ship = df['shipping_date'].isnull().sum()
    invalid_logic = (df['order_date'] > df['shipping_date']).sum()
    
    report['date_validation'] = {
        'invalid_order_dates': int(null_order),
        'invalid_shipping_dates': int(null_ship),
        'order_after_shipping': int(invalid_logic),
        'min_order_date': str(df['order_date'].min()),
        'max_order_date': str(df['order_date'].max()),
        'min_shipping_date': str(df['shipping_date'].min()),
        'max_shipping_date': str(df['shipping_date'].max())
    }
    return df

def validate_coordinates(df, report):
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        logger.warning("Latitude/Longitude not found for validation.")
        return df
    
    invalid_lat = ((df['latitude'] < -90) | (df['latitude'] > 90)).sum()
    invalid_lon = ((df['longitude'] < -180) | (df['longitude'] > 180)).sum()
    
    report['coordinate_validation'] = {
        'invalid_latitudes': int(invalid_lat),
        'invalid_longitudes': int(invalid_lon)
    }
    return df

def validate_target(df, report):
    if 'late_delivery_risk' not in df.columns:
        logger.warning("Target late_delivery_risk not found.")
        return df
    
    val_counts = df['late_delivery_risk'].value_counts(dropna=False).to_dict()
    invalid_targets = sum(v for k, v in val_counts.items() if k not in [0, 1])
    
    total = len(df)
    report['target_validation'] = {
        'target_values': list(val_counts.keys()),
        'target_counts': val_counts,
        'target_percentages': {k: v/total * 100 for k, v in val_counts.items()},
        'invalid_targets': invalid_targets
    }
    return df

def standardize_countries(df, report):
    if 'order_country' not in df.columns:
        return df
    
    unique_countries = df['order_country'].unique()
    iso_map = {c: DATACO_TO_ISO3.get(c, None) for c in unique_countries}
    
    df['order_country_iso3'] = df['order_country'].map(iso_map)
    unmatched = [c for c, iso in iso_map.items() if iso is None]
    
    report['country_validation'] = {
        'unique_countries_count': len(unique_countries),
        'iso3_mapped_count': len(unique_countries) - len(unmatched),
        'unmatched_countries_count': len(unmatched),
        'unmatched_countries': unmatched
    }
    return df

def handle_missing_values(df, report):
    missing_data = []
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            missing_data.append({
                'column': col,
                'missing_count': int(missing),
                'missing_percentage': float(missing / len(df) * 100),
                'dtype': str(df[col].dtype)
            })
    report['missing_values'] = missing_data
    return df

def remove_unwanted_columns(df, report):
    leakage_cols = ['delivery_status', 'days_shipping_real', 'shipping_date', 'order_status']
    unusable_cols = ['product_description', 'order_zipcode']
    
    excluded = {}
    to_drop = []
    for col in leakage_cols:
        if col in df.columns:
            to_drop.append(col)
            excluded[col] = "Post-outcome information / Leakage"
            
    for col in unusable_cols:
        if col in df.columns:
            to_drop.append(col)
            if col == 'product_description':
                excluded[col] = "100% missing - no usable information"
            elif col == 'order_zipcode':
                excluded[col] = "86.24% missing - redundant for our geographic strategy"
            
    df = df.drop(columns=to_drop, errors='ignore')
    report['excluded_columns'] = excluded
    return df

def create_temporal_features(df):
    if 'order_date' in df.columns:
        df['order_year'] = df['order_date'].dt.year
        df['order_month'] = df['order_date'].dt.month
        df['order_day'] = df['order_date'].dt.day
        df['order_day_of_week'] = df['order_date'].dt.dayofweek
        df['order_week'] = df['order_date'].dt.isocalendar().week.astype('Int32')
        df['is_weekend'] = df['order_day_of_week'].isin([5, 6]).astype(int)
    return df

def classify_columns(df, report):
    classification = {
        'SENSITIVE': [],
        'IDENTIFIER': [],
        'POTENTIAL_LEAKAGE': [],
        'KEEP': []
    }
    
    sensitive_cols = {'customer_password', 'customer_email', 'customer_fname', 'customer_lname', 'customer_street'}
    identifier_cols = {'customer_id', 'order_id', 'order_customer_id', 'order_item_id', 'order_item_cardprod_id', 
                       'product_card_id', 'department_id', 'product_category_id', 'category_id'}
    
    for col in df.columns:
        if col in sensitive_cols:
            classification['SENSITIVE'].append(col)
        elif col in identifier_cols:
            classification['IDENTIFIER'].append(col)
        elif col == 'late_delivery_risk':
            classification['POTENTIAL_LEAKAGE'].append(col + " (TARGET)")
        else:
            classification['KEEP'].append(col)
            
    report['column_classification'] = classification

def generate_report(report_data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=========================================\n")
        f.write(" DATACO CLEANING REPORT\n")
        f.write("=========================================\n\n")
        
        f.write("A. Dataset information\n")
        f.write(f"- Source file: {report_data['source_file']}\n")
        f.write(f"- Original rows: {report_data['original_rows']}\n")
        f.write(f"- Original columns: {report_data['original_columns']}\n\n")
        
        f.write("B. Cleaning summary\n")
        f.write(f"- Final rows: {report_data['final_rows']}\n")
        f.write(f"- Final columns: {report_data['final_columns']}\n")
        f.write(f"- Exact duplicate rows removed: {report_data['duplicates_removed']}\n")
        f.write(f"- Rows with invalid order dates: {report_data['date_validation']['invalid_order_dates']}\n")
        f.write(f"- Rows with invalid shipping dates: {report_data['date_validation']['invalid_shipping_dates']}\n")
        f.write(f"- Rows where order > shipping: {report_data['date_validation']['order_after_shipping']}\n")
        f.write(f"- Rows with invalid latitudes: {report_data['coordinate_validation'].get('invalid_latitudes', 0)}\n")
        f.write(f"- Rows with invalid longitudes: {report_data['coordinate_validation'].get('invalid_longitudes', 0)}\n")
        f.write(f"- Invalid target values: {report_data['target_validation'].get('invalid_targets', 0)}\n\n")
        
        f.write("C. Missing-value summary (after dropping unusable cols)\n")
        for m in report_data.get('missing_values', []):
            f.write(f"- {m['column']}: {m['missing_count']} missing ({m['missing_percentage']:.2f}%) [dtype: {m['dtype']}]\n")
        if not report_data.get('missing_values', []):
            f.write("- No missing values remaining.\n")
        f.write("\n")
        
        f.write("D. Dropped/excluded columns\n")
        for col, reason in report_data.get('excluded_columns', {}).items():
            f.write(f"- {col}: {reason}\n")
        f.write("\n")
        
        f.write("E. Country validation (ISO3 Mapping)\n")
        cv = report_data.get('country_validation', {})
        if cv:
            f.write(f"- Unique DataCo country names: {cv['unique_countries_count']}\n")
            f.write(f"- Successfully mapped to ISO3: {cv['iso3_mapped_count']}\n")
            f.write(f"- Unmatched countries: {cv['unmatched_countries_count']}\n")
            if cv['unmatched_countries_count'] > 0:
                f.write(f"- List of unmatched: {cv['unmatched_countries']}\n")
        f.write("\n")
        
        f.write("F. Column Classification (for ML feature selection)\n")
        class_data = report_data.get('column_classification', {})
        f.write("- SENSITIVE (should not enter ML):\n")
        for c in class_data.get('SENSITIVE', []): f.write(f"  * {c}\n")
        f.write("- IDENTIFIER (metadata):\n")
        for c in class_data.get('IDENTIFIER', []): f.write(f"  * {c}\n")
        f.write("- POTENTIAL LEAKAGE / TARGET:\n")
        for c in class_data.get('POTENTIAL_LEAKAGE', []): f.write(f"  * {c}\n")
        f.write("- KEEP (candidate feature):\n")
        for c in class_data.get('KEEP', []): f.write(f"  * {c}\n")
        f.write("\n")
        
        f.write("G. Final schema\n")
        for col, dtype in report_data.get('final_schema', {}).items():
            f.write(f"- {col}: {dtype}\n")

def main():
    report_data = {}
    
    logger.info("Starting DataCo cleaning pipeline Phase 2.1 (Revision 1)...")
    df, filename = load_dataco()
    report_data['source_file'] = filename
    report_data['original_rows'] = len(df)
    report_data['original_columns'] = len(df.columns)
    
    dup_count = df.duplicated().sum()
    report_data['duplicates_removed'] = int(dup_count)
    if dup_count > 0:
        df = df.drop_duplicates()
        
    df, mapping = normalize_columns(df)
    report_data['column_mapping'] = mapping
    
    df = parse_dates(df, report_data)
    df = validate_coordinates(df, report_data)
    df = validate_target(df, report_data)
    df = standardize_countries(df, report_data)
    
    df = remove_unwanted_columns(df, report_data)
    df = create_temporal_features(df)
    
    df = handle_missing_values(df, report_data)
    classify_columns(df, report_data)
    
    report_data['final_rows'] = len(df)
    report_data['final_columns'] = len(df.columns)
    report_data['final_schema'] = {col: str(df[col].dtype) for col in df.columns}
    
    out_csv = PROCESSED_DATA_DIR / "dataco_clean.csv"
    out_report = REPORT_DIR / "dataco_cleaning_report.txt"
    
    df.to_csv(out_csv, index=False, encoding='utf-8')
    generate_report(report_data, out_report)
    
    logger.info("Cleaning complete.")
    logger.info(f"Cleaned CSV saved to {out_csv}")
    logger.info(f"Report saved to {out_report}")

if __name__ == "__main__":
    main()
