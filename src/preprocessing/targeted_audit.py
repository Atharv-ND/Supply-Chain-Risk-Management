import os
import pandas as pd
import numpy as np
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_csv(rel_path):
    path = os.path.join(DATA_DIR, rel_path)
    try:
        return pd.read_csv(path, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin1")

def audit_dataco():
    print("\n" + "="*80)
    print("--- A. DataCo ---")
    print("="*80)
    df = load_csv("raw/DataCoSupplyChainDataset.csv")
    if df is None: return
    
    print("\n[Sample rows]")
    print(df.head(2).to_string())
    
    print(f"\n[Unique Values]")
    print(f"Unique Order Country: {df['Order Country'].nunique()}")
    print(f"Unique Order City: {df['Order City'].nunique()}")
    
    print("\n[Date range]")
    df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'], errors='coerce')
    df['shipping date (DateOrders)'] = pd.to_datetime(df['shipping date (DateOrders)'], errors='coerce')
    
    print(f"Min order date: {df['order date (DateOrders)'].min()}")
    print(f"Max order date: {df['order date (DateOrders)'].max()}")
    print(f"Min shipping date: {df['shipping date (DateOrders)'].min()}")
    print(f"Max shipping date: {df['shipping date (DateOrders)'].max()}")
    
    print("\n[Is shipping date = order date + shipping days (real)?]")
    diff = (df['shipping date (DateOrders)'] - df['order date (DateOrders)']).dt.days
    match = (diff == df['Days for shipping (real)']).mean() * 100
    print(f"Match percentage: {match:.2f}%")
    
    print("\n[Distribution of Late_delivery_risk]")
    print(df['Late_delivery_risk'].value_counts(normalize=True).round(4) * 100)
    
    print("\n[Relationship between Late_delivery_risk and Delivery Status]")
    print(pd.crosstab(df['Late_delivery_risk'], df['Delivery Status']))
    
    print("\n[Does Order City + Country uniquely identify coordinates?]")
    geo_check = df.groupby(['Order Country', 'Order City'])[['Latitude', 'Longitude']].nunique()
    multi_coords = geo_check[(geo_check['Latitude'] > 1) | (geo_check['Longitude'] > 1)]
    if len(multi_coords) == 0:
        print("Yes, each City/Country has exactly 1 coordinate pair.")
    else:
        print(f"No, {len(multi_coords)} City/Country pairs have multiple coordinates.")
        print("Examples of 1 city having multiple coordinates:")
        print(multi_coords.head())

def audit_fao(file_path, name):
    print("\n" + "="*80)
    print(f"--- {name} ---")
    print("="*80)
    df = load_csv(file_path)
    for col in ['INDICATOR', 'INDICATOR_LABEL', 'UNIT_MEASURE', 'UNIT_MEASURE_LABEL']:
        if col in df.columns:
            print(f"\nUnique {col}:")
            for val in df[col].dropna().unique():
                print(f"  - {val}")

def audit_wgi():
    print("\n" + "="*80)
    print("--- D. WGI ---")
    print("="*80)
    df = load_csv("external/governance/WB_WGI.csv")
    for col in ['INDICATOR', 'INDICATOR_LABEL', 'COMP_BREAKDOWN_1', 'COMP_BREAKDOWN_1_LABEL']:
        if col in df.columns:
            print(f"\nUnique {col}:")
            for val in df[col].dropna().unique():
                print(f"  - {val}")

def audit_lpi():
    print("\n" + "="*80)
    print("--- E. LPI ---")
    print("="*80)
    df = load_csv("external/logistics/WB_LPI_20.csv")
    for col in ['INDICATOR_LABEL', 'COMP_BREAKDOWN_1_LABEL', 'TIME_PERIOD', 'UNIT_MEASURE_LABEL']:
        if col in df.columns:
            print(f"\nUnique {col}:")
            for val in df[col].dropna().unique():
                print(f"  - {val}")

def audit_fuel():
    print("\n" + "="*80)
    print("--- F. Fuel ---")
    print("="*80)
    path = os.path.join(DATA_DIR, "external/transportation/Global_Fuel_Prices_Database.xlsx")
    try:
        xl = pd.ExcelFile(path)
        print(f"Sheet names: {xl.sheet_names}")
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            print(f"\nSheet: {sheet} (Rows: {len(df)}, Cols: {len(df.columns)})")
            print("First 15 rows (dropping fully empty rows/cols):")
            print(df.dropna(how='all').dropna(axis=1, how='all').head(15).to_string())
    except Exception as e:
        print(f"Error reading fuel database: {e}")

def audit_emdat():
    print("\n" + "="*80)
    print("--- G. EM-DAT ---")
    print("="*80)
    path = os.path.join(DATA_DIR, "external/environmental/public_emdat_2026-07-28.xlsx")
    try:
        # EM-DAT typically has metadata on first few rows, we can read first 10 rows to inspect
        df_raw = pd.read_excel(path, nrows=15)
        print("First 10 rows of EM-DAT to understand structure:")
        print(df_raw.to_string())
        
        # Usually header is at row 6 (index 5)
        df = pd.read_excel(path, header=6) 
        
        print("\n[Found Columns]")
        print(df.columns.tolist()[:20] + ["..."]) # print first 20 cols
        
        def find_col(keyword):
            for c in df.columns:
                if keyword.lower() in str(c).lower(): return c
            return None
        
        dt_col = find_col("Disaster Type")
        yr_col = find_col("Year")
        ct_col = find_col("Country")
        
        if dt_col: print(f"\nDisaster types ({dt_col}):", df[dt_col].dropna().unique())
        if yr_col: print(f"\nYears ({yr_col}): min={df[yr_col].min()}, max={df[yr_col].max()}")
        if ct_col: print(f"\nCountries ({ct_col}): {df[ct_col].nunique()} unique")
        
        print("\n[Available Damage & Affected Data Columns]")
        for c in df.columns:
            c_str = str(c).lower()
            if any(k in c_str for k in ["damage", "affected", "magnitude", "scale", "death"]):
                print(f"  - {c}")
        
    except Exception as e:
        print(f"Error reading EM-DAT database: {e}")

if __name__ == '__main__':
    log_path = os.path.join(RESULTS_DIR, "targeted_audit_output.txt")
    with open(log_path, 'w', encoding='utf-8') as f:
        sys.stdout = f
        audit_dataco()
        audit_fao("external/economics/FAO_CP.csv", "B. FAO_CP")
        audit_fao("external/economics/FAO_MK.csv", "C. FAO_MK")
        audit_wgi()
        audit_lpi()
        audit_fuel()
        audit_emdat()
    
    # Restore stdout and print completion
    sys.stdout = sys.__stdout__
    print(f"Targeted audit completed. Results saved to {log_path}")
