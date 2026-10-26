import pandas as pd
import numpy as np
import datetime
import os
import json

def merge_asof_backward(dataco, ext_df, ext_year_col='year', ext_country_col='country_code'):
    ext_df = ext_df.dropna(subset=[ext_year_col]).sort_values(ext_year_col)
    dataco = dataco.sort_values('order_year')
    merged = pd.merge_asof(
        dataco,
        ext_df,
        left_on='order_year',
        right_on=ext_year_col,
        left_by='order_country_iso3',
        right_by=ext_country_col,
        direction='backward'
    )
    merged.drop(columns=[ext_year_col, ext_country_col], inplace=True, errors='ignore')
    return merged

def merge_and_impute_fuel(dataco, fuel):
    merged = pd.merge(
        dataco,
        fuel[['country_code', 'year', 'month', 'diesel_price_usd_per_liter']].drop_duplicates(subset=['country_code', 'year', 'month']),
        how='left',
        left_on=['order_country_iso3', 'order_year', 'order_month'],
        right_on=['country_code', 'year', 'month']
    )
    
    merged['fuel_value_source'] = np.where(merged['diesel_price_usd_per_liter'].notna(), 'observed', 'missing')
    merged.drop(columns=['country_code', 'year', 'month'], inplace=True, errors='ignore')
    
    fuel['fuel_date'] = pd.to_datetime(fuel['year'].astype(str) + '-' + fuel['month'].astype(str) + '-01')
    fuel_sorted = fuel.dropna(subset=['diesel_price_usd_per_liter']).sort_values('fuel_date')
    
    dataco_dates = merged[['order_country_iso3', 'order_year', 'order_month']].drop_duplicates()
    dataco_dates['order_date_approx'] = pd.to_datetime(dataco_dates['order_year'].astype(str) + '-' + dataco_dates['order_month'].astype(str) + '-01')
    dataco_dates = dataco_dates.sort_values('order_date_approx')
    
    fallback = pd.merge_asof(
        dataco_dates,
        fuel_sorted[['country_code', 'fuel_date', 'diesel_price_usd_per_liter']],
        left_on='order_date_approx',
        right_on='fuel_date',
        left_by='order_country_iso3',
        right_by='country_code',
        direction='backward'
    )
    fallback = fallback.rename(columns={'diesel_price_usd_per_liter': 'hist_price'})
    
    global_proxy = fuel.groupby(['year', 'month'])['diesel_price_usd_per_liter'].mean().reset_index()
    global_proxy = global_proxy.rename(columns={'diesel_price_usd_per_liter': 'global_proxy_price'})
    
    merged = pd.merge(merged, fallback[['order_country_iso3', 'order_year', 'order_month', 'hist_price']], on=['order_country_iso3', 'order_year', 'order_month'], how='left')
    merged = pd.merge(merged, global_proxy, left_on=['order_year', 'order_month'], right_on=['year', 'month'], how='left')
    
    mask_hist = merged['diesel_price_usd_per_liter'].isna() & merged['hist_price'].notna()
    merged.loc[mask_hist, 'diesel_price_usd_per_liter'] = merged.loc[mask_hist, 'hist_price']
    merged.loc[mask_hist, 'fuel_value_source'] = 'historical_imputation'
    
    mask_proxy = merged['diesel_price_usd_per_liter'].isna() & merged['global_proxy_price'].notna()
    merged.loc[mask_proxy, 'diesel_price_usd_per_liter'] = merged.loc[mask_proxy, 'global_proxy_price']
    merged.loc[mask_proxy, 'fuel_value_source'] = 'regional_proxy'
    
    merged.drop(columns=['hist_price', 'global_proxy_price', 'year', 'month'], inplace=True, errors='ignore')
    return merged

def merge_holidays(dataco, holidays):
    holidays_unique = holidays[['country_code', 'date']].drop_duplicates().copy()
    holidays_unique['is_holiday_flag'] = 1
    holidays_unique['date_dt'] = pd.to_datetime(holidays_unique['date'])
    
    merged = pd.merge(
        dataco,
        holidays_unique[['country_code', 'date_dt', 'is_holiday_flag']],
        how='left',
        left_on=['order_country_iso3', 'order_date_dt'],
        right_on=['country_code', 'date_dt']
    )
    merged['is_holiday'] = merged['is_holiday_flag'].fillna(0).astype(int)
    merged.drop(columns=['country_code', 'date_dt', 'is_holiday_flag'], inplace=True, errors='ignore')
    return merged

def merge_emdat(dataco, emdat):
    emdat_nat = emdat[emdat['disaster_group'].str.lower() == 'natural'].copy()
    emdat_nat['event_date'] = pd.to_datetime(emdat_nat['event_date'])
    emdat_nat = emdat_nat.dropna(subset=['event_date']).sort_values('event_date')
    
    dataco_dates = dataco[['order_country_iso3', 'order_date_dt']].drop_duplicates().sort_values('order_date_dt')
    
    nearest_event = pd.merge_asof(
        dataco_dates,
        emdat_nat[['country_code', 'event_date']],
        left_on='order_date_dt',
        right_on='event_date',
        left_by='order_country_iso3',
        right_by='country_code',
        direction='backward',
        tolerance=pd.Timedelta(days=30)
    )
    
    nearest_event['is_disaster_30d_flag'] = nearest_event['event_date'].notna().astype(int)
    
    merged = pd.merge(
        dataco,
        nearest_event[['order_country_iso3', 'order_date_dt', 'is_disaster_30d_flag']].drop_duplicates(),
        on=['order_country_iso3', 'order_date_dt'],
        how='left'
    )
    merged['is_disaster_30d'] = merged['is_disaster_30d_flag'].fillna(0).astype(int)
    merged.drop(columns=['is_disaster_30d_flag'], inplace=True, errors='ignore')
    return merged

def main():
    print("Loading data...")
    dataco = pd.read_csv('data/processed/dataco/dataco_clean.csv')
    lpi = pd.read_csv('data/processed/lpi/lpi_clean.csv')
    wgi = pd.read_csv('data/processed/wgi/wgi_clean.csv')
    cpi = pd.read_csv('data/processed/fao/fao_cpi_clean.csv')
    gdp = pd.read_csv('data/processed/fao/fao_gdp_clean.csv')
    fuel = pd.read_csv('data/processed/fuel/fuel_clean.csv')
    holidays = pd.read_csv('data/processed/holidays/holidays_clean.csv')
    emdat = pd.read_csv('data/processed/emdat/emdat_clean.csv')
    
    original_rows = len(dataco)
    
    dataco['order_date_dt'] = pd.to_datetime(dataco['order_date']).dt.normalize()
    
    audit_data = []
    audit_data.append({'stage': 'initial_dataco', 'dataset': 'dataco', 'input_rows': original_rows, 'output_rows': original_rows, 'rows_removed': 0, 'notes': 'Base'})
    
    # 1. LPI
    dataco = merge_asof_backward(dataco, lpi, 'year', 'country_code')
    lpi_cols = ['lpi_overall', 'lpi_customs', 'lpi_infrastructure', 'lpi_international_shipments', 'lpi_logistics_competence', 'lpi_tracking_tracing', 'lpi_timeliness']
    curr_rows = len(dataco)
    dataco = dataco.dropna(subset=lpi_cols, how='any')
    audit_data.append({'stage': 'lpi', 'dataset': 'lpi', 'input_rows': curr_rows, 'output_rows': len(dataco), 'rows_removed': curr_rows - len(dataco), 'notes': 'Dropped missing LPI'})
    
    # 2. WGI
    curr_rows = len(dataco)
    wgi = wgi.drop_duplicates(subset=['country_code', 'year'])
    dataco = pd.merge(dataco, wgi, how='left', left_on=['order_country_iso3', 'order_year'], right_on=['country_code', 'year'])
    dataco.drop(columns=['country_code', 'year'], inplace=True, errors='ignore')
    wgi_cols = ['wgi_control_corruption', 'wgi_political_stability', 'wgi_government_effectiveness', 'wgi_voice_accountability', 'wgi_rule_of_law', 'wgi_regulatory_quality']
    dataco = dataco.dropna(subset=wgi_cols, how='any')
    audit_data.append({'stage': 'wgi', 'dataset': 'wgi', 'input_rows': curr_rows, 'output_rows': len(dataco), 'rows_removed': curr_rows - len(dataco), 'notes': 'Dropped missing WGI'})
    
    # 3. CPI
    curr_rows = len(dataco)
    dataco = merge_asof_backward(dataco, cpi, 'year', 'country_code')
    dataco = dataco.dropna(subset=['cpi_general'])
    audit_data.append({'stage': 'cpi', 'dataset': 'cpi', 'input_rows': curr_rows, 'output_rows': len(dataco), 'rows_removed': curr_rows - len(dataco), 'notes': 'Dropped missing CPI'})
    
    # 4. GDP
    curr_rows = len(dataco)
    dataco = merge_asof_backward(dataco, gdp, 'year', 'country_code')
    dataco = dataco.dropna(subset=['gdp'])
    audit_data.append({'stage': 'gdp', 'dataset': 'gdp', 'input_rows': curr_rows, 'output_rows': len(dataco), 'rows_removed': curr_rows - len(dataco), 'notes': 'Dropped missing GDP'})
    
    # 5. Fuel
    curr_rows = len(dataco)
    dataco = merge_and_impute_fuel(dataco, fuel)
    audit_data.append({'stage': 'fuel', 'dataset': 'fuel', 'input_rows': curr_rows, 'output_rows': len(dataco), 'rows_removed': curr_rows - len(dataco), 'notes': 'Imputed Fuel'})
    
    # 6. Holidays
    curr_rows = len(dataco)
    dataco = merge_holidays(dataco, holidays)
    audit_data.append({'stage': 'holidays', 'dataset': 'holidays', 'input_rows': curr_rows, 'output_rows': len(dataco), 'rows_removed': curr_rows - len(dataco), 'notes': 'Merged Holidays'})
    
    # 7. EM-DAT
    curr_rows = len(dataco)
    dataco = merge_emdat(dataco, emdat)
    audit_data.append({'stage': 'emdat', 'dataset': 'emdat', 'input_rows': curr_rows, 'output_rows': len(dataco), 'rows_removed': curr_rows - len(dataco), 'notes': 'Merged EM-DAT 30d'})
    
    if len(dataco) > original_rows:
        raise ValueError("Row count increased unexpectedly!")
    
    # ML Features
    approved_cols = [
        'order_year', 'order_month', 'order_day', 'order_day_of_week', 'order_week', 'is_weekend',
        'days_shipping_scheduled', 'shipping_mode', 'market', 'order_region', 'order_country', 'order_state', 'order_city',
        'customer_segment',
        'category_name', 'department_name', 'order_item_quantity', 'product_price', 'order_item_discount', 'order_item_discount_rate', 'order_item_profit_ratio', 'sales', 'order_item_total',
        'lpi_overall', 'lpi_customs', 'lpi_infrastructure', 'lpi_international_shipments', 'lpi_logistics_competence', 'lpi_tracking_tracing', 'lpi_timeliness',
        'wgi_control_corruption', 'wgi_political_stability', 'wgi_government_effectiveness', 'wgi_voice_accountability', 'wgi_rule_of_law', 'wgi_regulatory_quality',
        'cpi_general', 'gdp',
        'diesel_price_usd_per_liter',
        'is_holiday', 'is_disaster_30d',
        'late_delivery_risk'
    ]
    
    # Check if all exist, otherwise ignore missing
    existing_cols = [c for c in approved_cols if c in dataco.columns]
    
    # Add other columns from dataco just to preserve, but organized.
    audit_cols = ['order_id', 'order_item_id', 'order_country_iso3', 'order_date', 'fuel_value_source']
    
    other_cols = [c for c in dataco.columns if c not in existing_cols and c not in audit_cols and c != 'order_date_dt']
    dataco = dataco[audit_cols + existing_cols + other_cols]
    
    # Save output
    dataco.to_csv('data/integrated/dataco_integrated_phase5.csv', index=False)
    
    # Save audit
    audit_df = pd.DataFrame(audit_data)
    audit_df.to_csv('results/integration/phase5_integration_audit.csv', index=False)
    
    # Report
    with open('results/integration/phase5_integration_report.txt', 'w') as f:
        f.write("PHASE 5 INTEGRATION REPORT\n")
        f.write("==========================\n")
        f.write(f"Execution date: {datetime.datetime.now().isoformat()}\n")
        f.write("Input rows: {}\n".format(original_rows))
        f.write("Output rows: {}\n".format(len(dataco)))
        f.write("Retention %: {:.2f}%\n".format(100 * len(dataco) / original_rows))
        
        f.write("\nRow Loss:\n")
        for a in audit_data:
            f.write(f"{a['stage']}: {a['rows_removed']} rows removed\n")
            
        f.write("\nFuel Audit:\n")
        vc = dataco['fuel_value_source'].value_counts()
        for k, v in vc.items():
            f.write(f"{k}: {v} ({100 * v / len(dataco):.2f}%)\n")
            
        f.write("\nHoliday Audit:\n")
        f.write(f"is_holiday=1: {dataco['is_holiday'].sum()}\n")
        f.write(f"is_holiday=0: {len(dataco) - dataco['is_holiday'].sum()}\n")
        
        f.write("\nDisaster Audit:\n")
        f.write(f"is_disaster_30d=1: {dataco['is_disaster_30d'].sum()}\n")
        f.write(f"is_disaster_30d=0: {len(dataco) - dataco['is_disaster_30d'].sum()}\n")
        
        f.write("\nMissing Values:\n")
        for c in existing_cols:
            f.write(f"{c}: {dataco[c].isna().sum()}\n")
            
        f.write("\nTarget Audit:\n")
        f.write(f"Target missing: {dataco['late_delivery_risk'].isna().sum()}\n")
        
        f.write("\nApproved ML Features:\n")
        for c in existing_cols:
            f.write(f"- {c}\n")

if __name__ == "__main__":
    main()
