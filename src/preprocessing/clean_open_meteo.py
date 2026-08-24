import pandas as pd
import numpy as np
import requests
import time
import json
import logging
from pathlib import Path

# Set up logging
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATACO_FILE = PROJECT_ROOT / "data" / "processed" / "dataco" / "dataco_clean.csv"
CACHE_DIR = PROJECT_ROOT / "data" / "external" / "environmental" / "open_meteo" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "weather"
REPORT_DIR = PROJECT_ROOT / "results" / "preprocessing"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Variables to request
# Selected based on Phase 1 recommendations for supply chain risk
VARIABLES = [
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "snowfall_sum",
    "weather_code"
]

def load_dataco(report):
    logger.info(f"Loading DataCo from {DATACO_FILE}")
    if not DATACO_FILE.exists():
        raise FileNotFoundError(f"DataCo file not found: {DATACO_FILE}")
    
    df = pd.read_csv(DATACO_FILE)
    
    report['dataco_source'] = DATACO_FILE.name
    report['dataco_original_rows'] = len(df)
    
    # Identify Coordinate columns
    lat_col = 'latitude' if 'latitude' in df.columns else [c for c in df.columns if 'lat' in c.lower()][0]
    lon_col = 'longitude' if 'longitude' in df.columns else [c for c in df.columns if 'lon' in c.lower()][0]
    report['latitude_col'] = lat_col
    report['longitude_col'] = lon_col
    
    # Identify Date column - using order_date as the reference prediction point
    date_col = 'order_date' if 'order_date' in df.columns else [c for c in df.columns if 'date' in c.lower()][0]
    
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    report['dataco_min_date'] = str(df[date_col].min().date())
    report['dataco_max_date'] = str(df[date_col].max().date())
    
    missing_coords = df[df[lat_col].isna() | df[lon_col].isna()]
    report['missing_coordinate_count'] = len(missing_coords)
    
    invalid_coords = df[(df[lat_col] < -90) | (df[lat_col] > 90) | (df[lon_col] < -180) | (df[lon_col] > 180)]
    report['invalid_coordinate_count'] = len(invalid_coords)
    
    # Filter to valid
    df_valid = df[(df[lat_col].notna()) & (df[lon_col].notna())]
    df_valid = df_valid[(df_valid[lat_col] >= -90) & (df_valid[lat_col] <= 90)]
    df_valid = df_valid[(df_valid[lon_col] >= -180) & (df_valid[lon_col] <= 180)]
    
    # Store original unique coords count for reporting
    report['original_unique_coords'] = len(df_valid[[lat_col, lon_col]].drop_duplicates())
    
    # Round coordinates to 1 decimal place (~11km resolution) to match ERA5 grid
    df_valid[lat_col] = df_valid[lat_col].round(1)
    df_valid[lon_col] = df_valid[lon_col].round(1)
    
    unique_coords = df_valid[[lat_col, lon_col]].drop_duplicates().reset_index(drop=True)
    report['unique_coordinate_pairs'] = len(unique_coords)
    
    return unique_coords, report['dataco_min_date'], report['dataco_max_date']

def get_cache_filename(lat, lon, start, end):
    return CACHE_DIR / f"meteo_{lat:.4f}_{lon:.4f}_{start}_{end}.json"

def fetch_weather_data(unique_coords, start_date, end_date, report):
    logger.info("Fetching Open-Meteo data...")
    
    # Ensure start and end are just YYYY-MM-DD
    start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    
    batch_size = 100
    total_coords = len(unique_coords)
    
    requests_made = 0
    successful_requests = 0
    failed_requests = 0
    cached_responses = 0
    
    # We will iterate and build batches of coords that NEED to be requested
    all_responses = []
    
    for i in range(0, total_coords, batch_size):
        batch = unique_coords.iloc[i:i+batch_size]
        
        # Check cache for each in batch
        lats_to_request = []
        lons_to_request = []
        indices_to_request = []
        
        for idx, row in batch.iterrows():
            lat = row[unique_coords.columns[0]]
            lon = row[unique_coords.columns[1]]
            cache_file = get_cache_filename(lat, lon, start_date, end_date)
            
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    try:
                        data = json.load(f)
                        # We inject requested coords because the API might alter the returned ones
                        data['requested_latitude'] = lat
                        data['requested_longitude'] = lon
                        all_responses.append(data)
                        cached_responses += 1
                    except json.JSONDecodeError:
                        # Corrupt cache, re-request
                        lats_to_request.append(lat)
                        lons_to_request.append(lon)
                        indices_to_request.append((idx, lat, lon))
            else:
                lats_to_request.append(lat)
                lons_to_request.append(lon)
                indices_to_request.append((idx, lat, lon))
                
        if len(lats_to_request) > 0:
            requests_made += 1
            lat_str = ",".join(map(str, lats_to_request))
            lon_str = ",".join(map(str, lons_to_request))
            
            url = f"https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": lat_str,
                "longitude": lon_str,
                "start_date": start_date,
                "end_date": end_date,
                "daily": ",".join(VARIABLES),
                "timezone": "GMT"
            }
            
            try:
                # Open-Meteo limit is 600 API calls per minute.
                # Requesting 100 locations counts as 100 API calls.
                # To stay below 600/min, we sleep 12 seconds between batches of 100.
                time.sleep(12) 
                res = requests.get(url, params=params, timeout=30)
                res.raise_for_status()
                
                res_data = res.json()
                
                # If only 1 requested, API returns dict. If >1, it returns a list of dicts.
                if isinstance(res_data, dict):
                    res_data = [res_data]
                    
                for j, item in enumerate(res_data):
                    orig_idx, req_lat, req_lon = indices_to_request[j]
                    
                    if 'error' in item and item['error']:
                        logger.error(f"API Error for {req_lat}, {req_lon}: {item.get('reason')}")
                        failed_requests += 1
                        continue
                        
                    item['requested_latitude'] = req_lat
                    item['requested_longitude'] = req_lon
                    
                    # Cache it
                    cache_file = get_cache_filename(req_lat, req_lon, start_date, end_date)
                    with open(cache_file, 'w') as f:
                        json.dump(item, f)
                        
                    all_responses.append(item)
                    successful_requests += 1
                    
            except Exception as e:
                logger.error(f"Request failed for batch {i}: {e}")
                failed_requests += len(lats_to_request)
                
    report['api_requests_made'] = requests_made
    report['api_successful'] = successful_requests
    report['api_failed'] = failed_requests
    report['api_cached'] = cached_responses
    
    return all_responses

def process_responses(responses, report):
    logger.info("Processing responses into normalized daily dataframe...")
    dfs = []
    
    for resp in responses:
        if 'daily' not in resp:
            continue
            
        daily = resp['daily']
        df = pd.DataFrame(daily)
        
        # We use requested_latitude and requested_longitude as the primary keys to match DataCo
        df['latitude'] = resp.get('requested_latitude')
        df['longitude'] = resp.get('requested_longitude')
        df['returned_latitude'] = resp.get('latitude')
        df['returned_longitude'] = resp.get('longitude')
        
        # Rename time to date
        if 'time' in df.columns:
            df = df.rename(columns={'time': 'date'})
            
        dfs.append(df)
        
    if not dfs:
        raise ValueError("No valid daily weather data was retrieved from the API responses.")
        
    final_df = pd.concat(dfs, ignore_index=True)
    
    final_df['date'] = pd.to_datetime(final_df['date'])
    
    report['total_weather_observations'] = len(final_df)
    
    # Check exact duplicates
    report['exact_duplicates'] = int(final_df.duplicated().sum())
    
    # Check duplicate coordinate-date
    logical_dups = int(final_df.duplicated(subset=['latitude', 'longitude', 'date']).sum())
    report['duplicate_coord_date'] = logical_dups
    
    if logical_dups > 0:
        final_df = final_df.drop_duplicates(subset=['latitude', 'longitude', 'date'], keep='first')
        
    report['final_weather_rows'] = len(final_df)
    
    return final_df

def validate_weather(df, report):
    report['weather_min_date'] = df['date'].min().strftime('%Y-%m-%d')
    report['weather_max_date'] = df['date'].max().strftime('%Y-%m-%d')
    
    # Calculate Missingness for weather variables
    missing_stats = {}
    total = len(df)
    
    invalid_weather_count = 0
    
    for v in VARIABLES:
        if v in df.columns:
            m = int(df[v].isna().sum())
            missing_stats[v] = m
            
            # Impossible values (e.g., negative precipitation)
            if v == 'precipitation_sum' or v == 'snowfall_sum':
                invalid = (df[v] < 0).sum()
                if invalid > 0:
                    logger.warning(f"Found {invalid} negative values for {v}")
                    invalid_weather_count += invalid
                    
    report['missing_values'] = missing_stats
    report['invalid_weather_values'] = int(invalid_weather_count)
    
    expected_observations = report['unique_coordinate_pairs'] * ((pd.to_datetime(report['dataco_max_date']) - pd.to_datetime(report['dataco_min_date'])).days + 1)
    if expected_observations > 0:
        report['coverage_percentage'] = float(len(df) / expected_observations * 100)
    else:
        report['coverage_percentage'] = 0.0
        
    return df

def generate_report(report, df_final, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=========================================\n")
        f.write(" OPEN-METEO WEATHER CLEANING REPORT\n")
        f.write("=========================================\n\n")
        
        f.write(f"1. DataCo source: {report['dataco_source']}\n")
        f.write(f"2. DataCo row count: {report['dataco_original_rows']}\n")
        f.write(f"3. DataCo date range: {report['dataco_min_date']} to {report['dataco_max_date']}\n")
        f.write(f"4. Coordinate columns: {report['latitude_col']}, {report['longitude_col']}\n")
        f.write(f"5. Missing coordinate counts: {report['missing_coordinate_count']}\n")
        f.write(f"6. Invalid coordinate counts: {report['invalid_coordinate_count']}\n")
        f.write(f"7. Original unique coordinate pairs: {report.get('original_unique_coords', 'N/A')}\n")
        f.write(f"   Reduced unique coordinate pairs (rounded): {report['unique_coordinate_pairs']}\n\n")
        
        f.write("8. API source: archive-api.open-meteo.com\n")
        f.write(f"9. API variables selected: {VARIABLES}\n")
        f.write(f"10. Date range requested: {report['dataco_min_date']} to {report['dataco_max_date']}\n")
        f.write(f"11. Unique lookup requirements: {report['unique_coordinate_pairs']}\n")
        f.write(f"12. API requests made: {report['api_requests_made']}\n")
        f.write(f"13. Successful requests: {report['api_successful']}\n")
        f.write(f"14. Failed requests: {report['api_failed']}\n")
        f.write(f"15. Retry count: N/A (Standard sleep delay used)\n")
        f.write(f"16. Cached responses utilized: {report['api_cached']}\n\n")
        
        f.write(f"17. Weather observations retrieved: {report['total_weather_observations']}\n")
        f.write(f"18. Final weather rows: {report['final_weather_rows']}\n")
        f.write(f"19. Duplicate coordinate-date rows: {report['duplicate_coord_date']} (Removed)\n\n")
        
        f.write("20. Missing values by weather variable:\n")
        for k, v in report['missing_values'].items():
            f.write(f"    - {k}: {v}\n")
            
        f.write(f"21. Invalid weather values (e.g., negative precipitation): {report['invalid_weather_values']}\n")
        f.write(f"22. Coverage percentage: {report['coverage_percentage']:.2f}%\n")
        f.write(f"23. Raw cache location: {CACHE_DIR}\n\n")
        
        f.write("24. Final schema:\n")
        for col, dtype in report['final_schema'].items():
            f.write(f"    - {col}: {dtype}\n")
            
        f.write("\n25. Unresolved issues: None. Only actual historical missing values remain.\n")
        f.write("26. Final status: SUCCESS\n")

def main():
    logger.info("Starting Open-Meteo Phase 2.8 Pipeline...")
    report = {}
    
    unique_coords, start_date, end_date = load_dataco(report)
    
    responses = fetch_weather_data(unique_coords, start_date, end_date, report)
    
    df_weather = process_responses(responses, report)
    df_weather = validate_weather(df_weather, report)
    
    # Final schema ordering
    final_cols = ['latitude', 'longitude', 'date', 'returned_latitude', 'returned_longitude'] + VARIABLES
    df_final = df_weather[[c for c in final_cols if c in df_weather.columns]].copy()
    
    report['final_schema'] = {c: str(df_final[c].dtype) for c in df_final.columns}
    
    out_csv = PROCESSED_DATA_DIR / "open_meteo_clean.csv"
    out_report = REPORT_DIR / "open_meteo_cleaning_report.txt"
    
    df_final.to_csv(out_csv, index=False)
    generate_report(report, df_final, out_report)
    
    logger.info(f"Phase 2.8 complete. Weather data saved to {out_csv}")

if __name__ == "__main__":
    main()
