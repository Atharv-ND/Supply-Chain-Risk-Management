import streamlit as st
import pandas as pd
import numpy as np
import datetime
from src.ml.predict_supply_chain_risk import SupplyChainRiskPredictor
from .data_loader import load_lpi, load_wgi, load_fao, load_emdat, load_holidays, load_fuel, load_integrated_dataset

@st.cache_resource
def get_predictor():
    return SupplyChainRiskPredictor()

def construct_features(item_inputs):
    """
    Construct exactly 41 ML features based on the dictionary of user inputs.
    Derives temporal and external features automatically.
    Fields not collected from the UI (city, state, sales, item_total, raw discount)
    are auto-derived here so the model always receives all 41 features.
    """
    country = item_inputs.get('order_country', '')
    order_date_str = item_inputs.get('order_date', datetime.date.today().strftime('%Y-%m-%d'))
    order_date = pd.to_datetime(order_date_str)

    # Map country name to country_code
    df_int = load_integrated_dataset()
    country_mapping = {}
    if not df_int.empty and 'order_country' in df_int.columns and 'order_country_iso3' in df_int.columns:
        country_mapping = dict(zip(df_int['order_country'], df_int['order_country_iso3']))

    country_code = country_mapping.get(country, country)

    # ── 1. Temporal derivation ────────────────────────────────────────────────
    order_year = order_date.year
    order_month = order_date.month
    order_day = order_date.day
    order_day_of_week = order_date.dayofweek
    order_week = order_date.isocalendar()[1]
    is_weekend = 1 if order_day_of_week >= 5 else 0

    # ── 2. Auto-derive financial fields not collected from UI ─────────────────
    product_price = float(item_inputs.get('product_price', 0.0))
    qty = float(item_inputs.get('order_item_quantity', 1.0))
    disc_rate = float(item_inputs.get('order_item_discount_rate', 0.0))
    # Derive raw discount amount and totals from the user-provided rate and price
    order_item_discount = round(product_price * disc_rate, 4)
    sales = round(product_price * qty, 4)
    order_item_total = round(product_price * qty * (1.0 - disc_rate), 4)

    # ── 3. External features ──────────────────────────────────────────────────
    # LPI
    lpi_df = load_lpi()
    lpi_year = order_year
    if lpi_year == 2015:
        lpi_year = 2014
    elif lpi_year == 2017:
        lpi_year = 2016

    lpi_row = lpi_df[(lpi_df['country_code'] == country_code) & (lpi_df['year'] == lpi_year)]
    if len(lpi_row) == 0:
        country_lpi = lpi_df[lpi_df['country_code'] == country_code]
        if len(country_lpi) > 0:
            lpi_row = country_lpi.iloc[[0]]

    # WGI
    wgi_df = load_wgi()
    wgi_row = wgi_df[(wgi_df['country_code'] == country_code) & (wgi_df['year'] == order_year)]
    if len(wgi_row) == 0:
        country_wgi = wgi_df[wgi_df['country_code'] == country_code]
        if len(country_wgi) > 0:
            wgi_row = country_wgi.iloc[[0]]

    # GDP / CPI (FAO)
    fao_df = load_fao()
    fao_row = fao_df[(fao_df['country_code'] == country_code) & (fao_df['year'] == order_year)]
    if len(fao_row) == 0:
        country_fao = fao_df[fao_df['country_code'] == country_code]
        if len(country_fao) > 0:
            fao_row = country_fao.iloc[[0]]

    # Fuel
    fuel_df = load_fuel()
    fuel_row = fuel_df[
        (fuel_df['country_code'] == country_code) &
        (fuel_df['year'] == order_year) &
        (fuel_df['month'] == order_month)
    ]
    if len(fuel_row) == 0:
        country_fuel = fuel_df[fuel_df['country_code'] == country_code]
        if len(country_fuel) > 0:
            fuel_row = country_fuel.iloc[[0]]

    # Disaster
    emdat_df = load_emdat()
    is_disaster_30d = 0
    if len(emdat_df) > 0 and 'country_code' in emdat_df.columns:
        dis_match = emdat_df[
            (emdat_df['country_code'] == country_code) &
            (emdat_df['event_year'] == order_year)
        ]
        if len(dis_match) > 0:
            for _, row in dis_match.iterrows():
                try:
                    ev_date = pd.to_datetime(row['event_date'])
                    days_diff = (order_date - ev_date).days
                    if 0 <= days_diff <= 30:
                        is_disaster_30d = 1
                        break
                except Exception:
                    pass

    # Holiday
    hol_df = load_holidays()
    is_holiday = 0
    if len(hol_df) > 0 and 'country_code' in hol_df.columns:
        hol_match = hol_df[
            (hol_df['country_code'] == country_code) &
            (hol_df['date'] == order_date_str)
        ]
        if len(hol_match) > 0:
            is_holiday = 1

    # ── 4. Safe value extraction helper ──────────────────────────────────────
    def safe_extract(row, col, default=0.0):
        if len(row) > 0 and col in row.columns:
            val = row.iloc[0][col]
            if pd.notna(val):
                return float(val)
        return default

    # ── 5. Build exactly 41 features ─────────────────────────────────────────
    features = {
        'days_shipping_scheduled': float(item_inputs.get('days_shipping_scheduled', 0)),
        'shipping_mode':           item_inputs.get('shipping_mode', ''),
        'market':                  item_inputs.get('market', ''),
        'order_region':            item_inputs.get('order_region', ''),
        'order_country':           country,
        # city/state default to empty string; OrdinalEncoder treats as 'unknown'
        'order_state':             item_inputs.get('order_state', ''),
        'order_city':              item_inputs.get('order_city', ''),
        'customer_segment':        item_inputs.get('customer_segment', ''),
        'category_name':           item_inputs.get('category_name', ''),
        'department_name':         item_inputs.get('department_name', ''),
        'order_item_quantity':     qty,
        'product_price':           product_price,
        # auto-derived financial fields
        'order_item_discount':     order_item_discount,
        'order_item_discount_rate': disc_rate,
        'order_item_profit_ratio': float(item_inputs.get('order_item_profit_ratio', 0)),
        'sales':                   sales,
        'order_item_total':        order_item_total,
        # temporal
        'order_year':        int(order_year),
        'order_month':       int(order_month),
        'order_day':         int(order_day),
        'order_day_of_week': int(order_day_of_week),
        'order_week':        int(order_week),
        'is_weekend':        int(is_weekend),
        # LPI
        'lpi_overall':                 safe_extract(lpi_row, 'lpi_overall'),
        'lpi_customs':                 safe_extract(lpi_row, 'lpi_customs'),
        'lpi_infrastructure':          safe_extract(lpi_row, 'lpi_infrastructure'),
        'lpi_international_shipments': safe_extract(lpi_row, 'lpi_international_shipments'),
        'lpi_logistics_competence':    safe_extract(lpi_row, 'lpi_logistics_competence'),
        'lpi_tracking_tracing':        safe_extract(lpi_row, 'lpi_tracking_tracing'),
        'lpi_timeliness':              safe_extract(lpi_row, 'lpi_timeliness'),
        # WGI
        'wgi_control_corruption':        safe_extract(wgi_row, 'wgi_control_corruption'),
        'wgi_political_stability':       safe_extract(wgi_row, 'wgi_political_stability'),
        'wgi_government_effectiveness':  safe_extract(wgi_row, 'wgi_government_effectiveness'),
        'wgi_voice_accountability':      safe_extract(wgi_row, 'wgi_voice_accountability'),
        'wgi_rule_of_law':               safe_extract(wgi_row, 'wgi_rule_of_law'),
        'wgi_regulatory_quality':        safe_extract(wgi_row, 'wgi_regulatory_quality'),
        # FAO
        'cpi_general': safe_extract(fao_row, 'cpi_general'),
        'gdp':         safe_extract(fao_row, 'gdp'),
        # Fuel
        'diesel_price_usd_per_liter': safe_extract(fuel_row, 'diesel_price_usd_per_liter'),
        # Event flags
        'is_holiday':      int(is_holiday),
        'is_disaster_30d': int(is_disaster_30d),
    }

    assert len(features) == 41, f"Expected 41 features, built {len(features)}"
    return features
