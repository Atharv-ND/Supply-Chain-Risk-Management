import pandas as pd
import streamlit as st
import os

@st.cache_data
def load_feature_dataset():
    """Load the primary 41-feature modeling dataset."""
    path = "data/features/dataco_features.csv"
    if not os.path.exists(path):
        st.error(f"Required dataset not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_data
def load_integrated_dataset():
    """Load the Phase 5 integrated dataset (for extra historical context not in features)."""
    path = "data/integrated/dataco_integrated_phase5.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_data
def load_lpi():
    return pd.read_csv("data/processed/lpi/lpi_clean.csv")

@st.cache_data
def load_wgi():
    return pd.read_csv("data/processed/wgi/wgi_clean.csv")

@st.cache_data
def load_fao():
    return pd.read_csv("data/processed/fao/fao_economic_clean.csv")

@st.cache_data
def load_emdat():
    return pd.read_csv("data/processed/emdat/emdat_clean.csv")

@st.cache_data
def load_holidays():
    return pd.read_csv("data/processed/holidays/holidays_clean.csv")

@st.cache_data
def load_fuel():
    return pd.read_csv("data/processed/fuel/fuel_clean.csv")
