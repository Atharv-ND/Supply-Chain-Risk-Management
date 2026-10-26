import streamlit as st
import pandas as pd
import os

from src.dashboard.data_loader import load_feature_dataset, load_integrated_dataset
from src.dashboard.analytics import (render_overview_tab, render_geography_tab,
                                     render_economic_tab, render_external_risk_tab)
from src.dashboard.prediction_ui import render_prediction_tab
from src.dashboard.prediction_service import get_predictor

st.set_page_config(
    page_title="Multi-Risk Supply Chain Intelligence System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# Silent system readiness check (no output to main page)
# ─────────────────────────────────────────────────────────────────────────────
_checks = {
    "Historical dataset": os.path.exists("data/features/dataco_features.csv"),
    "Risk classifier":    os.path.exists("data/models/phase9_best_model.pkl"),
    "Delay regressor":    os.path.exists("data/models/phase9_delay_regressor.pkl"),
}
_all_ok = all(_checks.values())
_status_icon = "🟢" if _all_ok else "🔴"

# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────
st.title(f"{_status_icon} Multi-Risk Supply Chain Intelligence System")
st.caption("Historical Analysis & AI-Based Shipment Risk Prediction")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — navigation only
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🗺️ Navigation")
    active_tab = st.radio(
        "Select view:",
        [
            "📊 Overview",
            "🌍 Geography & Logistics",
            "💰 Economic & Operational",
            "⚠️ External Risk",
            "🤖 Predict Order Risk",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    # System status — tucked in sidebar, out of the way
    with st.expander("🔧 System Status", expanded=False):
        for name, ok in _checks.items():
            st.write(f"{'✅' if ok else '❌'} {name}")
        if not _all_ok:
            st.warning("Some system components are missing. Predictions may be unavailable.")

    st.markdown("---")
    st.caption("Supply Chain Intelligence v1.0")

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def get_global_data():
    df_feat = load_feature_dataset()
    df_int  = load_integrated_dataset()
    return df_feat, df_int

df_feat, df_int = get_global_data()

if df_feat.empty:
    st.error(
        "Historical features dataset could not be loaded. "
        "Please ensure data/features/dataco_features.csv exists."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Tab routing — each analytics tab owns its own inline filters
# ─────────────────────────────────────────────────────────────────────────────
if active_tab == "📊 Overview":
    render_overview_tab(df_feat, df_int)

elif active_tab == "🌍 Geography & Logistics":
    render_geography_tab(df_feat)

elif active_tab == "💰 Economic & Operational":
    render_economic_tab(df_feat)

elif active_tab == "⚠️ External Risk":
    render_external_risk_tab(df_feat)

elif active_tab == "🤖 Predict Order Risk":
    render_prediction_tab(df_feat)
