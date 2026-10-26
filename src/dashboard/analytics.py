import streamlit as st
import pandas as pd
from .charts import (plot_late_pie_chart, plot_late_trend, plot_bar_late_rate,
                     plot_scatter_lpi, plot_heatmap_shipping_market)


# ─────────────────────────────────────────────────────────────────────────────
# Shared inline filter helper
# ─────────────────────────────────────────────────────────────────────────────

def _inline_filters(df_feat: pd.DataFrame, tab_key: str) -> pd.DataFrame:
    """
    Render a compact row of filter widgets *inside* the calling tab.
    Returns the filtered dataframe. Uses tab_key to keep widget keys unique.
    """
    years          = ["All"] + sorted(df_feat['order_year'].dropna().unique().tolist())
    markets        = ["All"] + sorted(df_feat['market'].dropna().unique().tolist())
    regions        = ["All"] + sorted(df_feat['order_region'].dropna().unique().tolist())
    shipping_modes = ["All"] + sorted(df_feat['shipping_mode'].dropna().unique().tolist())

    f1, f2, f3, f4 = st.columns(4)
    sel_year   = f1.selectbox("Year",          years,          key=f"{tab_key}_yr")
    sel_market = f2.selectbox("Market",         markets,        key=f"{tab_key}_mkt")
    sel_region = f3.selectbox("Region",         regions,        key=f"{tab_key}_reg")
    sel_sm     = f4.selectbox("Shipping Mode",  shipping_modes, key=f"{tab_key}_sm")

    filtered = df_feat.copy()
    if sel_year   != "All": filtered = filtered[filtered['order_year']    == sel_year]
    if sel_market != "All": filtered = filtered[filtered['market']        == sel_market]
    if sel_region != "All": filtered = filtered[filtered['order_region']  == sel_region]
    if sel_sm     != "All": filtered = filtered[filtered['shipping_mode'] == sel_sm]
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Tab renderers
# ─────────────────────────────────────────────────────────────────────────────

def render_overview_tab(df_feat: pd.DataFrame, df_int: pd.DataFrame):
    st.header("Overview")

    # Inline filters
    with st.container():
        df = _inline_filters(df_feat, "overview")
    st.markdown("---")

    # ── KPI cards ─────────────────────────────────────────────────────────────
    total_records      = len(df)
    unique_orders      = df_int['order_id'].nunique() if 'order_id' in df_int.columns else "N/A"
    late_rate          = df['late_delivery_risk'].mean()
    on_time_rate       = 1.0 - late_rate
    countries_covered  = df['order_country'].nunique()
    avg_scheduled_days = df['days_shipping_scheduled'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Order-line Records",   f"{total_records:,}")
    col2.metric("Unique Orders",        f"{unique_orders:,}" if isinstance(unique_orders, int) else unique_orders)
    col3.metric("Late Delivery Rate",   f"{late_rate:.1%}")

    col4, col5, col6 = st.columns(3)
    col4.metric("On-Time Rate",         f"{on_time_rate:.1%}")
    col5.metric("Countries Covered",    f"{countries_covered}")
    col6.metric("Avg Scheduled Days",   f"{avg_scheduled_days:.1f}")

    st.markdown("---")

    # ── Target Analysis ────────────────────────────────────────────────────────
    st.subheader("Target Analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_late_pie_chart(df), use_container_width=True)
    with c2:
        time_agg = st.radio("Trend Aggregation", ["Monthly", "Yearly"], horizontal=True,
                            key="overview_trend_agg")
        st.plotly_chart(plot_late_trend(df, time_agg), use_container_width=True)

    c3, c4, c5 = st.columns(3)
    with c3:
        st.plotly_chart(plot_bar_late_rate(df, 'shipping_mode', 'Late Rate by Shipping Mode'),
                        use_container_width=True)
    with c4:
        st.plotly_chart(plot_bar_late_rate(df, 'market', 'Late Rate by Market'),
                        use_container_width=True)
    with c5:
        st.plotly_chart(plot_bar_late_rate(df, 'customer_segment', 'Late Rate by Customer Segment'),
                        use_container_width=True)

    st.markdown("---")

    # ── Product Analysis ───────────────────────────────────────────────────────
    st.subheader("Product Analysis")
    c6, c7 = st.columns(2)
    with c6:
        cat_grp = (df.groupby('category_name')
                     .agg(shipment_count=('late_delivery_risk', 'count'),
                          late_rate=('late_delivery_risk', 'mean'))
                     .reset_index()
                     .sort_values('late_rate', ascending=False))
        st.dataframe(cat_grp, height=250, use_container_width=True)
    with c7:
        dep_grp = (df.groupby('department_name')
                     .agg(shipment_count=('late_delivery_risk', 'count'),
                          late_rate=('late_delivery_risk', 'mean'))
                     .reset_index()
                     .sort_values('late_rate', ascending=False))
        st.dataframe(dep_grp, height=250, use_container_width=True)


def render_geography_tab(df_feat: pd.DataFrame):
    st.header("Geography & Logistics")

    with st.container():
        df = _inline_filters(df_feat, "geo")
    st.markdown("---")

    # ── Country Analysis ───────────────────────────────────────────────────────
    st.subheader("Country Analysis")
    min_shipments = st.slider("Minimum Shipment Count", min_value=1, max_value=5000, value=50,
                               key="geo_min_ship")

    cty_grp = (df.groupby('order_country')
                 .agg(Shipment_Count=('late_delivery_risk', 'count'),
                      Late_Count=('late_delivery_risk', 'sum'),
                      Late_Rate=('late_delivery_risk', 'mean'),
                      Average_Scheduled_Days=('days_shipping_scheduled', 'mean'),
                      Average_LPI=('lpi_overall', 'mean'))
                 .reset_index())

    cty_grp_filtered = (cty_grp[cty_grp['Shipment_Count'] >= min_shipments]
                        .sort_values('Late_Rate', ascending=False))

    st.dataframe(cty_grp_filtered.style.format({
        'Late_Rate': '{:.1%}',
        'Average_Scheduled_Days': '{:.1f}',
        'Average_LPI': '{:.2f}',
    }), use_container_width=True)

    st.markdown("---")

    # ── Region Analysis ────────────────────────────────────────────────────────
    st.subheader("Region Analysis")
    st.plotly_chart(plot_bar_late_rate(df, 'order_region', 'Late Rate by Region'),
                    use_container_width=True)

    st.markdown("---")

    # ── LPI Analysis ───────────────────────────────────────────────────────────
    st.subheader("LPI Analysis")
    st.plotly_chart(plot_scatter_lpi(df), use_container_width=True)

    st.markdown("---")

    lpi_components = [c for c in df.columns if c.startswith('lpi_')]
    if lpi_components:
        selected_lpi = st.selectbox("Select LPI Component", lpi_components, key="geo_lpi_comp")
        lpi_comp_grp = df.groupby(selected_lpi)['late_delivery_risk'].mean().reset_index()
        st.line_chart(lpi_comp_grp.set_index(selected_lpi))


def render_economic_tab(df_feat: pd.DataFrame):
    st.header("Economic & Operational")

    with st.container():
        df = _inline_filters(df_feat, "econ")
    st.markdown("---")

    # ── Economic KPIs ──────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Average GDP",          f"${df['gdp'].mean():,.0f}"                       if 'gdp' in df.columns else "N/A")
    c2.metric("Average CPI",          f"{df['cpi_general'].mean():.2f}"                 if 'cpi_general' in df.columns else "N/A")
    c3.metric("Average Diesel Price", f"${df['diesel_price_usd_per_liter'].mean():.2f}" if 'diesel_price_usd_per_liter' in df.columns else "N/A")

    st.markdown("---")

    # ── Operational Analysis ───────────────────────────────────────────────────
    st.subheader("Operational Analysis")
    st.plotly_chart(plot_heatmap_shipping_market(df), use_container_width=True)

    # Discount analysis
    df = df.copy()
    df['discount_bins'] = pd.cut(df['order_item_discount_rate'], bins=5)
    disc_grp = (df.groupby('discount_bins', observed=False)['late_delivery_risk']
                  .mean().reset_index())
    disc_grp['discount_bins'] = disc_grp['discount_bins'].astype(str)
    st.bar_chart(disc_grp.set_index('discount_bins'))


def render_external_risk_tab(df_feat: pd.DataFrame):
    st.header("External Risk")

    with st.container():
        df = _inline_filters(df_feat, "ext")
    st.markdown("---")

    c1, c2 = st.columns(2)

    # ── Disaster Analysis ──────────────────────────────────────────────────────
    with c1:
        st.subheader("Disaster Analysis")
        dis_grp = (df.groupby('is_disaster_30d')
                     .agg(Shipment_Count=('late_delivery_risk', 'count'),
                          Late_Rate=('late_delivery_risk', 'mean'))
                     .reset_index())
        dis_grp['is_disaster_30d'] = dis_grp['is_disaster_30d'].map(
            {0: 'No Recent Disaster', 1: 'Recent Disaster'})
        st.dataframe(dis_grp.style.format({'Late_Rate': '{:.1%}'}), use_container_width=True)

    # ── Holiday Analysis ───────────────────────────────────────────────────────
    with c2:
        st.subheader("Holiday Analysis")
        hol_grp = (df.groupby('is_holiday')
                     .agg(Shipment_Count=('late_delivery_risk', 'count'),
                          Late_Rate=('late_delivery_risk', 'mean'))
                     .reset_index())
        hol_grp['is_holiday'] = hol_grp['is_holiday'].map({0: 'Non-Holiday', 1: 'Holiday'})
        st.dataframe(hol_grp.style.format({'Late_Rate': '{:.1%}'}), use_container_width=True)

    st.markdown("---")

    # ── Governance Analysis ────────────────────────────────────────────────────
    st.subheader("Governance Analysis (WGI)")
    wgi_cols = [c for c in df.columns if c.startswith('wgi_')]
    if wgi_cols:
        sel_wgi = st.selectbox("Governance Indicator", wgi_cols, key="ext_wgi")
        df = df.copy()
        df['wgi_bin'] = pd.qcut(df[sel_wgi], q=5, duplicates='drop')
        wgi_grp = (df.groupby('wgi_bin', observed=False)['late_delivery_risk']
                     .mean().reset_index())
        wgi_grp['wgi_bin'] = wgi_grp['wgi_bin'].astype(str)
        st.bar_chart(wgi_grp.set_index('wgi_bin'))
