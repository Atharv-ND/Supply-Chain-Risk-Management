import streamlit as st
import pandas as pd
import datetime
from .prediction_service import get_predictor, construct_features


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_cascade_maps(df_feat: pd.DataFrame):
    """
    Pre-compute lookup dictionaries for cascading dropdowns.
    Returns dicts keyed by (country → {market, region, category, department}).
    """
    maps = {}
    for country, grp in df_feat.groupby('order_country'):
        maps[country] = {
            'markets':     sorted(grp['market'].dropna().unique().tolist()),
            'regions':     sorted(grp['order_region'].dropna().unique().tolist()),
            'categories':  sorted(grp['category_name'].dropna().unique().tolist()),
            'departments': sorted(grp['department_name'].dropna().unique().tolist()),
        }
    # Category → department mapping (global, not per-country)
    cat_dept_map = {}
    for cat, grp in df_feat.groupby('category_name'):
        cat_dept_map[cat] = sorted(grp['department_name'].dropna().unique().tolist())
    return maps, cat_dept_map


def _risk_badge(level: str) -> str:
    icons = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
    return f"{icons.get(level, '⚪')} {level}"


def _aggregate_order_risk(results: list[dict]) -> dict:
    """Derive order-level summary from item-level results."""
    delays = [r['predicted_delay_days'] for r in results]
    late_flags = [r['late_prediction'] for r in results]
    risk_levels = [r['risk_level'] for r in results]
    probs = [r['late_probability'] for r in results]

    max_delay = max(delays)
    worst_item = results[delays.index(max_delay)]['item_number']
    items_at_risk = sum(late_flags)
    risk_pct = items_at_risk / len(results) if results else 0
    max_risk = max(risk_levels, key=lambda x: ["LOW", "MEDIUM", "HIGH"].index(x))
    avg_prob = sum(probs) / len(probs) if probs else 0

    return {
        'max_delay': max_delay,
        'worst_item': worst_item,
        'items_at_risk': items_at_risk,
        'risk_pct': risk_pct,
        'overall_risk': max_risk,
        'avg_probability': avg_prob,
        'total_items': len(results),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────────

def render_prediction_tab(df_feat: pd.DataFrame):
    st.header("🤖 Predict Order Risk")
    st.caption(
        "Fill in the shipment details below, then click **Predict Order Risk**. "
        "The AI will estimate delivery risk and expected delay for each item."
    )

    # ── Session state ─────────────────────────────────────────────────────────
    if 'order_results' not in st.session_state:
        st.session_state.order_results = None
    if 'n_items' not in st.session_state:
        st.session_state.n_items = 1

    # ── Cascade maps (cached via session_state to avoid recompute on each run) ─
    if 'cascade_maps' not in st.session_state:
        st.session_state.cascade_maps, st.session_state.cat_dept_map = _build_cascade_maps(df_feat)
    cascade_maps = st.session_state.cascade_maps
    cat_dept_map = st.session_state.cat_dept_map

    # Global lists for fallbacks
    all_countries      = sorted(df_feat['order_country'].dropna().unique().tolist())
    all_shipping_modes = sorted(df_feat['shipping_mode'].dropna().unique().tolist())
    all_cust_segs      = sorted(df_feat['customer_segment'].dropna().unique().tolist())

    # ── Order-level controls ───────────────────────────────────────────────────
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 2, 1])
    with ctrl_col1:
        # Use the widget value directly — no rerun needed
        st.session_state.n_items = st.number_input(
            "Number of Items in Order",
            min_value=1, max_value=20,
            value=st.session_state.n_items,
            key="n_items_widget",
            help="How many line items does this order contain?"
        )
    with ctrl_col2:
        order_date = st.date_input(
            "Order Date",
            datetime.date(2015, 1, 1),
            key="order_date_global",
            help="Date on which the order is placed."
        )
    with ctrl_col3:
        st.write("")
        st.write("")
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.order_results = None

    st.markdown("---")

    # ── Per-item forms ────────────────────────────────────────────────────────
    items_data = []

    for i in range(1, int(st.session_state.n_items) + 1):
        with st.expander(f"📦 Item {i} Details", expanded=(i == 1)):

            # ── Row 1: Country (drives cascades) ──────────────────────────────
            country_key = f"cty_{i}"
            cty = st.selectbox(
                "Destination Country",
                all_countries,
                key=country_key,
                help="Select the order's destination country. Market and Region options will update automatically."
            )

            country_info = cascade_maps.get(cty, {
                'markets':     sorted(df_feat['market'].dropna().unique().tolist()),
                'regions':     sorted(df_feat['order_region'].dropna().unique().tolist()),
                'categories':  sorted(df_feat['category_name'].dropna().unique().tolist()),
                'departments': sorted(df_feat['department_name'].dropna().unique().tolist()),
            })

            # ── Row 2: Shipment columns ────────────────────────────────────────
            c_ship, c_cust, c_prod = st.columns(3)

            with c_ship:
                st.markdown("#### 🚚 Shipment")
                sm = st.selectbox(
                    "Shipping Mode", all_shipping_modes, key=f"sm_{i}",
                    help="Mode of transport for this shipment."
                )
                mkt = st.selectbox(
                    "Market",
                    country_info['markets'],
                    key=f"mkt_{i}",
                    help="Market region — filtered to choices available for the selected country."
                )
                reg = st.selectbox(
                    "Order Region",
                    country_info['regions'],
                    key=f"reg_{i}",
                    help="Geographic region of the order — filtered by country."
                )
                sched = st.number_input(
                    "Scheduled Delivery Days",
                    min_value=0, max_value=60, value=2, key=f"sched_{i}",
                    help="Number of days promised for delivery."
                )

            with c_cust:
                st.markdown("#### 👤 Customer")
                cust_seg = st.selectbox(
                    "Customer Segment", all_cust_segs, key=f"cust_seg_{i}",
                    help="Type of customer placing the order."
                )

            with c_prod:
                st.markdown("#### 📦 Product")
                # Category filtered by country; department filtered by category
                cat = st.selectbox(
                    "Category",
                    country_info['categories'],
                    key=f"cat_{i}",
                    help="Product category — filtered by country."
                )
                dept_options = cat_dept_map.get(cat, country_info['departments'])
                dept = st.selectbox(
                    "Department",
                    dept_options,
                    key=f"dept_{i}",
                    help="Department — filtered by selected category."
                )
                qty = st.number_input(
                    "Quantity", min_value=1, max_value=10000, value=1, key=f"qty_{i}",
                    help="Number of units ordered."
                )
                price = st.number_input(
                    "Product Price ($)", min_value=0.0, max_value=10000.0, value=100.0, key=f"price_{i}",
                    help="Unit price of the product."
                )
                disc_rate = st.number_input(
                    "Discount Rate (0–1)", min_value=0.0, max_value=1.0, value=0.0,
                    step=0.01, key=f"disc_rate_{i}",
                    help="Proportional discount applied (e.g. 0.10 = 10% off)."
                )
                prof_rat = st.number_input(
                    "Profit Ratio (−1 to 1)", min_value=-1.0, max_value=1.0, value=0.1,
                    step=0.01, key=f"prof_rat_{i}",
                    help="Expected profit ratio for this line item."
                )

            items_data.append({
                'order_date':                str(order_date),
                'shipping_mode':             sm,
                'market':                    mkt,
                'order_region':              reg,
                'order_country':             cty,
                'order_state':               '',   # not collected — model defaults to unknown
                'order_city':                '',   # not collected — model defaults to unknown
                'days_shipping_scheduled':   sched,
                'customer_segment':          cust_seg,
                'category_name':             cat,
                'department_name':           dept,
                'order_item_quantity':       qty,
                'product_price':             price,
                'order_item_discount_rate':  disc_rate,
                'order_item_profit_ratio':   prof_rat,
            })

    st.markdown("---")

    # ── Predict button — ALL processing happens only here ─────────────────────
    if st.button("🔮 Predict Order Risk", type="primary", use_container_width=True):
        predictor = get_predictor()
        all_results = []

        with st.spinner("Analysing shipment risk across all items…"):
            for idx, data in enumerate(items_data):
                try:
                    feat_dict = construct_features(data)
                    feat_df   = pd.DataFrame([feat_dict])
                    res       = predictor.predict_shipment(feat_df)[0]
                    res['item_number'] = idx + 1
                    all_results.append(res)
                except Exception as e:
                    st.error(f"Error predicting Item {idx + 1}: {e}")

        st.session_state.order_results = all_results

    # ── Results display ───────────────────────────────────────────────────────
    if st.session_state.order_results:
        results = st.session_state.order_results
        summary = _aggregate_order_risk(results)

        st.markdown("---")

        # ── Order-level banner ─────────────────────────────────────────────────
        risk = summary['overall_risk']
        risk_color = {"LOW": "#28a745", "MEDIUM": "#ffc107", "HIGH": "#dc3545"}.get(risk, "#6c757d")

        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {risk_color}22, {risk_color}11);
            border: 2px solid {risk_color};
            border-radius: 12px;
            padding: 20px 28px;
            margin-bottom: 20px;
        ">
            <h2 style="margin:0; color:{risk_color};">
                {_risk_badge(risk)} Overall Order Risk
            </h2>
            <p style="margin:6px 0 0; color:#555; font-size:0.95em;">
                The <strong>worst-case delay</strong> across all items in this order is
                <strong>{summary['max_delay']:.1f} days</strong>
                (driven by Item {summary['worst_item']}).
                {summary['items_at_risk']} of {summary['total_items']} item(s) flagged at risk
                ({summary['risk_pct']:.0%}).
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Key metric cards ───────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🕐 Max Delay",         f"{summary['max_delay']:.1f} days")
        m2.metric("⚠️ Items at Risk",      f"{summary['items_at_risk']} / {summary['total_items']}")
        m3.metric("📊 Avg Late Probability", f"{summary['avg_probability']:.1%}")
        m4.metric("🎯 Overall Risk",       risk)

        # ── Expandable item-wise breakdown ─────────────────────────────────────
        with st.expander("📋 View Item-by-Item Delay Breakdown", expanded=False):
            res_df = pd.DataFrame(results)

            # Build display table
            display_rows = []
            for r in results:
                level_icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(r['risk_level'], "⚪")
                worst = "⭐ Worst" if r['item_number'] == summary['worst_item'] and summary['max_delay'] > 0 else ""
                display_rows.append({
                    "Item #":            r['item_number'],
                    "Prediction":        "🔴 LATE" if r['late_prediction'] == 1 else "🟢 ON TIME",
                    "Late Probability":  f"{r['late_probability']:.1%}",
                    "Predicted Delay":   f"{r['predicted_delay_days']:.1f} days",
                    "Risk Level":        f"{level_icon} {r['risk_level']}",
                    "Note":              worst,
                })

            disp_df = pd.DataFrame(display_rows)
            st.dataframe(disp_df, use_container_width=True, hide_index=True)

            # Highlight worst item
            if summary['max_delay'] > 0:
                worst_res = next(r for r in results if r['item_number'] == summary['worst_item'])
                st.info(
                    f"**Item {summary['worst_item']}** is the critical path item — "
                    f"predicted delay of **{worst_res['predicted_delay_days']:.1f} days** "
                    f"with a late probability of **{worst_res['late_probability']:.1%}**."
                )

        # ── Download ───────────────────────────────────────────────────────────
        res_df_full = pd.DataFrame(results)
        st.download_button(
            label="📥 Download Full Prediction Results (CSV)",
            data=res_df_full.to_csv(index=False),
            file_name="order_risk_prediction.csv",
            mime="text/csv",
        )

        # ── Model information (collapsed by default) ───────────────────────────
        with st.expander("ℹ️ How does the prediction work?"):
            st.markdown("""
            **Two-stage AI prediction pipeline:**

            | Stage | Model | Role |
            |---|---|---|
            | **Stage 1** | Random Forest Classifier | Estimates the *probability* of late delivery |
            | **Stage 2** | XGBoost Regressor | If Stage 1 flags a risk, predicts *how many days late* |

            **Threshold:** Stage 1 flags a shipment as at-risk if the late-delivery probability ≥ **20%** (tuned for high recall — we prefer to over-alert than to miss a real delay).

            **41 predictors used:** Shipping mode, market, country, region, customer segment, product details, order timing, country-level infrastructure scores (LPI), governance indicators (WGI), economic conditions (GDP, CPI), diesel prices, disaster events, and public holidays.
            """)
