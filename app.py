# app.py
"""
Minimal Streamlit dashboard for UrbanMart sales dataset.
Auto-detects column types and builds only meaningful filters and visualizations.
Lightweight and Streamlit Cloud friendly.
"""

from typing import Dict, List, Tuple
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="UrbanMart — Minimal KPI", layout="wide")

# -------------------------
# Helpers & caching
# -------------------------
@st.cache_data
def load_data(path: str = "data/urbanmart_sales.csv") -> pd.DataFrame:
    """Load CSV and do light cleaning"""
    df = pd.read_csv(path, low_memory=False)
    return df

def try_parse_dates(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Try to detect and parse date-like columns; return list of parsed date columns"""
    date_cols = []
    for col in df.columns:
        # heuristic: column name includes 'date' or dtype object and parseable
        if "date" in col.lower() or "day" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                if df[col].notna().sum() > 0:
                    date_cols.append(col)
            except Exception:
                continue
        elif df[col].dtype == object:
            # try parse a sample
            try:
                parsed = pd.to_datetime(df[col].dropna().iloc[:20], errors="coerce")
                if parsed.notna().sum() > 0:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    date_cols.append(col)
            except Exception:
                continue
    return df, date_cols

def compute_line_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Create line_revenue if missing using quantity * unit_price - discount_applied"""
    if "line_revenue" not in df.columns:
        if {"quantity", "unit_price"}.issubset(df.columns):
            df["discount_applied"] = df.get("discount_applied", 0)
            try:
                df["line_revenue"] = (pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
                                      * pd.to_numeric(df["unit_price"], errors="coerce").fillna(0)
                                      - pd.to_numeric(df["discount_applied"], errors="coerce").fillna(0))
            except Exception:
                df["line_revenue"] = 0
        else:
            # fallback numeric columns sum as proxy if available
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if numeric_cols:
                df["line_revenue"] = df[numeric_cols].sum(axis=1).fillna(0)
            else:
                df["line_revenue"] = 0
    else:
        df["line_revenue"] = pd.to_numeric(df["line_revenue"], errors="coerce").fillna(0)
    return df

def summarize_missing_and_uniques(df: pd.DataFrame) -> pd.DataFrame:
    """Return a small table summarizing missing counts and unique counts for each column"""
    summary = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing": [int(df[c].isna().sum()) for c in df.columns],
        "unique": [int(df[c].nunique(dropna=True)) for c in df.columns]
    })
    return summary

def make_filters(df: pd.DataFrame, date_cols: List[str]) -> Dict:
    """
    Auto-generate filter widgets in sidebar.
    Rules:
      - Date columns => date_input range
      - Categorical columns (unique <= 50) => multiselect
      - Numeric columns => range slider (only for columns where unique > 10)
    """
    st.sidebar.header("Filters")
    filters = {}
    # Date filters first
    for d in date_cols:
        min_d = df[d].min()
        max_d = df[d].max()
        if pd.notna(min_d) and pd.notna(max_d):
            default = (min_d.date(), max_d.date())
            sel = st.sidebar.date_input(f"{d} range", value=default)
            # sel might be tuple or single date
            if isinstance(sel, tuple) and len(sel) == 2:
                filters[d] = (pd.to_datetime(sel[0]), pd.to_datetime(sel[1]))
            else:
                filters[d] = (pd.to_datetime(sel), pd.to_datetime(sel))
    # Categorical filters (reasonable cardinality)
    for col in df.columns:
        if col in date_cols:
            continue
        if df[col].dtype == object or pd.api.types.is_categorical_dtype(df[col]):
            uniq = df[col].dropna().unique()
            if 1 < len(uniq) <= 50:
                sel = st.sidebar.multiselect(f"{col}", options=sorted([str(x) for x in uniq]), default=list(sorted([str(x) for x in uniq]))[:6])
                # store as strings for filtering
                filters[col] = sel
        # numeric sliders
        elif pd.api.types.is_numeric_dtype(df[col]):
            # skip small uniques (like ID)
            if df[col].nunique(dropna=True) > 10:
                mn = float(df[col].min())
                mx = float(df[col].max())
                if np.isfinite(mn) and np.isfinite(mx) and mn != mx:
                    lo, hi = st.sidebar.slider(f"{col} range", min_value=mn, max_value=mx, value=(mn, mx))
                    filters[col] = (lo, hi)
    # Quick reset
    if st.sidebar.button("Clear filters"):
        st.experimental_rerun()
    return filters

def apply_filters(df: pd.DataFrame, filters: Dict, date_cols: List[str]) -> pd.DataFrame:
    df2 = df.copy()
    for k, v in filters.items():
        if k in date_cols:
            start, end = v
            df2 = df2[(df2[k] >= pd.to_datetime(start)) & (df2[k] <= pd.to_datetime(end))]
        else:
            if isinstance(v, list):
                # categorical multiselect (strings)
                if len(v) == 0:
                    # if empty selection, keep none
                    df2 = df2.iloc[0:0]
                else:
                    # cast to str for matching to avoid dtype mismatch
                    df2 = df2[df2[k].astype(str).isin(v)]
            elif isinstance(v, tuple) and len(v) == 2:
                lo, hi = v
                df2 = df2[(pd.to_numeric(df2[k], errors="coerce").fillna(-np.inf) >= lo) &
                          (pd.to_numeric(df2[k], errors="coerce").fillna(np.inf) <= hi)]
    return df2

# -------------------------
# Main app
# -------------------------
st.title("UrbanMart — Minimal Sales KPIs")

col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("**Load data**")
    uploaded = st.file_uploader("Upload CSV (optional). If none, app loads data/urbanmart_sales.csv", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded, low_memory=False)
            st.success("Uploaded dataset loaded")
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            st.stop()
    else:
        # default path; user should put file in data/urbanmart_sales.csv
        try:
            df = load_data("data/urbanmart_sales.csv")
        except FileNotFoundError:
            st.info("Put your CSV at data/urbanmart_sales.csv or upload one above.")
            st.stop()

with col2:
    st.markdown("Small auto-summary")
    st.write(f"Rows: **{df.shape[0]}** | Columns: **{df.shape[1]}**")
    st.write("Preview:")
    st.dataframe(df.head(5), use_container_width=True)

# parse dates and ensure revenue column
df, date_cols = try_parse_dates(df)
df = compute_line_revenue(df)

# Show summary table
summary_df = summarize_missing_and_uniques(df)
st.markdown("### Column summary")
st.dataframe(summary_df, use_container_width=True)

# Sidebar filters
filters = make_filters(df, date_cols)

# Apply filters
df_filtered = apply_filters(df, filters, date_cols)

st.markdown("---")

# KPI row (calculate only if line_revenue present)
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

total_revenue = float(df_filtered["line_revenue"].sum()) if "line_revenue" in df_filtered.columns else 0.0
units_sold = int(df_filtered["quantity"].sum()) if "quantity" in df_filtered.columns else int(df_filtered.shape[0])
unique_bills = int(df_filtered["bill_id"].nunique()) if "bill_id" in df_filtered.columns else df_filtered.shape[0]
aov = (total_revenue / unique_bills) if unique_bills and unique_bills > 0 else 0.0

kpi_col1.metric("Total revenue", f"₹{total_revenue:,.0f}")
kpi_col2.metric("Units sold", f"{units_sold:,}")
kpi_col3.metric("Unique bills", f"{unique_bills:,}")
kpi_col4.metric("Avg order value", f"₹{aov:,.2f}")

st.markdown("### Visualizations")

# Small set of lightweight visualizations depending on columns present
viz_col1, viz_col2 = st.columns(2)

# Revenue over time if date exists
with viz_col1:
    if date_cols:
        # choose primary date column
        primary_date = date_cols[0]
        ts = (df_filtered
              .dropna(subset=[primary_date, "line_revenue"])
              .groupby(pd.Grouper(key=primary_date, freq="W"))["line_revenue"]
              .sum()
              .reset_index())
        if ts.shape[0] > 0:
            fig = px.line(ts, x=primary_date, y="line_revenue", title="Revenue over time (weekly)")
            fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No date/time data available after filters to plot time series.")
    else:
        st.info("No date-like column detected for time series.")

# Category distribution if product_category exists
with viz_col2:
    if "product_category" in df_filtered.columns:
        cat = (df_filtered
               .groupby("product_category")["line_revenue"]
               .sum()
               .reset_index()
               .sort_values("line_revenue", ascending=False))
        if cat.shape[0] > 0:
            fig2 = px.bar(cat, x="line_revenue", y="product_category", orientation="h", title="Revenue by Category")
            fig2.update_layout(margin=dict(l=20, r=20, t=30, b=20), yaxis={"categoryorder":"total ascending"})
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No 'product_category' column to show category breakdown.")

# Top N products (if product_name exists)
if "product_name" in df_filtered.columns:
    topn = 10
    prod = (df_filtered.groupby("product_name")["line_revenue"].sum().reset_index()
            .sort_values("line_revenue", ascending=False).head(topn))
    if prod.shape[0] > 0:
        st.markdown("Top products by revenue")
        fig3 = px.bar(prod, x="line_revenue", y="product_name", orientation="h", title=f"Top {topn} products")
        fig3.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig3, use_container_width=True)

# Channel mix pie if channel column exists
if "channel" in df_filtered.columns:
    ch = df_filtered.groupby("channel")["line_revenue"].sum().reset_index()
    if ch.shape[0] > 0:
        fig4 = px.pie(ch, names="channel", values="line_revenue", title="Channel revenue mix")
        st.plotly_chart(fig4, use_container_width=True)

# Store performance small table/chart
if "store_location" in df_filtered.columns:
    store_rev = (df_filtered.groupby("store_location")["line_revenue"].sum().reset_index()
                 .sort_values("line_revenue", ascending=False))
    st.markdown("Revenue by store location (top 10)")
    st.dataframe(store_rev.head(10), use_container_width=True)

# Insights section: auto-generated small insights
st.markdown("---")
st.markdown("### Quick Insights (auto)")
insights = []
if total_revenue > 0:
    insights.append(f"Total revenue in current filter: ₹{total_revenue:,.0f}")
if "product_category" in df_filtered.columns:
    top_cat = df_filtered.groupby("product_category")["line_revenue"].sum().sort_values(ascending=False)
    if top_cat.shape[0] > 0:
        topc_name = top_cat.index[0]
        topc_pct = (top_cat.iloc[0] / top_cat.sum()) * 100
        insights.append(f"Top category: {topc_name} ({topc_pct:.1f}% of filtered revenue)")
if "channel" in df_filtered.columns:
    chmix = df_filtered.groupby("channel")["line_revenue"].sum().sort_values(ascending=False)
    if chmix.shape[0] > 0:
        insights.append(f"Top channel: {chmix.index[0]} ({(chmix.iloc[0]/chmix.sum()*100):.1f}%)")

if len(insights) == 0:
    st.write("No automated insights available for current dataset.")
else:
    for i in insights:
        st.write(f"- {i}")

# Footer
st.markdown("---")
st.caption("Lightweight dashboard autogenerated. Place your CSV in data/urbanmart_sales.csv or upload above.")
