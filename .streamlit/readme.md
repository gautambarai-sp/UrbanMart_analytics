# UrbanMart — Minimal Streamlit KPI Dashboard

A lightweight, single-page Streamlit dashboard to explore UrbanMart sales data.
The app auto-detects useful filters and visualizations and is optimized for Streamlit Cloud.

## How to use

1. Place your dataset at `data/urbanmart_sales.csv` OR upload via the app.
   - The app expects common columns such as: `date` (or a date-like column), `quantity`, `unit_price`, `discount_applied` (optional), `product_category`, `product_name`, `store_location`, `channel`, `bill_id`.
   - If `line_revenue` isn't present, the app will attempt to compute it as `(quantity * unit_price) - discount_applied`.

2. Run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
