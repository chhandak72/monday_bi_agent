import streamlit as st
import requests
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI

# ----------------------------
# CONFIG
# ----------------------------

load_dotenv()

MONDAY_API_KEY = os.getenv("MONDAY_API_KEY")
DEALS_BOARD_ID = os.getenv("DEALS_BOARD_ID")
WORK_ORDERS_BOARD_ID = os.getenv("WORK_ORDERS_BOARD_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MONDAY_URL = "https://api.monday.com/v2"
client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="Monday BI Agent", layout="wide")
st.title("📊 Monday.com Business Intelligence Agent")
st.markdown("Founder-level AI business intelligence across Sales & Operations")

# ----------------------------
# FETCH BOARD DATA
# ----------------------------

def fetch_board(board_id):

    query = f"""
    {{
      boards(ids: {board_id}) {{
        items_page(limit: 500) {{
          items {{
            name
            column_values {{
              text
              column {{
                title
              }}
            }}
          }}
        }}
      }}
    }}
    """

    headers = {"Authorization": MONDAY_API_KEY}
    response = requests.post(MONDAY_URL, json={"query": query}, headers=headers)

    if response.status_code != 200:
        raise Exception("Failed to fetch board")

    data = response.json()
    items = data["data"]["boards"][0]["items_page"]["items"]

    rows = []
    for item in items:
        row = {"Item Name": item["name"]}
        for col in item["column_values"]:
            row[col["column"]["title"]] = col["text"]
        rows.append(row)

    return pd.DataFrame(rows)

# ----------------------------
# CLEANING
# ----------------------------

def clean_deals(df):

    df = df.replace("", None)

    if "Masked Deal value" in df.columns:
        df["Masked Deal value"] = pd.to_numeric(
            df["Masked Deal value"], errors="coerce"
        )

    if "Closure Probability Numeric" in df.columns:
        df["Closure Probability Numeric"] = pd.to_numeric(
            df["Closure Probability Numeric"], errors="coerce"
        )

    if "Sector/service" in df.columns:
        df["Sector/service"] = (
            df["Sector/service"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

    return df


def clean_work_orders(df):

    df = df.replace("", None)

    for col in df.columns:
        if "amount" in col.lower():
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# ----------------------------
# SALES LOGIC
# ----------------------------

def calculate_pipeline(df):
    open_deals = df[df["Deal Status"] != "Closed Won"]

    total_pipeline = open_deals["Masked Deal value"].sum()

    weighted_pipeline = (
        open_deals["Masked Deal value"]
        * open_deals.get("Closure Probability Numeric", 0)
    ).sum()

    return total_pipeline, weighted_pipeline


def revenue_by_sector(df):
    closed = df[df["Deal Status"] == "Closed Won"]

    return (
        closed.groupby("Sector/service")["Masked Deal value"]
        .sum()
        .sort_values(ascending=False)
    )

# ----------------------------
# OPERATIONS LOGIC
# ----------------------------

def work_order_metrics(df):

    total_orders = len(df)

    status_counts = (
        df["Status"].value_counts()
        if "Status" in df.columns else {}
    )

    completed = status_counts.get("Completed", 0)
    in_progress = status_counts.get("In Progress", 0)
    delayed = status_counts.get("Delayed", 0)

    return total_orders, completed, in_progress, delayed

# ----------------------------
# DASHBOARD
# ----------------------------

def build_dashboard(deals_df, work_df):

    st.subheader("📈 Executive Dashboard")

    pipeline, weighted = calculate_pipeline(deals_df)

    closed = deals_df[deals_df["Deal Status"] == "Closed Won"]
    revenue = closed["Masked Deal value"].sum()

    total_orders, completed, in_progress, delayed = work_order_metrics(work_df)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Pipeline", f"₹{pipeline:,.0f}")
    col2.metric("Weighted Pipeline", f"₹{weighted:,.0f}")
    col3.metric("Closed Revenue", f"₹{revenue:,.0f}")
    col4.metric("Work Orders", total_orders)

    # Revenue by sector
    sector_data = revenue_by_sector(deals_df)
    if not sector_data.empty:
        st.subheader("Revenue by Sector")
        st.bar_chart(sector_data)

    # Work order status
    if "Status" in work_df.columns:
        st.subheader("Work Order Status Distribution")
        st.bar_chart(work_df["Status"].value_counts())

    st.subheader("⚠ Data Quality Overview")

    missing_prob = (
        deals_df["Closure Probability Numeric"].isna().mean() * 100
        if "Closure Probability Numeric" in deals_df.columns else 0
    )

    missing_value = (
        deals_df["Masked Deal value"].isna().mean() * 100
        if "Masked Deal value" in deals_df.columns else 0
    )

    st.write(f"- {missing_prob:.1f}% deals missing probability")
    st.write(f"- {missing_value:.1f}% deals missing deal value")

# ----------------------------
# LEADERSHIP SUMMARY
# ----------------------------

def generate_leadership_summary(deals_df, work_df):

    pipeline, weighted = calculate_pipeline(deals_df)
    revenue = deals_df[
        deals_df["Deal Status"] == "Closed Won"
    ]["Masked Deal value"].sum()

    total_orders, completed, in_progress, delayed = work_order_metrics(work_df)

    return f"""
### 📊 Leadership Summary

**Sales**
- Total Pipeline: ₹{pipeline:,.0f}
- Weighted Forecast: ₹{weighted:,.0f}
- Closed Revenue: ₹{revenue:,.0f}

**Operations**
- Total Work Orders: {total_orders}
- Completed: {completed}
- In Progress: {in_progress}
- Delayed: {delayed}

---

### Recommendations
- Improve probability capture for better forecasting
- Focus on late-stage deal conversion
- Investigate delayed work orders
"""

# ----------------------------
# QUERY INTERPRETER
# ----------------------------

def interpret_query(query):

    prompt = f"""
    Classify this query into:
    - pipeline
    - revenue
    - operations
    - leadership
    - sector
    - general

    Query: {query}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.lower()

# ----------------------------
# MAIN FLOW
# ----------------------------

try:
    with st.spinner("Fetching live data from monday.com..."):
        deals_df = clean_deals(fetch_board(DEALS_BOARD_ID))
        work_df = clean_work_orders(fetch_board(WORK_ORDERS_BOARD_ID))

except Exception:
    st.error("⚠ Unable to fetch monday.com data.")
    st.stop()

tab1, tab2 = st.tabs(["📊 Dashboard", "🤖 Chat Mode"])

with tab1:
    build_dashboard(deals_df, work_df)

with tab2:

    query = st.text_input("Ask a business question:")

    if query:

        intent = interpret_query(query)

        if "pipeline" in intent:
            pipeline, weighted = calculate_pipeline(deals_df)
            st.metric("Total Pipeline", f"₹{pipeline:,.0f}")
            st.metric("Weighted Forecast", f"₹{weighted:,.0f}")

        elif "revenue" in intent:
            revenue = deals_df[
                deals_df["Deal Status"] == "Closed Won"
            ]["Masked Deal value"].sum()
            st.metric("Closed Revenue", f"₹{revenue:,.0f}")

        elif "operations" in intent:
            total_orders, completed, in_progress, delayed = work_order_metrics(work_df)
            st.write(f"Total Orders: {total_orders}")
            st.write(f"Completed: {completed}")
            st.write(f"In Progress: {in_progress}")
            st.write(f"Delayed: {delayed}")

        elif "leadership" in intent:
            st.markdown(generate_leadership_summary(deals_df, work_df))

        elif "sector" in intent:
            st.bar_chart(revenue_by_sector(deals_df))

        else:
            st.write("Could you clarify your request?")
