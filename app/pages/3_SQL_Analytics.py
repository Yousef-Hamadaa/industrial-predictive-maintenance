import os
import sys
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.database import DatabaseManager

st.set_page_config(page_title="SQL Analytics", page_icon="🗄️", layout="wide")

st.title("🗄️ SQL Data Warehouse & Analytical Queries")
st.markdown("Interactive execution of 11 advanced SQL queries demonstrating **JOINs, CTEs, Window Functions (RANK, SUM OVER, Moving Avg)** and group aggregations.")

sql_file_path = "sql/queries.sql"

db = DatabaseManager()

if os.path.exists(sql_file_path):
    with open(sql_file_path, "r") as f:
        full_sql = f.read()

    queries = full_sql.split(";")
    query_list = [q.strip() for q in queries if q.strip() and not q.strip().startswith("-- ==")]

    query_options = {
        "Query 1: Failure Rate by Product Type (CTE + Group By)": query_list[0] if len(query_list) > 0 else "",
        "Query 2: Thermal Stress Analysis (Process vs Air Temp Diff)": query_list[1] if len(query_list) > 1 else "",
        "Query 3: Power Distribution per Failure Category": query_list[2] if len(query_list) > 2 else "",
        "Query 4: Top Worn Machines at Failure (Window Function: RANK)": query_list[3] if len(query_list) > 3 else "",
        "Query 5: Moving Average RPM & Torque (Window Function: Moving Avg)": query_list[4] if len(query_list) > 4 else "",
        "Query 6: Cumulative Tool Wear per Type (Window Function: SUM OVER)": query_list[5] if len(query_list) > 5 else "",
        "Query 7: High-Risk Machine Criteria CTE": query_list[6] if len(query_list) > 6 else "",
        "Query 8: Failure Rate Binned by Tool Wear Ranges": query_list[7] if len(query_list) > 7 else "",
        "Query 9: Multi-Failure Co-occurrence Matrix": query_list[8] if len(query_list) > 8 else "",
        "Query 10: Product Type Reliability Ranking (DENSE_RANK)": query_list[9] if len(query_list) > 9 else "",
        "Query 11: Recent Prediction Maintenance Logs": query_list[10] if len(query_list) > 10 else ""
    }

    selected_query_title = st.selectbox("Select Analytical SQL Query", list(query_options.keys()))
    selected_sql = query_options[selected_query_title]

    st.subheader("📜 SQL Statement")
    st.code(selected_sql, language="sql")

    if st.button("▶️ Execute Query"):
        try:
            df_res = db.execute_query(selected_sql)
            st.subheader(f"📊 Query Results ({len(df_res)} rows)")
            st.dataframe(df_res, use_container_width=True)
            
            # Interactive Chart if applicable
            if not df_res.empty and len(df_res.columns) >= 2:
                num_cols = df_res.select_dtypes(include=['float64', 'int64']).columns
                if len(num_cols) > 0:
                    st.bar_chart(df_res, x=df_res.columns[0], y=num_cols[0])
        except Exception as e:
            st.error(f"Error executing query: {e}")
else:
    st.warning("SQL queries file not found.")
