import streamlit as st
import pandas as pd
import os

from src import preprocess, categorize
from config import MODEL_FILE

st.set_page_config(page_title="Expense Classifier", layout="wide")

st.title("Expense Classifier")


uploaded_file = st.file_uploader("Upload a bank statement (.csv)", type=["csv"])

if uploaded_file is not None:
    st.success("File uploaded successfully!")

    df_raw = pd.read_csv(uploaded_file)
    df = df_raw.copy()

    try:
        df = preprocess.preprocess_csv(df, is_training=False)
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        st.stop()

    categorized_df = categorize.categorize_transactions_from_df(df, MODEL_FILE)
    
    st.subheader("Categorized Transactions")
    st.dataframe(categorized_df, use_container_width=True)

    st.subheader("Spending breakdown by category")
    category_totals = categorized_df.groupby('Predicted_Category')['Amount'].sum().sort_values()
    st.bar_chart(category_totals)

    csv = categorized_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Categorized CSV",
        data=csv,
        file_name="categorized_transactions.csv",
        mime="text/csv",
    )