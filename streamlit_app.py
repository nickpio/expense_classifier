import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

from src import preprocess, categorize
from config import MODEL_FILE, INCOME_CATEGORIES
from src.trainmodel import retrain_model

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
    st.subheader("Correct any wrong categories below:")
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    if st.button("Save & retrain"):
        labeled_path = os.path.join("data", "labeled", "corrections.csv")

        if os.path.exists(labeled_path):
            old_df = pd.read_csv(labeled_path)
            edited_df = edited_df.rename(columns={"Predicted_Category": "Category"})
            combined = pd.concat([old_df, edited_df], ignore_index=True)
            combined.to_csv(labeled_path, index=False)
        else:
            edited_df.to_csv(labeled_path, index=False)
        
        st.success(f"Corrections saved to {labeled_path}")
        st.info("Retraining model...")
        retrain_model_from_all_labeled(labeled_path)
    
    st.subheader("Spending breakdown by category")
    category_totals = categorized_df.groupby('Predicted_Category')['amount'].sum().sort_values()
    st.bar_chart(category_totals)

    spending_df = categorized_df[~categorized_df["Predicted_Category"].isin(INCOME_CATEGORIES)]
    spending_totals = spending_df.groupby('Predicted_Category')['amount'].sum().abs().sort_values(ascending=False)

    st.subheader("Spending distribution (excluding income)")

    fig, ax = plt.subplots()
    ax.pie(
        spending_totals,
        labels=spending_totals.index,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white"}
    )
    ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)
    
    csv = categorized_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Categorized CSV",
        data=csv,
        file_name="categorized_transactions.csv",
        mime="text/csv",
    )