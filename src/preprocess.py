import pandas as pd
import re

def clean_desc(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    return text


def preprocess_csv(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    if 'Description' not in df.columns or 'Amount' not in df.columns:
        raise ValueError("CSV must contain 'description' and 'Amount' columns")

    df['Cleaned_Description'] = df['Description'].apply(clean_desc)

    return df
 
