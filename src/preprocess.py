import pandas as pd
import re

def clean_desc(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    return text


def preprocess_csv(filepath_or_df, is_training=True):
    """
    Preprocesses a CSV file or DataFrame.
    - Cleans descriptions
    - Validates required columns
    """
    if isinstance(filepath_or_df, str):
        df = pd.read_csv(filepath_or_df)
    else:
        df = filepath_or_df.copy()

    df.columns = df.columns.str.strip()  # Clean column names

    required_columns = ['Description', 'Amount']
    if is_training:
        required_columns.append('Category')

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV must contain column: '{col}'")

    df['Cleaned_Description'] = df['Description'].apply(clean_desc)
    return df
 
