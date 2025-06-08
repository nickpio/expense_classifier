import pandas as pd
import joblib
from src.preprocess import clean_desc

def categorize_transactions(input_file, model_file, output_file):
    df = pd.read_csv(input_file)
    model = joblib.load(model_file)

    df['Cleaned_Description'] = df['Description'].apply(clean_desc)
    df['Predicted_Category'] = model.predict(df['Cleaned_Description'])

    df.to_csv(output_file, index=False)
    print(f"Categorized transactions saved to {output_file}")

def categorize_transactions_from_df(df, model_file):
    model = joblib.load(model_file)

    X_test = df['Cleaned_Description']
    predictions = model.predict(X_test)
    df['Predicted_Category'] = predictions
    return df