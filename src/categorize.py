import pandas as pd
import joblib
from src.preprocess import clean_desc

def categorize_transactions(input_path, model_path, output_path):
    df = pd.read_csv(input_path)
    model = joblib.load(model_path)

    df['Cleaned_Description'] = df['Description'].apply(clean_desc)
    df['Predicted_Category'] = model.predict(df['Cleaned_Description'])

    df.to_csv(output_path, index=False)
    print(f"Categorized transactions saved to {output_path}")
