import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib


def train_classifier(csv_path, model_path):
    df = pd.read_csv(csv_path)
    if 'Category' not in df.columns:
        raise ValueError("Training data must include a 'Category' column")

    X = df['Cleaned_Description'] if 'Cleaned_Description' in df else df['Description']
    y = df['Category']

    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X, y)

    joblib.dump(model, model_path)
    print(f"Model trained and saved to {model_path}")