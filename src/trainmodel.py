import pandas as pd
from src.preprocess import preprocess_csv
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import joblib
import os
import shutil
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from config import MODEL_FILE


def train_classifier(csv_path, model_path):
    df = pd.read_csv(csv_path)
    if 'Category' not in df.columns:
        raise ValueError("Training data must include a 'Category' column")

    X = df['Cleaned_Description'] if 'Cleaned_Description' in df else df['description']
    y = df['Category']

    model = make_pipeline(TfidfVectorizer(), LogisticRegression(class_weight='balanced', max_iter=1000))
    model.fit(X, y)

    joblib.dump(model, model_path)
    print(f"Model trained and saved to {model_path}")

def retrain_model(labeled_csv_path):
    # Make backup of corrections
    backup_dir = os.path.join("data", "backups")
    os.makedirs(backup_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_name = os.path.basename(labeled_csv_path).replace(".csv", "")
    backup_filename = f"{base_name}_{timestamp}.csv"
    backup_path = os.path.join(backup_dir, backup_filename)

    shutil.copy(labeled_csv_path, backup_path)
    print(f"📁 Backup saved to: {backup_path}")

    # Retrain model
    print(f"Retraining model from: {labeled_csv_path}")

    df = preprocess_csv(labeled_csv_path, is_training=True)
    df = df[['Cleaned_Description', 'Category']].dropna()

    X = df['Cleaned_Description']
    y = df['Category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", LogisticRegression(class_weight='balanced', max_iter=1000))
    ])

    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_FILE)
    print(f"Model retrained and saved to {MODEL_FILE}")

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:\n")
    print(report)

    return model
