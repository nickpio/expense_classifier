from src.preprocess import preprocess_csv
from src.trainmodel import train_classifier
from src.categorize import categorize_transactions

train_file = "data/labeled/labeled_transactions.csv"
model_file = "models/classifier.pkl"
new_data = "data/raw/example_bank_statement.csv"
output_file = "data/processed/categorized_transactions.csv"

print("Preprocessing training data...")
train_df = preprocess_csv(train_file)
train_df.to_csv(train_file, index=False)

print("Training model...")
train_classifier(train_file, model_file)

print("Categorizing new transactions...")
categorize_transactions(new_data, model_file, output_file)
print("Categorization complete. Results saved to", output_file)