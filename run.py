import argparse
from src import preprocess, trainmodel, categorize
import os

from config import TRAINING_DATA, MODEL_FILE, OUTPUT_FOLDER
def main():
    parser = argparse.ArgumentParser(description="Expense Categorizer")
    parser.add_argument('--train', action='store_true', help='Train the model using labeled CSVs')
    parser.add_argument('--predict', type=str, help="CSV file with new transactions to categorize")
    args = parser.parse_args()

    if args.train:
        print("Preprocessing and training the model...")
        df = preprocess.preprocess_csv(TRAINING_DATA, is_training=True)
        df.to_csv(TRAINING_DATA, index=False)
        trainmodel.train_classifier(TRAINING_DATA, MODEL_FILE)
    elif args.predict:
        print(f"Preprocessing and predicting for: {args.predict}")
        df = preprocess.preprocess_csv(args.predict, is_training=False)
        df.to_csv(args.predict, index=False)

        output_path = os.path.join(OUTPUT_FOLDER, "categorized_output.csv")
        categorize.categorize_transactions(args.predict, MODEL_FILE, output_path)
        print(f"Done. Output saved to: {output_path}")

    else:
        parser.print_help()

    
if __name__ == "__main__":
    main()