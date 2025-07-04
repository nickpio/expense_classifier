import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") #for OpenAI API usage, if needed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


TRAINING_DATA = os.path.join(BASE_DIR, "data", "labeled", "realstatement.csv")
RAW_DATA_FOLDER = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_FOLDER = os.path.join(BASE_DIR, "data", "processed")
MODEL_FILE = os.path.join(BASE_DIR, "models", "expense_classifier_model.pkl")
INCOME_CATEGORIES = ["Income", "Salary", "Paycheck", "Paycheque", "Deposit", "Transfer In", "Refund", "Reimbursement"]
