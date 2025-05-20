import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import mlflow
import json
import os
import datetime

# Load model, tokenizer, label encoder
model_path = './model/bert_intent_model'
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

with open('./model/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Path for logging chat inputs/outputs
log_path = './model/chat_logs.json'
os.makedirs(os.path.dirname(log_path), exist_ok=True)

def get_response(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        tag = le.inverse_transform([pred])[0]

    # Get current date and time (no milliseconds)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Prepare the chat log entry with timestamp
    chat_entry = {
        "timestamp": timestamp,
        "input": text,
        "output": tag
    }

    # Append the entry to the JSONL file
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(chat_entry) + '\n')

    # Log last input/output as mlflow params (truncated to 100 chars)
    try:
        mlflow.log_param("last_user_input", text[:100])
        mlflow.log_param("last_bot_response", tag[:100])
    except Exception as e:
        print(f"MLflow log_param error: {e}")

    return tag
