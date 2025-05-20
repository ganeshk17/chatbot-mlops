import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch
import os
import pickle
import time

# Load intents
with open('data/intents.json') as f:
    data = json.load(f)

texts = []
labels = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        texts.append(pattern)
        labels.append(intent['tag'])

# Encode labels
le = LabelEncoder()
labels_enc = le.fit_transform(labels)

# Tokenizer and model
model_name = "huawei-noah/TinyBERT_General_4L_312D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(le.classes_))

# Dataset
class IntentDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Train/val split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels_enc, test_size=0.2, random_state=42)

train_dataset = IntentDataset(train_texts, train_labels)
val_dataset = IntentDataset(val_texts, val_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Optimizer
learning_rate = 5e-5
optimizer = AdamW(model.parameters(), lr=learning_rate)
optimizer_name = "AdamW"

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

epochs = 3
batch_size = 8
start_time = time.time()

# MLflow tracking for training
with mlflow.start_run(run_name="intent_classification_training"):
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("optimizer", optimizer_name)
    mlflow.log_param("dataset_version", "1.0")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
        mlflow.log_metric("loss", avg_loss, step=epoch + 1)
        mlflow.log_metric("accuracy", accuracy, step=epoch + 1)

    training_time = time.time() - start_time
    mlflow.log_metric("training_time_sec", training_time)

    # Save model and tokenizer
    save_path = './model/bert_intent_model'
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Save label encoder
    with open('./model/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    mlflow.log_artifact('./model/label_encoder.pkl')

    # Log model to MLflow
    mlflow.pytorch.log_model(model, "model")
