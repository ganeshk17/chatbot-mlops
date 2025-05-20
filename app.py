from flask import Flask, render_template, request, jsonify
from chat import get_response
import mlflow
import json
import os
import datetime

app = Flask(__name__)

# File to store chat logs for artifact 
CHAT_LOG_FILE = "./chat_logs.json"

# Ensure chat log file exists
if not os.path.exists(CHAT_LOG_FILE):
    open(CHAT_LOG_FILE, 'w').close()

# Start an MLflow run for inference logging (nested=True allows nesting under training if needed)
inference_run = mlflow.start_run(run_name="chat_inference_run", nested=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json.get('message')
    response = get_response(user_msg)

    # Get current date and time (no milliseconds)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Append input-output pair with timestamp as JSON line
    record = {
        "timestamp": timestamp,
        "user_message": user_msg,
        "bot_response": response
    }
    with open(CHAT_LOG_FILE, "a", encoding='utf-8') as f:
        f.write(json.dumps(record) + "\n")

    # Log/update the artifact (overwrite previous)
    try:
        mlflow.log_artifact(CHAT_LOG_FILE, artifact_path="chat_logs")
    except Exception as e:
        print(f"MLflow log_artifact error: {e}")

    return jsonify({'response': response})

if __name__ == '__main__':
    try:
        app.run(port=5000, debug=True)
    finally:
        mlflow.end_run()
