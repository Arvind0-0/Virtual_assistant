from flask import Flask, request, jsonify
import torch
import pickle
import sqlite3
from transformers import BertTokenizer

# Load Model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define Intents
intents = ["schedule_meeting", "set_reminder", "get_weather", "unknown"]

# Flask App
app = Flask(__name__)

# Database Setup
def create_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS reminders (id INTEGER PRIMARY KEY, message TEXT, time TEXT)")
    conn.commit()
    conn.close()

create_db()

# Predict Intent
def predict_intent(text):
    tokens = tokenizer(text, padding="max_length", max_length=20, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(tokens["input_ids"])
    predicted_intent = intents[torch.argmax(outputs.logits)]
    return predicted_intent

# API Routes
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    user_input = data.get("text", "")
    intent = predict_intent(user_input)
    return jsonify({"intent": intent})

@app.route("/set_reminder", methods=["POST"])
def set_reminder():
    data = request.json
    message = data.get("message")
    time = data.get("time")

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO reminders (message, time) VALUES (?, ?)", (message, time))
    conn.commit()
    conn.close()

    return jsonify({"message": "Reminder set successfully!"})

if __name__ == "__main__":
    app.run(debug=True)
