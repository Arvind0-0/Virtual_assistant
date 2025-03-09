import streamlit as st
import requests

# API URL
API_URL = "http://127.0.0.1:5000"

st.title("AI Virtual Assistant 🤖")

# User Input
user_input = st.text_input("Enter your command:")

if st.button("Submit"):
    response = requests.post(f"{API_URL}/predict", json={"text": user_input})
    result = response.json()

    st.write(f"**Predicted Intent:** {result['intent']}")

    if result["intent"] == "set_reminder":
        reminder_msg = st.text_input("Reminder Message:")
        reminder_time = st.text_input("Reminder Time (YYYY-MM-DD HH:MM):")

        if st.button("Set Reminder"):
            res = requests.post(f"{API_URL}/set_reminder", json={"message": reminder_msg, "time": reminder_time})
            st.success(res.json()["message"])
