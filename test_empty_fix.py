import requests
import time
import json
import sys

# Generate IDs
sid = f"regtest-empty-fix-{int(time.time())}"
uid = f"regtest-user-{int(time.time())}"

messages = [
    "My name is Jordan Hale.",
    "I work as a doctor at the General Hospital.",
    "I have two cats: Mango and Pixel.",
    "I drive a white 2018 Honda Civic.",
    "Pixel is allergic to chicken.",
    "I sold the Civic and bought a Tesla.",
    "What's my name and what do I drive?"
]

print(f"Session ID: {sid}")
print(f"User ID: {uid}\n")

for i, msg in enumerate(messages, 1):
    print(f"Message {i}: {msg}")
    payload = {
        "session_id": sid,
        "user_id": uid,
        "content": msg,
        "mode": "auto",
        "use_mem0": True,
        "recall_k": 6
    }
    try:
        response = requests.post("http://localhost:7100/generate", json=payload, timeout=180)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            resp_content = data.get("response", "")
            print(f"Response (first 200 chars): {resp_content[:200]}")
            if not resp_content:
                print("WARNING: Empty response field!")
        else:
            print(f"Error Body: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")
    print("-" * 20)

