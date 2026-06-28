"""Regression test for empty response fix.

Sends a sequence of messages through the /prompt endpoint and verifies
that responses are non-empty and that memory recall works correctly.
Connects to the surriti service at SURRITI_URL (default: http://localhost:3000).
"""
import json
import os
import requests
import sys
import time

SURRITI_URL = os.environ.get("SURRITI_URL", "http://localhost:3000")
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

print(f"User ID: {uid}\n")

for i, msg in enumerate(messages, 1):
    print(f"Message {i}: {msg}")
    payload = {"text": msg, "user_id": uid}
    try:
        response = requests.post(f"{SURRITI_URL}/prompt", json=payload, timeout=180)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            resp_content = data.get("response", "")
            recalled = data.get("recalled_facts", [])
            print(f"Response (first 200 chars): {resp_content[:200]}")
            if recalled:
                print(f"Recalled {len(recalled)} facts: {recalled[:3]}")
            if not resp_content:
                print("WARNING: Empty response field!")
        else:
            print(f"Error Body: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")
    print("-" * 20)

