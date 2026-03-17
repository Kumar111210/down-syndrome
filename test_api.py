# test_api.py
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv()

# Get the API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Basic safety check
if not api_key:
    print("Error: OPENAI_API_KEY not found!")
    print("Please do one of the following:")
    print("  1. Create a file called .env in this folder with this line:")
    print("     OPENAI_API_KEY=sk-proj-YourActualKeyHere...")
    print("  2. Or set it manually in your terminal before running:")
    print("     set OPENAI_API_KEY=sk-proj-...     (Windows CMD)")
    print("     $env:OPENAI_API_KEY = 'sk-proj-...'  (PowerShell)")
    exit(1)

# If we reach here → we have a key
print("API key loaded successfully (length:", len(api_key), "characters)")

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

# ────────────────────────────────────────────────
# Example: Simple test call to make sure it works
# ────────────────────────────────────────────────
try:
    response = client.chat.completions.create(
      model="gpt-3.5-turbo",   # ← change to this for super-cheap test        # or "gpt-3.5-turbo", "gpt-4o", etc.
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello from Hyderabad!' in a pirate voice 🏴‍☠️"}
        ],
        max_tokens=200,
        temperature=0.7
    )
    
    print("\nSuccess! Here's the response:")
    print("-" * 50)
    print(response.choices[0].message.content.strip())
    print("-" * 50)

except Exception as e:
    print("Error during API call:")
    print(str(e))