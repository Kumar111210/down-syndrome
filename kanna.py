import os
from google.genai import TextGenerationModel, configure

# Get GEMINI API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY not found!")
    exit(1)

# Configure the API
configure(api_key=GEMINI_API_KEY)
model = TextGenerationModel()

# Test prompt
prompt = "Say hello in one sentence."

try:
    response = model.generate(prompt=prompt)
    print("✅ Gemini API working!")
    print("Response:", response.output_text)

except Exception as e:
    print("❌ Error calling Gemini API:", e)
