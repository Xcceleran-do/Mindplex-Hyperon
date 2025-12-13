import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY4")
print(f"API Key found: {'Yes' if api_key else 'No'}")

if not api_key:
    print("Error: GEMINI_API_KEY4 not found in environment.")
    exit(1)

try:
    genai.configure(api_key=api_key)
    print("Listing available models...")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")
