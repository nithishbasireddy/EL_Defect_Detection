"""Quick test to verify your Gemini API key works."""
import os
from src.agents.gemini_client import ask_gemini

key = os.environ.get("GEMINI_API_KEY", "")
print(f"Key loaded: {'Yes' if key else 'No'} ({key[:10]}...)")

print("\nTesting Gemini API...")
result = ask_gemini("Say 'API is working' in exactly those 3 words.")
print(f"Response: {result}")
