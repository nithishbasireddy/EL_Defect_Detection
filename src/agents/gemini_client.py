#Groq
import os
from groq import Groq


# Groq model — free, fast, 14400 requests/day
MODEL = "llama-3.3-70b-versatile"


def load_api_key():
    """Load Groq API key from .env file or environment variable."""
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key

    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    env_path = os.path.abspath(env_path)

    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("GROQ_API_KEY="):
                    return line.split("=", 1)[1]

    raise ValueError("GROQ_API_KEY not found in environment or .env file")


def ask_gemini(prompt, temperature=0.4):
    """Send a prompt to the LLM and return the response text.
    
    Function name kept as ask_gemini for backward compatibility
    with other agents that import it.
    """
    try:
        api_key = load_api_key()
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2048,
        )

        return response.choices[0].message.content

    except ValueError as e:
        return f"API Error: {str(e)}"

    except Exception as e:
        return f"LLM request failed: {str(e)}"
