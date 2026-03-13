import os
import requests

def get_gemini_key():
    return os.getenv("GEMINI_API_KEY")


def generate_embedding(text: str):
    embed_url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-embedding-001:embedContent?key={get_gemini_key()}"
    )

    payload = {
        "content": {
            "parts": [{"text": text}]
        },
        "outputDimensionality": 3072
    }

    response = requests.post(embed_url, json=payload, timeout=30)

    if response.status_code != 200:
        raise Exception(f"Gemini error: {response.text}")

    return response.json()["embedding"]["values"]