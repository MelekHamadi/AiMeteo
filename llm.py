import requests

OLLAMA_URL = "http://172.29.64.1:11434/api/generate"
MODEL = "qwen2.5:7b"         
TIMEOUT = 120

def ask_llm(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 500   
        }
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json().get("response", "").strip()