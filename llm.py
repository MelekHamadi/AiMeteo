import requests
import subprocess

def get_ollama_url():
    """Détecte automatiquement la bonne URL Ollama depuis WSL."""
    # Essayer localhost d'abord
    for url in [
        "http://localhost:11434",
        "http://127.0.0.1:11434",
        "http://172.30.128.1:11434",
        "http://host.docker.internal:11434",
    ]:
        try:
            r = requests.get(f"{url}/api/tags", timeout=3)
            if r.status_code == 200:
                print(f"✅ Ollama trouvé sur : {url}")
                return f"{url}/api/generate"
        except:
            continue
    print("❌ Ollama introuvable — utilisation URL par défaut")
    return "http://172.30.128.1:11434/api/generate"

OLLAMA_URL = get_ollama_url()
MODEL = "qwen2.5:14b"
TIMEOUT = 300

def ask_llm(prompt, max_tokens=1500):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": max_tokens,
            "num_gpu": 99,
            "num_ctx": 3072,
            "repeat_penalty": 1.1
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ReadTimeout:
        print(f"⚠️ Timeout après {TIMEOUT}s — réponse trop longue")
        return "La génération a pris trop de temps. Veuillez réessayer avec une question plus courte."
    except Exception as e:
        print(f"❌ Erreur LLM : {e}")
        return "Erreur de connexion au modèle. Veuillez réessayer."