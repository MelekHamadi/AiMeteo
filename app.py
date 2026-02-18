from flask import Flask, request, jsonify, send_from_directory
from rag import retrieve_context_all_projects, load_project, project_indexes
from llm import ask_llm
import os
import pandas as pd
import re

app = Flask(__name__, static_folder="frontend", static_url_path="")
DATA_FOLDER = "data"


def extract_project(question, available_projects):
    question_lower = question.lower()
    for proj in available_projects:
        if proj.lower() in question_lower:
            return proj
    return None


def extract_week(question):
    match = re.search(r'S\d{2}-\d{2}', question)
    return match.group(0) if match else None


def filter_context_by_project_and_week(context, target_project=None, target_week=None):
    if not context:
        return ""
    lines = context.split('\n')
    filtered = []
    current_proj = None

    for line in lines:
        if line.startswith("[PROJET="):
            match = re.match(r'\[PROJET=([^\]]+)\]', line)
            if match:
                current_proj = match.group(1).strip()
        keep = True
        if target_project and current_proj != target_project:
            keep = False
        if target_week and f"[Semaine={target_week}]" not in line:
            keep = False
        if keep:
            filtered.append(line)
    return "\n".join(filtered)


@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/projects", methods=["GET"])
def list_projects():
    projects = []
    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".xlsx"):
            try:
                df = pd.read_excel(os.path.join(DATA_FOLDER, file), sheet_name="Infos_projet")
                if "Projet" in df.columns and not df.empty:
                    projects.append(str(df["Projet"].iloc[0]).strip())
            except:
                continue
    return jsonify({"projects": projects})


@app.route("/upload_project", methods=["POST"])
def upload_project():
    if "file" not in request.files:
        return jsonify({"message": "Aucun fichier"})
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"message": "Nom invalide"})
    path = os.path.join(DATA_FOLDER, file.filename)
    file.save(path)
    load_project(file.filename)
    return jsonify({"message": "Projet ajouté"})


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        question = data.get("message", "").strip()
        ui_project = data.get("project")

        if not question:
            return jsonify({"answer": "Veuillez poser une question."})

        raw_context = retrieve_context_all_projects(question, k_per_project=10, k_final=50)

        if not raw_context:
            return jsonify({"answer": "Information non disponible."})

        available_projects = list(project_indexes.keys())
        mentioned_project = extract_project(question, available_projects)
        if not mentioned_project and ui_project:
            mentioned_project = ui_project

        mentioned_week = extract_week(question)

        filtered_context = filter_context_by_project_and_week(
            raw_context,
            target_project=mentioned_project,
            target_week=mentioned_week
        )

        if not filtered_context.strip():
            if mentioned_project and mentioned_week:
                return jsonify({"answer": f"Information non disponible pour le projet {mentioned_project} à la semaine {mentioned_week}."})
            elif mentioned_project:
                return jsonify({"answer": f"Information non disponible pour le projet {mentioned_project}."})
            elif mentioned_week:
                return jsonify({"answer": f"Information non disponible pour la semaine {mentioned_week}."})
            else:
                return jsonify({"answer": "Information non disponible."})

        # --- Optimisation pour les questions de budget KTND ---
        if "ktnd" in question.lower() and "budget total" in question.lower():
            # Chercher dans le contexte filtré la ligne avec [INFO=Budget_initial_KTND]
            lines = filtered_context.split('\n')
            for line in lines:
                if "[INFO=Budget_initial_KTND]" in line:
                    # La phrase est après le tag, sur la même ligne ou la suivante ?
                    # Dans notre format, c'est sur la ligne suivante
                    parts = line.split('\n', 1)
                    if len(parts) > 1:
                        answer = parts[1].strip()
                        # Vérifier que la phrase contient bien le nom du projet mentionné
                        if mentioned_project and mentioned_project not in answer:
                            continue
                        return jsonify({"answer": answer})
            # Si on ne trouve pas, on continue avec le LLM

        # --- Affichage pour débogage ---
        print("\n--- CONTEXTE FILTRÉ ---")
        print(filtered_context)
        print("-----------------------\n")

        prompt = f"""
Tu es un assistant PMO senior. Tu réponds uniquement à partir du CONTEXTE fourni.

# 📌 RÈGLES ABSOLUES
1. **N'invente jamais rien.** Si l'information n'est pas textuellement présente dans le contexte, réponds exactement "Information non disponible."
2. **Ne fais aucun calcul, aucune déduction, aucune interprétation.** Ne combine pas des chiffres de différents passages.
3. **Chaque passage du contexte est préfixé par `[PROJET=NomDuProjet]` et peut contenir d'autres tags comme `[Semaine=...]`, `[KPI=...]`, `[INFO=...]`.**
4. **Ne cite jamais ces tags dans ta réponse.** Utilise uniquement le contenu textuel après le tag.
5. **Si la question mentionne un projet précis, utilise uniquement les informations de ce projet.**
6. **Si la question mentionne une semaine précise (ex: S10-26), utilise uniquement les informations de cette semaine.**
7. **Pour les questions générales (ex: "liste tous les projets"), tu peux utiliser les informations de tous les projets.**
8. **Si la réponse se trouve dans le contexte sous forme d'une phrase complète, recopie cette phrase exactement, sans modification.**
9. **Si la question demande une valeur numérique (ex: "combien de J/H ont été consommés ?"), cherche dans le contexte la phrase contenant "budget consommé J/H" suivie du nombre. Utilise ce nombre exact, ne l'invente pas.**
10. **Si la question demande la liste des semaines avec un certain statut, cherche toutes les phrases contenant ce statut et liste les semaines correspondantes.**
11. **Pour le budget total en KTND, ne cherche pas à convertir ou interpréter. Cherche la phrase exacte contenant "budget total prévu en KTND". Si elle existe, recopie-la. Sinon, réponds "Information non disponible."**
12. **Réponds en une seule phrase, claire et professionnelle.**

🔍 **Précisions importantes** :
- **Pour les questions sur le budget total en KTND sans précision, utilise le budget initial (tag [INFO=Budget_initial_KTND]).**
- **Pour les questions sur le budget total consommé en KTND, utilise [INFO=Budget_total_consomme_KTND].**
- **Si la question demande "Quels KPI étaient en amélioration ?", liste ceux dont la phrase contient "tendance Amélioration".**
- **Ne combine jamais des informations de plusieurs documents pour en déduire une nouvelle.**

Exemples de bonnes réponses :
- "Le code du projet Data Fraud Detection est PC003."
- "Le sponsor du projet Scoring Crédit Retail est Direction Risques."
- "En semaine S10-26, la phase du projet Data Fraud Detection était Spécification."
- "Le budget total consommé en J/H pour le projet Data Fraud Detection est de 4000."
- "Les risques actuels pour le projet Data Fraud Detection sont Faible."
- "Le budget total prévu en KTND pour le projet Orion Data Platform est de 1350000."
- "Information non disponible."

# 📋 CONTEXTE :
{filtered_context}

# ❓ QUESTION :
{question}

# ✏️ RÉPONSE :
"""

        answer = ask_llm(prompt)
        return jsonify({"answer": answer})

    except Exception as e:
        print(f"❌ Erreur interne dans /chat : {e}")
        return jsonify({"answer": "Une erreur technique est survenue. Veuillez réessayer."})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)