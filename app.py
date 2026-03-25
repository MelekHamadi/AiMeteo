from flask import Flask, request, jsonify, send_from_directory
from rag import (
    project_names, project_info, project_current_phase, project_chef, project_sponsor,
    project_documents,
    extract_risks_from_faits_marquants, get_kpi_for_week, get_general_status,
    get_latest_week, get_budget_consumption, get_phase_transition,
    retrieve_filtered_context, rerank_passages,
    advanced_risk_synthesis, suggest_actions,
    compute_health_score_advanced, generate_health_explanation_advanced,
    predict_problems,
    produce_risk_report, aggregate_risks_all_projects, compare_projects,
    get_project_with_most_critical_risks
)
from llm import ask_llm
import os
import pandas as pd
import re

app = Flask(__name__, static_folder="frontend", static_url_path="")
DATA_FOLDER = "data"

# ==================== UTILITAIRES ====================
def format_date(date_str):
    if not date_str or date_str == "Information non disponible":
        return date_str
    date_str = re.sub(r'\s+\d{2}:\d{2}:\d{2}$', '', date_str)
    patterns = [
        r'(\d{4})-(\d{2})-(\d{2})',
        r'(\d{2})/(\d{2})/(\d{4})',
        r'(\d{2})-(\d{2})-(\d{4})'
    ]
    mois_fr = {
        '01': 'janvier', '02': 'février', '03': 'mars', '04': 'avril',
        '05': 'mai', '06': 'juin', '07': 'juillet', '08': 'août',
        '09': 'septembre', '10': 'octobre', '11': 'novembre', '12': 'décembre'
    }
    for pattern in patterns:
        match = re.search(pattern, date_str)
        if match:
            annee, mois, jour = match.groups()
            if len(annee) == 4:
                pass
            else:
                annee, mois, jour = match.groups()
            mois_nom = mois_fr.get(mois, mois)
            return f"{int(jour)} {mois_nom} {annee}"
    return date_str

def remove_tags(text):
    return re.sub(r'\[[^\]]+\]', '', text).strip()

def extract_week(question):
    match = re.search(r'S\d{2}-\d{2}', question)
    return match.group(0) if match else None

def is_list_question(question):
    keywords = ["quelles", "quels", "liste", "listes", "les semaines", "les projets", "tous", "chaque"]
    return any(kw in question.lower() for kw in keywords)

def is_kpi_question(question):
    kpi_keywords = [
        "avancement", "respect des délais", "respect des delais", "périmètre", "perimetre",
        "risques", "budget", "dépendances", "dependances", "satisfaction client",
        "ressources humaines", "qualité du delivery", "qualite du delivery", "kpi"
    ]
    return any(kw in question.lower() for kw in kpi_keywords)

def clean_text(text):
    text = text.strip()
    text = re.sub(r'[?.!,;:]$', '', text)
    return text.strip()

def find_project(project_query):
    project_query = project_query.lower().strip()
    project_query = re.sub(r'^(le|la|les|du|de la|des)\s+', '', project_query)
    best_match = None
    best_score = 0
    for proj in project_names:
        proj_lower = proj.lower()
        if proj_lower == project_query:
            return proj
        if project_query in proj_lower:
            score = len(project_query) / len(proj_lower)
            if score > best_score:
                best_score = score
                best_match = proj
        if proj_lower in project_query:
            score = len(proj_lower) / len(project_query)
            if score > best_score:
                best_score = score
                best_match = proj
    if best_score > 0.5:
        return best_match
    return None

def extract_project_from_question(question):
    match = re.search(r'(?:du\s+projet\s+|projet\s+)([A-Za-zÀ-ÿ\s]+?)(?:\s+en|\s+et|\s+ainsi|\?|$)', question, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        candidate = re.sub(r'\b(et|ou|ainsi que|le|la|les|du|de la|des)\b', '', candidate, flags=re.IGNORECASE).strip()
        for proj in project_names:
            if proj.lower() == candidate.lower():
                return proj
            if candidate.lower() in proj.lower() or proj.lower() in candidate.lower():
                return proj
    for proj in project_names:
        if proj.lower() in question.lower():
            return proj
    return None

def get_latest_risk_level(project_name):
    docs = project_documents.get(project_name, [])
    risk_docs = []
    for doc in docs:
        if "[FEUILLE=KPI]" not in doc or "[KPI=Risques]" not in doc:
            continue
        semaine_match = re.search(r'\[Semaine=(S\d{2}-\d{2})\]', doc)
        if not semaine_match:
            continue
        # ✅ Nouveau pattern aligné avec rag.py
        statut_match = re.search(r"statut='([^']+)'", doc)
        if statut_match:
            risk_docs.append((semaine_match.group(1), statut_match.group(1)))

    if not risk_docs:
        return "inconnu"

    risk_docs.sort(key=lambda x: int(x[0][1:3]), reverse=True)
    latest_statut = risk_docs[0][1].lower()

    if latest_statut in ["à surveiller", "à redresser"]:
        return "critique"
    elif latest_statut == "en contrôle":
        return "faible"
    else:
        return "inconnu"

def split_question(question):
    separators = [r"\s+et\s+", r"\s+ainsi que\s+", r"\s*,\s*"]
    for sep in separators:
        parts = re.split(sep, question)
        if len(parts) > 1:
            return [p.strip() for p in parts if p.strip()]
    return [question]

def resolve_pronouns(sub_question, context_project):
    if not context_project:
        return sub_question
    sub_question = re.sub(r'\bce projet\b', context_project, sub_question, flags=re.IGNORECASE)
    sub_question = re.sub(r'\bcet projet\b', context_project, sub_question, flags=re.IGNORECASE)
    sub_question = re.sub(r'\bce dernier\b', context_project, sub_question, flags=re.IGNORECASE)
    sub_question = re.sub(r'\bcelle-ci\b', context_project, sub_question, flags=re.IGNORECASE)
    return sub_question

# ==================== TRAITEMENT DES QUESTIONS ====================
def process_single_question(question, ui_project, context_project=None):
    if context_project:
        question = resolve_pronouns(question, context_project)
    q_lower = question.lower()

    # ========== QUESTIONS GLOBALES ==========
    if "budget total en j/h de tous les projets" in q_lower:
        budgets = [f"{proj}: {project_info.get(proj, {}).get('budget_jh', 'N/A')} J/H" for proj in project_names if project_info.get(proj, {}).get('budget_jh')]
        return ", ".join(budgets) if budgets else "Information non disponible."

    if "budget total en ktnd de tous les projets" in q_lower:
        budgets = [f"{proj}: {project_info.get(proj, {}).get('budget_ktnd', 'N/A')} KTND" for proj in project_names if project_info.get(proj, {}).get('budget_ktnd')]
        return ", ".join(budgets) if budgets else "Information non disponible."

    if "phase actuelle de chaque projet" in q_lower or "phase actuelle de tous les projets" in q_lower:
        if project_current_phase:
            return " ".join([f"La phase actuelle du projet {proj} est {phase}." for proj, phase in project_current_phase.items()])
        return "Information non disponible."

    if "sponsor de chaque projet" in q_lower:
        return ", ".join([f"{proj}: {sponsor}" for proj, sponsor in project_sponsor.items()]) if project_sponsor else "Information non disponible."

    if "lister tous les projets avec leur chef de projet" in q_lower or "liste des projets et chef de projet" in q_lower:
        return ", ".join([f"{proj}: {chef}" for proj, chef in project_chef.items()]) if project_chef else "Information non disponible."

    if "quel projet a la date de fin la plus tardive" in q_lower:
        latest_date = None
        latest_proj = None
        for proj in project_names:
            for doc in project_documents.get(proj, []):
                if "[FEUILLE=Infos_projet]" in doc and "Date fin" in doc:
                    m = re.search(r'Date fin\s*:\s*(\d{4}-\d{2}-\d{2})', doc)
                    if m:
                        end_date = pd.to_datetime(m.group(1))
                        if latest_date is None or end_date > latest_date:
                            latest_date = end_date
                            latest_proj = proj
                        break
        return latest_proj if latest_proj else "Information non disponible."

    if "quel projet a le plus gros budget initial en ktnd" in q_lower:
        max_budget = -1
        max_proj = None
        for proj in project_names:
            b = project_info.get(proj, {}).get("budget_ktnd")
            if b and b > max_budget:
                max_budget = b
                max_proj = proj
        return max_proj if max_proj else "Information non disponible."

    if "quels projets étaient en phase d'homologation en s18-26" in q_lower:
        projs = []
        for proj in project_names:
            if get_phase_transition(proj, "Homologation") is not None:
                for doc in project_documents.get(proj, []):
                    if "[FEUILLE=Météo]" in doc and "S18-26" in doc and "phase" in doc and "Homologation" in doc:
                        projs.append(proj)
                        break
        return ", ".join(projs) if projs else "Aucun projet trouvé."

    # ========== QUESTIONS SPÉCIFIQUES AVEC PROJET ==========
    proj = extract_project_from_question(question) or ui_project or context_project

    if "chef de projet" in q_lower:
        if proj:
            chef = project_chef.get(proj)
            return f"Le chef de projet du projet {proj} est {chef}." if chef else f"Information non disponible pour {proj}."
        return "Veuillez préciser un projet."

    if "sponsor" in q_lower and "projet" in q_lower:
        if proj:
            sponsor = project_sponsor.get(proj)
            return f"Le sponsor du projet {proj} est {sponsor}." if sponsor else f"Information non disponible pour {proj}."
        return "Veuillez préciser un projet."

    patterns_jh = [r"budget(?:\s+total)?\s+en\s+j/?h\s+(?:du\s+projet\s+)?(.*)", r"budget\s+j/?h\s+(?:du\s+projet\s+)?(.*)", r"j/?h\s+du\s+projet\s+(.*)"]
    if any(re.search(p, q_lower) for p in patterns_jh):
        if proj:
            budget = project_info.get(proj, {}).get("budget_jh")
            return f"Le budget total en J/H du projet {proj} est de {budget}." if budget else f"Information non disponible pour {proj}."
        return "Veuillez préciser un projet."

    patterns_ktnd = [r"budget(?:\s+total)?\s+en\s+ktnd?\s+(?:du\s+projet\s+)?(.*)", r"budget\s+ktnd?\s+(?:du\s+projet\s+)?(.*)", r"ktnd?\s+du\s+projet\s+(.*)"]
    if any(re.search(p, q_lower) for p in patterns_ktnd):
        if proj:
            budget = project_info.get(proj, {}).get("budget_ktnd")
            return f"Le budget total en KTND du projet {proj} est de {budget}." if budget else f"Information non disponible pour {proj}."
        return "Veuillez préciser un projet."

    match = re.search(r"phase actuelle(?:\s+du\s+projet)?\s+([^?]+)", q_lower)
    if match:
        if proj:
            phase = project_current_phase.get(proj)
            return f"La phase actuelle du projet {proj} est {phase}." if phase else f"Information non disponible pour {proj}."
        return "Veuillez préciser un projet."

    # ========== AXE 1 : Synthèse des risques ==========
    if "synthèse des risques" in q_lower or "résumé des risques" in q_lower or "risques encourus" in q_lower:
        if not proj:
            return "Veuillez préciser un projet."
        weeks = re.findall(r'S\d{2}-\d{2}', question)
        start_week = weeks[0] if weeks else None
        end_week = weeks[1] if len(weeks) > 1 else weeks[0] if weeks else None
        return advanced_risk_synthesis(proj, start_week, end_week)

    # ========== AXE 2 : Propositions d'actions ==========
    if "que faire" in q_lower or "action" in q_lower or "recommandation" in q_lower or "propose" in q_lower:
        if not proj:
            return "Pour des propositions d'actions, veuillez préciser un projet."
        week = extract_week(question)
        actions = suggest_actions(proj, week)
        if len(actions) == 1:
            return actions[0]
        else:
            return "Voici les actions recommandées :\n- " + "\n- ".join(actions)

    # ========== AXE 3 : Santé du projet ==========
    if "santé" in q_lower or "health" in q_lower or "score de santé" in q_lower:
        if not proj:
            return "Veuillez préciser un projet pour évaluer sa santé."
        week = extract_week(question)
        health_data = compute_health_score_advanced(proj, week)
        explanation = generate_health_explanation_advanced(proj, health_data)
        return f"**Score de santé du projet {proj} : {health_data['score']}/100** – {health_data['level']}\n\n{explanation}"

    # ========== AXE 4 : Prédiction des problèmes ==========
    if "prédiction" in q_lower or "anticipation" in q_lower or "futurs" in q_lower or "problèmes" in q_lower:
        if not proj:
            return "Veuillez préciser un projet pour une prédiction."
        return predict_problems(proj)

    # ========== AUTRES : Rapports, comparaison, etc. ==========
    if "rapport de risques" in q_lower or "analyse des risques" in q_lower or "rapport détaillé" in q_lower:
        if not proj:
            return "Veuillez préciser un projet."
        weeks = re.findall(r'S\d{2}-\d{2}', question)
        start_week = weeks[0] if weeks else None
        end_week = weeks[1] if len(weeks) > 1 else weeks[0] if weeks else None
        return produce_risk_report(proj, start_week, end_week)

    if "bilan des risques sur tous les projets" in q_lower:
        weeks = re.findall(r'S\d{2}-\d{2}', question)
        start_week = weeks[0] if weeks else None
        end_week = weeks[1] if len(weeks) > 1 else weeks[0] if weeks else None
        return aggregate_risks_all_projects(start_week, end_week)

    if "compare" in q_lower:
        projs = [p for p in project_names if p.lower() in q_lower]
        if len(projs) >= 2:
            weeks = re.findall(r'S\d{2}-\d{2}', question)
            start_week = weeks[0] if weeks else None
            end_week = weeks[1] if len(weeks) > 1 else weeks[0] if weeks else None
            return compare_projects(projs[0], projs[1], start_week, end_week)
        return "Veuillez mentionner deux projets à comparer."

    # ========== RAG GÉNÉRAL ==========
    if is_kpi_question(question):
        feuille = "KPI"
        k_final = 50
        top_k_rerank = 10
        kpi_instruction = "Pour une question sur un KPI spécifique, réponds avec le statut et la tendance si disponibles."
    else:
        feuille = None
        k_final = 30
        top_k_rerank = 5
        kpi_instruction = ""

    if is_list_question(question):
        k_final = 100
        top_k_rerank = 20

    raw_context = retrieve_filtered_context(question, k_final=k_final, feuille=feuille, force_project=proj)
    if not raw_context:
        return f"Information non disponible pour le projet {proj}." if proj else "Information non disponible."

    blocks = []
    current_block = []
    for line in raw_context.split('\n'):
        if line.startswith("[PROJET="):
            if current_block:
                blocks.append("\n".join(current_block))
            current_block = [line]
        else:
            current_block.append(line)
    if current_block:
        blocks.append("\n".join(current_block))

    if len(blocks) > top_k_rerank:
        best_blocks = rerank_passages(question, blocks, top_k=top_k_rerank)
    else:
        best_blocks = blocks

    final_context = "\n".join(best_blocks)

    prompt = f"""
Tu es un assistant spécialisé en gestion de projet. Tu dois répondre UNIQUEMENT en utilisant les passages du CONTEXTE ci-dessous.

RÈGLES IMPÉRATIVES :
- {('Liste tous les éléments demandés, séparés par des virgules, sans ajout de texte.' if is_list_question(question) else 'Donne une réponse concise, en une seule phrase si possible.')}
- {kpi_instruction}
- Ne donne AUCUNE explication, commentaire ou introduction.
- Si la question est de type factuelle, réponds uniquement par le statut ou l'information demandée.
- N'invente pas d'actions si la question ne le demande pas.
- Réponds en utilisant les informations du contexte.
- Si la question mentionne un projet, assure-toi que la phrase concerne ce projet.
- Si la question mentionne une semaine, assure-toi que l'information correspond à cette semaine.
- Si AUCUNE information ne répond, réponds : "Information non disponible."
- Ne cite jamais les tags comme [PROJET=...], [Semaine=...], [INFO=...].
- Transforme les dates au format français.
- Réponds toujours en français.

CONTEXTE :
{final_context}

QUESTION : {question}

RÉPONSE :
"""
    answer = ask_llm(prompt)
    answer = remove_tags(answer).strip()
    # Post-traitement des dates
    date_pattern = r'(\d{4})-(\d{2})-(\d{2})(?:\s+\d{2}:\d{2}:\d{2})?'
    matches = re.findall(date_pattern, answer)
    for annee, mois, jour in matches:
        old_date = f"{annee}-{mois}-{jour}"
        for heure in ["", " 00:00:00"]:
            candidate = old_date + heure
            if candidate in answer:
                new_date = format_date(candidate)
                answer = answer.replace(candidate, new_date)
    answer = re.sub(r'(\d{1,2}\s+\w+\s+\d{4})\s+\d{2}:\d{2}:\d{2}', r'\1', answer)

    if not answer or answer.lower() in ["je ne sais pas", "inconnu"]:
        answer = "Information non disponible."
    return answer

# ==================== ROUTES ====================
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

@app.route("/project_status", methods=["GET"])
def project_status():
    statuses = {}
    for proj in project_names:
        statuses[proj] = {"risk": get_latest_risk_level(proj)}
    return jsonify(statuses)

@app.route("/upload_project", methods=["POST"])
def upload_project():
    if "file" not in request.files:
        return jsonify({"message": "Aucun fichier"})
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"message": "Nom invalide"})
    path = os.path.join(DATA_FOLDER, file.filename)
    file.save(path)
    from rag import load_project
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

        sub_questions = split_question(question)
        if len(sub_questions) == 1:
            answer = process_single_question(question, ui_project)
            return jsonify({"answer": answer})
        else:
            answers = []
            last_project = None
            for subq in sub_questions:
                proj_in_sub = extract_project_from_question(subq)
                if proj_in_sub:
                    last_project = proj_in_sub
                ans = process_single_question(subq, ui_project, last_project)
                answers.append(ans)
            return jsonify({"answer": "\n\n".join(answers)})

    except Exception as e:
        print(f"❌ Erreur interne dans /chat : {e}")
        return jsonify({"answer": "Une erreur technique est survenue. Veuillez réessayer."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)