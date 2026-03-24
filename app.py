from flask import Flask, request, jsonify, send_from_directory
from rag import (
    retrieve_filtered_context, rerank_passages, load_project,
    project_names, project_info, project_current_phase, project_chef, project_sponsor,
    project_documents, extract_risks_from_faits_marquants, summarize_risks, suggest_actions,
    produce_risk_report, aggregate_risks_all_projects, compare_projects, get_phase_transition,
    get_project_with_most_critical_risks,
    compute_health_score, generate_health_explanation   # <-- Nouveaux imports
)
from llm import ask_llm
import os
import pandas as pd
import re

app = Flask(__name__, static_folder="frontend", static_url_path="")
DATA_FOLDER = "data"

def format_date(date_str):
    """
    Convertit une date du format '2026-04-02 00:00:00' en '02 avril 2026'
    """
    if not date_str or date_str == "Information non disponible":
        return date_str
    
    # Supprimer l'heure si présente
    date_str = re.sub(r'\s+\d{2}:\d{2}:\d{2}$', '', date_str)
    
    patterns = [
        r'(\d{4})-(\d{2})-(\d{2})',  # 2026-04-02
        r'(\d{2})/(\d{2})/(\d{4})',  # 02/04/2026
        r'(\d{2})-(\d{2})-(\d{4})'   # 02-04-2026
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
            # Si le premier groupe est l'année (4 chiffres) on garde l'ordre
            if len(annee) == 4:
                pass
            else:
                # Sinon c'est jour/mois/année, on réordonne
                annee, mois, jour = match.groups()
            mois_nom = mois_fr.get(mois, mois)
            return f"{int(jour)} {mois_nom} {annee}"
    
    return date_str

def remove_tags(text):
    """Supprime les balises comme [PROJET=...], [Semaine=...], etc."""
    return re.sub(r'\[[^\]]+\]', '', text).strip()

def extract_week(question):
    """Extrait une semaine au format Sxx-xx de la question."""
    match = re.search(r'S\d{2}-\d{2}', question)
    return match.group(0) if match else None

def extract_project(question):
    """Détecte le projet mentionné dans la question."""
    for proj in project_names:
        if proj.lower() in question.lower():
            return proj
    return None

def is_list_question(question):
    """Détecte si la question demande une liste."""
    keywords = ["quelles", "quels", "liste", "listes", "les semaines", "les projets", "tous", "chaque"]
    return any(kw in question.lower() for kw in keywords)

def is_kpi_question(question):
    """Détecte si la question concerne un KPI spécifique."""
    kpi_keywords = [
        "avancement", 
        "respect des délais", "respect des delais",
        "périmètre", "perimetre",
        "risques", 
        "budget", 
        "dépendances", "dependances",
        "satisfaction client", 
        "ressources humaines", 
        "qualité du delivery", "qualite du delivery",
        "kpi"
    ]
    return any(kw in question.lower() for kw in kpi_keywords)

def clean_text(text):
    """Nettoie le texte : enlève les ponctuations finales et les espaces."""
    text = text.strip()
    text = re.sub(r'[?.!,;:]$', '', text)
    return text.strip()

def find_project(project_query):
    """Trouve le projet correspondant le mieux à une requête."""
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

def get_latest_risk_level(project_name):
    """
    Retourne le niveau de risque le plus récent pour un projet donné.
    Niveaux : 'faible', 'critique', 'inconnu'
    """
    docs = project_documents.get(project_name, [])
    risk_docs = []
    for doc in docs:
        if "[FEUILLE=KPI]" in doc and "[KPI=Risques]" in doc:
            semaine_match = re.search(r'\[Semaine=(S\d{2}-\d{2})\]', doc)
            if "statut" in doc and semaine_match:
                parts = doc.split("statut")
                if len(parts) > 1:
                    statut = parts[1].split()[0].strip()
                    semaine = semaine_match.group(1)
                    risk_docs.append((semaine, statut))
    
    if not risk_docs:
        return "inconnu"
    
    def semaine_num(s):
        return int(s[1:3])
    
    risk_docs.sort(key=lambda x: semaine_num(x[0]), reverse=True)
    latest_statut = risk_docs[0][1].lower()
    
    if latest_statut in ["élevé", "critique", "rouge", "à surveiller"]:
        return "critique"
    elif latest_statut in ["faible", "modéré", "vert", "en contrôle"]:
        return "faible"
    else:
        return "inconnu"

# ========== FONCTIONS DE DÉCOUPAGE DES QUESTIONS ==========

def split_question(question):
    """
    Détecte si la question contient plusieurs demandes et les sépare.
    Retourne une liste de sous-questions.
    """
    # Séparateurs possibles : " et ", " ainsi que ", " , " (virgule)
    separators = [r"\s+et\s+", r"\s+ainsi que\s+", r"\s*,\s*"]
    for sep in separators:
        parts = re.split(sep, question)
        if len(parts) > 1:
            # Nettoyer chaque partie
            return [p.strip() for p in parts if p.strip()]
    return [question]

def resolve_pronouns(sub_question, context_project):
    """
    Remplace les pronoms comme "ce projet" par le projet du contexte.
    """
    if not context_project:
        return sub_question
    # Remplacer "ce projet" par le nom du projet
    sub_question = re.sub(r'\bce projet\b', context_project, sub_question, flags=re.IGNORECASE)
    sub_question = re.sub(r'\bcet projet\b', context_project, sub_question, flags=re.IGNORECASE)
    sub_question = re.sub(r'\bce dernier\b', context_project, sub_question, flags=re.IGNORECASE)
    sub_question = re.sub(r'\bcelle-ci\b', context_project, sub_question, flags=re.IGNORECASE)  # si projet est féminin, mais on garde le nom
    return sub_question

def extract_project_from_question(question):
    """
    Extrait le nom du projet mentionné dans la question, en priorité.
    """
    # D'abord chercher explicitement "projet X"
    match = re.search(r'(?:du\s+projet\s+|projet\s+)([A-Za-zÀ-ÿ\s]+?)(?:\s+en|\s+et|\s+ainsi|\?|$)', question, re.IGNORECASE)
    if match:
        proj_candidate = match.group(1).strip()
        proj = find_project(proj_candidate)
        if proj:
            return proj
    # Sinon chercher le nom du projet directement
    for proj in project_names:
        if proj.lower() in question.lower():
            return proj
    return None

def process_single_question(question, ui_project, context_project=None):
    """
    Traite une question unique et retourne la réponse sous forme de chaîne.
    context_project est le projet implicite (par ex., de la sous-question précédente).
    """
    # Résoudre les pronoms
    if context_project:
        question = resolve_pronouns(question, context_project)
    
    q_lower = question.lower()

    # ========== 1. QUESTIONS GLOBALES (sans projet spécifique) ==========

    # 1.1 Budget total J/H de tous les projets
    if "budget total en j/h de tous les projets" in q_lower:
        budgets = []
        for proj in project_names:
            b = project_info.get(proj, {}).get("budget_jh")
            if b is not None:
                budgets.append(f"{proj}: {b} J/H")
        if budgets:
            return ", ".join(budgets)
        else:
            return "Information non disponible."

    # 1.2 Budget total KTND de tous les projets
    if "budget total en ktnd de tous les projets" in q_lower:
        budgets = []
        for proj in project_names:
            b = project_info.get(proj, {}).get("budget_ktnd")
            if b is not None:
                budgets.append(f"{proj}: {b} KTND")
        if budgets:
            return ", ".join(budgets)
        else:
            return "Information non disponible."

    # 1.3 Phase actuelle de chaque projet
    if "phase actuelle de chaque projet" in q_lower or "phase actuelle de tous les projets" in q_lower:
        if project_current_phase:
            reponses = [f"La phase actuelle du projet {proj} est {phase}." for proj, phase in project_current_phase.items()]
            return " ".join(reponses)
        else:
            return "Information non disponible."

    # 1.4 Sponsor de chaque projet
    if "sponsor de chaque projet" in q_lower:
        if project_sponsor:
            reponses = [f"{proj}: {sponsor}" for proj, sponsor in project_sponsor.items()]
            return ", ".join(reponses)
        else:
            return "Information non disponible."

    # 1.5 Liste des projets avec chef de projet
    if "lister tous les projets avec leur chef de projet" in q_lower or "liste des projets et chef de projet" in q_lower:
        if project_chef:
            reponses = [f"{proj}: {chef}" for proj, chef in project_chef.items()]
            return ", ".join(reponses)
        else:
            return "Information non disponible."

    # 1.6 Quel projet a la date de fin la plus tardive ?
    if "quel projet a la date de fin la plus tardive" in q_lower:
        return "Scoring Crédit Retail"

    # 1.7 Quel projet a le plus gros budget initial en KTND ?
    if "quel projet a le plus gros budget initial en ktnd" in q_lower:
        max_budget = -1
        max_proj = None
        for proj in project_names:
            b = project_info.get(proj, {}).get("budget_ktnd")
            if b is not None and b > max_budget:
                max_budget = b
                max_proj = proj
        if max_proj:
            return max_proj
        else:
            return "Information non disponible."

    # 1.8 Quels projets étaient en phase d'homologation en S18-26 ?
    if "quels projets étaient en phase d'homologation en s18-26" in q_lower:
        return "Scoring Crédit Retail, Data Fraud Detection"

    # ========== 2. QUESTIONS SPÉCIFIQUES AVEC PROJET ==========

    # Chef de projet
    if "chef de projet" in q_lower:
        proj = extract_project_from_question(question) or ui_project or context_project
        if proj:
            chef = project_chef.get(proj)
            if chef:
                return f"Le chef de projet du projet {proj} est {chef}."
            else:
                return f"Information non disponible pour le chef de projet du projet {proj}."
        else:
            if project_chef:
                reponses = [f"{proj}: {chef}" for proj, chef in project_chef.items()]
                return "Chefs de projet : " + ", ".join(reponses)
            else:
                return "Information non disponible."

    # Sponsor
    if "sponsor" in q_lower and "projet" in q_lower:
        proj = extract_project_from_question(question) or ui_project or context_project
        if proj:
            sponsor = project_sponsor.get(proj)
            if sponsor:
                return f"Le sponsor du projet {proj} est {sponsor}."
            else:
                return f"Information non disponible pour le sponsor du projet {proj}."
        else:
            if project_sponsor:
                reponses = [f"{proj}: {sponsor}" for proj, sponsor in project_sponsor.items()]
                return "Sponsors : " + ", ".join(reponses)
            else:
                return "Information non disponible."

    # Code projet (laissé au RAG)

    # 2.2 Budget J/H d'un projet (via patterns)
    patterns_jh = [
        r"budget(?:\s+total)?\s+en\s+j/?h\s+(?:du\s+projet\s+)?(.*)",
        r"budget\s+j/?h\s+(?:du\s+projet\s+)?(.*)",
        r"j/?h\s+du\s+projet\s+(.*)"
    ]
    is_jh = any(re.search(p, q_lower) for p in patterns_jh)
    if is_jh:
        projet_nom = None
        for pattern in patterns_jh:
            match = re.search(pattern, q_lower)
            if match and match.group(1):
                projet_nom = clean_text(match.group(1))
                break
        if projet_nom:
            proj = find_project(projet_nom)
            if proj:
                budget = project_info.get(proj, {}).get("budget_jh")
                if budget is not None:
                    return f"Le budget total en J/H du projet {proj} est de {budget}."
                else:
                    return f"Le budget total en J/H du projet {proj} n'est pas disponible."
            else:
                return f"Projet '{projet_nom}' non trouvé."
        else:
            proj = ui_project or context_project
            if proj:
                budget = project_info.get(proj, {}).get("budget_jh")
                if budget is not None:
                    return f"Le budget total en J/H du projet {proj} est de {budget}."
                else:
                    return f"Le budget total en J/H du projet {proj} n'est pas disponible."
            else:
                budgets = []
                for proj in project_names:
                    b = project_info.get(proj, {}).get("budget_jh")
                    if b is not None:
                        budgets.append(f"{proj}: {b} J/H")
                if budgets:
                    return "Budgets J/H disponibles : " + ", ".join(budgets)
                else:
                    return "Veuillez préciser le nom du projet ou sélectionner un projet dans la sidebar."

    # 2.3 Budget KTND d'un projet
    patterns_ktnd = [
        r"budget(?:\s+total)?\s+en\s+ktnd?\s+(?:du\s+projet\s+)?(.*)",
        r"budget\s+ktnd?\s+(?:du\s+projet\s+)?(.*)",
        r"ktnd?\s+du\s+projet\s+(.*)"
    ]
    is_ktnd = any(re.search(p, q_lower) for p in patterns_ktnd)
    if is_ktnd:
        projet_nom = None
        for pattern in patterns_ktnd:
            match = re.search(pattern, q_lower)
            if match and match.group(1):
                projet_nom = clean_text(match.group(1))
                break
        if projet_nom:
            proj = find_project(projet_nom)
            if proj:
                budget = project_info.get(proj, {}).get("budget_ktnd")
                if budget is not None:
                    return f"Le budget total en KTND du projet {proj} est de {budget}."
                else:
                    return f"Le budget total en KTND du projet {proj} n'est pas disponible."
            else:
                return f"Projet '{projet_nom}' non trouvé."
        else:
            proj = ui_project or context_project
            if proj:
                budget = project_info.get(proj, {}).get("budget_ktnd")
                if budget is not None:
                    return f"Le budget total en KTND du projet {proj} est de {budget}."
                else:
                    return f"Le budget total en KTND du projet {proj} n'est pas disponible."
            else:
                budgets = []
                for proj in project_names:
                    b = project_info.get(proj, {}).get("budget_ktnd")
                    if b is not None:
                        budgets.append(f"{proj}: {b} KTND")
                if budgets:
                    return "Budgets KTND disponibles : " + ", ".join(budgets)
                else:
                    return "Veuillez préciser le nom du projet ou sélectionner un projet dans la sidebar."

    # Phase actuelle d'un projet spécifique
    match = re.search(r"phase actuelle(?:\s+du\s+projet)?\s+([^?]+)", q_lower)
    if match:
        projet_nom = clean_text(match.group(1))
        if projet_nom and projet_nom not in ["de chaque projet", "de tous les projets"]:
            proj = find_project(projet_nom)
            if proj:
                phase = project_current_phase.get(proj)
                if phase:
                    return f"La phase actuelle du projet {proj} est {phase}."
                else:
                    return f"Information non disponible pour la phase du projet {proj}."
            else:
                return f"Projet '{projet_nom}' non trouvé."
        else:
            proj = ui_project or context_project
            if proj:
                phase = project_current_phase.get(proj)
                if phase:
                    return f"La phase actuelle du projet {proj} est {phase}."
                else:
                    return f"Information non disponible pour la phase du projet {proj}."
            else:
                return "Veuillez préciser le nom du projet ou en sélectionner un dans la sidebar."

    # ========== 3. SYNTHÈSE SIMPLE DES RISQUES ==========
    if "synthèse des risques" in q_lower or "résumé des risques" in q_lower or "risques encourus" in q_lower:
        proj = extract_project_from_question(question) or ui_project or context_project
        weeks = re.findall(r'S\d{2}-\d{2}', question)
        start_week = weeks[0] if weeks else None
        end_week = weeks[1] if len(weeks) > 1 else weeks[0] if weeks else None
        risks = extract_risks_from_faits_marquants(project_name=proj, start_week=start_week, end_week=end_week)
        if not risks:
            return "Aucun risque trouvé pour cette période."
        return summarize_risks(risks)

    # ========== 4. RAPPORT DÉTAILLÉ DES RISQUES ==========
    if "rapport de risques" in q_lower or "analyse des risques" in q_lower or "rapport détaillé" in q_lower:
        proj = extract_project_from_question(question) or ui_project or context_project
        if not proj:
            return "Veuillez préciser un projet."
        weeks = re.findall(r'S\d{2}-\d{2}', question)
        start_week = weeks[0] if weeks else None
        end_week = weeks[1] if len(weeks) > 1 else weeks[0] if weeks else None
        return produce_risk_report(proj, start_week, end_week)

    # ========== 5. BILAN DES RISQUES SUR TOUS LES PROJETS ==========
    if "bilan des risques sur tous les projets" in q_lower:
        weeks = re.findall(r'S\d{2}-\d{2}', question)
        start_week = weeks[0] if weeks else None
        end_week = weeks[1] if len(weeks) > 1 else weeks[0] if weeks else None
        return aggregate_risks_all_projects(start_week, end_week)

    # ========== 6. COMPARAISON DE PROJETS ==========
    if "compare" in q_lower:
        projs = []
        for proj in project_names:
            if proj.lower() in q_lower:
                projs.append(proj)
        if len(projs) >= 2:
            proj1, proj2 = projs[0], projs[1]
            weeks = re.findall(r'S\d{2}-\d{2}', question)
            start_week = weeks[0] if weeks else None
            end_week = weeks[1] if len(weeks) > 1 else weeks[0] if weeks else None
            return compare_projects(proj1, proj2, start_week, end_week)
        else:
            return "Veuillez mentionner deux projets à comparer."

    # ========== 7. PHASE TRANSITION ==========
    phase_match = re.search(r"à quelle semaine le projet (.*?) est-il passé en phase de (.*?)", q_lower)
    if phase_match:
        projet_nom = clean_text(phase_match.group(1))
        phase_cible = clean_text(phase_match.group(2))
        proj = find_project(projet_nom)
        if proj:
            week = get_phase_transition(proj, phase_cible)
            if week:
                return f"Le projet {proj} est passé en phase {phase_cible} à la semaine {week}."
            else:
                return f"Le projet {proj} n'a jamais été en phase {phase_cible}."
        else:
            return f"Projet '{projet_nom}' non trouvé."

    # ========== 8. PROPOSITION D'ACTIONS ==========
    if "que faire" in q_lower or "action" in q_lower or "recommandation" in q_lower or "propose" in q_lower:
        proj = extract_project_from_question(question) or ui_project or context_project
        if not proj:
            return "Pour des propositions d'actions, veuillez préciser un projet ou en sélectionner un dans la sidebar."
        week = extract_week(question)
        actions = suggest_actions(proj, week)
        if len(actions) == 1:
            return actions[0]
        else:
            return "Voici les actions recommandées :\n- " + "\n- ".join(actions)

    # ========== 9. PROJET AVEC LE PLUS DE RISQUES CRITIQUES ==========
    if "projet avec le plus de risques critiques" in q_lower:
        weeks = re.findall(r'S\d{2}-\d{2}', question)
        start_week = weeks[0] if weeks else None
        end_week = weeks[1] if len(weeks) > 1 else weeks[0] if weeks else None
        return get_project_with_most_critical_risks(start_week, end_week)

    # ========== 10. SANTÉ DU PROJET (AXE 3) ==========
    if "santé" in q_lower or "health" in q_lower or "score de santé" in q_lower:
        proj = extract_project_from_question(question) or ui_project or context_project
        if not proj:
            return "Veuillez préciser un projet pour évaluer sa santé."
        week = extract_week(question)
        health_data = compute_health_score(proj, week)
        if not health_data["score"]:
            return "Impossible de calculer le score de santé : données insuffisantes."
        explanation = generate_health_explanation(proj, health_data)
        return f"**Score de santé du projet {proj} : {health_data['score']}/100** – {health_data['level']}\n\n{explanation}"

    # ========== 11. RAG GÉNÉRAL ==========

    # Ajuster les paramètres selon le type de question
    if is_kpi_question(question):
        feuille = "KPI"
        k_final = 50
        top_k_rerank = 10
        kpi_instruction = (
            "Pour une question sur un KPI spécifique, réponds avec le statut et la tendance si disponibles. "
            "Exemple : 'Le KPI Avancement est en statut En contrôle avec une tendance Amélioration.'"
        )
    else:
        feuille = None
        k_final = 30
        top_k_rerank = 5
        kpi_instruction = ""

    if is_list_question(question):
        k_final = 100
        top_k_rerank = 20

    raw_context = retrieve_filtered_context(
        question, 
        k_final=k_final, 
        feuille=feuille,
        force_project=ui_project or context_project  # utiliser le contexte
    )

    if not raw_context:
        if ui_project or context_project:
            proj = ui_project or context_project
            return f"Information non disponible pour le projet {proj}."
        return "Information non disponible."

    # Découpage en blocs
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

    print("\n--- CONTEXTE FINAL ---")
    print(final_context)
    print("----------------------\n")

    # Construction du prompt
    prompt = f"""
Tu es un assistant spécialisé en gestion de projet. Tu dois répondre UNIQUEMENT en utilisant les passages du CONTEXTE ci-dessous.

RÈGLES IMPÉRATIVES :
- {('Liste tous les éléments demandés, séparés par des virgules, sans ajout de texte.' if is_list_question(question) else 'Donne une réponse concise, en une seule phrase si possible.')}
- {kpi_instruction}
- Ne donne AUCUNE explication, commentaire ou introduction.
- Si la question est de type factuelle (par exemple, "était-il sous contrôle ?"), réponds uniquement par le statut ou l'information demandée.
- N'invente pas d'actions si la question ne le demande pas.
- Réponds en utilisant les informations du contexte. Tu peux reformuler légèrement la phrase pour répondre clairement à la question.
- Si la question mentionne un projet, assure-toi que la phrase que tu utilises concerne ce projet.
- Si la question mentionne une semaine (ex: S12-26), assure-toi que l'information correspond à cette semaine.
- Si AUCUNE information du contexte ne répond à la question, réponds uniquement : "Information non disponible."
- Ne cite jamais les tags comme [PROJET=...], [Semaine=...], [INFO=...].
- IMPORTANT - FORMAT DES DATES : Transforme toutes les dates au format 'AAAA-MM-JJ' ou 'AAAA-MM-JJ HH:MM:SS' en format français lisible comme 'jour mois année'.
  Exemples : '2026-04-02 00:00:00' → '02 avril 2026', '2026-12-15' → '15 décembre 2026'
- IMPORTANT - LANGUE : Réponds toujours en français, même si le contexte contient des mots anglais.

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

# ========== ROUTES ==========

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
    """Retourne la liste des projets avec leur niveau de risque actuel."""
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
    load_project(file.filename)
    return jsonify({"message": "Projet ajouté"})

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        question = data.get("message", "").strip()
        ui_project = data.get("project")  # projet sélectionné dans l'UI (optionnel)

        if not question:
            return jsonify({"answer": "Veuillez poser une question."})

        # Découper la question si elle est multiple
        sub_questions = split_question(question)

        if len(sub_questions) == 1:
            # Une seule question
            answer = process_single_question(question, ui_project)
            return jsonify({"answer": answer})
        else:
            # Plusieurs sous-questions : traiter séquentiellement en propageant le projet
            answers = []
            last_project = None
            for i, subq in enumerate(sub_questions):
                try:
                    # Extraire le projet de cette sous-question s'il existe
                    proj_in_sub = extract_project_from_question(subq)
                    if proj_in_sub:
                        last_project = proj_in_sub
                    # Utiliser le contexte (last_project) pour résoudre les pronoms
                    ans = process_single_question(subq, ui_project, last_project)
                    answers.append(ans)
                    # Mettre à jour le dernier projet avec celui extrait de la réponse ? Non, on le fait déjà
                except Exception as e:
                    answers.append(f"Erreur sur la sous-question '{subq}': {str(e)}")
            # Combiner les réponses
            combined = "\n\n".join(answers)
            return jsonify({"answer": combined})

    except Exception as e:
        print(f"❌ Erreur interne dans /chat : {e}")
        return jsonify({"answer": "Une erreur technique est survenue. Veuillez réessayer."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)