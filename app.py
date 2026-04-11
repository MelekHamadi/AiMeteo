"""
app.py v3 — PMO AI Copilot
Corrections v3 :
- Handler portfolio_budget complet (J/H, KTND, max budget, total portefeuille)
- Handler trend_analysis pour questions d'évolution globale
- Handler global_analytical enrichi (phase de transition, projets en phase X, code projet)
- Résolution KPI multi-projets sans projet spécifié (UI project = fallback)
- Seuil get_project_with_most_critical_risks ajusté (threshold=2)
- Réponses "quels projets en phase X" corrigées (MEP = production, Prod = production)
- Clarification intelligente pour questions vagues
- Contexte conversationnel persisté
"""

from flask import Flask, request, jsonify, send_from_directory
from rag import (
    project_names, project_info, project_current_phase, project_chef, project_sponsor,
    project_documents,
    extract_risks_from_faits_marquants, get_kpi_for_week, get_general_status,
    get_latest_week, get_budget_consumption, get_phase_transition, find_kpi,
    retrieve_filtered_context, rerank_passages,
    advanced_risk_synthesis, suggest_actions,
    compute_health_score_advanced, generate_health_explanation_advanced,
    predict_problems, produce_risk_report, aggregate_risks_all_projects,
    compare_projects, get_project_with_most_critical_risks,
    get_kpi_evolution, get_risk_signals, compute_weekly_risk_scores
)
from intent_classifier import classify_intent, _extract_kpi_name, _extract_entities
from llm import ask_llm
import os, re, pandas as pd

app = Flask(__name__, static_folder="frontend", static_url_path="")
DATA_FOLDER = "data"

MOIS_FR = {
    '01': 'janvier', '02': 'février', '03': 'mars', '04': 'avril',
    '05': 'mai', '06': 'juin', '07': 'juillet', '08': 'août',
    '09': 'septembre', '10': 'octobre', '11': 'novembre', '12': 'décembre'
}

# Alias de phases pour normaliser "MEP", "Mise en production", "Prod" → famille "production"
PHASE_PRODUCTION_ALIASES = {
    "production", "mise en production", "mep", "prod", "pré-prod", "pre-prod",
    "mise en prod", "deploiement", "déploiement"
}

PHASE_HOMOLOGATION_ALIASES = {
    "homologation", "hom", "recette", "validation", "test", "tests"
}


# ==================== UTILITAIRES ====================

def format_date(date_str):
    if not date_str or date_str == "Information non disponible":
        return date_str
    date_str = re.sub(r'\s+\d{2}:\d{2}:\d{2}$', '', str(date_str))
    for pattern in [r'(\d{4})-(\d{2})-(\d{2})', r'(\d{2})/(\d{2})/(\d{4})', r'(\d{2})-(\d{2})-(\d{4})']:
        m = re.search(pattern, date_str)
        if m:
            g = m.groups()
            a, mo, j = (g[0], g[1], g[2]) if len(g[0]) == 4 else (g[2], g[1], g[0])
            return f"{int(j)} {MOIS_FR.get(mo, mo)} {a}"
    return date_str


def remove_tags(text):
    return re.sub(r'\[[^\]]+\]', '', text).strip()


def extract_week(q):
    m = re.search(r'S\d{2}-\d{2}', q)
    return m.group(0) if m else None


def is_list_question(q):
    return any(kw in q.lower() for kw in ["quelles", "quels", "liste", "tous", "chaque"])


def is_kpi_question(q):
    patterns = [
        r'\bavancement\b', r'\bkpi\b', r'\brespect\s+des\s+d[eé]lais\b',
        r'\bp[eé]rim[eè]tre\b', r'\brisques?\b', r'\bbudget\b', r'\bd[eé]pendances?\b',
        r'\bressources\s+humaines\b', r'\bqualit[eé]\s+du\s+delivery\b', r'\bsatisfaction\s+client\b'
    ]
    return any(re.search(p, q.lower()) for p in patterns)


def extract_project_from_question(question):
    # Chercher après "du projet X", "projet X"
    m = re.search(
        r'(?:du\s+projet\s+|projet\s+)([A-Za-zÀ-ÿ\s]+?)(?:\s+en\b|\s+et\b|\s+ainsi\b|\?|$)',
        question, re.IGNORECASE
    )
    if m:
        c = re.sub(r'\b(et|ou|ainsi que|le|la|les|du|de la|des)\b', '', m.group(1), flags=re.IGNORECASE).strip()
        for proj in project_names:
            if proj.lower() == c.lower() or c.lower() in proj.lower() or proj.lower() in c.lower():
                return proj
    # Chercher le nom de projet directement dans la question
    for proj in sorted(project_names, key=len, reverse=True):
        if proj.lower() in question.lower():
            return proj
    return None


def get_latest_risk_level(project_name):
    docs = project_documents.get(project_name, [])
    risk_docs = []
    for doc in docs:
        if "[FEUILLE=KPI]" not in doc or "[KPI=Risques]" not in doc:
            continue
        sm = re.search(r'\[Semaine=(S\d{2}-\d{2})\]', doc)
        stm = re.search(r"statut='([^']+)'", doc)
        if sm and stm:
            risk_docs.append((sm.group(1), stm.group(1)))
    if not risk_docs:
        return "inconnu"
    risk_docs.sort(key=lambda x: int(x[0][1:3]), reverse=True)
    s = risk_docs[0][1].lower()
    if s in ["à surveiller", "à redresser"]:
        return "critique"
    elif s == "en contrôle":
        return "faible"
    return "inconnu"


def resolve_pronouns(q, ctx):
    if not ctx:
        return q
    for p in [r'\bce projet\b', r'\bcet projet\b', r'\bce dernier\b', r'\bcelle-ci\b']:
        q = re.sub(p, ctx, q, flags=re.IGNORECASE)
    return q


def _get_project_phase_at_week(project_name, week):
    """Retourne la phase du projet à une semaine donnée."""
    for doc in project_documents.get(project_name, []):
        if f"[Semaine={week}]" in doc and "[FEUILLE=Météo]" in doc:
            m = re.search(r'phase\s+([^,\n]+)', doc)
            if m:
                return m.group(1).strip()
    return None


def _phase_matches_query(phase_str, query_phase):
    """Vérifie si une phase correspond à la phase cherchée, avec normalisation."""
    if not phase_str:
        return False
    phase_norm = phase_str.lower().strip()
    query_norm = query_phase.lower().strip()

    # Correspondance directe
    if query_norm in phase_norm or phase_norm in query_norm:
        return True

    # Famille "production" : MEP, Mise en production, Prod, Pré-Prod
    if query_norm in PHASE_PRODUCTION_ALIASES and phase_norm in PHASE_PRODUCTION_ALIASES:
        return True

    # Famille "homologation"
    if query_norm in PHASE_HOMOLOGATION_ALIASES and phase_norm in PHASE_HOMOLOGATION_ALIASES:
        return True

    return False


# ==================== HANDLER PORTFOLIO BUDGET ====================

def handle_portfolio_budget(question):
    """
    Gère toutes les questions de budget au niveau portefeuille :
    - Budget total J/H de tous les projets
    - Budget total KTND de tous les projets
    - Budget total du portefeuille
    - Quel projet a le plus gros budget
    """
    q = question.lower()

    # Quel projet a le plus gros budget (J/H ou KTND)
    if re.search(r'\bquel\s+projet\b.*\bplus\s+(gros|grand)\s+budget\b', q):
        # Détecter J/H ou KTND
        is_ktnd = re.search(r'\bktnd?\b', q)
        key = "budget_ktnd" if is_ktnd else "budget_jh"
        unit = "KTND" if is_ktnd else "J/H"
        max_budget = -1
        max_proj = None
        for proj in project_names:
            b = project_info.get(proj, {}).get(key)
            if b and b > max_budget:
                max_budget = b
                max_proj = proj
        if max_proj:
            return f"Le projet avec le plus gros budget initial en {unit} est **{max_proj}** avec {max_budget} {unit}."
        return "Information non disponible."

    # Budget total en KTND de tous les projets
    if re.search(r'\bktnd?\b', q):
        budgets = []
        for proj in project_names:
            b = project_info.get(proj, {}).get("budget_ktnd")
            if b:
                budgets.append(f"{proj}: {b} KTND")
        total = sum(project_info.get(p, {}).get("budget_ktnd", 0) or 0 for p in project_names)
        detail = " | ".join(budgets) if budgets else "Non disponible"
        return f"Budget total en KTND de tous les projets : **{total} KTND**.\nDétail : {detail}"

    # Budget total J/H de tous les projets (ou budget total du portefeuille)
    total_jh = sum(project_info.get(p, {}).get("budget_jh", 0) or 0 for p in project_names)
    total_ktnd = sum(project_info.get(p, {}).get("budget_ktnd", 0) or 0 for p in project_names)
    details_jh = " | ".join([
        f"{p}: {project_info.get(p, {}).get('budget_jh', '?')} J/H"
        for p in project_names
    ])
    return (
        f"Budget total du portefeuille : **{total_jh} J/H** et **{total_ktnd} KTND**.\n"
        f"Détail J/H : {details_jh}"
    )


# ==================== HANDLER TREND ANALYSIS ====================

def handle_trend_analysis(question, proj):
    """
    Gère les questions d'évolution/tendance dans le temps :
    - Comment a évolué la santé de X ?
    - Évolution des risques de X sur tout le projet
    - La santé de X s'améliore-t-elle ?
    """
    q = question.lower()

    if not proj:
        return "Veuillez préciser un projet pour analyser son évolution."

    # Détecter le sujet de l'évolution
    is_health = re.search(r'\bsant[eé]\b|\bscore\b', q)
    is_risks = re.search(r'\brisques?\b', q)
    is_kpi = re.search(r'\bavancement\b|\bbudget\b|\bqualit[eé]\b|\bsatisfaction\b', q)

    if is_health:
        # Évolution de la santé : calculer les scores sur toutes les semaines disponibles
        all_weeks = sorted(
            {m.group(1) for doc in project_documents.get(proj, [])
             for m in [re.search(r'\[Semaine=(S\d{2}-\d{2})\]', doc)] if m},
            key=lambda w: int(w[1:3])
        )

        if not all_weeks:
            return f"Aucune donnée disponible pour analyser l'évolution de la santé du projet {proj}."

        # Calculer le score sur les semaines clés (début, milieu, fin)
        sample_weeks = all_weeks[::max(1, len(all_weeks) // 6)][:6]  # max 6 points
        scores_data = []
        for w in sample_weeks:
            h = compute_health_score_advanced(proj, w)
            if h["score"] > 0:
                scores_data.append(f"  - {w} : {h['score']}/100 ({h['level']}) — phase {h.get('phase', 'N/A')}")

        if not scores_data:
            return f"Données insuffisantes pour analyser l'évolution de la santé du projet {proj}."

        scores_text = "\n".join(scores_data)
        latest = compute_health_score_advanced(proj)

        return ask_llm(f"""
Tu es un analyste PMO senior. Voici l'évolution du score de santé du projet {proj} :

SCORES PAR PÉRIODE :
{scores_text}

SCORE ACTUEL : {latest['score']}/100 — {latest['level']} (semaine {latest['week']})

Rédige une analyse de l'évolution comprenant :
1. TENDANCE GÉNÉRALE : le projet s'améliore-t-il, se dégrade-t-il, ou reste-t-il stable ?
2. PHASES CRITIQUES : identifier les périodes de dégradation et leurs causes probables
3. SITUATION ACTUELLE : verdict sur l'état actuel
4. RECOMMANDATION : que surveiller pour maintenir ou améliorer la trajectoire

Réponds en français, professionnel, concis, sans emojis.
""", max_tokens=800)

    elif is_risks:
        # Évolution des risques sur tout le projet
        return advanced_risk_synthesis(proj)

    else:
        # Évolution générale d'un KPI
        kpi_name = _extract_kpi_name(question)
        if kpi_name and kpi_name != "budget_consomme":
            evolution = get_kpi_evolution(proj)
            kpi_history = []
            for e in evolution:
                all_kpis = get_kpi_for_week(proj, e["week"])
                kd = find_kpi(all_kpis, kpi_name)
                if kd.get("statut"):
                    kpi_history.append(f"  - {e['week']} : {kd['statut']} (tendance: {kd.get('tendance', 'N/A')})")

            if not kpi_history:
                return f"Aucune donnée disponible pour le KPI {kpi_name} du projet {proj}."

            history_text = "\n".join(kpi_history)
            return ask_llm(f"""
Tu es un analyste PMO. Voici l'historique du KPI {kpi_name} pour le projet {proj} :

{history_text}

Rédige une analyse concise (3-4 phrases) de l'évolution :
- Tendance générale (amélioration / dégradation / stable)
- Périodes critiques identifiées
- Situation actuelle et recommandation

Réponds en français, sans emojis.
""", max_tokens=400)

        # Évolution générale du projet
        return advanced_risk_synthesis(proj)


# ==================== HANDLER GLOBAL ANALYTICAL ====================

def handle_global_analytical(question, proj):
    """
    Gère les questions analytiques globales sans hardcoding de questions.
    """
    q = question.lower()

    # ── Budget total portefeuille ──
    if re.search(r'\bbudget\s+total\b.*\bportefeuille\b|'
                 r'\bbudget\s+total\b.*\btous\b|'
                 r'\bportefeuille\b.*\bbudget\b', q):
        return handle_portfolio_budget(question)

    # ── Quel projet le plus critique / le plus exposé ──
    if re.search(r'\bquel\s+projet\b.*\b(le\s+plus\s+critique|plus\s+expos[eé]|plus\s+de\s+risques?)\b|'
                 r'\bprojet\b.*\b(le\s+plus\s+critique|plus\s+expos[eé])\b', q):
        result = get_project_with_most_critical_risks(threshold=2)
        return result

    # ── Classer les projets par niveau de risque ──
    if re.search(r'\bclas(se[rz]?|sement)\b.*\bprojets?\b', q):
        return aggregate_risks_all_projects()

    # ── Quel projet a commencé le plus tôt ──
    if re.search(r'\bcommenc[eé]\s+le\s+plus\s+t[oô]t\b|\bplus?\s+ancien\b', q):
        earliest_date, earliest_proj = None, None
        for p in project_names:
            for doc in project_documents.get(p, []):
                if "[FEUILLE=Infos_projet]" in doc:
                    m = re.search(r'[Dd]ate\s+d[eé]but.*?(\d{4}-\d{2}-\d{2})', doc)
                    if m:
                        d = pd.to_datetime(m.group(1))
                        if earliest_date is None or d < earliest_date:
                            earliest_date = d
                            earliest_proj = p
        return f"Le projet qui a commencé le plus tôt est **{earliest_proj}** ({format_date(str(earliest_date.date()))})." if earliest_proj else "Information non disponible."

    # ── Date de fin la plus tardive ──
    if re.search(r'\bdat.*fin\s+la\s+plus\s+tardive\b|\btermine\s+le\s+plus\s+tard\b', q):
        latest_date, latest_proj = None, None
        for p in project_names:
            for doc in project_documents.get(p, []):
                if "[FEUILLE=Infos_projet]" in doc:
                    m = re.search(r'Date fin\s*:\s*(\d{4}-\d{2}-\d{2})', doc)
                    if m:
                        d = pd.to_datetime(m.group(1))
                        if latest_date is None or d > latest_date:
                            latest_date = d
                            latest_proj = p
        return f"Le projet avec la date de fin la plus tardive est **{latest_proj}** ({format_date(str(latest_date.date()))})." if latest_proj else "Information non disponible."

    # ── Date de fin d'un projet spécifique ──
    if re.search(r'\bquand\s+se\s+termine\b|\bdate\s+(de\s+)?fin\b', q) and proj:
        for doc in project_documents.get(proj, []):
            if "[FEUILLE=Infos_projet]" in doc:
                m = re.search(r'Date fin\s*:\s*(\d{4}-\d{2}-\d{2})', doc)
                if m:
                    return f"Le projet **{proj}** se termine le {format_date(m.group(1))}."
        return f"Information non disponible pour {proj}."

    # ── À quelle semaine le projet X est passé en phase Y ──
    if re.search(r'\b[aà]\s+quelle\s+semaine\b.*\bpass[eé]\b', q) and proj:
        phase_m = re.search(r'(?:phase\s+|en\s+phase\s+)([A-Za-zÀ-ÿ/\s\-]+?)(?:\s+\?|$)', question, re.IGNORECASE)
        if phase_m:
            target_phase = phase_m.group(1).strip()
            w = get_phase_transition(proj, target_phase)
            if w:
                return f"Le projet **{proj}** est passé en phase **{target_phase}** à la semaine {w}."
            return f"Aucune transition vers la phase {target_phase} trouvée pour {proj}."
        return "Veuillez préciser la phase cible."

    # ── Quels projets étaient en phase X en semaine Y ──
    if re.search(r'\bquels?\s+projets?\b.*\bphase\b', q):
        week_q = extract_week(question)

        # Extraire la phase cherchée
        phase_m = re.search(r'(?:phase\s+(?:de\s+)?|en\s+phase\s+)([A-Za-zÀ-ÿ/\s\-]+?)(?:\s+en\s+S|\?|$)', question, re.IGNORECASE)
        target_phase = phase_m.group(1).strip() if phase_m else None

        if not target_phase:
            return handle_rag_question(question, proj)

        found = []
        for p in project_names:
            for doc in project_documents.get(p, []):
                if "[FEUILLE=Météo]" not in doc:
                    continue
                if week_q and week_q not in doc:
                    continue
                phase_in_doc = None
                m = re.search(r'phase\s+([^,\n]+)', doc)
                if m:
                    phase_in_doc = m.group(1).strip()
                if phase_in_doc and _phase_matches_query(phase_in_doc, target_phase):
                    if p not in found:
                        found.append(p)

        if found:
            period = f" en {week_q}" if week_q else ""
            return f"Les projets en phase **{target_phase}**{period} sont : {', '.join(found)}."
        return f"Aucun projet trouvé en phase {target_phase}" + (f" en {week_q}." if week_q else ".")

    # ── Quel projet a tel code ──
    if re.search(r'\bcode\b.*\b(P[A-Z0-9]+)\b', q, re.IGNORECASE):
        code_m = re.search(r'\b(P[A-Z0-9]+|PC\d+)\b', question, re.IGNORECASE)
        if code_m:
            target_code = code_m.group(1).upper()
            for p in project_names:
                for doc in project_documents.get(p, []):
                    if "[FEUILLE=Infos_projet]" in doc and target_code.lower() in doc.lower():
                        return f"Le projet avec le code **{target_code}** est **{p}**."
            return f"Aucun projet trouvé avec le code {target_code}."

    # ── Diagnostic portefeuille ──
    if re.search(r'\bdiagnostic\b.*\bportefeuille\b|\b[eé]tat\s+des\s+lieux\b.*\bportefeuille\b', q):
        return handle_global_health(question)

    # Fallback RAG
    return handle_rag_question(question, proj)


# ==================== HANDLER GLOBAL HEALTH ====================

def handle_global_health(question):
    """
    Gère les questions analytiques sur la santé globale du portefeuille.
    """
    scores = []
    for p in project_names:
        h = compute_health_score_advanced(p)
        scores.append({
            "project": p, "score": h["score"],
            "level": h["level"], "week": h["week"]
        })
    scores.sort(key=lambda x: x["score"], reverse=True)

    q = question.lower()

    if re.search(r'\b(meilleure?|mieux|plus\s+haute?)\b', q):
        best = scores[0]
        return (f"Le projet en meilleure santé est **{best['project']}** "
                f"avec un score de {best['score']}/100 ({best['level']}) "
                f"à la semaine {best['week']}.")

    if re.search(r'\b(plus?\s+d[eé]grad|pire|plus?\s+mauvais|moins\s+bonne?|plus?\s+basse?)\b', q):
        worst = scores[-1]
        return (f"Le projet avec la santé la plus dégradée est **{worst['project']}** "
                f"avec un score de {worst['score']}/100 ({worst['level']}) "
                f"à la semaine {worst['week']}.")

    # Comparaison complète de tous les projets
    summary = "\n".join([
        f"- {s['project']}: {s['score']}/100 ({s['level']}) — semaine {s['week']}"
        for s in scores
    ])
    return ask_llm(f"""
Tu es un analyste PMO senior. Voici les scores de santé des projets du portefeuille :

{summary}

Rédige une comparaison structurée comprenant :
1. Classement des projets du meilleur au plus dégradé
2. Points forts du projet le mieux classé
3. Points d'attention du projet le moins bien classé
4. Recommandation globale

Réponds en français, professionnel, sans emojis.
""", max_tokens=800)


# ==================== HANDLER KPI SIMPLE ====================

def handle_kpi_simple(classified, proj, question):
    """
    Répond aux questions sur un KPI précis à une semaine donnée.
    """
    kpi_name = classified.get("kpi_name") or _extract_kpi_name(question)
    week = classified.get("week") or extract_week(question)

    if not proj:
        return handle_rag_question(question, None)

    # Budget consommé spécial
    if kpi_name == "budget_consomme":
        consumed = get_budget_consumption(proj, week)
        wk = week or get_latest_week(proj)
        if consumed is not None:
            return f"Le budget consommé du projet **{proj}** en {wk} est de **{consumed} J/H**."
        return f"Budget consommé non renseigné pour {proj}" + (f" en {wk}." if wk else ".")

    # KPI classique
    if not kpi_name:
        return handle_rag_question(question, proj)

    target_week = week or get_latest_week(proj)
    if not target_week:
        return f"Aucune semaine disponible pour {proj}."

    all_kpis = get_kpi_for_week(proj, target_week)
    kpi_data = find_kpi(all_kpis, kpi_name)

    statut = kpi_data.get("statut")
    tendance = kpi_data.get("tendance")

    if not statut:
        # Chercher dans les semaines proches si la semaine exacte manque
        all_weeks = sorted(
            {m.group(1) for doc in project_documents.get(proj, [])
             for m in [re.search(r'\[Semaine=(S\d{2}-\d{2})\]', doc)] if m},
            key=lambda w: int(w[1:3])
        )
        # Trouver la semaine disponible la plus proche
        if all_weeks:
            target_num = int(target_week[1:3])
            closest = min(all_weeks, key=lambda w: abs(int(w[1:3]) - target_num))
            all_kpis_closest = get_kpi_for_week(proj, closest)
            kpi_data_closest = find_kpi(all_kpis_closest, kpi_name)
            if kpi_data_closest.get("statut"):
                statut = kpi_data_closest["statut"]
                tendance = kpi_data_closest.get("tendance")
                target_week = closest

        if not statut:
            # Statut général
            if kpi_name.lower() in ["statut général", "statut general"]:
                sg = get_general_status(proj, target_week)
                return f"Le statut général du projet **{proj}** en {target_week} est '{sg}'." if sg else "Information non disponible."
            return f"Information non disponible pour le KPI **{kpi_name}** du projet {proj} en {target_week}."

    q_lower = question.lower()
    if re.search(r'\btendance\b', q_lower):
        return f"La tendance du KPI **{kpi_name}** du projet **{proj}** en {target_week} est '{tendance}'."
    if re.search(r'\bstatut\b', q_lower):
        return f"Le statut du KPI **{kpi_name}** du projet **{proj}** en {target_week} est '{statut}'."
    if re.search(r'\b[eé]volue\b|\b[eé]volution\b', q_lower):
        return f"Le KPI **{kpi_name}** du projet **{proj}** est à '{statut}' (tendance : {tendance}) en {target_week}."

    return f"Le KPI **{kpi_name}** du projet **{proj}** en {target_week} : statut='{statut}', tendance='{tendance}'."


# ==================== HANDLER MULTI-ENTITÉS ====================

def handle_multi_entities(classified, proj, question):
    """
    Répond aux questions composées demandant plusieurs informations d'un même projet.
    """
    entities = classified.get("entities_requested", []) or _extract_entities(question)
    week = classified.get("week") or extract_week(question)

    if not entities or not proj:
        return handle_rag_question(question, proj)

    parts = []
    for entity in entities:
        if entity == "budget":
            jh = project_info.get(proj, {}).get("budget_jh")
            ktnd = project_info.get(proj, {}).get("budget_ktnd")
            b_parts = []
            if jh: b_parts.append(f"{jh} J/H")
            if ktnd: b_parts.append(f"{ktnd} KTND")
            parts.append(f"Budget : {' et '.join(b_parts)}" if b_parts else "Budget : Information non disponible")

        elif entity == "chef":
            chef = project_chef.get(proj, "Information non disponible")
            parts.append(f"Chef de projet : {chef}")

        elif entity == "phase":
            phase = project_current_phase.get(proj, "Information non disponible")
            parts.append(f"Phase actuelle : {phase}")

        elif entity == "sponsor":
            sponsor = project_sponsor.get(proj, "Information non disponible")
            parts.append(f"Sponsor : {sponsor}")

        elif entity == "date_debut":
            found = False
            for doc in project_documents.get(proj, []):
                if "[FEUILLE=Infos_projet]" in doc:
                    m = re.search(r'[Dd]ate\s+d[eé]but.*?(\d{4}-\d{2}-\d{2})', doc)
                    if m:
                        parts.append(f"Date de début : {format_date(m.group(1))}")
                        found = True
                        break
            if not found:
                parts.append("Date de début : Information non disponible")

        elif entity == "date_fin":
            found = False
            for doc in project_documents.get(proj, []):
                if "[FEUILLE=Infos_projet]" in doc:
                    m = re.search(r'Date fin\s*:\s*(\d{4}-\d{2}-\d{2})', doc)
                    if m:
                        parts.append(f"Date de fin : {format_date(m.group(1))}")
                        found = True
                        break
            if not found:
                parts.append("Date de fin : Information non disponible")

        elif entity == "code":
            found = False
            for doc in project_documents.get(proj, []):
                if "[FEUILLE=Infos_projet]" in doc:
                    m = re.search(r'[Cc]ode\s+projet.*?:\s*(P[\w]+)', doc)
                    if m:
                        parts.append(f"Code projet : {m.group(1)}")
                        found = True
                        break
            if not found:
                parts.append("Code projet : Information non disponible")

        elif entity == "statut_general":
            target_week = week or get_latest_week(proj)
            sg = get_general_status(proj, target_week)
            parts.append(f"Statut général ({target_week}) : {sg or 'Information non disponible'}")

        elif entity == "budget_consomme":
            target_week = week or get_latest_week(proj)
            consumed = get_budget_consumption(proj, target_week)
            parts.append(
                f"Budget consommé ({target_week}) : {consumed} J/H" if consumed is not None
                else f"Budget consommé ({target_week}) : non renseigné"
            )

        elif entity.startswith("kpi_statut:"):
            kpi_n = entity.split(":", 1)[1].strip()
            target_week = week or get_latest_week(proj)
            all_kpis = get_kpi_for_week(proj, target_week)
            kd = find_kpi(all_kpis, kpi_n)
            if kd.get("statut"):
                parts.append(f"KPI {kpi_n} ({target_week}) : statut='{kd['statut']}', tendance='{kd.get('tendance', 'N/A')}'")
            else:
                parts.append(f"KPI {kpi_n} ({target_week}) : Information non disponible")

        elif entity.startswith("kpi_tendance:"):
            kpi_n = entity.split(":", 1)[1].strip()
            target_week = week or get_latest_week(proj)
            all_kpis = get_kpi_for_week(proj, target_week)
            kd = find_kpi(all_kpis, kpi_n)
            if kd.get("tendance"):
                parts.append(f"Tendance {kpi_n} ({target_week}) : '{kd['tendance']}'")
            else:
                parts.append(f"Tendance {kpi_n} ({target_week}) : Information non disponible")

    if not parts:
        return handle_rag_question(question, proj)

    return f"**{proj}**\n" + "\n".join([f"• {p}" for p in parts])


# ==================== HANDLER FACTUEL GLOBAL ====================

def handle_global_factual(question):
    """
    Gère les questions factuelles concernant tous les projets (chefs, sponsors, phases).
    """
    q = question.lower()

    # Chef de projet de tous les projets
    if re.search(r'\bchef\b', q):
        if project_chef:
            return "\n".join([f"• **{p}** : {c}" for p, c in project_chef.items()])
        return "Information non disponible."

    # Sponsor de tous les projets
    if re.search(r'\bsponsor\b', q):
        if project_sponsor:
            return "\n".join([f"• **{p}** : {s}" for p, s in project_sponsor.items()])
        return "Information non disponible."

    # Phase actuelle de tous les projets
    if re.search(r'\bphase\b', q):
        if project_current_phase:
            return "\n".join([f"• **{p}** : {ph}" for p, ph in project_current_phase.items()])
        return "Information non disponible."

    # Budget de tous les projets
    if re.search(r'\bbudget\b', q):
        return handle_portfolio_budget(question)

    return handle_rag_question(question, None)


# ==================== HANDLER CLARIFICATION ====================

def handle_clarification(question):
    """
    Retourne une réponse de clarification intelligente pour les questions trop vagues.
    """
    q = question.lower().strip().rstrip("?").strip()

    if re.search(r'\bbudget\b', q):
        return ("Souhaitez-vous le budget d'un projet spécifique ou le budget total du portefeuille ?\n"
                "Exemples : 'Quel est le budget du projet Data Fraud Detection ?' "
                "ou 'Budget total de tous les projets'.")

    if re.search(r'\bsant[eé]\b', q):
        return ("Pour évaluer la santé d'un projet, précisez lequel.\n"
                "Exemples : 'Santé du projet Orion Data Platform' "
                "ou 'Quel projet est en meilleure santé ?'.")

    if re.search(r'\brisques?\b', q):
        return ("Pour une synthèse des risques, précisez un projet ou demandez un bilan global.\n"
                "Exemples : 'Risques du projet Scoring Crédit Retail' "
                "ou 'Bilan des risques sur tous les projets'.")

    if re.search(r'\bcompare\b', q):
        return ("Pour comparer des projets, précisez lesquels.\n"
                "Exemple : 'Compare Data Fraud Detection et Scoring Crédit Retail'.")

    return ("Pourriez-vous préciser votre question ? Vous pouvez demander :\n"
            "• La santé, les risques, les actions ou une prédiction pour un projet spécifique\n"
            "• Un bilan global du portefeuille\n"
            "• Des informations sur les KPI, budgets ou équipes")


# ==================== RAG GÉNÉRAL ====================

def handle_rag_question(question, proj):
    """Réponse RAG générale pour les questions factuelles."""
    feuille = "KPI" if is_kpi_question(question) else None
    k_final = 50 if feuille else 30
    top_k = 10 if feuille else 5
    if is_list_question(question):
        k_final, top_k = 100, 20

    raw_ctx = retrieve_filtered_context(question, k_final=k_final, feuille=feuille, force_project=proj)
    if not raw_ctx:
        return f"Information non disponible pour {proj}." if proj else "Information non disponible."

    blocks, cur = [], []
    for line in raw_ctx.split('\n'):
        if line.startswith("[PROJET="):
            if cur: blocks.append("\n".join(cur))
            cur = [line]
        else:
            cur.append(line)
    if cur: blocks.append("\n".join(cur))

    best = rerank_passages(question, blocks, top_k=top_k) if len(blocks) > top_k else blocks
    ctx = "\n".join(best)

    prompt = f"""
Tu es un assistant PMO spécialisé. Réponds UNIQUEMENT depuis le CONTEXTE.
RÈGLES :
- {"Liste tous les éléments demandés." if is_list_question(question) else "Réponse concise, une phrase si possible."}
- Si question sur un KPI : donne statut et tendance.
- Pas d'introduction ni d'explication inutile.
- Transforme les dates en français (ex: 15 décembre 2026).
- Si absent du contexte : "Information non disponible."
- Ne cite jamais les tags [PROJET=...], [Semaine=...].
- Réponds toujours en français.

CONTEXTE :
{ctx}

QUESTION : {question}
RÉPONSE :"""

    answer = ask_llm(prompt)
    answer = remove_tags(answer).strip()

    # Post-traitement dates
    for annee, mois, jour in re.findall(r'(\d{4})-(\d{2})-(\d{2})(?:\s+\d{2}:\d{2}:\d{2})?', answer):
        old = f"{annee}-{mois}-{jour}"
        for suf in ["", " 00:00:00"]:
            if old + suf in answer:
                answer = answer.replace(old + suf, format_date(old + suf))
    answer = re.sub(r'(\d{1,2}\s+\w+\s+\d{4})\s+\d{2}:\d{2}:\d{2}', r'\1', answer)

    return answer or "Information non disponible."


# ==================== DISPATCH PRINCIPAL ====================

def process_single_question(question, ui_project, context_project=None):
    if context_project:
        question = resolve_pronouns(question, context_project)

    classified = classify_intent(question, project_names, use_llm=True)
    intent = classified.get("intent", "factual")
    proj = classified.get("project") or extract_project_from_question(question) or ui_project or context_project
    week = classified.get("week") or extract_week(question)
    week_range = classified.get("week_range")
    start_week = week_range[0] if week_range else week
    end_week = week_range[1] if week_range and len(week_range) > 1 else (week if week_range else None)

    print(f"[dispatch] intent={intent} proj={proj} week={week} "
          f"kpi={classified.get('kpi_name')} entities={classified.get('entities_requested')} "
          f"is_global={classified.get('is_global')} src={classified.get('_source', '?')}")

    # ── Clarification ──
    if intent == "clarification":
        return handle_clarification(question)

    # ── Portfolio budget ──
    if intent == "portfolio_budget":
        return handle_portfolio_budget(question)

    # ── Global health ──
    if intent == "global_health":
        return handle_global_health(question)

    # ── Global analytical ──
    if intent == "global_analytical":
        return handle_global_analytical(question, proj)

    # ── Global info (bilan risques portefeuille) ──
    if intent == "global_info":
        weeks_found = re.findall(r'S\d{2}-\d{2}', question)
        return aggregate_risks_all_projects(
            weeks_found[0] if weeks_found else None,
            weeks_found[1] if len(weeks_found) > 1 else None
        )

    # ── Factuel global (chef/sponsor/phase de tous les projets) ──
    if intent == "factual" and classified.get("is_global"):
        return handle_global_factual(question)

    # ── Trend analysis ──
    if intent == "trend_analysis":
        return handle_trend_analysis(question, proj)

    # ── KPI simple ──
    if intent == "kpi_simple":
        return handle_kpi_simple(classified, proj, question)

    # ── Multi-entités composées ──
    entities = classified.get("entities_requested", [])
    if intent == "factual" and len(entities) >= 2 and proj:
        return handle_multi_entities(classified, proj, question)

    # ── Axes analytiques par projet ──
    if intent == "risks":
        if not proj:
            return "Veuillez préciser un projet pour la synthèse des risques."
        return advanced_risk_synthesis(proj, start_week, end_week)

    if intent == "prediction":
        if not proj:
            return "Veuillez préciser un projet pour une prédiction."
        return predict_problems(proj)

    if intent == "actions":
        if not proj:
            proj = extract_project_from_question(question) or ui_project
        if not proj:
            return "Pour des propositions d'actions, veuillez préciser un projet."
        actions = suggest_actions(proj, week, user_question=question)
        if isinstance(actions, list) and actions:
            lines = []
            for a in actions:
                if isinstance(a, dict):
                    line = a.get("action", "")
                    if a.get("impact"): line += f" | Impact : {a['impact']}"
                    if a.get("effort"): line += f" | Effort : {a['effort']}"
                    if a.get("delai"): line += f" | Délai : {a['delai']}"
                    lines.append(line)
                else:
                    lines.append(str(a))
            return "Voici les actions recommandées :\n\n" + "\n".join(lines)
        return str(actions)

    if intent == "health":
        if not proj:
            return "Veuillez préciser un projet pour évaluer sa santé."
        h = compute_health_score_advanced(proj, week)
        expl = generate_health_explanation_advanced(proj, h)
        return f"**Score de santé du projet {proj} : {h['score']}/100** – {h['level']}\n\n{expl}"

    if intent == "report":
        if not proj:
            return "Veuillez préciser un projet pour le rapport de risques."
        return produce_risk_report(proj, start_week, end_week)

    if intent == "compare":
        projs = [p for p in project_names if p.lower() in question.lower()]
        if len(projs) >= 2:
            return compare_projects(projs[0], projs[1], start_week, end_week)
        return aggregate_risks_all_projects()

    # ── Factuel simple ──
    if intent == "factual":
        q = question.lower()

        if entities and proj:
            return handle_multi_entities(classified, proj, question)

        # Chef de projet
        if re.search(r'\bchef\s+de\s+projet\b', q):
            if proj:
                chef = project_chef.get(proj)
                return f"Le chef de projet du projet **{proj}** est **{chef}**." if chef else f"Information non disponible pour {proj}."
            return handle_global_factual(question)

        # Sponsor
        if re.search(r'\bsponsor\b', q):
            if proj:
                sp = project_sponsor.get(proj)
                return f"Le sponsor du projet **{proj}** est **{sp}**." if sp else f"Information non disponible pour {proj}."
            return handle_global_factual(question)

        # Budget
        is_jh = bool(re.search(r"budget.*en\s+j/?h|budget\s+j/?h", q))
        is_ktnd = bool(re.search(r"budget.*en\s+ktnd?|budget\s+ktnd?", q))
        has_budget = bool(re.search(r'\bbudget\b', q))

        if (is_jh or is_ktnd or has_budget) and proj:
            jh = project_info.get(proj, {}).get("budget_jh")
            ktnd = project_info.get(proj, {}).get("budget_ktnd")
            if is_jh:
                return f"Le budget J/H du projet **{proj}** est de **{jh} J/H**." if jh else f"Information non disponible pour {proj}."
            if is_ktnd:
                return f"Le budget KTND du projet **{proj}** est de **{ktnd} KTND**." if ktnd else f"Information non disponible pour {proj}."
            b_parts = []
            if jh: b_parts.append(f"{jh} J/H")
            if ktnd: b_parts.append(f"{ktnd} KTND")
            return f"Le budget du projet **{proj}** est de {' et '.join(b_parts)}." if b_parts else f"Information non disponible pour {proj}."

        # Phase actuelle d'un projet
        if re.search(r'\bphase\b', q) and proj and not re.search(r'\bquels?\s+projets?\b', q):
            phase = project_current_phase.get(proj)
            return f"La phase actuelle du projet **{proj}** est **{phase}**." if phase else f"Information non disponible pour {proj}."

        # Date de fin d'un projet
        if re.search(r'\bquand\s+se\s+termine\b|\bdate\s+(de\s+)?fin\b', q) and proj:
            return handle_global_analytical(question, proj)

        # Quel projet n'a pas de sponsor
        if re.search(r'\bquel\s+projet\b.*\bpas\s+de\s+sponsor\b|\baucun\s+sponsor\b', q):
            no_sponsor = [p for p in project_names if not project_sponsor.get(p)]
            return f"Projets sans sponsor renseigné : {', '.join(no_sponsor)}." if no_sponsor else "Tous les projets ont un sponsor renseigné."

    # ── Fallback RAG ──
    return handle_rag_question(question, proj)


# ==================== ROUTES ====================

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/projects", methods=["GET"])
def list_projects():
    projects = []
    for f in os.listdir(DATA_FOLDER):
        if f.endswith(".xlsx"):
            try:
                df = pd.read_excel(os.path.join(DATA_FOLDER, f), sheet_name="Infos_projet")
                if "Projet" in df.columns and not df.empty:
                    projects.append(str(df["Projet"].iloc[0]).strip())
            except:
                pass
    return jsonify({"projects": projects})


@app.route("/project_status", methods=["GET"])
def project_status():
    return jsonify({p: {"risk": get_latest_risk_level(p)} for p in project_names})


@app.route("/upload_project", methods=["POST"])
def upload_project():
    if "file" not in request.files:
        return jsonify({"message": "Aucun fichier"})
    f = request.files["file"]
    if not f.filename:
        return jsonify({"message": "Nom invalide"})
    path = os.path.join(DATA_FOLDER, f.filename)
    f.save(path)
    from rag import load_project
    load_project(f.filename)
    return jsonify({"message": "Projet ajouté"})


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        question = data.get("message", "").strip()
        ui_project = data.get("project")
        context_project = data.get("context_project")

        if not question:
            return jsonify({"answer": "Veuillez poser une question.", "detected_project": None})

        answer = process_single_question(question, ui_project, context_project)
        detected = extract_project_from_question(question) or ui_project

        return jsonify({"answer": answer, "detected_project": detected})

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Erreur interne dans /chat : {e}")
        return jsonify({"answer": "Une erreur technique est survenue. Veuillez réessayer.", "detected_project": None})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)