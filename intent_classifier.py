"""
intent_classifier.py v3 — Classification LLM enrichie pour PMO AI Copilot
Corrections v3 :
- Nouveau intent "portfolio_budget" pour les questions budget portefeuille
- Détection robuste de "phase de production" incluant MEP/Prod/Mise en production
- Correction détection KPI multi-projets sans projet spécifié
- Nouveau intent "trend_analysis" pour les questions d'évolution/tendance globale
- Meilleure détection des questions chef/sponsor globales
- Seuil critique ajusté dans le classifier
"""

import re
import json
from llm import ask_llm

# ==================== SCHÉMA JSON v3 ====================
INTENT_SCHEMA = """
Réponds UNIQUEMENT en JSON valide, sans markdown ni preamble.

Schéma attendu :
{{
  "intent": "risks|actions|health|prediction|report|compare|global_info|global_health|global_analytical|portfolio_budget|kpi_simple|trend_analysis|factual|clarification",
  "project": "<nom exact parmi {project_list} ou null>",
  "week": "<S##-## ou null>",
  "week_range": ["<start>", "<end>"] ou null,
  "kpi_name": "<nom exact du KPI ou null>",
  "entities_requested": [],
  "needs_comparison": false,
  "is_global": false,
  "confidence": 0.9
}}

INTENTS :
- "risks"             → synthèse/résumé/analyse des risques d'UN projet précis
- "actions"           → que faire, plan d'action, recommandations, comment améliorer X, comment réduire X, comment réagir
- "health"            → santé, score de santé, état du projet, comment se porte UN projet précis
- "prediction"        → prédiction, anticipe, problèmes futurs, va-t-il prendre du retard, signaux d'alerte
- "report"            → rapport détaillé des risques
- "compare"           → comparaison entre deux projets nommés explicitement
- "global_info"       → bilan global risques, vue portefeuille risques, risques sur tous les projets
- "global_health"     → quel projet en meilleure/pire santé, compare santé de plusieurs projets, diagnostic portefeuille santé
- "global_analytical" → quel projet le plus critique/exposé, quel a commencé le plus tôt/tard, date fin la plus tardive, à quelle semaine un projet a changé de phase, quel projet a tel code, quels projets en phase X en semaine Y
- "portfolio_budget"  → budget total du portefeuille, budget total en J/H de tous les projets, budget en KTND de tous les projets, quel projet a le plus gros budget, somme des budgets
- "trend_analysis"    → comment a évolué X, évolution de Y dans le temps, tendance générale sur tout le projet, s'améliore-t-il
- "kpi_simple"        → statut OU tendance d'un KPI précis (Avancement, Budget, Risques, Qualité du delivery, Satisfaction client, Ressources humaines, Dépendances, Périmètre, Respect des délais) à une semaine donnée. Nécessite UN projet ET une semaine OU le projet dans le sélecteur UI.
- "factual"           → chef de projet d'un projet, sponsor d'un projet, budget d'un projet, phase d'un projet, date début/fin d'un projet, code projet, statut général d'une semaine. Peut être global (tous les chefs, tous les sponsors, phase de tous les projets).
- "clarification"     → question trop vague sans projet ni contexte suffisant (ex: "Budget ?", "Santé", "Risques" seuls sans projet)

CHAMP entities_requested — remplir quand la question demande PLUSIEURS informations factuelles d'un même projet :
Valeurs : "budget", "chef", "phase", "sponsor", "date_debut", "date_fin", "code", "statut_general", "budget_consomme", "kpi_statut:NomKPI", "kpi_tendance:NomKPI"
Exemples :
- "Budget et chef de projet de DFD" → entities_requested: ["budget", "chef"]
- "Sponsor et phase actuelle de Scoring" → entities_requested: ["sponsor", "phase"]
- "Statut général, KPI Avancement et KPI Budget en S08" → entities_requested: ["statut_general", "kpi_statut:Avancement", "kpi_statut:Budget"]

CHAMP kpi_name — remplir UNIQUEMENT pour intent "kpi_simple" :
- "Statut du KPI Avancement pour DFD en S12-26" → kpi_name: "Avancement"
- "Tendance du KPI Risques d'Orion" → kpi_name: "Risques"
- "Budget consommé en J/H en S16-26" → kpi_name: "budget_consomme"
- "Comment évolue l'avancement d'Orion" → kpi_name: "Avancement" (mais intent = kpi_simple car semaine ou projet précis)
- "Satisfaction client de DFD en S22-26" → kpi_name: "Satisfaction client"
- "KPI Ressources humaines d'Orion en S10-26" → kpi_name: "Ressources humaines"

RÈGLES CRITIQUES — lire attentivement :
1. "satisfaction client" = kpi_simple ou factual, JAMAIS actions
2. "comment évolue X" avec semaine précise = kpi_simple. Sans semaine et sans projet précis = trend_analysis
3. "tendance du KPI X en SYY-ZZ" = kpi_simple
4. "statut du KPI X" = kpi_simple
5. "budget consommé" = kpi_simple avec kpi_name="budget_consomme"
6. "budget total de tous les projets" / "budget en KTND de tous les projets" / "quel projet a le plus gros budget" = portfolio_budget
7. "quel projet est le plus critique" / "projet le plus exposé" = global_analytical
8. "quel projet en meilleure santé" / "quel projet en moins bonne santé" = global_health
9. "quels projets sont en phase X" = global_analytical (pas factual)
10. "bilan des risques sur tous les projets" = global_info
11. "lister tous les projets avec leur chef" / "qui est le chef de chaque projet" = factual avec is_global=true et entities_requested=["chef"]
12. "le sponsor de chaque projet" = factual avec is_global=true et entities_requested=["sponsor"]
13. "phase actuelle de chaque projet" = factual avec is_global=true et entities_requested=["phase"]
14. "comment a évolué la santé" / "s'améliore-t-il" / "évolution sur tout le projet" = trend_analysis
15. "compare les trois projets" / "compare tous les projets" = global_info (pas compare, car compare = 2 projets nommés)
16. "diagnostique du portefeuille" = global_health
17. actions SEULEMENT pour : "que faire", "comment améliorer X spécifique", "comment réduire X spécifique", "plan d'action", "recommandations pour améliorer", "quelles actions urgentes"
18. Une question vague sans projet (juste "Budget ?", "Santé", "Risques") = clarification
19. "à quelle semaine le projet X est passé en phase Y" = global_analytical
20. "quel projet a le code PC003" = global_analytical
21. "KPI Avancement et KPI Budget en S14-26" sans projet = kpi_simple avec project=null (UI project résoudra)
22. "évolution des risques sur tout le projet" = trend_analysis (pas risks ni kpi_simple)

Projets connus : {project_list}

Question : {question}
JSON :"""

# ==================== KPI NAMES ====================
KPI_NAMES = [
    "Avancement", "Budget", "Risques", "Qualité du delivery", "Satisfaction client",
    "Ressources humaines", "Dépendances", "Périmètre", "Respect des délais"
]

KPI_NAME_ALIASES = {
    "avancement": "Avancement",
    "budget": "Budget",
    "risques": "Risques",
    "risque": "Risques",
    "qualite du delivery": "Qualité du delivery",
    "qualité du delivery": "Qualité du delivery",
    "qualite": "Qualité du delivery",
    "qualité": "Qualité du delivery",
    "satisfaction client": "Satisfaction client",
    "satisfaction": "Satisfaction client",
    "ressources humaines": "Ressources humaines",
    "ressources": "Ressources humaines",
    "dependances": "Dépendances",
    "dépendances": "Dépendances",
    "dependance": "Dépendances",
    "périmètre": "Périmètre",
    "perimetre": "Périmètre",
    "respect des delais": "Respect des délais",
    "respect des délais": "Respect des délais",
    "delais": "Respect des délais",
    "délais": "Respect des délais",
}

# ==================== PATTERNS FALLBACK ROBUSTES ====================

def _extract_kpi_name(question):
    """Extrait le nom normalisé du KPI depuis la question."""
    q = question.lower()
    # budget consommé
    if re.search(r'budget\s+consomm[eé]', q):
        return "budget_consomme"
    # Chercher les noms connus par ordre de longueur décroissante (évite "budget" avant "qualité du delivery")
    sorted_aliases = sorted(KPI_NAME_ALIASES.keys(), key=len, reverse=True)
    for alias in sorted_aliases:
        if alias in q:
            return KPI_NAME_ALIASES[alias]
    return None


def _extract_entities(question):
    """Extrait les entités demandées dans une question composée."""
    q = question.lower()
    entities = []
    seen = set()

    def add(e):
        if e not in seen:
            seen.add(e)
            entities.append(e)

    # Budget global (pas budget consommé)
    if re.search(r'\bbudget\b', q) and not re.search(r'budget\s+consomm[eé]', q) \
            and not re.search(r'\bkpi\b.*\bbudget\b', q):
        add("budget")
    if re.search(r'\bbudget\s+consomm[eé]\b', q):
        add("budget_consomme")
    if re.search(r'\bchef\b', q):
        add("chef")
    if re.search(r'\bphase\b', q):
        add("phase")
    if re.search(r'\bsponsor\b', q):
        add("sponsor")
    if re.search(r'\bdate\s+(de\s+)?d[eé]but\b', q):
        add("date_debut")
    if re.search(r'\bdate\s+(de\s+)?fin\b|\bquand\s+se\s+termine\b', q):
        add("date_fin")
    if re.search(r'\bcode\s*(projet)?\b', q):
        add("code")
    if re.search(r'\bstatut\s+g[eé]n[eé]ral\b', q):
        add("statut_general")

    # KPI dans entities (ex: "KPI Avancement et KPI Budget")
    kpi_mentions = re.findall(r'kpi\s+([\w\s]+?)(?:\s+et\s+(?:kpi\s+)?|\s+en\s+S|\?|$)', q)
    for ke in kpi_mentions:
        kpi_n = _extract_kpi_name(ke.strip())
        if kpi_n:
            add(f"kpi_statut:{kpi_n}")

    # Tendance dans entities
    if re.search(r'\btendance\b', q):
        kpi_n = _extract_kpi_name(q)
        if kpi_n:
            add(f"kpi_tendance:{kpi_n}")

    return entities


# ==================== PATTERNS FALLBACK RÉGEX ====================

RISK_PATTERNS = [
    r'\bsynth[eè]se\b.*\brisques?\b',
    r'\brisques?\s+encourrus?\b',
    r'\brisques?\s+du\s+projet\b',
    r'\br[eé]sum[eé]\s+des\s+risques?\b',
    r'\bquels?\s+sont\s+les\s+risques?\b',
    r'\banalyse\s+des\s+risques?\b',
    r'\brisques?\s+apparus?\b',
    r'\brisques?\s+majeurs?\b',
]

ACTION_PATTERNS = [
    r'\bque\s+faire\b',
    r'\bplan\s+d[\'e]\s*action\b',
    r'\bcomment\s+(am[eé]liorer|r[eé]duire|respecter|g[eé]rer)\b',
    r'\bcomment\s+r[eé]agir\b',
    r'\brecommandations?\s+pour\s+(am[eé]liorer|le\s+projet|r[eé]duire)\b',
    r'\bque\s+recommandes?\b',
    r'\bque\s+conseilles?\b',
    r'\bpropose[sz]?\s+(des\s+)?(?:actions?|recommandations?)\b',
    r'\bquelles?\s+actions?\s+urgentes?\b',
    r'\bpropose\s+des\s+recommandations?\b',
    r'\brecommandations?\s+pour\s+am[eé]liorer\b',
]

HEALTH_PATTERNS = [
    r'\bsant[eé]\s+du\s+projet\b',
    r'\bscore\s+de\s+sant[eé]\b',
    r'\b[eé]value\s+la\s+sant[eé]\b',
    r'\bcomment\s+se\s+porte\b',
    r'\bsant[eé]\s+(d[\'e]|du|de)\s+\w',
    r'\b[eé]valuation\s+de\s+la\s+sant[eé]\b',
]

PREDICTION_PATTERNS = [
    r'\bpr[eé]diction\b',
    r'\banticip[ae]\b',
    r'\bprobl[eè]mes?\s+futurs?\b',
    r'\bva.?t.?il\s+(prendre|d[eé]passer)\b',
    r'\bsignaux?\s+d[\'e]alerte\b',
    r'\brisques?\s+pour\s+les\s+\d+\s+prochaines?\s+semaines?\b',
]

REPORT_PATTERNS = [
    r'\brapport\s+(?:d[eé]taill[eé]|de\s+risques?|complet)\b',
    r'\banalyse\s+compl[eè]te?\s+du\s+projet\b',
]

COMPARE_PATTERNS = [r'\bcompar[ez]\b', r'\bversus?\b', r'\bvs\b']

GLOBAL_INFO_PATTERNS = [
    r'\bbilan\s+(global\s+)?des\s+risques?\s+sur\s+tous\b',
    r'\brisques?\s+communs?\b',
    r'\bcompare[rz]?\s+(?:les\s+)?trois\s+projets?\b',
    r'\bcompare[rz]?\s+tous\s+les\s+projets?\b',
]

GLOBAL_HEALTH_PATTERNS = [
    r'\bquel\s+projet\b.*\b(meilleure?|mieux)\b.*\bsant[eé]\b',
    r'\bquel\s+projet\b.*\bsant[eé]\b.*\b(d[eé]grad|pire|plus\s+mauvais|moins\s+bonne?)\b',
    r'\bquel\s+projet\b.*\b(moins\s+bonne?|plus\s+d[eé]grad)\b.*\bsant[eé]\b',
    r'\bcompare[rz]?\b.*\bsant[eé]\b.*\b(tous|trois|3|chaque)\b',
    r'\bsant[eé]\b.*\bcompare[rz]?\b',
    r'\bdiagnostic\s+du\s+portefeuille\b',
    r'\bfais\s+un\s+diagnostic\b',
]

GLOBAL_ANALYTICAL_PATTERNS = [
    r'\bquel\s+projet\b.*\b(le\s+plus\s+critique|plus\s+expos[eé])\b',
    r'\bprojet\b.*\b(le\s+plus\s+critique|plus\s+expos[eé])\b',
    r'\bquel\s+projet\b.*\bcommenc[eé]\s+le\s+plus\s+t[oô]t\b',
    r'\bquel\s+projet\b.*\bdat.*fin\s+la\s+plus\s+tardive\b',
    r'\bquels?\s+projets?\s+[eé]taient?\s+en\s+phase\b',
    r'\bquels?\s+projets?\s+sont\s+en\s+phase\b',
    r'\b[aà]\s+quelle\s+semaine\b.*\bpass[eé]\b',
    r'\bquel\s+projet\b.*\bcode\b',
    r'\bclasse[rz]?\s+les\s+projets?\b',
    r'\bquel\s+projet\b.*\bplus\s+(gros|grand)\s+budget\b',
]

PORTFOLIO_BUDGET_PATTERNS = [
    r'\bbudget\s+total\s+(en\s+j/?h\s+)?de\s+tous\s+les\s+projets?\b',
    r'\bbudget\s+total\s+en\s+ktnd?\s+(de\s+tous)?\b',
    r'\bdonnes?\s*(moi)?\s*le\s+budget\s+total\s+en\s+ktnd?\b',
    r'\bbudget\s+total\s+du\s+portefeuille\b',
    r'\bquel\s+projet\b.*\bplus\s+(gros|grand)\s+budget\s+(initial\s+)?en\s+ktnd?\b',
    r'\bquel\s+projet\b.*\bplus\s+(gros|grand)\s+budget\b',
]

KPI_SIMPLE_PATTERNS = [
    r'\bstatut\b.*\bkpi\b',
    r'\bkpi\b.*\bstatut\b',
    r'\btendance\b.*\bkpi\b',
    r'\bkpi\b.*\btendance\b',
    r'\bbudget\s+consomm[eé]\b',
    r'\bkpi\s+(?:satisfaction|avancement|risques?|p[eé]rim[eè]tre|d[eé]pendances?|ressources\s+humaines|qualit[eé]|budget)\b',
    r'\bstatut\s+g[eé]n[eé]ral\b.*\bS\d{2}-\d{2}\b',
]

TREND_ANALYSIS_PATTERNS = [
    r'\bcomment\s+a\s+[eé]volu[eé]\b',
    r'\b[eé]volution\b.*\btemps\b',
    r'\bsur\s+tout\s+le\s+projet\b',
    r'\bs[\'e]am[eé]liore.?t.?il\b',
    r'\btendance\s+g[eé]n[eé]rale\b',
    r'\btrajectoire\b',
    r'\b[eé]volution\s+des\s+risques?\b',
    r'\bcomment\s+[eé]volue\b',
]

GLOBAL_FACTUAL_PATTERNS = [
    r'\bchef\s+de\s+projet\s+de\s+chaque\b',
    r'\bchef\s+de\s+projet\s+de\s+tous\b',
    r'\bliste[rz]?\s+tous\s+les\s+projets?\s+avec\s+leur\s+chef\b',
    r'\bliste\s+des\s+projets?\s+et\s+(chef|sponsor)\b',
    r'\bsponsor\s+de\s+chaque\s+projet\b',
    r'\bsponsor\s+de\s+tous\b',
    r'\bphase\s+actuelle\s+de\s+chaque\s+projet\b',
    r'\bphase\s+actuelle\s+de\s+tous\b',
    r'\bqui\s+est\s+le\s+chef\s+de\s+chaque\b',
    r'\bqui\s+est\s+le\s+chef\s+de\s+tous\b',
]

CLARIFICATION_PATTERNS = [
    r'^budget\s*\??$',
    r'^santé\s*\??$',
    r'^risques?\s*\??$',
    r'^actions?\s*\??$',
    r'^pr[eé]diction\s*\??$',
    r'^compare\s*\??$',
]


def _regex_fallback(question, project_names):
    """Fallback regex robuste si le LLM échoue."""
    q = question.lower().strip()
    weeks = re.findall(r'S\d{2}-\d{2}', question)
    week = weeks[0] if weeks else None
    week_range = [weeks[0], weeks[1]] if len(weeks) >= 2 else None
    project = next((p for p in project_names if p.lower() in q), None)

    def m(patterns):
        return any(re.search(p, q) for p in patterns)

    # Ordre de priorité strict
    if m(CLARIFICATION_PATTERNS):
        intent = "clarification"
    elif m(GLOBAL_HEALTH_PATTERNS):
        intent = "global_health"
    elif m(PORTFOLIO_BUDGET_PATTERNS):
        intent = "portfolio_budget"
    elif m(GLOBAL_ANALYTICAL_PATTERNS):
        intent = "global_analytical"
    elif m(COMPARE_PATTERNS) and sum(1 for p in project_names if p.lower() in q) >= 2:
        intent = "compare"
    elif m(GLOBAL_INFO_PATTERNS):
        intent = "global_info"
    elif m(GLOBAL_FACTUAL_PATTERNS):
        intent = "factual"
        # is_global = True pour ces cas
    elif m(REPORT_PATTERNS):
        intent = "report"
    elif m(TREND_ANALYSIS_PATTERNS):
        intent = "trend_analysis"
    elif m(KPI_SIMPLE_PATTERNS) and (project or week):
        intent = "kpi_simple"
    elif m(PREDICTION_PATTERNS):
        intent = "prediction"
    elif m(HEALTH_PATTERNS):
        intent = "health"
    elif m(ACTION_PATTERNS):
        intent = "actions"
    elif m(RISK_PATTERNS):
        intent = "risks"
    else:
        intent = "factual"

    kpi_name = _extract_kpi_name(question) if intent == "kpi_simple" else None
    entities = _extract_entities(question)
    is_global = m(GLOBAL_FACTUAL_PATTERNS) or intent in ("global_info", "global_health", "global_analytical", "portfolio_budget")

    return {
        "intent": intent,
        "project": project,
        "week": week,
        "week_range": week_range,
        "kpi_name": kpi_name,
        "entities_requested": entities,
        "needs_comparison": intent == "compare",
        "is_global": is_global,
        "confidence": 0.7,
        "_source": "regex_fallback",
    }


def classify_intent(question, project_names, use_llm=True):
    """Classifie l'intention de la question avec LLM + fallback regex."""
    if not use_llm:
        return _regex_fallback(question, project_names)

    prompt = INTENT_SCHEMA.format(
        project_list=", ".join(project_names),
        question=question
    )

    try:
        raw = ask_llm(prompt, max_tokens=300)
        raw = re.sub(r'```(?:json)?|```', '', raw).strip()
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)

        parsed = json.loads(raw)

        # Valeurs par défaut
        defaults = {
            "intent": "factual", "project": None, "week": None,
            "week_range": None, "kpi_name": None, "entities_requested": [],
            "needs_comparison": False, "is_global": False, "confidence": 0.8
        }
        for k, v in defaults.items():
            parsed.setdefault(k, v)

        valid_intents = {
            "risks", "actions", "health", "prediction", "report", "compare",
            "global_info", "global_health", "global_analytical", "portfolio_budget",
            "kpi_simple", "trend_analysis", "factual", "clarification"
        }
        if parsed["intent"] not in valid_intents:
            raise ValueError(f"intent invalide: {parsed['intent']}")

        parsed["_source"] = "llm"

        # ── POST-CORRECTIONS ANTI-FAUX-POSITIFS ──

        q_lower = question.lower()

        # 1. "satisfaction" ≠ actions
        if (parsed["intent"] == "actions" and "satisfaction" in q_lower
                and not any(re.search(p, q_lower) for p in ACTION_PATTERNS)):
            parsed["intent"] = "kpi_simple"
            parsed["kpi_name"] = "Satisfaction client"
            parsed["_source"] = "llm_corrected"

        # 2. Budget KTND de tous les projets → portfolio_budget
        if parsed["intent"] != "portfolio_budget" and any(re.search(p, q_lower) for p in PORTFOLIO_BUDGET_PATTERNS):
            parsed["intent"] = "portfolio_budget"
            parsed["_source"] = "llm_corrected"

        # 3. Questions globales sur chef/sponsor/phase → factual is_global
        if parsed["intent"] != "factual" and any(re.search(p, q_lower) for p in GLOBAL_FACTUAL_PATTERNS):
            parsed["intent"] = "factual"
            parsed["is_global"] = True
            parsed["_source"] = "llm_corrected"

        # 4. Évolution / tendance globale → trend_analysis
        if parsed["intent"] not in ("kpi_simple", "trend_analysis") and any(re.search(p, q_lower) for p in TREND_ANALYSIS_PATTERNS):
            # Seulement si c'est vraiment une question d'évolution globale (pas une semaine précise)
            if not re.search(r'S\d{2}-\d{2}', question):
                parsed["intent"] = "trend_analysis"
                parsed["_source"] = "llm_corrected"

        # 5. kpi_simple sans kpi_name → compléter
        if parsed["intent"] == "kpi_simple" and not parsed.get("kpi_name"):
            parsed["kpi_name"] = _extract_kpi_name(question)

        # 6. factual avec plusieurs entités → compléter entities
        if parsed["intent"] == "factual" and not parsed.get("entities_requested"):
            entities = _extract_entities(question)
            if len(entities) >= 2:
                parsed["entities_requested"] = entities

        # 7. Question vague sans projet → clarification si vraiment aucun contexte
        if parsed["intent"] in ("kpi_simple", "health", "risks", "actions", "prediction") and not parsed.get("project"):
            q_stripped = q_lower.strip().rstrip("?").strip()
            if len(q_stripped.split()) <= 2:
                parsed["intent"] = "clarification"
                parsed["_source"] = "llm_corrected"

        return parsed

    except Exception as e:
        print(f"[classifier] LLM failed ({e}) → fallback regex")
        return _regex_fallback(question, project_names)