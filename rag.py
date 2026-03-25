import os
import re
import json
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.cluster import KMeans

DATA_FOLDER = "data"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

# Variables globales
project_indexes = {}
project_documents = {}
project_names = []
project_info = {}
project_current_phase = {}
project_chef = {}
project_sponsor = {}

# ==================== FONCTIONS UTILITAIRES ====================
def clean_columns(df):
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", na=False)]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(lvl).strip() for lvl in col if lvl]).strip()
                      for col in df.columns.values]
    else:
        df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)
    return df

def safe_str(val):
    if pd.isna(val):
        return ""
    return str(val).strip()

def extract_number(val):
    if pd.isna(val):
        return ""
    digits = re.sub(r'[^\d]', '', str(val))
    return digits

# ==================== CHARGEMENT DES PROJETS ====================
def load_project(project_filename):
    path = os.path.join(DATA_FOLDER, project_filename)
    if not os.path.exists(path):
        print(f"❌ Fichier introuvable : {path}")
        return

    try:
        xls = pd.ExcelFile(path)
        documents = []

        df_info = pd.read_excel(xls, sheet_name="Infos_projet")
        df_info = clean_columns(df_info)
        if "Projet" not in df_info.columns or df_info.empty:
            print(f"❌ Pas de colonne Projet dans Infos_projet pour {project_filename}")
            return
        real_project_name = str(df_info["Projet"].iloc[0]).strip()

        budget_jh = None
        budget_ktnd = None
        chef = None
        sponsor = None
        if not df_info.empty:
            row = df_info.iloc[0]
            if "Budget J/H" in df_info.columns:
                val = row["Budget J/H"]
                if pd.notna(val):
                    num = extract_number(val)
                    if num:
                        budget_jh = int(num)
            if "Budget KTND" in df_info.columns:
                val = row["Budget KTND"]
                if pd.notna(val):
                    num = extract_number(val)
                    if num:
                        budget_ktnd = int(num)
            if "Chef de Projet" in df_info.columns:
                chef = safe_str(row["Chef de Projet"])
            if "Sponsor" in df_info.columns:
                sponsor = safe_str(row["Sponsor"])

        project_info[real_project_name] = {"budget_jh": budget_jh, "budget_ktnd": budget_ktnd}
        if chef:
            project_chef[real_project_name] = chef
        if sponsor:
            project_sponsor[real_project_name] = sponsor

        if not df_info.empty:
            row = df_info.iloc[0]
            all_parts = []
            for col in df_info.columns:
                val = row[col]
                if pd.notna(val):
                    all_parts.append(f"{col}: {val}")
            if all_parts:
                doc = f"[PROJET={real_project_name}][FEUILLE=Infos_projet][INFO=Complet]\n" + " | ".join(all_parts)
                documents.append(doc)

            key_fields = {
                "Code projet": "code_projet",
                "Sponsor": "sponsor",
                "Reponsable Métier": "responsable_metier",
                "Chef de Projet": "chef_projet",
                "Référent technique": "referent_technique",
                "Date début": "date_debut",
                "Date fin": "date_fin",
            }
            for col, tag in key_fields.items():
                if col in df_info.columns:
                    val = row[col]
                    if pd.notna(val):
                        phrase = f"Le {col.lower()} du projet {real_project_name} est {val}."
                        doc = f"[PROJET={real_project_name}][FEUILLE=Infos_projet][INFO={tag}]\n{phrase}"
                        documents.append(doc)

            if "Budget J/H" in df_info.columns:
                val = row["Budget J/H"]
                if pd.notna(val):
                    for label in ["total", "initial"]:
                        phrase = f"Le budget {label} prévu en J/H pour le projet {real_project_name} est de {val}."
                        doc = f"[PROJET={real_project_name}][FEUILLE=Infos_projet][INFO=Budget_{label}_JH]\n{phrase}"
                        documents.append(doc)
            if "Budget KTND" in df_info.columns:
                val = row["Budget KTND"]
                if pd.notna(val):
                    num = extract_number(val)
                    if num:
                        for label in ["total", "initial"]:
                            phrase = f"Le budget {label} prévu en KTND pour le projet {real_project_name} est de {num}."
                            doc = f"[PROJET={real_project_name}][FEUILLE=Infos_projet][INFO=Budget_{label}_KTND]\n{phrase}"
                            documents.append(doc)

        for sheet in xls.sheet_names:
            if sheet.lower() == "infos_projet":
                continue

            if sheet.lower() == "kpi":
                df_sheet = pd.read_excel(xls, sheet_name=sheet, header=[0, 1])
                df_sheet.columns = [
                    re.sub(r'\s+', ' ', '_'.join([str(lvl).strip() for lvl in col if str(lvl) != 'nan'])).strip()
                    for col in df_sheet.columns
                ]
                semaine_col = next((col for col in df_sheet.columns if 'Semaine' in col), None)
                projet_col = next((col for col in df_sheet.columns if 'Projet' in col), None)
                if not semaine_col or not projet_col:
                    print("Colonnes Semaine/Projet introuvables dans KPI, feuille ignorée.")
                    continue

                statut_cols = [col for col in df_sheet.columns if col.endswith('_Statut')]
                for _, row in df_sheet.iterrows():
                    semaine = safe_str(row.get(semaine_col))
                    projet = safe_str(row.get(projet_col))
                    if not semaine or not projet or projet != real_project_name:
                        continue
                    for statut_col in statut_cols:
                        kpi_name = statut_col.replace('_Statut', '')
                        statut_val = row.get(statut_col)
                        if pd.notna(statut_val):
                            statut_str = str(statut_val).strip()
                            tend_col = statut_col.replace('_Statut', '_Tendence')
                            tendence_val = row.get(tend_col) if tend_col in df_sheet.columns else None
                            tendence_str = str(tendence_val).strip() if pd.notna(tendence_val) else ""
                            # ✅ FIX : stocker statut et tendance complets (pas tronqués)
                            phrase = (
                                f"En semaine {semaine}, le KPI {kpi_name} du projet {projet} "
                                f"a le statut='{statut_str}' et la tendance='{tendence_str}'."
                            )
                            doc = f"[PROJET={real_project_name}][FEUILLE=KPI][Semaine={semaine}][KPI={kpi_name}]\n{phrase}"
                            documents.append(doc)
                continue

            df_sheet = xls.parse(sheet)
            df_sheet = clean_columns(df_sheet)

            if sheet.lower() in ("météo générale", "meteo generale"):
                for _, row in df_sheet.iterrows():
                    semaine = safe_str(row.get("Semaine"))
                    projet = safe_str(row.get("Projet"))
                    if not semaine or not projet or projet != real_project_name:
                        continue
                    phase = safe_str(row.get("Phase du projet", ""))
                    statut_gen = safe_str(row.get("Statut Générale", row.get("Statut Général", "")))
                    tend_gen = safe_str(row.get("Tendence générale", row.get("Tendance générale", "")))
                    conso_jh = safe_str(row.get("Budget consommé J/H", ""))
                    conso_ktnd = safe_str(row.get("Budget consommé KTND", ""))
                    reste_jh = safe_str(row.get("Reste à faire J/H", ""))
                    reste_ktnd = safe_str(row.get("Reste à consommer KTND", ""))

                    elements = []
                    if phase: elements.append(f"phase {phase}")
                    if statut_gen: elements.append(f"statut_general='{statut_gen}'")
                    if tend_gen: elements.append(f"tendance_generale='{tend_gen}'")
                    if conso_jh: elements.append(f"budget consommé J/H {conso_jh}")
                    if conso_ktnd: elements.append(f"budget consommé KTND {conso_ktnd}")
                    if reste_jh: elements.append(f"reste à faire J/H {reste_jh}")
                    if reste_ktnd: elements.append(f"reste à consommer KTND {reste_ktnd}")

                    phrase = f"En semaine {semaine}, pour le projet {projet} : " + ", ".join(elements) + "."
                    doc = f"[PROJET={real_project_name}][FEUILLE=Météo][Semaine={semaine}]\n{phrase}"
                    documents.append(doc)

            elif sheet.lower() == "faits marquants":
                for _, row in df_sheet.iterrows():
                    semaine = safe_str(row.get("Semaine"))
                    projet = safe_str(row.get("Projet"))
                    if not semaine or not projet or projet != real_project_name:
                        continue
                    periode = safe_str(row.get("Période écoulé", ""))
                    prochains_chant = safe_str(row.get("Prochains chantier", ""))
                    risques = safe_str(row.get("Risques encourus", ""))
                    derniers_liv = safe_str(row.get("Derniers livrables", ""))
                    prochains_liv = safe_str(row.get("Prochains livrables", ""))
                    dernier_copil = safe_str(row.get("Date du dernier COPIL", ""))
                    prochain_copil = safe_str(row.get("Date du prochain COPIL", ""))

                    parts = []
                    if periode and periode not in ["-", "nan"]: parts.append(f"période écoulée : {periode}")
                    if prochains_chant and prochains_chant not in ["-", "nan"]: parts.append(f"prochains chantiers : {prochains_chant}")
                    if risques and risques not in ["-", "nan"]: parts.append(f"risques encourus : {risques}")
                    if derniers_liv and derniers_liv not in ["-", "nan"]: parts.append(f"derniers livrables : {derniers_liv}")
                    if prochains_liv and prochains_liv not in ["-", "nan"]: parts.append(f"prochains livrables : {prochains_liv}")
                    if dernier_copil and dernier_copil not in ["-", "nan"]: parts.append(f"dernier COPIL : {dernier_copil}")
                    if prochain_copil and prochain_copil not in ["-", "nan"]: parts.append(f"prochain COPIL : {prochain_copil}")

                    if parts:
                        phrase = f"En semaine {semaine}, pour le projet {projet}, " + ", ".join(parts) + "."
                        doc = f"[PROJET={real_project_name}][FEUILLE=Faits marquants][Semaine={semaine}]\n{phrase}"
                        documents.append(doc)
            else:
                for _, row in df_sheet.iterrows():
                    projet = safe_str(row.get("Projet")) if "Projet" in df_sheet.columns else real_project_name
                    semaine = safe_str(row.get("Semaine")) if "Semaine" in df_sheet.columns else ""
                    if projet != real_project_name:
                        continue
                    parts = [f"{col}: {row[col]}" for col in df_sheet.columns if pd.notna(row[col])]
                    if parts:
                        prefix = f"[PROJET={real_project_name}][FEUILLE={sheet}]"
                        if semaine:
                            prefix += f"[Semaine={semaine}]"
                        documents.append(prefix + "\n" + " | ".join(parts))

        # Post-traitements
        if "Faits marquants" in xls.sheet_names:
            df_faits = xls.parse("Faits marquants")
            df_faits = clean_columns(df_faits)
            if not df_faits.empty and "Semaine" in df_faits.columns:
                df_faits = df_faits[df_faits["Projet"].astype(str).str.strip() == real_project_name]
                if not df_faits.empty:
                    df_faits['semaine_num'] = df_faits['Semaine'].str.extract(r'S(\d+)').astype(int)
                    df_faits = df_faits.sort_values('semaine_num', ascending=False)
                    latest = df_faits.iloc[0]
                    for col, tag in [
                        ("Date du dernier COPIL", "Dernier_COPIL"),
                        ("Derniers livrables", "Derniers_livrables"),
                        ("Prochains livrables", "Prochains_livrables"),
                        ("Risques encourus", "Risques_actuels"),
                    ]:
                        if col in df_faits.columns and pd.notna(latest[col]):
                            phrase = f"[INFO={tag}] {col} du projet {real_project_name} : {latest[col]}."
                            doc = f"[PROJET={real_project_name}][FEUILLE=Faits marquants][INFO={tag}]\n{phrase}"
                            documents.append(doc)

        sheet_name_meteo = next(
            (s for s in xls.sheet_names if s.lower() in ("météo générale", "meteo generale")), None
        )
        if sheet_name_meteo:
            df_meteo = xls.parse(sheet_name_meteo)
            df_meteo = clean_columns(df_meteo)
            if not df_meteo.empty and "Semaine" in df_meteo.columns:
                df_meteo = df_meteo[df_meteo["Projet"].astype(str).str.strip() == real_project_name]
                if not df_meteo.empty:
                    df_meteo['semaine_num'] = df_meteo['Semaine'].str.extract(r'S(\d+)').astype(int)
                    df_meteo = df_meteo.sort_values('semaine_num', ascending=False)
                    latest = df_meteo.iloc[0]
                    phase_actuelle = safe_str(latest.get("Phase du projet", ""))
                    if phase_actuelle:
                        phrase = f"La phase actuelle du projet {real_project_name} est {phase_actuelle}."
                        documents.append(
                            f"[PROJET={real_project_name}][FEUILLE=Météo][INFO=Phase_actuelle]\n{phrase}"
                        )
                        project_current_phase[real_project_name] = phase_actuelle

        if not documents:
            print(f"⚠️ Aucun document pour {real_project_name}")
            return

        embeddings = model.encode(documents, convert_to_numpy=True).astype("float32")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        project_indexes[real_project_name] = index
        project_documents[real_project_name] = documents
        project_names.append(real_project_name)
        print(f"✅ Projet chargé : {real_project_name} ({len(documents)} documents)")

    except Exception as e:
        import traceback
        print(f"❌ Erreur chargement {project_filename} : {e}")
        traceback.print_exc()

def load_all_projects():
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER, exist_ok=True)
    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".xlsx"):
            load_project(file)

load_all_projects()

# ==================== RAG ====================
def retrieve_filtered_context(question, k_final=5, feuille=None, force_project=None):
    mentioned_projects = [force_project] if force_project else [
        proj for proj in project_names if proj.lower() in question.lower()
    ]

    week_match = re.search(r'S\d{2}-\d{2}', question)
    week = week_match.group(0) if week_match else None

    candidate_docs = [
        (proj, doc)
        for proj, docs in project_documents.items()
        if not mentioned_projects or proj in mentioned_projects
        for doc in docs
        if not feuille or f"[FEUILLE={feuille}]" in doc
    ]

    if not candidate_docs:
        return ""

    texts = [doc.split('\n', 1)[-1] for _, doc in candidate_docs]
    embeddings = model.encode(texts, convert_to_numpy=True).astype("float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0, 1, norms)

    q_emb = model.encode([question], convert_to_numpy=True).astype("float32")
    q_emb = q_emb / np.linalg.norm(q_emb)
    similarities = np.dot(embeddings, q_emb.T).flatten()
    sorted_indices = np.argsort(similarities)[::-1]

    if week:
        results = [
            candidate_docs[i][1] for i in sorted_indices
            if week in candidate_docs[i][1]
        ][:k_final]
    else:
        results = [candidate_docs[i][1] for i in sorted_indices[:k_final]]

    return "\n".join(results)

def rerank_passages(question, passages, top_k=5):
    if not passages:
        return []
    texts = [p.split('\n', 1)[-1] if '\n' in p else p for p in passages]
    scores = reranker.predict([(question, text) for text in texts])
    return [p for _, p in sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)[:top_k]]

# ==================== FONCTIONS D'EXTRACTION ====================
def extract_risks_from_faits_marquants(project_name=None, start_week=None, end_week=None):
    risks = []
    for proj, docs in project_documents.items():
        if project_name and proj != project_name:
            continue
        for doc in docs:
            if "[FEUILLE=Faits marquants]" not in doc:
                continue
            semaine_match = re.search(r'\[Semaine=(S\d{2}-\d{2})\]', doc)
            if not semaine_match:
                continue
            semaine = semaine_match.group(1)
            if start_week and semaine < start_week:
                continue
            if end_week and semaine > end_week:
                continue
            # ✅ FIX : regex plus large qui ne tronque pas sur les points
            risque_match = re.search(r'risques encourus\s*:\s*(.+?)(?:,\s*(?:derniers|prochains|dernier|prochain)|$)', doc, re.IGNORECASE)
            if risque_match:
                risque = risque_match.group(1).strip().rstrip('.')
                if risque and risque not in ["—", "-", "nan", ""]:
                    risks.append({"projet": proj, "semaine": semaine, "risque": risque})
    return risks

def get_kpi_for_week(project_name, week):
    """
    ✅ FIX CRITIQUE : utilise les nouveaux patterns statut='...' et tendance='...'
    pour capturer des valeurs multi-mots comme 'En contrôle' ou 'À surveiller'.
    """
    docs = project_documents.get(project_name, [])
    kpis = {}
    for doc in docs:
        if "[FEUILLE=KPI]" not in doc:
            continue
        if f"[Semaine={week}]" not in doc:
            continue
        kpi_match = re.search(r'\[KPI=([^\]]+)\]', doc)
        if not kpi_match:
            continue
        kpi_name = kpi_match.group(1)
        statut_match = re.search(r"statut='([^']+)'", doc)
        tendance_match = re.search(r"tendance='([^']+)'", doc)
        statut = statut_match.group(1) if statut_match else "inconnu"
        tendance = tendance_match.group(1) if tendance_match else "inconnu"
        kpis[kpi_name] = {"statut": statut, "tendance": tendance}
    return kpis

def get_general_status(project_name, week=None):
    """✅ FIX : utilise le nouveau pattern statut_general='...' pour capturer multi-mots."""
    if week is None:
        week = get_latest_week(project_name)
    for doc in project_documents.get(project_name, []):
        if f"[Semaine={week}]" in doc and "[FEUILLE=Météo]" in doc:
            m = re.search(r"statut_general='([^']+)'", doc)
            if m:
                return m.group(1)
    return None

def get_latest_week(project_name):
    weeks = {
        m.group(1)
        for doc in project_documents.get(project_name, [])
        for m in [re.search(r'\[Semaine=(S\d{2}-\d{2})\]', doc)]
        if m
    }
    return sorted(weeks, key=lambda w: int(w[1:3]))[-1] if weeks else None

def get_budget_consumption(project_name, week=None):
    if week is None:
        week = get_latest_week(project_name)
    for doc in project_documents.get(project_name, []):
        if f"[Semaine={week}]" in doc and "[FEUILLE=Météo]" in doc:
            m = re.search(r'budget consommé J/H (\d+)', doc)
            if m:
                return int(m.group(1))
    return 0

def get_phase_transition(project_name, target_phase):
    weeks = []
    for doc in project_documents.get(project_name, []):
        if "[FEUILLE=Météo]" not in doc:
            continue
        semaine_match = re.search(r'\[Semaine=(S\d{2}-\d{2})\]', doc)
        if not semaine_match:
            continue
        if target_phase.lower() in doc.lower():
            weeks.append(semaine_match.group(1))
    if not weeks:
        return None
    return sorted(weeks, key=lambda w: int(w[1:3]))[0]

def get_risk_signals(project_name, start_week=None, end_week=None):
    all_weeks = sorted(
        {m.group(1) for doc in project_documents.get(project_name, [])
         for m in [re.search(r'\[Semaine=(S\d{2}-\d{2})\]', doc)] if m},
        key=lambda w: int(w[1:3])
    )
    signals = []
    for week in all_weeks:
        if start_week and week < start_week:
            continue
        if end_week and week > end_week:
            continue
        signal = {
            "week": week,
            "risks_text": [r["risque"] for r in extract_risks_from_faits_marquants(project_name, week, week)],
            "kpi_risk_status": None,
            "budget_consumed": get_budget_consumption(project_name, week),
            "reste_a_faire": None,
        }
        kpi_risques = get_kpi_for_week(project_name, week).get("Risques", {})
        signal["kpi_risk_status"] = kpi_risques.get("statut")
        for doc in project_documents.get(project_name, []):
            if f"[Semaine={week}]" in doc and "[FEUILLE=Météo]" in doc:
                m = re.search(r'reste à faire J/H (\d+)', doc)
                if m:
                    signal["reste_a_faire"] = int(m.group(1))
                break
        signals.append(signal)
    return signals

# ==================== AXE 1 : SYNTHÈSE AVANCÉE DES RISQUES ====================
def batch_classify_risks(risks):
    """
    ✅ FIX : un seul appel LLM pour classifier tous les risques (au lieu de N appels).
    """
    if not risks:
        return []
    from llm import ask_llm
    risks_text = "\n".join([f'{i+1}. "{r["risque"]}"' for i, r in enumerate(risks)])
    prompt = f"""
Classe chacun des risques suivants selon ces dimensions :
- type : technique / organisationnel / budget / externe
- severity : faible / moyen / critique
- trend : augmentation / stable / réduction

Réponds UNIQUEMENT en JSON valide, tableau ordonné dans le même ordre :
[
  {{"type": "...", "severity": "...", "trend": "..."}},
  ...
]

Risques :
{risks_text}
"""
    try:
        response = ask_llm(prompt)
        # Nettoyer les backticks markdown si présents
        response = re.sub(r'```(?:json)?', '', response).strip().rstrip('`').strip()
        classifications = json.loads(response)
        if isinstance(classifications, list) and len(classifications) == len(risks):
            return classifications
    except Exception as e:
        print(f"⚠️ batch_classify_risks échoué : {e}")
    # Fallback : valeurs par défaut
    return [{"type": "inconnu", "severity": "inconnu", "trend": "inconnu"}] * len(risks)

def advanced_risk_synthesis(project_name, start_week=None, end_week=None):
    risks = extract_risks_from_faits_marquants(project_name, start_week, end_week)
    if not risks:
        return "Aucun risque trouvé pour cette période."

    classifications = batch_classify_risks(risks)
    enriched = [{**r, **c} for r, c in zip(risks, classifications)]

    types = [e.get("type", "inconnu") for e in enriched]
    severities = [e.get("severity", "inconnu") for e in enriched]
    trends = [e.get("trend", "inconnu") for e in enriched]

    most_freq_type = max(set(types), key=types.count) if types else "inconnu"
    most_critical = next((s for s in ["critique", "moyen", "faible"] if s in severities), "inconnu")
    trend_summary = next((t for t in ["augmentation", "stable", "réduction"] if t in trends), "inconnu")

    from llm import ask_llm
    risks_summary = "\n".join([
        f"- S{r['semaine']} : {r['risque']} [type={r.get('type','?')}, sévérité={r.get('severity','?')}]"
        for r in enriched
    ])
    prompt = f"""
Tu es un analyste PMO senior. Voici les risques du projet {project_name} (période {start_week or 'début'} → {end_week or 'fin'}) :

{risks_summary}

Statistiques :
- Type dominant : {most_freq_type}
- Sévérité la plus élevée : {most_critical}
- Tendance dominante : {trend_summary}

Rédige un résumé professionnel (3-4 phrases) : principaux risques, sévérité, recommandation.
"""
    return ask_llm(prompt)

def compute_weekly_risk_scores(project_name, signals):
    total_budget = max(project_info.get(project_name, {}).get("budget_jh", 1), 1)
    scores = []
    for s in signals:
        score = 0
        kpi_map = {"En contrôle": 0, "À surveiller": 1, "À redresser": 2, None: 0}
        score += kpi_map.get(s.get("kpi_risk_status"), 0) * 2
        score += min(len(s.get("risks_text", [])), 3)
        if s.get("budget_consumed"):
            score += (s["budget_consumed"] / total_budget) * 3
        scores.append({"week": s["week"], "risk_score": score})
    return scores

def cluster_risks(risks_list):
    if len(risks_list) < 2:
        return {0: risks_list}
    texts = [r["risque"] for r in risks_list]
    emb = model.encode(texts)
    k = min(3, len(texts))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(emb)
    clusters = {}
    for i, r in enumerate(risks_list):
        clusters.setdefault(int(labels[i]), []).append(r)
    return clusters

def detect_risk_change_points(weekly_scores, threshold=1.5):
    if len(weekly_scores) < 3:
        return []
    scores = [s["risk_score"] for s in weekly_scores]
    change_points = []
    for i in range(1, len(scores) - 1):
        window = scores[max(0, i - 2):i]
        prev_avg = np.mean(window)
        std = np.std(scores[:i]) if i > 1 else 1
        if abs(scores[i] - prev_avg) > threshold * max(std, 0.01):
            change_points.append(weekly_scores[i]["week"])
    return change_points

def produce_risk_report(project_name, start_week=None, end_week=None):
    signals = get_risk_signals(project_name, start_week, end_week)
    if not signals:
        return "Aucune donnée de risque disponible pour ce projet sur la période spécifiée."

    weekly_scores = compute_weekly_risk_scores(project_name, signals)
    avg_score = np.mean([s["risk_score"] for s in weekly_scores]) if weekly_scores else 0

    all_risks = [
        {"semaine": s["week"], "risque": r, "projet": project_name}
        for s in signals for r in s["risks_text"]
    ]
    clusters = cluster_risks(all_risks) if all_risks else {}
    change_points = detect_risk_change_points(weekly_scores)
    level = "FAIBLE" if avg_score < 3 else "MODÉRÉ" if avg_score < 6 else "ÉLEVÉ"

    trend = "stable"
    if len(weekly_scores) >= 2:
        scores = [s["risk_score"] for s in weekly_scores]
        slope = np.polyfit(range(len(scores)), scores, 1)[0]
        trend = "dégradation" if slope > 0.5 else "amélioration" if slope < -0.5 else "stable"

    clusters_text = "\n".join([
        f"- Groupe {cl+1} ({len(risks)} risques) : {', '.join([r['risque'][:50] for r in risks[:3]])}"
        for cl, risks in clusters.items()
    ])

    from llm import ask_llm
    prompt = f"""
Tu es un analyste PMO senior. Rapport de risques pour {project_name} ({start_week or 'début'} → {end_week or 'fin'}).

- Score moyen : {avg_score:.1f}/10 → {level}
- Tendance : {trend}
- Semaines critiques : {', '.join(change_points) if change_points else 'aucune'}
- Clusters :
{clusters_text}

Rédige un rapport (5-7 phrases) : résumé exécutif, thèmes récurrents, risques critiques, évolution, 2 recommandations.
"""
    return ask_llm(prompt)

def aggregate_risks_all_projects(start_week=None, end_week=None):
    projets_text = {}
    for proj in project_names:
        risks = extract_risks_from_faits_marquants(proj, start_week, end_week)
        if risks:
            projets_text[proj] = [f"- S{r['semaine']} : {r['risque']}" for r in risks]
    if not projets_text:
        return "Aucun risque trouvé sur aucun projet."
    prompt = "Risques sur tous les projets :\n\n" + "\n\n".join(
        [f"Projet {p} :\n" + "\n".join(r) for p, r in projets_text.items()]
    ) + "\n\nSynthèse globale (4-5 phrases) : tendances communes, projets les plus exposés."
    from llm import ask_llm
    return ask_llm(prompt)

def compare_projects(proj1, proj2, start_week=None, end_week=None):
    r1 = produce_risk_report(proj1, start_week, end_week)
    r2 = produce_risk_report(proj2, start_week, end_week)
    from llm import ask_llm
    return ask_llm(f"""
Compare ces deux projets :

**{proj1}** : {r1}

**{proj2}** : {r2}

Comparaison (4-5 phrases) : différences de risque, spécificités, similitudes, lequel est le plus critique et pourquoi.
""")

# ==================== AXE 2 : PROPOSITION D'ACTIONS ====================
def suggest_actions(project_name, week=None):
    if week is None:
        week = get_latest_week(project_name)
        if not week:
            return ["Aucune semaine disponible pour ce projet."]

    risks = extract_risks_from_faits_marquants(project_name, week, week)
    kpis = get_kpi_for_week(project_name, week)
    phase = project_current_phase.get(project_name, "inconnue")

    # ✅ FIX : requête RAG ciblée sur les faits marquants de la semaine
    context = retrieve_filtered_context(
        f"risques livrables semaine {week} {project_name}",
        k_final=8,
        feuille="Faits marquants",
        force_project=project_name
    )

    risks_text = "\n".join([f"- {r['risque']}" for r in risks]) if risks else "Aucun risque signalé."
    kpis_text = "\n".join([
        f"- {k}: statut={v['statut']}, tendance={v['tendance']}"
        for k, v in kpis.items()
    ]) if kpis else "Aucun KPI disponible."

    from llm import ask_llm
    prompt = f"""
Tu es un expert PMO senior. Projet : {project_name} | Semaine : {week} | Phase : {phase}

Risques actuels :
{risks_text}

KPI :
{kpis_text}

Contexte additionnel :
{context}

Propose 3 actions prioritaires. Pour chaque action :
- Action: description claire
- Impact: fort / moyen / faible
- Effort: fort / moyen / faible

Format strict : "- Action: ... | Impact: ... | Effort: ..."
"""
    response = ask_llm(prompt)
    return [line.strip() for line in response.split('\n') if line.strip()] if response else ["Aucune action recommandée."]

# ==================== AXE 3 : SANTÉ PROJET AVANCÉE ====================
HEALTH_WEIGHTS = {
    "Cadrage":           {"Avancement": 0.30, "Budget": 0.20, "Risques": 0.20, "Qualité du delivery": 0.15, "Satisfaction client": 0.15},
    "Spécification":     {"Avancement": 0.25, "Budget": 0.20, "Risques": 0.25, "Qualité du delivery": 0.15, "Satisfaction client": 0.15},
    "Dev/Intégration":   {"Avancement": 0.20, "Budget": 0.25, "Risques": 0.25, "Qualité du delivery": 0.20, "Satisfaction client": 0.10},
    "Homologation":      {"Avancement": 0.20, "Budget": 0.20, "Risques": 0.30, "Qualité du delivery": 0.20, "Satisfaction client": 0.10},
    "Mise en production":{"Avancement": 0.15, "Budget": 0.30, "Risques": 0.25, "Qualité du delivery": 0.20, "Satisfaction client": 0.10},
    "MEP":               {"Avancement": 0.15, "Budget": 0.30, "Risques": 0.25, "Qualité du delivery": 0.20, "Satisfaction client": 0.10},
}
DEFAULT_WEIGHTS = {"Avancement": 0.20, "Budget": 0.25, "Risques": 0.25, "Qualité du delivery": 0.20, "Satisfaction client": 0.10}

def get_kpi_score(kpi_name, statut, tendance):
    """Score KPI avec pénalité tendance."""
    base = {"En contrôle": 100, "À surveiller": 50, "À redresser": 0}.get(statut, 0)
    penalty = -10 if tendance == "Détérioration" else 5 if tendance == "Amélioration" else 0
    return max(0, min(100, base + penalty))

def compute_health_score_advanced(project_name, week=None):
    if week is None:
        week = get_latest_week(project_name)
        if not week:
            return {"score": 0, "level": "INCONNU", "components": {}, "week": None, "phase": None}

    phase = project_current_phase.get(project_name, "Dev/Intégration")
    weights = HEALTH_WEIGHTS.get(phase, DEFAULT_WEIGHTS)

    statut_gen = get_general_status(project_name, week)
    statut_score = {"En contrôle": 100, "À surveiller": 50, "À redresser": 0}.get(statut_gen, 0)

    kpi_list = ["Avancement", "Budget", "Risques", "Qualité du delivery", "Satisfaction client"]
    all_kpis = get_kpi_for_week(project_name, week)
    kpi_scores = {}
    for kpi in kpi_list:
        data = all_kpis.get(kpi, {"statut": "inconnu", "tendance": "inconnu"})
        kpi_scores[kpi] = get_kpi_score(kpi, data["statut"], data["tendance"])

    # ✅ Score KPI pondéré selon la phase
    weighted_kpi = sum(kpi_scores[k] * weights.get(k, 0) for k in kpi_list)

    total_budget = max(project_info.get(project_name, {}).get("budget_jh", 1), 1)
    consumed = get_budget_consumption(project_name, week)
    budget_health = max(0, 100 - (consumed / total_budget * 100))

    risk_score = kpi_scores.get("Risques", 0)

    final_score = round(
        0.20 * statut_score +
        0.50 * weighted_kpi +
        0.20 * budget_health +
        0.10 * risk_score,
        1
    )

    level = (
        "VERT (excellente santé)" if final_score >= 80 else
        "VERT CLAIR (bonne santé)" if final_score >= 60 else
        "ORANGE (vigilance requise)" if final_score >= 40 else
        "ORANGE ROUGE (santé dégradée)" if final_score >= 20 else
        "ROUGE (santé critique)"
    )

    return {
        "score": final_score,
        "level": level,
        "week": week,
        "phase": phase,
        "components": {
            "statut_general": statut_gen,
            "statut_score": statut_score,
            "kpi_scores": kpi_scores,
            "weighted_kpi": round(weighted_kpi, 1),
            "budget_health": round(budget_health, 1),
            "risk_score": risk_score,
        }
    }

def generate_health_explanation_advanced(project_name, health_data):
    comp = health_data.get("components", {})
    kpi_lines = "\n".join([f"  - {k}: {v}/100" for k, v in comp.get("kpi_scores", {}).items()])
    from llm import ask_llm
    prompt = f"""
Tu es un analyste PMO. Score de santé du projet {project_name} – semaine {health_data['week']} (phase {health_data['phase']}).

Résultat global : {health_data['score']}/100 → {health_data['level']}
- Statut général : {comp.get('statut_general')} ({comp.get('statut_score')}/100)
- KPI pondérés : {comp.get('weighted_kpi')}/100
{kpi_lines}
- Santé budgétaire : {comp.get('budget_health')}/100
- Score risques : {comp.get('risk_score')}/100

Rédige 3-4 phrases : points forts, points faibles, recommandation générale.
"""
    return ask_llm(prompt)

# ==================== AXE 4 : PRÉDICTION ====================
def get_time_series(project_name, metrics, weeks_count=6):
    """
    ✅ FIX : récupère toutes les métriques en un seul parcours des semaines.
    Retourne (weeks, {metric: [values]})
    """
    all_weeks = sorted(
        {m.group(1) for doc in project_documents.get(project_name, [])
         for m in [re.search(r'\[Semaine=(S\d{2}-\d{2})\]', doc)] if m},
        key=lambda w: int(w[1:3])
    )[-weeks_count:]

    series = {metric: [] for metric in metrics}
    for w in all_weeks:
        all_kpis = get_kpi_for_week(project_name, w)
        for metric in metrics:
            if metric == "budget":
                series[metric].append(get_budget_consumption(project_name, w))
            elif metric in ("avancement", "risques", "qualite"):
                kpi_key = {
                    "avancement": "Avancement",
                    "risques": "Risques",
                    "qualite": "Qualité du delivery"
                }[metric]
                data = all_kpis.get(kpi_key, {})
                series[metric].append(get_kpi_score(kpi_key, data.get("statut"), data.get("tendance")))
            else:
                series[metric].append(None)
    return all_weeks, series

def predict_problems(project_name, horizon_weeks=2):
    metrics = ["budget", "avancement", "risques", "qualite"]
    weeks, series = get_time_series(project_name, metrics)

    def slope(vals):
        clean = [v for v in vals if v is not None]
        if len(clean) < 2:
            return 0.0
        return float(np.polyfit(range(len(clean)), clean, 1)[0])

    slopes = {m: slope(series[m]) for m in metrics}

    # Identifier les signaux d'alerte
    alerts = []
    if slopes["budget"] > 50:
        alerts.append(f"Consommation budgétaire accélérée (pente +{slopes['budget']:.0f} J/H/semaine)")
    if slopes["avancement"] < -5:
        alerts.append(f"Dégradation de l'avancement (pente {slopes['avancement']:.1f}/semaine)")
    if slopes["risques"] < -5:
        alerts.append(f"Dégradation du score risques (pente {slopes['risques']:.1f}/semaine)")
    if slopes["qualite"] < -5:
        alerts.append(f"Baisse de qualité du delivery (pente {slopes['qualite']:.1f}/semaine)")

    alerts_text = "\n".join([f"- {a}" for a in alerts]) if alerts else "Aucun signal d'alerte détecté."

    last_values = {m: series[m][-1] if series[m] else "N/A" for m in metrics}

    from llm import ask_llm
    prompt = f"""
Projet {project_name} – Prédiction sur {horizon_weeks} semaines ({len(weeks)} semaines analysées).

Dernières valeurs :
- Budget consommé : {last_values['budget']} J/H
- Score avancement : {last_values['avancement']}/100
- Score risques : {last_values['risques']}/100
- Score qualité : {last_values['qualite']}/100

Signaux d'alerte détectés :
{alerts_text}

En tant qu'expert PMO, prédit les problèmes probables. Pour chaque problème :
- Description précise
- Probabilité : haute / moyenne / faible
- Impact potentiel
- Mesure préventive concrète

Rédige en 4-5 phrases structurées.
"""
    return ask_llm(prompt)

# ==================== COMPATIBILITÉ ====================
def compute_health_score(project_name, week=None):
    return compute_health_score_advanced(project_name, week)

def generate_health_explanation(project_name, health_data):
    return generate_health_explanation_advanced(project_name, health_data)

def summarize_risks(risks_list):
    if not risks_list:
        return "Aucun risque signalé sur cette période."
    from llm import ask_llm
    risques_text = "\n".join([f"- S{r['semaine']} ({r['projet']}) : {r['risque']}" for r in risks_list])
    return ask_llm(f"""
Risques sur différents projets/semaines :
{risques_text}

Synthèse concise (2-3 phrases) : risques récurrents, plus critiques, tendance générale.
""")

def get_project_with_most_critical_risks(start_week=None, end_week=None, threshold=6):
    project_counts = {
        proj: sum(1 for s in compute_weekly_risk_scores(proj, get_risk_signals(proj, start_week, end_week))
                  if s["risk_score"] > threshold)
        for proj in project_names
        if get_risk_signals(proj, start_week, end_week)
    }
    project_counts = {p: c for p, c in project_counts.items() if c > 0}
    if not project_counts:
        return "Aucun projet avec risques critiques détecté sur cette période."
    max_proj = max(project_counts, key=project_counts.get)
    return f"Le projet le plus exposé est {max_proj} avec {project_counts[max_proj]} semaines critiques (score > {threshold})."