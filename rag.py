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

project_indexes = {}
project_documents = {}
project_names = []
project_info = {}
project_current_phase = {}
project_chef = {}
project_sponsor = {}

# ==================== UTILITAIRES ====================
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
    return re.sub(r'[^\d]', '', str(val))

# ==================== CHARGEMENT ====================
def load_project(project_filename):
    path = os.path.join(DATA_FOLDER, project_filename)
    if not os.path.exists(path):
        print(f"Fichier introuvable : {path}")
        return
    try:
        xls = pd.ExcelFile(path)
        documents = []

        df_info = pd.read_excel(xls, sheet_name="Infos_projet")
        df_info = clean_columns(df_info)
        if "Projet" not in df_info.columns or df_info.empty:
            print(f"Pas de colonne Projet dans Infos_projet pour {project_filename}")
            return
        real_project_name = str(df_info["Projet"].iloc[0]).strip()

        budget_jh = None
        budget_ktnd = None
        chef = None
        sponsor = None
        row = df_info.iloc[0]

        if "Budget J/H" in df_info.columns and pd.notna(row["Budget J/H"]):
            num = extract_number(row["Budget J/H"])
            if num: budget_jh = int(num)
        if "Budget KTND" in df_info.columns and pd.notna(row["Budget KTND"]):
            num = extract_number(row["Budget KTND"])
            if num: budget_ktnd = int(num)
        if "Chef de Projet" in df_info.columns:
            chef = safe_str(row["Chef de Projet"])
        if "Sponsor" in df_info.columns:
            sponsor = safe_str(row["Sponsor"])

        project_info[real_project_name] = {"budget_jh": budget_jh, "budget_ktnd": budget_ktnd}
        if chef: project_chef[real_project_name] = chef
        if sponsor: project_sponsor[real_project_name] = sponsor

        all_parts = [f"{col}: {row[col]}" for col in df_info.columns if pd.notna(row[col])]
        if all_parts:
            documents.append(f"[PROJET={real_project_name}][FEUILLE=Infos_projet][INFO=Complet]\n" + " | ".join(all_parts))

        key_fields = {
            "Code projet": "code_projet", "Sponsor": "sponsor",
            "Reponsable Métier": "responsable_metier", "Chef de Projet": "chef_projet",
            "Référent technique": "referent_technique", "Date début": "date_debut", "Date fin": "date_fin",
        }
        for col, tag in key_fields.items():
            if col in df_info.columns and pd.notna(row[col]):
                documents.append(
                    f"[PROJET={real_project_name}][FEUILLE=Infos_projet][INFO={tag}]\n"
                    f"Le {col.lower()} du projet {real_project_name} est {row[col]}."
                )

        for label in ["total", "initial"]:
            if "Budget J/H" in df_info.columns and pd.notna(row["Budget J/H"]):
                documents.append(
                    f"[PROJET={real_project_name}][FEUILLE=Infos_projet][INFO=Budget_{label}_JH]\n"
                    f"Le budget {label} prévu en J/H pour le projet {real_project_name} est de {row['Budget J/H']}."
                )
            if "Budget KTND" in df_info.columns and pd.notna(row["Budget KTND"]):
                num = extract_number(row["Budget KTND"])
                if num:
                    documents.append(
                        f"[PROJET={real_project_name}][FEUILLE=Infos_projet][INFO=Budget_{label}_KTND]\n"
                        f"Le budget {label} prévu en KTND pour le projet {real_project_name} est de {num}."
                    )

        for sheet in xls.sheet_names:
            if sheet.lower() == "infos_projet":
                continue

            if sheet.lower() == "kpi":
                df_sheet = pd.read_excel(xls, sheet_name=sheet, header=[0, 1])
                df_sheet.columns = [
                    re.sub(r'\s+', ' ', '_'.join([str(lvl).strip() for lvl in col if str(lvl) != 'nan'])).strip()
                    for col in df_sheet.columns
                ]
                semaine_col = next((c for c in df_sheet.columns if 'Semaine' in c), None)
                projet_col = next((c for c in df_sheet.columns if 'Projet' in c), None)
                if not semaine_col or not projet_col:
                    continue
                statut_cols = [c for c in df_sheet.columns if c.endswith('_Statut')]
                for _, r in df_sheet.iterrows():
                    semaine = safe_str(r.get(semaine_col))
                    projet = safe_str(r.get(projet_col))
                    if not semaine or not projet or projet != real_project_name:
                        continue
                    for statut_col in statut_cols:
                        kpi_name = statut_col.replace('_Statut', '')
                        statut_val = r.get(statut_col)
                        if pd.notna(statut_val):
                            statut_str = str(statut_val).strip()
                            tend_col = statut_col.replace('_Statut', '_Tendence')
                            tendence_str = str(r.get(tend_col, "")).strip() if tend_col in df_sheet.columns and pd.notna(r.get(tend_col)) else ""
                            phrase = (
                                f"En semaine {semaine}, le KPI {kpi_name} du projet {projet} "
                                f"a le statut='{statut_str}' et la tendance='{tendence_str}'."
                            )
                            documents.append(f"[PROJET={real_project_name}][FEUILLE=KPI][Semaine={semaine}][KPI={kpi_name}]\n{phrase}")
                continue

            df_sheet = xls.parse(sheet)
            df_sheet = clean_columns(df_sheet)

            if sheet.lower() in ("météo générale", "meteo generale"):
                for _, r in df_sheet.iterrows():
                    semaine = safe_str(r.get("Semaine"))
                    projet = safe_str(r.get("Projet"))
                    if not semaine or not projet or projet != real_project_name:
                        continue
                    phase = safe_str(r.get("Phase du projet", ""))
                    statut_gen = safe_str(r.get("Statut Générale", r.get("Statut Général", "")))
                    tend_gen = safe_str(r.get("Tendence générale", r.get("Tendance générale", "")))
                    conso_jh = safe_str(r.get("Budget consommé J/H", ""))
                    conso_ktnd = safe_str(r.get("Budget consommé KTND", ""))
                    reste_jh = safe_str(r.get("Reste à faire J/H", ""))
                    reste_ktnd = safe_str(r.get("Reste à consommer KTND", ""))
                    elements = []
                    if phase: elements.append(f"phase {phase}")
                    if statut_gen: elements.append(f"statut_general='{statut_gen}'")
                    if tend_gen: elements.append(f"tendance_generale='{tend_gen}'")
                    if conso_jh: elements.append(f"budget_consomme_jh={conso_jh}")
                    if conso_ktnd: elements.append(f"budget_consomme_ktnd={conso_ktnd}")
                    if reste_jh: elements.append(f"reste_jh={reste_jh}")
                    if reste_ktnd: elements.append(f"reste_ktnd={reste_ktnd}")
                    phrase = f"En semaine {semaine}, pour le projet {projet} : " + ", ".join(elements) + "."
                    documents.append(f"[PROJET={real_project_name}][FEUILLE=Météo][Semaine={semaine}]\n{phrase}")

            elif sheet.lower() == "faits marquants":
                for _, r in df_sheet.iterrows():
                    semaine = safe_str(r.get("Semaine"))
                    projet = safe_str(r.get("Projet"))
                    if not semaine or not projet or projet != real_project_name:
                        continue
                    parts = []
                    field_map = {
                        "Période écoulé": "période écoulée",
                        "Prochains chantier": "prochains chantiers",
                        "Risques encourus": "risques encourus",
                        "Derniers livrables": "derniers livrables",
                        "Prochains livrables": "prochains livrables",
                        "Date du dernier COPIL": "dernier COPIL",
                        "Date du prochain COPIL": "prochain COPIL",
                    }
                    for col, label in field_map.items():
                        val = safe_str(r.get(col, ""))
                        if val and val not in ["-", "nan"]:
                            parts.append(f"{label} : {val}")
                    if parts:
                        phrase = f"En semaine {semaine}, pour le projet {projet}, " + ", ".join(parts) + "."
                        documents.append(f"[PROJET={real_project_name}][FEUILLE=Faits marquants][Semaine={semaine}]\n{phrase}")
            else:
                for _, r in df_sheet.iterrows():
                    projet = safe_str(r.get("Projet")) if "Projet" in df_sheet.columns else real_project_name
                    semaine = safe_str(r.get("Semaine")) if "Semaine" in df_sheet.columns else ""
                    if projet != real_project_name:
                        continue
                    parts = [f"{col}: {r[col]}" for col in df_sheet.columns if pd.notna(r[col])]
                    if parts:
                        prefix = f"[PROJET={real_project_name}][FEUILLE={sheet}]"
                        if semaine: prefix += f"[Semaine={semaine}]"
                        documents.append(prefix + "\n" + " | ".join(parts))

        if "Faits marquants" in xls.sheet_names:
            df_faits = xls.parse("Faits marquants")
            df_faits = clean_columns(df_faits)
            if not df_faits.empty and "Semaine" in df_faits.columns:
                df_faits = df_faits[df_faits["Projet"].astype(str).str.strip() == real_project_name]
                if not df_faits.empty:
                    df_faits['semaine_num'] = df_faits['Semaine'].str.extract(r'S(\d+)').astype(int)
                    latest = df_faits.sort_values('semaine_num', ascending=False).iloc[0]
                    for col, tag in [
                        ("Date du dernier COPIL", "Dernier_COPIL"),
                        ("Derniers livrables", "Derniers_livrables"),
                        ("Prochains livrables", "Prochains_livrables"),
                        ("Risques encourus", "Risques_actuels"),
                    ]:
                        if col in df_faits.columns and pd.notna(latest[col]):
                            documents.append(
                                f"[PROJET={real_project_name}][FEUILLE=Faits marquants][INFO={tag}]\n"
                                f"{col} du projet {real_project_name} : {latest[col]}."
                            )

        sheet_name_meteo = next((s for s in xls.sheet_names if s.lower() in ("météo générale", "meteo generale")), None)
        if sheet_name_meteo:
            df_meteo = xls.parse(sheet_name_meteo)
            df_meteo = clean_columns(df_meteo)
            if not df_meteo.empty and "Semaine" in df_meteo.columns:
                df_meteo = df_meteo[df_meteo["Projet"].astype(str).str.strip() == real_project_name]
                if not df_meteo.empty:
                    df_meteo['semaine_num'] = df_meteo['Semaine'].str.extract(r'S(\d+)').astype(int)
                    latest = df_meteo.sort_values('semaine_num', ascending=False).iloc[0]
                    phase_actuelle = safe_str(latest.get("Phase du projet", ""))
                    if phase_actuelle:
                        documents.append(
                            f"[PROJET={real_project_name}][FEUILLE=Météo][INFO=Phase_actuelle]\n"
                            f"La phase actuelle du projet {real_project_name} est {phase_actuelle}."
                        )
                        project_current_phase[real_project_name] = phase_actuelle

        if not documents:
            print(f"Aucun document pour {real_project_name}")
            return

        embeddings = model.encode(documents, convert_to_numpy=True).astype("float32")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        project_indexes[real_project_name] = index
        project_documents[real_project_name] = documents
        project_names.append(real_project_name)
        print(f"Projet chargé : {real_project_name} ({len(documents)} documents)")

    except Exception as e:
        import traceback
        print(f"Erreur chargement {project_filename} : {e}")
        traceback.print_exc()

def load_all_projects():
    os.makedirs(DATA_FOLDER, exist_ok=True)
    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".xlsx"):
            load_project(file)

load_all_projects()

# ==================== RAG ====================
def retrieve_filtered_context(question, k_final=5, feuille=None, force_project=None):
    mentioned_projects = [force_project] if force_project else [
        p for p in project_names if p.lower() in question.lower()
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
        results = [candidate_docs[i][1] for i in sorted_indices if week in candidate_docs[i][1]][:k_final]
    else:
        results = [candidate_docs[i][1] for i in sorted_indices[:k_final]]
    return "\n".join(results)

def rerank_passages(question, passages, top_k=5):
    if not passages:
        return []
    texts = [p.split('\n', 1)[-1] if '\n' in p else p for p in passages]
    scores = reranker.predict([(question, t) for t in texts])
    return [p for _, p in sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)[:top_k]]

# ==================== EXTRACTION ====================
def extract_risks_from_faits_marquants(project_name=None, start_week=None, end_week=None):
    risks = []
    for proj, docs in project_documents.items():
        if project_name and proj != project_name:
            continue
        for doc in docs:
            if "[FEUILLE=Faits marquants]" not in doc:
                continue
            m = re.search(r'\[Semaine=(S\d{2}-\d{2})\]', doc)
            if not m:
                continue
            semaine = m.group(1)
            if start_week and semaine < start_week:
                continue
            if end_week and semaine > end_week:
                continue
            risque_match = re.search(
                r'risques encourus\s*:\s*(.+?)(?:,\s*(?:derniers|prochains|dernier|prochain)\s+|$)',
                doc, re.IGNORECASE
            )
            if risque_match:
                risque = risque_match.group(1).strip().rstrip('.')
                if risque and risque not in ["—", "-", "nan", "", "Faible"]:
                    risks.append({"projet": proj, "semaine": semaine, "risque": risque})
    return risks

def get_kpi_for_week(project_name, week):
    kpis = {}
    for doc in project_documents.get(project_name, []):
        if "[FEUILLE=KPI]" not in doc or f"[Semaine={week}]" not in doc:
            continue
        kpi_match = re.search(r'\[KPI=([^\]]+)\]', doc)
        if not kpi_match:
            continue
        kpi_name = kpi_match.group(1)
        statut_match = re.search(r"statut='([^']+)'", doc)
        tendance_match = re.search(r"tendance='([^']+)'", doc)
        kpis[kpi_name] = {
            "statut": statut_match.group(1) if statut_match else None,
            "tendance": tendance_match.group(1) if tendance_match else None,
        }
    return kpis

def find_kpi(all_kpis, kpi_name):
    """Recherche insensible aux accents et casse."""
    if kpi_name in all_kpis:
        return all_kpis[kpi_name]
    def normalize(s):
        return s.lower().replace('é','e').replace('è','e').replace('ê','e').replace('à','a').replace('ù','u').replace('î','i').replace('ô','o')
    kpi_norm = normalize(kpi_name)
    for k, v in all_kpis.items():
        if normalize(k) == kpi_norm:
            return v
    return {"statut": None, "tendance": None}

def get_general_status(project_name, week=None):
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
        for m in [re.search(r'\[Semaine=(S\d{2}-\d{2})\]', doc)] if m
    }
    return sorted(weeks, key=lambda w: int(w[1:3]))[-1] if weeks else None

def get_budget_consumption(project_name, week=None):
    if week is None:
        week = get_latest_week(project_name)
    for doc in project_documents.get(project_name, []):
        if f"[Semaine={week}]" in doc and "[FEUILLE=Météo]" in doc:
            m = re.search(r'budget_consomme_jh=(\d+)', doc)
            if m:
                return int(m.group(1))
    return None  # FIX : None au lieu de 0 pour distinguer "non renseigné" de "zéro réel"

def get_phase_transition(project_name, target_phase):
    weeks = [
        m.group(1)
        for doc in project_documents.get(project_name, [])
        if "[FEUILLE=Météo]" in doc and target_phase.lower() in doc.lower()
        for m in [re.search(r'\[Semaine=(S\d{2}-\d{2})\]', doc)] if m
    ]
    return sorted(weeks, key=lambda w: int(w[1:3]))[0] if weeks else None

def get_risk_signals(project_name, start_week=None, end_week=None):
    all_weeks = sorted(
        {m.group(1) for doc in project_documents.get(project_name, [])
         for m in [re.search(r'\[Semaine=(S\d{2}-\d{2})\]', doc)] if m},
        key=lambda w: int(w[1:3])
    )
    signals = []
    for week in all_weeks:
        if start_week and week < start_week: continue
        if end_week and week > end_week: continue
        kpi_risques = get_kpi_for_week(project_name, week).get("Risques", {})
        reste_jh = None
        for doc in project_documents.get(project_name, []):
            if f"[Semaine={week}]" in doc and "[FEUILLE=Météo]" in doc:
                m = re.search(r'reste_jh=(\d+)', doc)
                if m: reste_jh = int(m.group(1))
                break
        signals.append({
            "week": week,
            "risks_text": [r["risque"] for r in extract_risks_from_faits_marquants(project_name, week, week)],
            "kpi_risk_status": kpi_risques.get("statut"),
            "budget_consumed": get_budget_consumption(project_name, week),
            "reste_a_faire": reste_jh,
        })
    return signals

def get_kpi_evolution(project_name, start_week=None, end_week=None):
    all_weeks = sorted(
        {m.group(1) for doc in project_documents.get(project_name, [])
         for m in [re.search(r'\[Semaine=(S\d{2}-\d{2})\]', doc)] if m},
        key=lambda w: int(w[1:3])
    )
    evolution = []
    for week in all_weeks:
        if start_week and week < start_week: continue
        if end_week and week > end_week: continue
        statut = get_general_status(project_name, week)
        phase = None
        for doc in project_documents.get(project_name, []):
            if f"[Semaine={week}]" in doc and "[FEUILLE=Météo]" in doc:
                m = re.search(r'phase ([^,]+)', doc)
                if m: phase = m.group(1).strip()
                break
        evolution.append({"week": week, "statut": statut, "phase": phase})
    return evolution

def get_kpi_degraded_as_risks(project_name, start_week=None, end_week=None):
    """
    FIX PRIORITÉ 1 — Fallback KPI pour les projets sans risques textuels (ex: Orion).
    Retourne les KPI dégradés (À surveiller / À redresser) comme signaux de risque de substitution.
    """
    risk_signals = []
    all_weeks = sorted(
        {m.group(1) for doc in project_documents.get(project_name, [])
         for m in [re.search(r'\[Semaine=(S\d{2}-\d{2})\]', doc)] if m},
        key=lambda w: int(w[1:3])
    )
    for week in all_weeks:
        if start_week and week < start_week: continue
        if end_week and week > end_week: continue
        all_kpis = get_kpi_for_week(project_name, week)
        for kpi_name, kpi_data in all_kpis.items():
            statut = kpi_data.get("statut")
            if statut in ["À surveiller", "À redresser"]:
                risk_signals.append({
                    "projet": project_name,
                    "semaine": week,
                    "risque": f"KPI {kpi_name} en statut {statut}",
                    "source": "kpi_fallback"
                })
        # Vérifier aussi le statut général
        statut_gen = get_general_status(project_name, week)
        if statut_gen in ["À surveiller", "À redresser"]:
            existing = [r for r in risk_signals if r["semaine"] == week and "statut général" in r["risque"]]
            if not existing:
                risk_signals.append({
                    "projet": project_name,
                    "semaine": week,
                    "risque": f"Statut général du projet : {statut_gen}",
                    "source": "kpi_fallback"
                })
    return risk_signals

# ==================== AXE 1 : SYNTHÈSE RISQUES ====================
def batch_classify_risks(risks):
    if not risks:
        return []
    from llm import ask_llm
    risks_text = "\n".join([f'{i+1}. "{r["risque"]}"' for i, r in enumerate(risks)])
    prompt = f"""
Classe chacun des risques suivants :
- type : technique / organisationnel / budget / externe
- severity : faible / moyen / critique
- trend : augmentation / stable / reduction

Reponds UNIQUEMENT en JSON valide, tableau dans le meme ordre :
[{{"type":"...","severity":"...","trend":"..."}}, ...]

Risques :
{risks_text}
"""
    try:
        response = ask_llm(prompt)
        response = re.sub(r'```(?:json)?', '', response).strip().rstrip('`').strip()
        classifications = json.loads(response)
        if isinstance(classifications, list) and len(classifications) == len(risks):
            return classifications
    except Exception as e:
        print(f"batch_classify_risks échoué : {e}")
    return [{"type": "inconnu", "severity": "inconnu", "trend": "inconnu"}] * len(risks)

def advanced_risk_synthesis(project_name, start_week=None, end_week=None):
    """
    AXE 1 — Synthèse enrichie.
    FIX PRIORITÉ 1 : fallback KPI si aucun risque textuel (couvre les projets sans Faits marquants remplis).
    FIX BUG SS : suppression du "S" redondant dans le formatage des semaines.
    """
    risks = extract_risks_from_faits_marquants(project_name, start_week, end_week)
    used_fallback = False

    # FIX PRIORITÉ 1 — Fallback KPI si pas de risques textuels
    if not risks:
        risks = get_kpi_degraded_as_risks(project_name, start_week, end_week)
        used_fallback = True

    evolution = get_kpi_evolution(project_name, start_week, end_week)
    evolution_text = "\n".join([
        f"  - {e['week']} : {e['statut'] or 'N/A'} (phase {e['phase'] or 'N/A'})"
        for e in evolution
    ]) if evolution else "  Aucune donnée disponible."

    from llm import ask_llm

    if not risks:
        return ask_llm(f"""
Tu es un analyste PMO senior. Le projet {project_name} sur la periode {start_week or 'debut'} a {end_week or 'fin'}
ne presente pas de risques formellement documentes.

Evolution du statut general :
{evolution_text}

Redige une analyse structuree comprenant :
1. Une observation sur l'absence de risques documentes
2. Les signaux faibles identifies via l'evolution des statuts
3. Une recommandation de vigilance

Reponds en francais, de facon professionnelle et structuree, sans emojis.
""")

    classifications = batch_classify_risks(risks)
    enriched = [{**r, **c} for r, c in zip(risks, classifications)]

    by_type = {}
    for e in enriched:
        by_type.setdefault(e.get("type", "inconnu"), []).append(e)

    types = [e.get("type", "inconnu") for e in enriched]
    severities = [e.get("severity", "inconnu") for e in enriched]
    most_freq_type = max(set(types), key=types.count)
    most_critical = next((s for s in ["critique", "moyen", "faible"] if s in severities), "inconnu")
    trends_list = [e.get("trend", "inconnu") for e in enriched]
    trend_summary = next((t for t in ["augmentation", "stable", "reduction"] if t in trends_list), "inconnu")

    # FIX BUG SS — semaine contient déjà "S06-26", ne pas ajouter de "S" devant
    detail_par_type = ""
    for type_name, items in by_type.items():
        detail_par_type += f"\n  [{type_name.upper()}]\n"
        for item in items:
            # FIX : utiliser directement item['semaine'] sans préfixe "S"
            detail_par_type += f"    - {item['semaine']} : {item['risque']} (severite : {item.get('severity','?')})\n"

    source_note = "\nNote : ces risques ont été extraits des KPI dégradés (aucun risque textuel disponible dans les Faits marquants)." if used_fallback else ""

    return ask_llm(f"""
Tu es un analyste PMO senior. Voici l'analyse complete des risques du projet {project_name}
sur la periode {start_week or 'debut'} a {end_week or 'fin'}.{source_note}

RISQUES IDENTIFIES PAR CATEGORIE :
{detail_par_type}

EVOLUTION DU STATUT GENERAL :
{evolution_text}

STATISTIQUES :
- Type de risque dominant : {most_freq_type}
- Niveau de severite le plus eleve : {most_critical}
- Tendance generale : {trend_summary}
- Nombre total de signaux : {len(risks)}

Redige une synthese professionnelle et structuree comprenant :

1. RISQUES MAJEURS IDENTIFIES
   - Groupe chaque risque par categorie (technique, organisationnel, budget, externe)
   - Pour chaque categorie : description du risque et impact potentiel

2. EVOLUTION DU RISQUE DANS LE TEMPS
   - Decris la trajectoire du projet sur la periode
   - Identifie les semaines critiques

3. CONCLUSION GLOBALE
   - Niveau de risque global (faible / modere / eleve / critique)
   - Resume en 2 phrases

4. RECOMMANDATIONS (3 recommandations concretes et actionnables)

Reponds en francais, de facon professionnelle et detaillee, sans emojis.
""")

def compute_weekly_risk_scores(project_name, signals):
    total_budget = max(project_info.get(project_name, {}).get("budget_jh", 1), 1)
    return [
        {
            "week": s["week"],
            "risk_score": (
                {"En contrôle": 0, "À surveiller": 1, "À redresser": 2}.get(s.get("kpi_risk_status"), 0) * 2
                + min(len(s.get("risks_text", [])), 3)
                # FIX : budget_consumed peut être None (non renseigné), traiter comme 0 dans ce calcul
                + ((s["budget_consumed"] or 0) / total_budget * 3 if s.get("budget_consumed") else 0)
            )
        }
        for s in signals
    ]

def cluster_risks(risks_list):
    if len(risks_list) < 2:
        return {0: risks_list}
    texts = [r["risque"] for r in risks_list]
    emb = model.encode(texts)
    k = min(3, len(texts))
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(emb)
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
        prev_avg = np.mean(scores[max(0, i - 2):i])
        std = max(np.std(scores[:i]) if i > 1 else 1, 0.01)
        if abs(scores[i] - prev_avg) > threshold * std:
            change_points.append(weekly_scores[i]["week"])
    return change_points

def produce_risk_report(project_name, start_week=None, end_week=None):
    signals = get_risk_signals(project_name, start_week, end_week)
    if not signals:
        return "Aucune donnée de risque disponible pour ce projet sur la période spécifiée."
    weekly_scores = compute_weekly_risk_scores(project_name, signals)
    avg_score = np.mean([s["risk_score"] for s in weekly_scores])
    all_risks = [{"semaine": s["week"], "risque": r, "projet": project_name} for s in signals for r in s["risks_text"]]
    # FIX PRIORITÉ 1 — fallback KPI si aucun risque textuel
    if not all_risks:
        all_risks = get_kpi_degraded_as_risks(project_name, start_week, end_week)
    clusters = cluster_risks(all_risks) if all_risks else {}
    change_points = detect_risk_change_points(weekly_scores)
    level = "FAIBLE" if avg_score < 3 else "MODERE" if avg_score < 6 else "ELEVE"
    trend = "stable"
    if len(weekly_scores) >= 2:
        sc = [s["risk_score"] for s in weekly_scores]
        slope = np.polyfit(range(len(sc)), sc, 1)[0]
        trend = "degradation" if slope > 0.5 else "amelioration" if slope < -0.5 else "stable"
    clusters_text = "\n".join([
        f"  - Groupe {cl+1} ({len(r)} risques) : {', '.join([x['risque'][:60] for x in r[:3]])}"
        for cl, r in clusters.items()
    ])
    evolution = get_kpi_evolution(project_name, start_week, end_week)
    evolution_text = "\n".join([f"  - {e['week']} : {e['statut'] or 'N/A'}" for e in evolution])
    from llm import ask_llm
    return ask_llm(f"""
Tu es un analyste PMO senior. Genere un rapport detaille des risques pour le projet {project_name}
sur la periode {start_week or 'debut'} a {end_week or 'fin'}.

DONNEES :
- Score de risque moyen : {avg_score:.1f}/10 -> niveau {level}
- Tendance : {trend}
- Semaines avec changement critique : {', '.join(change_points) if change_points else 'aucune'}

CLUSTERS DE RISQUES :
{clusters_text if clusters_text else '  Aucun risque textuel identifie - signaux KPI utilises en substitution.'}

EVOLUTION DU STATUT :
{evolution_text}

Redige un rapport structure comprenant :

1. RESUME EXECUTIF
   - Niveau de risque global et justification

2. THEMES RECURRENTS
   - Les grands themes de risques identifies avec exemples concrets

3. RISQUES CRITIQUES
   - Les risques les plus importants avec leur impact potentiel

4. ANALYSE DE L'EVOLUTION
   - Description de la trajectoire semaine par semaine
   - Points de crise identifies

5. RECOMMANDATIONS
   - 2 recommandations concretes, actionnables et priorisees

Reponds en francais, de facon professionnelle et detaillee, sans emojis.
""")

def aggregate_risks_all_projects(start_week=None, end_week=None):
    projets_text = {}
    projets_evolution = {}
    for proj in project_names:
        risks = extract_risks_from_faits_marquants(proj, start_week, end_week)
        # FIX PRIORITÉ 1 — fallback KPI
        if not risks:
            risks = get_kpi_degraded_as_risks(proj, start_week, end_week)
        if risks:
            projets_text[proj] = [f"  - {r['semaine']} : {r['risque']}" for r in risks]
        evolution = get_kpi_evolution(proj, start_week, end_week)
        if evolution:
            critiques = [e for e in evolution if e['statut'] in ["À surveiller", "À redresser"]]
            projets_evolution[proj] = {
                "total_semaines": len(evolution),
                "semaines_critiques": len(critiques),
                "niveau": "CRITIQUE" if any(e['statut'] == "À redresser" for e in evolution)
                          else "MODERE" if critiques else "MAITRISE"
            }
    from llm import ask_llm
    risques_body = "\n\n".join([
        f"Projet {p} :\n" + "\n".join(r) for p, r in projets_text.items()
    ]) if projets_text else "Aucun risque documente."
    evolution_body = "\n".join([
        f"  - {p} : niveau {v['niveau']} ({v['semaines_critiques']}/{v['total_semaines']} semaines critiques)"
        for p, v in projets_evolution.items()
    ])
    return ask_llm(f"""
Tu es un analyste PMO senior. Voici le bilan global des risques sur le portefeuille de projets
(periode {start_week or 'debut'} a {end_week or 'fin'}).

RISQUES PAR PROJET :
{risques_body}

NIVEAU DE CRITICITE PAR PROJET :
{evolution_body}

Redige un bilan structure comprenant :

1. VUE D'ENSEMBLE DU PORTEFEUILLE
   - Classement des projets par niveau de risque (critique / modere / maitrise)

2. RISQUES TRANSVERSES
   - Risques communs a plusieurs projets
   - Tendances globales du portefeuille

3. PROJET LE PLUS CRITIQUE
   - Identification et justification

4. RECOMMANDATIONS STRATEGIQUES
   - 3 recommandations prioritaires pour le portefeuille

Reponds en francais, de facon professionnelle et structuree, sans emojis.
""")

def compare_projects(proj1, proj2, start_week=None, end_week=None):
    from llm import ask_llm
    r1 = produce_risk_report(proj1, start_week, end_week)
    r2 = produce_risk_report(proj2, start_week, end_week)
    return ask_llm(f"""
Tu es un analyste PMO senior. Compare les deux projets suivants :

PROJET {proj1} :
{r1}

PROJET {proj2} :
{r2}

Redige une comparaison structuree comprenant :
1. Tableau comparatif (niveau de risque, tendance, risques principaux)
2. Differences cles entre les deux projets
3. Similitudes et risques communs
4. Verdict : lequel est le plus critique et pourquoi
5. Recommandations specifiques pour chaque projet

Reponds en francais, de facon professionnelle, sans emojis.
""")

def summarize_risks(risks_list):
    if not risks_list:
        return "Aucun risque signalé sur cette période."
    from llm import ask_llm
    text = "\n".join([f"- {r['semaine']} ({r['projet']}) : {r['risque']}" for r in risks_list])
    return ask_llm(f"""
Risques identifies :
{text}

Synthese concise (3-4 phrases) : risques recurrents, plus critiques, tendance generale, recommandation.
Reponds en francais, sans emojis.
""")

# ==================== AXE 2 : ACTIONS ====================
def suggest_actions(project_name, week=None, user_question=""):
    """
    AXE 2 — Plan d'action structuré.
    user_question : question originale pour contextualiser les actions.
    """
    if week is None:
        week = get_latest_week(project_name)
        if not week:
            return [{"action": "Aucune semaine disponible pour ce projet."}]

    risks = extract_risks_from_faits_marquants(project_name, week, week)
    all_kpis = get_kpi_for_week(project_name, week)
    phase = project_current_phase.get(project_name, "inconnue")
    statut_gen = get_general_status(project_name, week)

    # Contexte élargi si la question concerne un sujet spécifique
    context_query = user_question if user_question else f"risques livrables chantiers semaine {week} {project_name}"
    context = retrieve_filtered_context(
        context_query,
        k_final=10, feuille="Faits marquants", force_project=project_name
    )

    risks_text = "\n".join([f"  - {r['risque']}" for r in risks]) if risks else "  Aucun risque textuel signale."

    kpis_degraded = [
        f"  - {k}: statut={v['statut']}, tendance={v['tendance']}"
        for k, v in all_kpis.items() if v.get('statut') in ["À surveiller", "À redresser"]
    ]
    kpis_ok = [f"  - {k}: {v['statut']}" for k, v in all_kpis.items() if v.get('statut') == "En contrôle"]
    kpis_degraded_text = "\n".join(kpis_degraded) if kpis_degraded else "  Aucun KPI dégradé."
    kpis_ok_text = "\n".join(kpis_ok) if kpis_ok else "  Aucun KPI en contrôle."

    from llm import ask_llm
    response = ask_llm(f"""
Tu es un expert PMO senior. Analyse la situation du projet {project_name}
a la semaine {week} (phase : {phase}, statut general : {statut_gen or 'inconnu'}).

RISQUES ACTUELS :
{risks_text}

KPI DEGRADÉS (necessitent attention) :
{kpis_degraded_text}

KPI EN CONTROLE :
{kpis_ok_text}

CONTEXTE ADDITIONNEL (faits marquants) :
{context}

QUESTION SPECIFIQUE DE L'UTILISATEUR : {user_question if user_question else 'Plan d action general'}
IMPORTANT : Si la question porte sur un sujet precis (qualite des donnees, delais, regles metier, budget...),
adapte le DIAGNOSTIC et les ACTIONS PRIORITAIRES specifiquement a ce sujet.
Ne pas repondre generiquement si la question est specifique.

Produis un plan d'action structure comprenant :

1. DIAGNOSTIC RAPIDE
   - Resume de la situation actuelle en 2-3 phrases, en lien avec la question posee
   - Niveau d'urgence : critique / eleve / modere / faible

2. ACTIONS PRIORITAIRES (3 actions)
   Pour chaque action :
   - Description concrete et actionnable (adaptee a la question specifique)
   - KPI ou risque cible
   - Impact attendu : fort / moyen / faible
   - Effort requis : fort / moyen / faible
   - Delai : immediat (cette semaine) / court terme (2-3 semaines) / moyen terme

3. POINT DE VIGILANCE PRINCIPAL
   - Le risque le plus critique a surveiller en priorite

Reponds en francais, de facon professionnelle et detaillee, sans emojis.
""")

    # FIX PRIORITÉ 3 — parser proprement chaque ligne, supprimer les guillemets parasites
    lines = []
    for line in response.split('\n'):
        line = line.strip().strip('"').strip("'").strip('-').strip()
        if line:
            lines.append(line)
    return lines if lines else [{"action": "Aucune action recommandée."}]

# ==================== AXE 3 : SANTÉ ====================
HEALTH_WEIGHTS = {
    "Cadrage":            {"Avancement": 0.30, "Budget": 0.20, "Risques": 0.20, "Qualité du delivery": 0.15, "Satisfaction client": 0.15},
    "Spécification":      {"Avancement": 0.25, "Budget": 0.20, "Risques": 0.25, "Qualité du delivery": 0.15, "Satisfaction client": 0.15},
    "Dev/Intégration":    {"Avancement": 0.20, "Budget": 0.25, "Risques": 0.25, "Qualité du delivery": 0.20, "Satisfaction client": 0.10},
    "Dev/intégration":    {"Avancement": 0.20, "Budget": 0.25, "Risques": 0.25, "Qualité du delivery": 0.20, "Satisfaction client": 0.10},
    "Dev":                {"Avancement": 0.20, "Budget": 0.25, "Risques": 0.25, "Qualité du delivery": 0.20, "Satisfaction client": 0.10},
    "Homologation":       {"Avancement": 0.20, "Budget": 0.20, "Risques": 0.30, "Qualité du delivery": 0.20, "Satisfaction client": 0.10},
    "Mise en production": {"Avancement": 0.15, "Budget": 0.30, "Risques": 0.25, "Qualité du delivery": 0.20, "Satisfaction client": 0.10},
    "MEP":                {"Avancement": 0.15, "Budget": 0.30, "Risques": 0.25, "Qualité du delivery": 0.20, "Satisfaction client": 0.10},
    "Pré-Prod":           {"Avancement": 0.20, "Budget": 0.25, "Risques": 0.25, "Qualité du delivery": 0.20, "Satisfaction client": 0.10},
    "Test":               {"Avancement": 0.20, "Budget": 0.20, "Risques": 0.30, "Qualité du delivery": 0.20, "Satisfaction client": 0.10},
    "Prod":               {"Avancement": 0.15, "Budget": 0.30, "Risques": 0.25, "Qualité du delivery": 0.20, "Satisfaction client": 0.10},
}
DEFAULT_WEIGHTS = {"Avancement": 0.20, "Budget": 0.25, "Risques": 0.25, "Qualité du delivery": 0.20, "Satisfaction client": 0.10}

def get_kpi_score(kpi_name, statut, tendance):
    """
    FIX PRIORITÉ 5 — KPI absent (None) retourne 50 (neutre) au lieu de 0.
    Évite de pénaliser les projets en début de vie ou sans données renseignées.
    """
    if statut is None:
        return 50  # neutre : donnée non disponible, ne pas pénaliser
    base = {"En contrôle": 100, "À surveiller": 50, "À redresser": 0}.get(statut, 50)
    bonus = 5 if tendance == "Amélioration" else -10 if tendance == "Détérioration" else 0
    return max(0, min(100, base + bonus))

def compute_health_score_advanced(project_name, week=None):
    if week is None:
        week = get_latest_week(project_name)
        if not week:
            return {"score": 0, "level": "INCONNU", "components": {}, "week": None, "phase": None}

    # FIX : récupérer la phase réelle de la semaine demandée, pas seulement la phase actuelle
    phase = project_current_phase.get(project_name, "Dev/Intégration")
    for doc in project_documents.get(project_name, []):
        if f"[Semaine={week}]" in doc and "[FEUILLE=Météo]" in doc:
            m = re.search(r'phase ([^,]+)', doc)
            if m:
                phase = m.group(1).strip()
                break

    weights = HEALTH_WEIGHTS.get(phase, DEFAULT_WEIGHTS)
    statut_gen = get_general_status(project_name, week)
    statut_score = {"En contrôle": 100, "À surveiller": 50, "À redresser": 0}.get(statut_gen, 50)

    kpi_list = ["Avancement", "Budget", "Risques", "Qualité du delivery", "Satisfaction client"]
    all_kpis = get_kpi_for_week(project_name, week)

    kpi_scores = {
        kpi: get_kpi_score(kpi, find_kpi(all_kpis, kpi).get("statut"), find_kpi(all_kpis, kpi).get("tendance"))
        for kpi in kpi_list
    }
    weighted_kpi = sum(kpi_scores[k] * weights.get(k, 0) for k in kpi_list)

    total_budget = max(project_info.get(project_name, {}).get("budget_jh", 1), 1)
    consumed = get_budget_consumption(project_name, week)

    # FIX : consumed peut être None (non renseigné) — traiter différemment de 0
    if consumed is None:
        # Budget non renseigné : score neutre, sauf si statut dégradé
        if statut_gen in ["À surveiller", "À redresser"]:
            budget_health = 50.0
        else:
            budget_health = 70.0  # légèrement positif si projet sain mais budget non renseigné
    elif consumed == 0:
        budget_health = 50.0 if statut_gen in ["À surveiller", "À redresser"] else 100.0
    else:
        budget_health = max(0.0, 100.0 - (consumed / total_budget * 100))

    risk_score = kpi_scores.get("Risques", 50)
    final_score = round(0.20 * statut_score + 0.50 * weighted_kpi + 0.20 * budget_health + 0.10 * risk_score, 1)

    level = (
        "VERT (excellente santé)" if final_score >= 80 else
        "VERT CLAIR (bonne santé)" if final_score >= 60 else
        "ORANGE (vigilance requise)" if final_score >= 40 else
        "ORANGE ROUGE (santé dégradée)" if final_score >= 20 else
        "ROUGE (santé critique)"
    )
    return {
        "score": final_score, "level": level, "week": week, "phase": phase,
        "components": {
            "statut_general": statut_gen, "statut_score": statut_score,
            "kpi_scores": kpi_scores, "weighted_kpi": round(weighted_kpi, 1),
            "budget_health": round(budget_health, 1), "risk_score": risk_score,
            "weights": weights,
            "budget_consumed": consumed,
        }
    }

def generate_health_explanation_advanced(project_name, health_data):
    """AXE 3 — Explication narrative riche du score de santé."""
    comp = health_data.get("components", {})
    kpi_scores = comp.get("kpi_scores", {})
    weights = comp.get("weights", {})

    kpi_forts = {k: v for k, v in kpi_scores.items() if v >= 80}
    kpi_moyens = {k: v for k, v in kpi_scores.items() if 40 <= v < 80}
    kpi_faibles = {k: v for k, v in kpi_scores.items() if v < 40}

    kpi_forts_text = "\n".join([f"  - {k}: {v}/100 (poids {weights.get(k,0)*100:.0f}%)" for k, v in kpi_forts.items()]) or "  Aucun"
    kpi_moyens_text = "\n".join([f"  - {k}: {v}/100 (poids {weights.get(k,0)*100:.0f}%)" for k, v in kpi_moyens.items()]) or "  Aucun"
    kpi_faibles_text = "\n".join([f"  - {k}: {v}/100 (poids {weights.get(k,0)*100:.0f}%) -> PRIORITE" for k, v in kpi_faibles.items()]) or "  Aucun"

    budget_consumed = comp.get("budget_consumed")
    budget_note = f"{budget_consumed} J/H consommés" if budget_consumed is not None else "budget non renseigné dans les données"

    from llm import ask_llm
    return ask_llm(f"""
Tu es un analyste PMO senior. Voici l'evaluation de la sante du projet {project_name}
pour la semaine {health_data['week']} (phase : {health_data['phase']}).

SCORE GLOBAL : {health_data['score']}/100 -> {health_data['level']}

COMPOSANTES :
- Statut general : {comp.get('statut_general')} ({comp.get('statut_score')}/100)
- Score KPI pondere : {comp.get('weighted_kpi')}/100
- Sante budgetaire : {comp.get('budget_health')}/100 ({budget_note})
- Score risques : {comp.get('risk_score')}/100

DETAIL DES KPI :
RAPPEL ECHELLE : 100=excellent (En contrôle + Amélioration), 50=a surveiller (ou donnée absente), 0=critique (A redresser + Détérioration).

KPI FORTS (>= 80) :
{kpi_forts_text}

KPI MOYENS (40-79) :
{kpi_moyens_text}

KPI FAIBLES (< 40) — NECESSITENT ACTION IMMEDIATE :
{kpi_faibles_text}

Redige une evaluation structuree comprenant :

1. VERDICT GLOBAL
   - Resume en 1-2 phrases du niveau de sante

2. POINTS FORTS
   - Les indicateurs bien maitrise et pourquoi c'est positif

3. POINTS FAIBLES A CORRIGER
   - Les indicateurs critiques avec leur impact sur le projet

4. ANALYSE BUDGETAIRE
   - Commentaire sur la consommation budgetaire

5. RECOMMANDATION GENERALE
   - 1 action prioritaire concrete pour ameliorer la sante du projet

Reponds en francais, de facon professionnelle et structuree, sans emojis.
""")

# ==================== AXE 4 : PRÉDICTION ====================
def get_time_series(project_name, metrics, weeks_count=6):
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
                val = get_budget_consumption(project_name, w)
                # FIX PRIORITÉ 2 — None signifie "non renseigné", ne pas l'inclure dans la série
                series[metric].append(val)  # peut être None
            elif metric in ("avancement", "risques", "qualite"):
                key = {"avancement": "Avancement", "risques": "Risques", "qualite": "Qualité du delivery"}[metric]
                data = find_kpi(all_kpis, key)
                score = get_kpi_score(key, data.get("statut"), data.get("tendance"))
                series[metric].append(score)
            else:
                series[metric].append(None)
    return all_weeks, series

def _compute_slope(vals, zero_is_structural=False):
    """
    FIX PRIORITÉ 2 — Calcul de pente robuste.
    Filtre les None ET les zéros structurels (budget non renseigné).
    Retourne None si pas assez de données réelles.
    """
    clean = [v for v in vals if v is not None]
    if zero_is_structural:
        clean = [v for v in clean if v != 0]
    if len(clean) < 2:
        return None
    return float(np.polyfit(range(len(clean)), clean, 1)[0])

def _interpret_kpi_score(score):
    """
    FIX PRIORITÉ 2 — Interprétation correcte du score KPI.
    100 = excellent (En contrôle + Amélioration), PAS un problème.
    """
    if score is None:
        return "données insuffisantes"
    if score >= 80:
        return "excellent (En contrôle)"
    if score >= 55:
        return "satisfaisant (En contrôle)"
    if score >= 40:
        return "à surveiller"
    if score >= 20:
        return "dégradé (À surveiller)"
    return "critique (À redresser)"

def predict_problems(project_name, horizon_weeks=2):
    """
    AXE 4 — Prédiction structurée.
    FIX PRIORITÉ 2 :
    - Zéros structurels filtrés (budget Orion non renseigné ≠ budget = 0)
    - Score KPI 100 interprété comme EXCELLENT, pas comme problème
    - Pente None = pas assez de données → pas d'alerte générée
    """
    metrics = ["budget", "avancement", "risques", "qualite"]
    weeks, series = get_time_series(project_name, metrics)

    # Calcul pentes avec filtrage zéros structurels
    slopes = {
        "budget":    _compute_slope(series["budget"], zero_is_structural=True),
        "avancement": _compute_slope(series["avancement"]),
        "risques":   _compute_slope(series["risques"]),
        "qualite":   _compute_slope(series["qualite"]),
    }

    last = {m: next((v for v in reversed(series[m]) if v is not None), None) for m in metrics}

    recent_risks = extract_risks_from_faits_marquants(project_name)
    # FIX PRIORITÉ 1 — fallback KPI pour la prédiction aussi
    if not recent_risks:
        recent_risks = get_kpi_degraded_as_risks(project_name)
    recent_risks_text = "\n".join([
        f"  - {r['semaine']} : {r['risque']}" for r in recent_risks[-5:]
    ]) if recent_risks else "  Aucun risque documente."

    next_deliverables = ""
    for doc in project_documents.get(project_name, []):
        if "[FEUILLE=Faits marquants]" in doc and "prochains livrables" in doc.lower():
            m = re.search(r'prochains livrables\s*:\s*([^,\n]+)', doc, re.IGNORECASE)
            if m:
                next_deliverables = m.group(1).strip()
                break

    # FIX PRIORITÉ 2 — alertes uniquement sur signaux réels, en évitant les faux positifs
    alerts = []
    if slopes["budget"] is not None and slopes["budget"] > 50:
        alerts.append(f"Consommation budgetaire acceleree (+{slopes['budget']:.0f} J/H/semaine) — risque depassement budget")
    elif slopes["budget"] is None:
        alerts.append("Budget non renseigne dans les donnees — impossible de calculer la tendance budgetaire")

    if slopes["avancement"] is not None and slopes["avancement"] < -5:
        alerts.append(f"Degradation de l'avancement (pente {slopes['avancement']:.1f}/semaine) — risque retard livraison")

    if slopes["risques"] is not None and slopes["risques"] < -5:
        alerts.append(f"Degradation du score risques (pente {slopes['risques']:.1f}/semaine) — vigilance accrue")

    if slopes["qualite"] is not None and slopes["qualite"] < -5:
        alerts.append(f"Baisse de qualite du delivery (pente {slopes['qualite']:.1f}/semaine) — risque livrable non conforme")

    alerts_text = "\n".join([f"  - {a}" for a in alerts]) if alerts else "  Aucun signal d'alerte mathematique detecte."

    # Résumé état actuel avec interprétation correcte
    etat_budget = f"{last['budget']} J/H consommes — {_interpret_kpi_score(last['budget'])}" if last['budget'] is not None else "non renseigne dans les donnees"
    etat_avancement = f"{last['avancement']}/100 — {_interpret_kpi_score(last['avancement'])}" if last['avancement'] is not None else "N/A"
    etat_risques = f"{last['risques']}/100 — {_interpret_kpi_score(last['risques'])}" if last['risques'] is not None else "N/A"
    etat_qualite = f"{last['qualite']}/100 — {_interpret_kpi_score(last['qualite'])}" if last['qualite'] is not None else "N/A"

    from llm import ask_llm
    return ask_llm(f"""
Tu es un expert PMO senior. Analyse les tendances du projet {project_name}
sur les {len(weeks)} dernieres semaines ({weeks[0] if weeks else 'N/A'} a {weeks[-1] if weeks else 'N/A'})
et predit les problemes probables dans les {horizon_weeks} prochaines semaines.

ETAT ACTUEL DES INDICATEURS (avec interpretation) :
- Budget consomme : {etat_budget}
- Score avancement : {etat_avancement}
- Score risques : {etat_risques}
- Score qualite : {etat_qualite}

RAPPEL CRITIQUE : dans notre systeme, un score KPI de 100/100 signifie "En contrôle avec Amélioration", c'est EXCELLENT.
Un score de 50 signifie "A surveiller". Un score de 0 signifie "A redresser", c'est CRITIQUE.
Ne pas signaler un score de 100 comme un probleme — c'est le meilleur resultat possible.

SIGNAUX D'ALERTE MATHEMATIQUES DETECTES :
{alerts_text}

RISQUES ET SIGNAUX RECENTS DOCUMENTES :
{recent_risks_text}

PROCHAINS LIVRABLES : {next_deliverables or 'Non specifies'}

Redige une analyse predictive structuree comprenant :

1. CONTEXTE ET HISTORIQUE
   - Resume de la trajectoire recente du projet (2-3 phrases)
   - Ne signale PAS les indicateurs excellents (>= 80) comme des problemes

2. PROBLEMES PROBABLES (2 a 4 problemes)
   Identifier uniquement les VRAIS problemes (indicateurs < 60 ou en degradation)
   Pour chaque probleme :
   - Description precise et contextualisee
   - Probabilite : haute / moyenne / faible (avec justification)
   - Impact potentiel sur le projet
   - Mesure preventive concrete

3. RECOMMANDATION DE SURVEILLANCE
   - Le KPI ou risque le plus important a surveiller en priorite

Reponds en francais, de facon professionnelle et structuree, sans emojis.
""")

# ==================== COMPATIBILITÉ ====================
def compute_health_score(project_name, week=None):
    return compute_health_score_advanced(project_name, week)

def generate_health_explanation(project_name, health_data):
    return generate_health_explanation_advanced(project_name, health_data)

def get_project_with_most_critical_risks(start_week=None, end_week=None, threshold=6):
    counts = {
        proj: sum(1 for s in compute_weekly_risk_scores(proj, get_risk_signals(proj, start_week, end_week))
                  if s["risk_score"] > threshold)
        for proj in project_names
    }
    counts = {p: c for p, c in counts.items() if c > 0}
    if not counts:
        return "Aucun projet avec risques critiques détecté sur cette période."
    max_proj = max(counts, key=counts.get)
    return f"Le projet le plus exposé est {max_proj} avec {counts[max_proj]} semaines critiques (score > {threshold})."