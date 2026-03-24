import os
import re
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

        # Récupération des informations structurées
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

            # Document complet des infos projet
            all_parts = []
            for col in df_info.columns:
                val = row[col]
                if pd.notna(val):
                    all_parts.append(f"{col}: {val}")
            if all_parts:
                doc = f"[PROJET={real_project_name}][FEUILLE=Infos_projet][INFO=Complet]\n" + " | ".join(all_parts)
                documents.append(doc)

            # Documents spécifiques pour chaque champ clé
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

            # Budgets initiaux (deux formulations)
            if "Budget J/H" in df_info.columns:
                val = row["Budget J/H"]
                if pd.notna(val):
                    phrase1 = f"Le budget total prévu en J/H pour le projet {real_project_name} est de {val}."
                    phrase2 = f"Le budget initial prévu en J/H pour le projet {real_project_name} est de {val}."
                    doc1 = f"[PROJET={real_project_name}][FEUILLE=Infos_projet][INFO=Budget_total_JH]\n{phrase1}"
                    doc2 = f"[PROJET={real_project_name}][FEUILLE=Infos_projet][INFO=Budget_initial_JH]\n{phrase2}"
                    documents.append(doc1)
                    documents.append(doc2)
            if "Budget KTND" in df_info.columns:
                val = row["Budget KTND"]
                if pd.notna(val):
                    num = extract_number(val)
                    if num:
                        phrase1 = f"Le budget total prévu en KTND pour le projet {real_project_name} est de {num}."
                        phrase2 = f"Le budget initial prévu en KTND pour le projet {real_project_name} est de {num}."
                        doc1 = f"[PROJET={real_project_name}][FEUILLE=Infos_projet][INFO=Budget_total_KTND]\n{phrase1}"
                        doc2 = f"[PROJET={real_project_name}][FEUILLE=Infos_projet][INFO=Budget_initial_KTND]\n{phrase2}"
                        documents.append(doc1)
                        documents.append(doc2)

        # Parcours des autres feuilles
        for sheet in xls.sheet_names:
            if sheet.lower() == "infos_projet":
                continue

            # ---------- Feuille KPI (multi-index) ----------
            if sheet.lower() == "kpi":
                # Lecture avec deux lignes d'en-tête
                df_sheet = pd.read_excel(xls, sheet_name=sheet, header=[0, 1])
                # Aplatir les colonnes : concaténer les deux niveaux avec un underscore
                df_sheet.columns = ['_'.join([str(lvl).strip() for lvl in col if str(lvl) != 'nan']).strip() for col in df_sheet.columns]
                # Nettoyer les noms (espaces multiples)
                df_sheet.columns = [re.sub(r'\s+', ' ', col).strip() for col in df_sheet.columns]
                print(f"Colonnes après flatten : {list(df_sheet.columns)}")
                
                # Identifier les colonnes de Semaine et Projet (elles contiennent ces mots)
                semaine_col = None
                projet_col = None
                for col in df_sheet.columns:
                    if 'Semaine' in col:
                        semaine_col = col
                    if 'Projet' in col:
                        projet_col = col
                if not semaine_col or not projet_col:
                    print("Impossible de trouver les colonnes Semaine ou Projet, abandon de la feuille KPI")
                    continue
                
                # Identifier les colonnes de statut (celles finissant par '_Statut')
                statut_cols = [col for col in df_sheet.columns if col.endswith('_Statut')]
                print(f"Colonnes de statut trouvées : {statut_cols}")

                kpi_count = 0
                for _, row in df_sheet.iterrows():
                    semaine = safe_str(row.get(semaine_col))
                    projet = safe_str(row.get(projet_col))
                    if not semaine or not projet or projet != real_project_name:
                        continue
                    semaine_tag = f"[Semaine={semaine}]"
                    for statut_col in statut_cols:
                        kpi_name = statut_col.replace('_Statut', '')
                        statut = row.get(statut_col)
                        if pd.notna(statut):
                            tend_col = statut_col.replace('_Statut', '_Tendence')
                            tendence = row.get(tend_col) if tend_col in df_sheet.columns else None
                            phrase = f"En semaine {semaine}, le KPI {kpi_name} du projet {projet} est en statut {statut}."
                            if pd.notna(tendence):
                                phrase += f" Sa tendance est {tendence}."
                            doc = f"[PROJET={real_project_name}][FEUILLE=KPI]{semaine_tag}[KPI={kpi_name}]\n{phrase}"
                            documents.append(doc)
                            kpi_count += 1
                print(f"Nombre de documents KPI ajoutés pour {real_project_name} : {kpi_count}")
                continue

            # ---------- Autres feuilles (lecture normale) ----------
            df_sheet = xls.parse(sheet)
            df_sheet = clean_columns(df_sheet)

            # Feuille Météo générale
            if sheet.lower() in ("météo générale", "meteo generale"):
                for _, row in df_sheet.iterrows():
                    semaine = safe_str(row.get("Semaine"))
                    projet = safe_str(row.get("Projet"))
                    if not semaine or not projet or projet != real_project_name:
                        continue

                    semaine_tag = f"[Semaine={semaine}]"
                    phase = safe_str(row.get("Phase du projet", ""))
                    statut_gen = safe_str(row.get("Statut Générale", row.get("Statut Général", "")))
                    tend_gen = safe_str(row.get("Tendence générale", row.get("Tendance générale", "")))
                    conso_jh = safe_str(row.get("Budget consommé J/H", ""))
                    conso_ktnd = safe_str(row.get("Budget consommé KTND", ""))
                    reste_jh = safe_str(row.get("Reste à faire J/H", ""))
                    reste_ktnd = safe_str(row.get("Reste à consommer KTND", ""))

                    elements = []
                    if phase:
                        elements.append(f"phase {phase}")
                    if statut_gen:
                        elements.append(f"statut général {statut_gen}")
                    if tend_gen:
                        elements.append(f"tendance générale {tend_gen}")
                    if conso_jh:
                        elements.append(f"budget consommé J/H {conso_jh}")
                    if conso_ktnd:
                        elements.append(f"budget consommé KTND {conso_ktnd}")
                    if reste_jh:
                        elements.append(f"reste à faire J/H {reste_jh}")
                    if reste_ktnd:
                        elements.append(f"reste à consommer KTND {reste_ktnd}")

                    phrase = f"En semaine {semaine}, pour le projet {projet} : " + ", ".join(elements) + "."
                    doc = f"[PROJET={real_project_name}][FEUILLE=Météo]{semaine_tag}\n{phrase}"
                    documents.append(doc)

            # Feuille Faits marquants
            elif sheet.lower() == "faits marquants":
                for _, row in df_sheet.iterrows():
                    semaine = safe_str(row.get("Semaine"))
                    projet = safe_str(row.get("Projet"))
                    if not semaine or not projet or projet != real_project_name:
                        continue

                    semaine_tag = f"[Semaine={semaine}]"
                    periode = safe_str(row.get("Période écoulé", ""))
                    prochains_chant = safe_str(row.get("Prochains chantier", ""))
                    risques = safe_str(row.get("Risques encourus", ""))
                    derniers_liv = safe_str(row.get("Derniers livrables", ""))
                    prochains_liv = safe_str(row.get("Prochains livrables", ""))
                    dernier_copil = safe_str(row.get("Date du dernier COPIL", ""))
                    prochain_copil = safe_str(row.get("Date du prochain COPIL", ""))

                    parts = []
                    if periode and periode not in ["-", "nan"]:
                        parts.append(f"période écoulée : {periode}")
                    if prochains_chant and prochains_chant not in ["-", "nan"]:
                        parts.append(f"prochains chantiers : {prochains_chant}")
                    if risques and risques not in ["-", "nan"]:
                        parts.append(f"risques encourus : {risques}")
                    if derniers_liv and derniers_liv not in ["-", "nan"]:
                        parts.append(f"derniers livrables : {derniers_liv}")
                    if prochains_liv and prochains_liv not in ["-", "nan"]:
                        parts.append(f"prochains livrables : {prochains_liv}")
                    if dernier_copil and dernier_copil not in ["-", "nan"]:
                        parts.append(f"dernier COPIL : {dernier_copil}")
                    if prochain_copil and prochain_copil not in ["-", "nan"]:
                        parts.append(f"prochain COPIL : {prochain_copil}")

                    if parts:
                        phrase = f"En semaine {semaine}, pour le projet {projet}, " + ", ".join(parts) + "."
                        doc = f"[PROJET={real_project_name}][FEUILLE=Faits marquants]{semaine_tag}\n{phrase}"
                        documents.append(doc)

            # Autres feuilles (générique)
            else:
                for _, row in df_sheet.iterrows():
                    projet = safe_str(row.get("Projet")) if "Projet" in df_sheet.columns else real_project_name
                    semaine = safe_str(row.get("Semaine")) if "Semaine" in df_sheet.columns else ""
                    if projet != real_project_name:
                        continue
                    parts = []
                    for col in df_sheet.columns:
                        val = row[col]
                        if pd.notna(val):
                            parts.append(f"{col}: {val}")
                    if parts:
                        prefix = f"[PROJET={real_project_name}][FEUILLE={sheet}]"
                        if semaine:
                            prefix += f"[Semaine={semaine}]"
                        doc = prefix + "\n" + " | ".join(parts)
                        documents.append(doc)

        # Post-traitements : derniers COPIL, derniers livrables, etc.
        if "Faits marquants" in xls.sheet_names:
            df_faits = xls.parse("Faits marquants")
            df_faits = clean_columns(df_faits)
            if not df_faits.empty and "Semaine" in df_faits.columns:
                df_faits = df_faits[df_faits["Projet"].astype(str).str.strip() == real_project_name]
                if not df_faits.empty:
                    df_faits['semaine_num'] = df_faits['Semaine'].str.extract(r'S(\d+)').astype(int)
                    df_faits = df_faits.sort_values('semaine_num', ascending=False)
                    latest = df_faits.iloc[0]

                    if "Date du dernier COPIL" in df_faits.columns and pd.notna(latest["Date du dernier COPIL"]):
                        phrase = f"Le dernier COPIL du projet {real_project_name} a eu lieu le {latest['Date du dernier COPIL']}."
                        doc = f"[PROJET={real_project_name}][FEUILLE=Faits marquants][INFO=Dernier_COPIL]\n{phrase}"
                        documents.append(doc)

                    if "Derniers livrables" in df_faits.columns and pd.notna(latest["Derniers livrables"]):
                        phrase = f"Le dernier livrable du projet {real_project_name} est {latest['Derniers livrables']}."
                        doc = f"[PROJET={real_project_name}][FEUILLE=Faits marquants][INFO=Derniers_livrables]\n{phrase}"
                        documents.append(doc)

                    if "Prochains livrables" in df_faits.columns and pd.notna(latest["Prochains livrables"]):
                        phrase = f"Les prochains livrables du projet {real_project_name} sont {latest['Prochains livrables']}."
                        doc = f"[PROJET={real_project_name}][FEUILLE=Faits marquants][INFO=Prochains_livrables]\n{phrase}"
                        documents.append(doc)

                    if "Risques encourus" in df_faits.columns and pd.notna(latest["Risques encourus"]):
                        phrase = f"Les risques actuels pour le projet {real_project_name} sont {latest['Risques encourus']}."
                        doc = f"[PROJET={real_project_name}][FEUILLE=Faits marquants][INFO=Risques_actuels]\n{phrase}"
                        documents.append(doc)

        # Déterminer la phase actuelle (dernière semaine dans la météo)
        if "Météo générale" in xls.sheet_names or "meteo generale" in xls.sheet_names:
            sheet_name = "Météo générale" if "Météo générale" in xls.sheet_names else "meteo generale"
            df_meteo = xls.parse(sheet_name)
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
                        doc = f"[PROJET={real_project_name}][FEUILLE=Météo][INFO=Phase_actuelle]\n{phrase}"
                        documents.append(doc)
                        project_current_phase[real_project_name] = phase_actuelle

        if not documents:
            print(f"⚠️ Aucun document pour {real_project_name}")
            return

        # Création de l'index FAISS
        embeddings = model.encode(documents, convert_to_numpy=True)
        embeddings = np.array(embeddings).astype("float32")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        project_indexes[real_project_name] = index
        project_documents[real_project_name] = documents
        project_names.append(real_project_name)

        print(f"✅ Projet chargé : {real_project_name} ({len(documents)} documents)")

    except Exception as e:
        print(f"❌ Erreur chargement {project_filename} : {e}")

def load_all_projects():
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER, exist_ok=True)
    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".xlsx"):
            load_project(file)

load_all_projects()

def retrieve_filtered_context(question, k_final=5, feuille=None, force_project=None):
    """
    Recherche contextuelle : d'abord on récupère les documents selon projet/feuille,
    puis on fait une recherche sémantique, et on applique le filtre semaine après.
    """
    mentioned_projects = []
    if force_project:
        mentioned_projects = [force_project]
    else:
        for proj in project_names:
            if proj.lower() in question.lower():
                mentioned_projects.append(proj)

    week_match = re.search(r'S\d{2}-\d{2}', question)
    week = week_match.group(0) if week_match else None

    # Collecte des documents avec filtres projet et feuille uniquement
    candidate_docs = []
    for proj, docs in project_documents.items():
        if mentioned_projects and proj not in mentioned_projects:
            continue
        for doc in docs:
            if feuille and f"[FEUILLE={feuille}]" not in doc:
                continue
            candidate_docs.append((proj, doc))

    if not candidate_docs:
        print("Aucun document candidat (projet/feuille)")
        return ""

    # Recherche sémantique sur tous les candidats
    texts = [doc.split('\n', 1)[-1] for _, doc in candidate_docs]
    embeddings = model.encode(texts, convert_to_numpy=True).astype("float32")
    q_emb = model.encode([question], convert_to_numpy=True).astype("float32")
    similarities = np.dot(embeddings, q_emb.T).flatten()

    sorted_indices = np.argsort(similarities)[::-1]

    # Filtrer par semaine après le classement, et ne garder que k_final
    results = []
    for idx in sorted_indices:
        proj, doc = candidate_docs[idx]
        if week and week not in doc:
            continue
        results.append(doc)
        if len(results) >= k_final:
            break

    # Si pas de semaine spécifiée, on prend les k_final meilleurs
    if not week:
        results = [candidate_docs[i][1] for i in sorted_indices[:k_final]]

    print(f"Docs candidats initiaux: {len(candidate_docs)}, après filtrage semaine: {len(results)}")
    return "\n".join(results)

def retrieve_context_all_projects(question, k_per_project=10, k_final=50):
    """
    Recherche globale sans filtrage projet (fallback).
    """
    if not project_indexes:
        return ""

    question_embedding = model.encode([question], convert_to_numpy=True).astype("float32")
    global_results = []

    for project_name, index in project_indexes.items():
        documents = project_documents.get(project_name, [])
        if not documents:
            continue
        k = min(k_per_project, len(documents))
        distances, indices = index.search(question_embedding, k)
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(documents):
                global_results.append((dist, project_name, documents[idx]))

    global_results.sort(key=lambda x: x[0])
    best_chunks = global_results[:k_final]

    final_context = []
    for _, proj, doc in best_chunks:
        final_context.append(f"[PROJET={proj}]\n{doc}")

    return "\n".join(final_context)

def rerank_passages(question, passages, top_k=5):
    """
    Réordonne les passages avec un cross-encodeur.
    """
    if not passages:
        return []
    texts = [p.split('\n', 1)[-1] if '\n' in p else p for p in passages]
    scores = reranker.predict([(question, text) for text in texts])
    scored = list(zip(scores, passages))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_k]]

# ========== FONCTIONS POUR L'AXE 1 : SYNTHÈSE DES RISQUES ==========

def extract_risks_from_faits_marquants(project_name=None, start_week=None, end_week=None):
    """
    Extrait les risques encourus à partir des documents Faits marquants.
    Retourne une liste de dictionnaires avec les clés : projet, semaine, risque.
    """
    risks = []
    for proj, docs in project_documents.items():
        if project_name and proj != project_name:
            continue
        for doc in docs:
            if "[FEUILLE=Faits marquants]" not in doc:
                continue
            # Extraire la semaine
            semaine_match = re.search(r'\[Semaine=(S\d{2}-\d{2})\]', doc)
            if not semaine_match:
                continue
            semaine = semaine_match.group(1)
            # Filtrer par semaine si demandé
            if start_week and semaine < start_week:
                continue
            if end_week and semaine > end_week:
                continue
            # Extraire le champ "risques encourus"
            # Le format attendu : "risques encourus : texte"
            risque_match = re.search(r'risques encourus\s*:\s*([^,\n]+)', doc)
            if risque_match:
                risque = risque_match.group(1).strip()
                if risque and risque not in ["—", "-", "nan", ""]:
                    risks.append({
                        "projet": proj,
                        "semaine": semaine,
                        "risque": risque
                    })
    return risks

def get_risk_signals(project_name, start_week=None, end_week=None):
    """
    Rassemble tous les signaux de risque pour un projet sur une plage de semaines.
    Retourne une liste de dictionnaires avec les clés : semaine, risques_text, kpi_risk_status,
    budget_consomme, reste_a_faire, etc.
    """
    signals = []
    # Récupérer toutes les semaines du projet (à partir des documents)
    weeks = set()
    for doc in project_documents.get(project_name, []):
        m = re.search(r'\[Semaine=(S\d{2}-\d{2})\]', doc)
        if m:
            weeks.add(m.group(1))
    weeks = sorted(weeks, key=lambda w: int(w[1:3]))

    total_budget = project_info.get(project_name, {}).get("budget_jh", 1)

    for week in weeks:
        if start_week and week < start_week:
            continue
        if end_week and week > end_week:
            continue

        signal = {"week": week, "risks_text": [], "kpi_risk_status": None,
                  "budget_consumed": None, "reste_a_faire": None}

        # Risques textuels
        risks = extract_risks_from_faits_marquants(project_name, week, week)
        signal["risks_text"] = [r["risque"] for r in risks]

        # KPI Risques
        for doc in project_documents.get(project_name, []):
            if f"[Semaine={week}]" in doc and "[FEUILLE=KPI]" in doc and "[KPI=Risques]" in doc:
                m = re.search(r'statut (\w+)', doc)
                if m:
                    signal["kpi_risk_status"] = m.group(1)
                break

        # Données budgétaires (Météo)
        for doc in project_documents.get(project_name, []):
            if f"[Semaine={week}]" in doc and "[FEUILLE=Météo]" in doc:
                m = re.search(r'budget consommé J/H (\d+)', doc)
                if m:
                    signal["budget_consumed"] = int(m.group(1))
                m = re.search(r'reste à faire J/H (\d+)', doc)
                if m:
                    signal["reste_a_faire"] = int(m.group(1))
                break

        signals.append(signal)

    return signals

def compute_weekly_risk_scores(project_name, signals):
    """
    Calcule un score de risque composite pour chaque semaine.
    Le score est une somme pondérée de plusieurs facteurs.
    """
    scores = []
    total_budget = project_info.get(project_name, {}).get("budget_jh", 1)
    for s in signals:
        score = 0
        # 1. KPI Risques
        kpi_map = {"En contrôle": 0, "À surveiller": 1, "À redresser": 2, None: 0}
        score += kpi_map.get(s.get("kpi_risk_status"), 0) * 2  # poids 2

        # 2. Nombre de risques textuels (max 3)
        score += min(len(s.get("risks_text", [])), 3)

        # 3. Ratio de consommation budgétaire
        if s.get("budget_consumed") and total_budget:
            ratio = s["budget_consumed"] / total_budget
            score += ratio * 3  # poids 3

        # 4. Tendance du reste à faire (si disponible)
        # (optionnel, on peut ajouter d'autres indicateurs)

        scores.append({"week": s["week"], "risk_score": score})
    return scores

def cluster_risks(risks_list):
    """
    Regroupe les risques textuels par similarité sémantique.
    Utilise le modèle d'embedding déjà chargé et KMeans.
    Retourne un dictionnaire {cluster_id: [risks]}.
    """
    if len(risks_list) < 2:
        return {0: risks_list}  # un seul groupe
    texts = [r["risque"] for r in risks_list]
    emb = model.encode(texts)
    k = min(3, len(texts))  # maximum 3 clusters
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(emb)
    clusters = {}
    for i, r in enumerate(risks_list):
        clusters.setdefault(labels[i], []).append(r)
    return clusters

def detect_risk_change_points(weekly_scores, threshold=1.5):
    """
    Détecte les semaines où le score de risque change significativement.
    Utilise une méthode simple basée sur la différence avec la moyenne mobile.
    Retourne une liste de semaines (indices).
    """
    if len(weekly_scores) < 3:
        return []
    scores = [s["risk_score"] for s in weekly_scores]
    change_points = []
    for i in range(1, len(scores)-1):
        # Comparer avec la moyenne des deux précédentes
        prev_avg = np.mean(scores[max(0,i-2):i])
        if abs(scores[i] - prev_avg) > threshold * np.std(scores[:i]) if i>1 else 1:
            change_points.append(weekly_scores[i]["week"])
    return change_points

def summarize_risks(risks_list):
    """
    Utilise le LLM pour générer une synthèse des risques.
    """
    if not risks_list:
        return "Aucun risque signalé sur cette période."
    risques_text = "\n".join([f"- Semaine {r['semaine']} (projet {r['projet']}) : {r['risque']}" for r in risks_list])
    prompt = f"""
Voici la liste des risques encourus sur différents projets/semaines :

{risques_text}

Fais-en une synthèse très concise (2-3 phrases) en mettant en évidence :
- Les risques récurrents
- Les risques les plus critiques
- Une tendance générale (amélioration/dégradation)
"""
    from llm import ask_llm
    return ask_llm(prompt)

def produce_risk_report(project_name, start_week=None, end_week=None):
    """
    Génère un rapport structuré de risques pour un projet donné.
    Retourne une chaîne de caractères formatée.
    """
    # 1. Récupérer les signaux
    signals = get_risk_signals(project_name, start_week, end_week)
    if not signals:
        return "Aucune donnée de risque disponible pour ce projet sur la période spécifiée."

    # 2. Calculer les scores hebdomadaires
    weekly_scores = compute_weekly_risk_scores(project_name, signals)
    avg_score = np.mean([s["risk_score"] for s in weekly_scores]) if weekly_scores else 0

    # 3. Extraire tous les risques textuels
    all_risks = []
    for s in signals:
        for r_text in s["risks_text"]:
            all_risks.append({"semaine": s["week"], "risque": r_text, "projet": project_name})

    # 4. Clustering des risques
    clusters = cluster_risks(all_risks) if all_risks else {}

    # 5. Détection des points de changement
    change_points = detect_risk_change_points(weekly_scores)

    # 6. Déterminer le niveau de risque global
    if avg_score < 3:
        level = "FAIBLE"
    elif avg_score < 6:
        level = "MODÉRÉ"
    else:
        level = "ÉLEVÉ"

    # 7. Calculer la tendance (pente)
    if len(weekly_scores) >= 2:
        scores = [s["risk_score"] for s in weekly_scores]
        x = list(range(len(scores)))
        slope = np.polyfit(x, scores, 1)[0]
        if slope > 0.5:
            trend = "dégradation"
        elif slope < -0.5:
            trend = "amélioration"
        else:
            trend = "stable"
    else:
        trend = "stable"

    # 8. Construire le prompt pour le rapport final
    clusters_text = ""
    for cl, risks in clusters.items():
        exemples = ", ".join([r["risque"][:50] for r in risks[:3]])  # premiers risques
        clusters_text += f"- Groupe {cl+1} (fréquence {len(risks)}) : {exemples}\n"

    prompt = f"""
Tu es un analyste PMO senior. Génère un rapport de synthèse des risques pour le projet {project_name} sur la période {start_week or 'début'} à {end_week or 'fin'}.

Données :
- Score de risque moyen : {avg_score:.1f}/10 → niveau {level}
- Tendance : {trend}
- Semaines avec changement notable : {', '.join(change_points) if change_points else 'aucune'}
- Clusters de risques identifiés :
{clusters_text}

Rédige un rapport professionnel (5-7 phrases) comprenant :
- Un résumé exécutif du niveau de risque.
- Les principaux thèmes de risques récurrents.
- Les risques les plus critiques.
- Une analyse de l'évolution dans le temps.
- Deux recommandations d'actions concrètes.
"""
    from llm import ask_llm
    return ask_llm(prompt)

def aggregate_risks_all_projects(start_week=None, end_week=None):
    """
    Rassemble les risques de tous les projets et produit un rapport global.
    """
    all_risks = []
    for proj in project_names:
        risks = extract_risks_from_faits_marquants(project_name=proj, start_week=start_week, end_week=end_week)
        all_risks.extend(risks)
    if not all_risks:
        return "Aucun risque trouvé sur aucun projet."
    # Grouper par projet pour le prompt
    projets_text = {}
    for r in all_risks:
        projets_text.setdefault(r["projet"], []).append(f"- Semaine {r['semaine']} : {r['risque']}")
    prompt = "Voici les risques encourus sur tous les projets :\n\n"
    for proj, risks in projets_text.items():
        prompt += f"Projet {proj} :\n" + "\n".join(risks) + "\n\n"
    prompt += "Fais une synthèse globale (4-5 phrases) des principaux risques, en identifiant les tendances communes et les projets les plus exposés."
    from llm import ask_llm
    return ask_llm(prompt)

def compare_projects(proj1, proj2, start_week=None, end_week=None):
    """
    Compare les risques et la santé de deux projets.
    """
    report1 = produce_risk_report(proj1, start_week, end_week)
    report2 = produce_risk_report(proj2, start_week, end_week)
    prompt = f"""
Tu dois comparer les deux projets suivants :

**Projet {proj1}** :
{report1}

**Projet {proj2}** :
{report2}

Rédige une comparaison structurée (4-5 phrases) mettant en évidence :
- Les différences de niveau de risque
- Les risques spécifiques à chaque projet
- Les similitudes éventuelles
- Le projet qui semble le plus critique et pourquoi.
"""
    from llm import ask_llm
    return ask_llm(prompt)

def get_phase_transition(project_name, target_phase):
    """
    Retourne la première semaine où le projet est entré dans la phase cible.
    """
    docs = project_documents.get(project_name, [])
    weeks = []
    for doc in docs:
        if "[FEUILLE=Météo]" in doc:
            semaine_match = re.search(r'\[Semaine=(S\d{2}-\d{2})\]', doc)
            if not semaine_match:
                continue
            semaine = semaine_match.group(1)
            phase_match = re.search(r'phase ([^,]+)', doc)
            if phase_match and target_phase.lower() in phase_match.group(1).lower():
                weeks.append(semaine)
    if not weeks:
        return None
    weeks.sort(key=lambda w: int(w[1:3]))
    return weeks[0]

def get_project_with_most_critical_risks(start_week=None, end_week=None, threshold=6):
    """
    Retourne le projet ayant le plus grand nombre de semaines avec un score de risque > threshold.
    """
    project_counts = {}
    for proj in project_names:
        signals = get_risk_signals(proj, start_week, end_week)
        if not signals:
            continue
        scores = compute_weekly_risk_scores(proj, signals)
        critical_weeks = sum(1 for s in scores if s["risk_score"] > threshold)
        if critical_weeks > 0:
            project_counts[proj] = critical_weeks
    if not project_counts:
        return "Aucun projet avec risques critiques détecté sur cette période."
    max_proj = max(project_counts, key=project_counts.get)
    return f"Le projet avec le plus de risques critiques est {max_proj} avec {project_counts[max_proj]} semaines critiques (score > {threshold})."

# ========== FONCTIONS POUR L'AXE 2 : PROPOSITION D'ACTIONS ==========

def get_kpi_for_week(project_name, week):
    """
    Récupère tous les KPI d'un projet pour une semaine donnée.
    Retourne un dictionnaire {kpi: {"statut": statut, "tendance": tendance}}.
    """
    docs = project_documents.get(project_name, [])
    kpis = {}
    for doc in docs:
        if "[FEUILLE=KPI]" not in doc:
            continue
        semaine_match = re.search(r'\[Semaine=(S\d{2}-\d{2})\]', doc)
        if not semaine_match or semaine_match.group(1) != week:
            continue
        kpi_match = re.search(r'\[KPI=([^\]]+)\]', doc)
        if not kpi_match:
            continue
        kpi_name = kpi_match.group(1)
        # Extraire statut et tendance
        statut_match = re.search(r'statut (\w+)', doc)
        tendance_match = re.search(r'tendance est (\w+)', doc)
        statut = statut_match.group(1) if statut_match else "inconnu"
        tendance = tendance_match.group(1) if tendance_match else "inconnu"
        kpis[kpi_name] = {"statut": statut, "tendance": tendance}
    return kpis

# Règles métier pour les actions
ACTION_RULES = {
    "risques": {
        "Performance": "Organiser un audit de performance et allouer des ressources supplémentaires.",
        "Conformité": "Vérifier les exigences réglementaires et planifier une revue de conformité.",
        "Retard validation": "Accélérer le processus de validation en réunissant les parties prenantes.",
        "Qualité données": "Mettre en place des contrôles de qualité supplémentaires et nettoyer les données.",
        "Règles métier instables": "Documenter et valider les règles métier avec les experts.",
        "Dépendance IT non planifiée": "Renégocier les dépendances et ajuster le planning.",
        "Risque surcharge production": "Prévoir une montée en charge et tester la scalabilité.",
        "Risque dérive modèle": "Revoir les hypothèses du modèle et recalibrer.",
        "Données historiques incomplètes": "Compléter les données historiques et vérifier leur fiabilité.",
        "Qualité données douteuse": "Auditer la qualité des données et mettre en place des correctifs.",
    },
    "kpi": {
        "Avancement": {
            "À surveiller": "Analyser les causes du retard et renforcer l'équipe si nécessaire.",
            "À redresser": "Mettre en place un plan d'action d'urgence avec des jalons rapprochés.",
            "Détérioration": "Revoir le planning et allouer des ressources supplémentaires.",
        },
        "Budget": {
            "À surveiller": "Revoir les dépenses et identifier des économies potentielles.",
            "Détérioration": "Geler les dépenses non essentielles et réviser le budget.",
        },
        "Risques": {
            "À surveiller": "Surveiller de près les risques identifiés et mettre en place des plans de mitigation.",
            "Détérioration": "Activer les plans de contingence et revoir l'analyse des risques.",
        },
        "Respect des délais": {
            "À surveiller": "Analyser les retards et ajuster le calendrier.",
            "Détérioration": "Mettre en place un suivi quotidien et renforcer l'équipe.",
        },
        "Périmètre": {
            "À surveiller": "Revoir les exigences et éviter les dérives de périmètre.",
            "Détérioration": "Geler les nouvelles demandes et stabiliser le périmètre.",
        },
        "Dépendances": {
            "À surveiller": "Renforcer la communication avec les équipes externes.",
            "Détérioration": "Mettre en place des réunions de coordination régulières.",
        },
        "Satisfaction client": {
            "À surveiller": "Recueillir les retours clients et ajuster les livrables.",
            "Détérioration": "Organiser une réunion avec le client pour comprendre les problèmes.",
        },
        "Ressources humaines": {
            "À surveiller": "Vérifier la charge de travail et prévenir l'épuisement.",
            "Détérioration": "Recruter ou réaffecter des ressources.",
        },
        "Qualité du delivery": {
            "À surveiller": "Renforcer les tests et les revues de code.",
            "Détérioration": "Mettre en place un processus de qualité plus strict.",
        }
    }
}

def suggest_actions(project_name, week=None):
    """
    Propose des actions basées sur les risques et KPI du projet pour une semaine donnée.
    Si week est None, on prend la dernière semaine disponible.
    Retourne une liste de chaînes (actions).
    """
    actions = []
    
    # Déterminer la semaine si non spécifiée (prendre la plus récente)
    if week is None:
        # Récupérer toutes les semaines du projet (à partir des documents)
        weeks = set()
        for doc in project_documents.get(project_name, []):
            semaine_match = re.search(r'\[Semaine=(S\d{2}-\d{2})\]', doc)
            if semaine_match:
                weeks.add(semaine_match.group(1))
        if weeks:
            # Trier par numéro de semaine (ex: S01-26 < S02-26)
            week = sorted(weeks, key=lambda w: int(w[1:3]))[-1]
        else:
            week = None
    
    # 1. Actions liées aux risques (faits marquants)
    risks = extract_risks_from_faits_marquants(project_name=project_name, start_week=week, end_week=week)
    for r in risks:
        risque_text = r['risque']
        # Chercher une correspondance dans les règles
        matched = False
        for key, action in ACTION_RULES["risques"].items():
            if key.lower() in risque_text.lower():
                actions.append(f"Risque '{risque_text}' : {action}")
                matched = True
                break
        if not matched:
            # Utiliser le LLM pour proposer une action
            prompt = f"Pour le risque suivant : '{risque_text}', propose une action corrective en une phrase."
            from llm import ask_llm
            action_llm = ask_llm(prompt)
            actions.append(f"Risque '{risque_text}' : {action_llm}")
    
    # 2. Actions liées aux KPI
    if week:
        kpis = get_kpi_for_week(project_name, week)
        for kpi_name, data in kpis.items():
            statut = data['statut']
            tendance = data['tendance']
            # Chercher dans les règles KPI
            if kpi_name in ACTION_RULES["kpi"]:
                # Vérifier d'abord le statut
                if statut in ACTION_RULES["kpi"][kpi_name]:
                    actions.append(f"KPI {kpi_name} en statut {statut} : {ACTION_RULES['kpi'][kpi_name][statut]}")
                elif tendance == "Détérioration":
                    # Si pas de règle pour le statut, mais tendance défavorable, on utilise une règle générique
                    actions.append(f"KPI {kpi_name} en tendance Détérioration : Surveiller de près et analyser les causes.")
            else:
                # KPI non couvert par les règles, utiliser LLM
                prompt = f"Le KPI {kpi_name} est en statut {statut} avec une tendance {tendance}. Propose une action corrective en une phrase."
                from llm import ask_llm
                action_llm = ask_llm(prompt)
                actions.append(f"KPI {kpi_name} : {action_llm}")
    
    return actions if actions else ["Aucune action spécifique recommandée pour ce projet/semaine."]


    # ========== FONCTIONS POUR L'AXE 3 : ÉVALUATION DE LA SANTÉ ==========

def get_latest_week(project_name):
    """Retourne la semaine la plus récente pour un projet donné."""
    weeks = set()
    for doc in project_documents.get(project_name, []):
        m = re.search(r'\[Semaine=(S\d{2}-\d{2})\]', doc)
        if m:
            weeks.add(m.group(1))
    if not weeks:
        return None
    return sorted(weeks, key=lambda w: int(w[1:3]))[-1]

def get_general_status(project_name, week=None):
    """Récupère le statut général (statut générale) pour une semaine donnée."""
    if week is None:
        week = get_latest_week(project_name)
    for doc in project_documents.get(project_name, []):
        if f"[Semaine={week}]" in doc and "[FEUILLE=Météo]" in doc:
            m = re.search(r'statut général (\w+)', doc)
            if m:
                return m.group(1)
    return None

def get_kpi_status(project_name, kpi_name, week=None):
    """Récupère le statut d'un KPI spécifique pour une semaine donnée."""
    if week is None:
        week = get_latest_week(project_name)
    for doc in project_documents.get(project_name, []):
        if f"[Semaine={week}]" in doc and "[FEUILLE=KPI]" in doc and f"[KPI={kpi_name}]" in doc:
            m = re.search(r'statut (\w+)', doc)
            if m:
                return m.group(1)
    return None

def get_budget_consumption(project_name, week=None):
    """Récupère le budget consommé J/H pour une semaine donnée."""
    if week is None:
        week = get_latest_week(project_name)
    for doc in project_documents.get(project_name, []):
        if f"[Semaine={week}]" in doc and "[FEUILLE=Météo]" in doc:
            m = re.search(r'budget consommé J/H (\d+)', doc)
            if m:
                return int(m.group(1))
    return 0

def compute_health_score(project_name, week=None):
    """
    Calcule un score de santé pour un projet à une semaine donnée (par défaut la plus récente).
    Retourne un dictionnaire avec score (0-100), niveau, et les composantes individuelles.
    """
    if week is None:
        week = get_latest_week(project_name)
        if not week:
            return {"score": 0, "level": "INCONNU", "components": {}}

    # 1. Statut général (0-100)
    statut_gen = get_general_status(project_name, week)
    statut_score = {
        "En contrôle": 100,
        "À surveiller": 50,
        "À redresser": 0,
        None: 0
    }.get(statut_gen, 0)

    # 2. KPI principaux (moyenne des statuts)
    kpi_list = ["Avancement", "Budget", "Risques", "Respect des delais", "Périmètre"]
    kpi_scores = []
    for kpi in kpi_list:
        statut = get_kpi_status(project_name, kpi, week)
        score = {
            "En contrôle": 100,
            "À surveiller": 50,
            "À redresser": 0,
            None: 0
        }.get(statut, 0)
        kpi_scores.append(score)
    avg_kpi_score = sum(kpi_scores) / len(kpi_scores) if kpi_scores else 0

    # 3. Santé budgétaire
    total_budget = project_info.get(project_name, {}).get("budget_jh", 1)
    consumed = get_budget_consumption(project_name, week)
    budget_health = max(0, 100 - (consumed / total_budget * 100)) if total_budget else 0

    # 4. Score de risque (basé sur le KPI Risques)
    risk_statut = get_kpi_status(project_name, "Risques", week)
    risk_score = {
        "En contrôle": 100,
        "À surveiller": 50,
        "À redresser": 0,
        None: 0
    }.get(risk_statut, 0)

    # Pondération
    final_score = (
        0.25 * statut_score +
        0.40 * avg_kpi_score +
        0.20 * budget_health +
        0.15 * risk_score
    )

    # Détermination du niveau
    if final_score >= 70:
        level = "VERT (bonne santé)"
    elif final_score >= 40:
        level = "ORANGE (santé moyenne, vigilance requise)"
    else:
        level = "ROUGE (santé critique, actions urgentes)"

    return {
        "score": round(final_score, 1),
        "level": level,
        "week": week,
        "components": {
            "statut_general": statut_gen,
            "statut_score": statut_score,
            "kpi_scores": {kpi: s for kpi, s in zip(kpi_list, kpi_scores)},
            "avg_kpi_score": avg_kpi_score,
            "budget_health": budget_health,
            "risk_score": risk_score
        }
    }

def generate_health_explanation(project_name, health_data):
    """
    Utilise le LLM pour générer une explication textuelle du score de santé.
    """
    comp = health_data["components"]
    prompt = f"""
Tu es un analyste PMO. Explique le score de santé du projet {project_name} pour la semaine {health_data['week']}.

Données :
- Score global : {health_data['score']}/100 → niveau {health_data['level']}
- Statut général : {comp['statut_general']} (score {comp['statut_score']})
- Moyenne des KPI : {comp['avg_kpi_score']:.1f}/100
- Santé budgétaire (consommation) : {comp['budget_health']:.1f}/100
- Score des risques : {comp['risk_score']}/100

Rédige une explication concise (3-4 phrases) mettant en évidence :
- Les points forts du projet.
- Les points faibles à surveiller.
- Une recommandation générale.
"""
    from llm import ask_llm
    return ask_llm(prompt)