import os
import pandas as pd
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer

DATA_FOLDER = "data"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

project_indexes = {}
project_documents = {}


def clean_columns(df):
    """Nettoie les colonnes et aplatit les MultiIndex."""
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", na=False)]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(lvl).strip() for lvl in col if lvl]).strip()
                      for col in df.columns.values]
    else:
        df.columns = df.columns.str.strip()
    return df


def safe_str(val):
    """Convertit une valeur en chaîne, gère les NaNs."""
    if pd.isna(val):
        return ""
    return str(val).strip()


def extract_number(val):
    """Extrait tous les chiffres d'une chaîne (utile pour les montants formatés)."""
    if pd.isna(val):
        return ""
    # Supprime tout ce qui n'est pas un chiffre
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

        # --- 1. Lecture de Infos_projet pour obtenir le vrai nom du projet ---
        df_info = pd.read_excel(xls, sheet_name="Infos_projet")
        df_info = clean_columns(df_info)
        if "Projet" not in df_info.columns or df_info.empty:
            print(f"❌ Pas de colonne Projet dans Infos_projet pour {project_filename}")
            return
        real_project_name = str(df_info["Projet"].iloc[0]).strip()

        # --- 2. Traitement de Infos_projet : créer des phrases naturelles pour chaque champ ---
        if not df_info.empty:
            row = df_info.iloc[0]

            # Document complet (toutes les infos en une seule phrase)
            all_parts = []
            for col in df_info.columns:
                val = row[col]
                if pd.notna(val):
                    all_parts.append(f"{col}: {val}")
            if all_parts:
                doc = f"[PROJET={real_project_name}][FEUILLE=Infos_projet][INFO=Complet]\n" + " | ".join(all_parts)
                documents.append(doc)

            # Champs individuels sous forme de phrase naturelle
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
                        documents.append(doc)  # duplication pour renforcer

            # Budgets initiaux
            if "Budget J/H" in df_info.columns:
                val = row["Budget J/H"]
                if pd.notna(val):
                    phrase = f"Le budget total prévu en J/H pour le projet {real_project_name} est de {val}."
                    doc = f"[PROJET={real_project_name}][FEUILLE=Infos_projet][INFO=Budget_initial_JH]\n{phrase}"
                    documents.append(doc)
                    documents.append(doc)
            if "Budget KTND" in df_info.columns:
                val = row["Budget KTND"]
                if pd.notna(val):
                    num = extract_number(val)
                    if num:
                        num_int = int(num)  # conversion en entier
                        phrase = f"Le budget total prévu en KTND pour le projet {real_project_name} est de {num_int}."
                        doc = f"[PROJET={real_project_name}][FEUILLE=Infos_projet][INFO=Budget_initial_KTND]\n{phrase}"
                        documents.append(doc)
                        documents.append(doc)
                        print(f"✅ Budget KTND créé pour {real_project_name}: {num_int}")  # LOG

        # --- 3. Autres feuilles ---
        for sheet in xls.sheet_names:
            if sheet.lower() == "infos_projet":
                continue

            df_sheet = xls.parse(sheet)
            df_sheet = clean_columns(df_sheet)

            # ----- KPI -----
            if sheet.lower() == "kpi":
                for _, row in df_sheet.iterrows():
                    semaine = safe_str(row.get("Semaine"))
                    projet = safe_str(row.get("Projet"))
                    if not semaine or not projet or projet != real_project_name:
                        continue

                    semaine_tag = f"[Semaine={semaine}]"
                    kpi_names = [
                        "Avancement", "Respect des delais", "Périmètre", "Risques",
                        "Budget", "Dépendances", "Satisfaction client",
                        "Ressources humaines", "Qualité du delivery"
                    ]
                    for kpi in kpi_names:
                        statut_col = f"{kpi}_Statut"
                        tend_col = f"{kpi}_Tendence"
                        if statut_col in df_sheet.columns:
                            statut = row.get(statut_col)
                            tendence = row.get(tend_col) if tend_col in df_sheet.columns else None
                            if pd.notna(statut):
                                phrase = f"En semaine {semaine}, le KPI {kpi} du projet {projet} est en statut {statut}."
                                if pd.notna(tendence):
                                    phrase += f" Sa tendance est {tendence}."
                                doc = f"[PROJET={real_project_name}][FEUILLE=KPI]{semaine_tag}[KPI={kpi}]\n{phrase}"
                                documents.append(doc)

            # ----- Météo générale -----
            elif sheet.lower() in ("météo générale", "meteo generale"):
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

                    # Construction d'une phrase complète avec tous les éléments
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

            # ----- Faits marquants -----
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

            # ----- Autres feuilles (fallback générique) -----
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

        # --- 4. Documents résumés agrégés (totaux, état courant) ---
        if "Météo générale" in xls.sheet_names or "meteo generale" in xls.sheet_names:
            sheet_name = "Météo générale" if "Météo générale" in xls.sheet_names else "meteo generale"
            df_meteo = xls.parse(sheet_name)
            df_meteo = clean_columns(df_meteo)
            if not df_meteo.empty and "Semaine" in df_meteo.columns:
                df_meteo = df_meteo[df_meteo["Projet"].astype(str).str.strip() == real_project_name]
                if not df_meteo.empty:
                    if 'Budget consommé J/H' in df_meteo.columns:
                        consos = pd.to_numeric(df_meteo['Budget consommé J/H'], errors='coerce')
                        total_jh = consos.sum()
                        if pd.notna(total_jh) and total_jh > 0:
                            phrase = f"Le budget total consommé en J/H pour le projet {real_project_name} est de {total_jh}."
                            doc = f"[PROJET={real_project_name}][FEUILLE=Météo][INFO=Budget_total_consomme_JH]\n{phrase}"
                            documents.append(doc)
                    if 'Budget consommé KTND' in df_meteo.columns:
                        consos = pd.to_numeric(df_meteo['Budget consommé KTND'], errors='coerce')
                        total_ktnd = consos.sum()
                        if pd.notna(total_ktnd) and total_ktnd > 0:
                            phrase = f"Le budget total consommé en KTND pour le projet {real_project_name} est de {total_ktnd}."
                            doc = f"[PROJET={real_project_name}][FEUILLE=Météo][INFO=Budget_total_consomme_KTND]\n{phrase}"
                            documents.append(doc)

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

        # --- 5. Indexation FAISS ---
        if not documents:
            print(f"⚠️ Aucun document pour {real_project_name}")
            return

        embeddings = model.encode(documents, convert_to_numpy=True)
        embeddings = np.array(embeddings).astype("float32")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        project_indexes[real_project_name] = index
        project_documents[real_project_name] = documents

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


def retrieve_context_all_projects(question, k_per_project=10, k_final=50):
    """
    Recherche dans tous les projets et retourne les passages les plus pertinents,
    chacun préfixé par [PROJET=...].
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

    global_results.sort(key=lambda x: x[0])  # plus petite distance = plus pertinent
    best_chunks = global_results[:k_final]

    final_context = []
    for _, proj, doc in best_chunks:
        final_context.append(f"[PROJET={proj}]\n{doc}")

    return "\n".join(final_context)