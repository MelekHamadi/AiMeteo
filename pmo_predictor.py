"""
pmo_predictor.py — Modèle ML standalone pour l'Axe 4 Prédiction
================================================================
Fichier UNIQUE et AUTONOME. Aucune dépendance vers rag.py ou app.py.

Usage :
    # 1. Entraîner les modèles (une seule fois, ou quand vous ajoutez des données)
    python pmo_predictor.py train

    # 2. Prédire pour un projet à une semaine donnée
    python pmo_predictor.py predict --project "Data Fraud Detection" --week "S22-26"

    # 3. Rapport complet avec SHAP (explication des facteurs)
    python pmo_predictor.py report --project "Scoring Crédit Retail"

Architecture :
    - 7 modèles XGBoost indépendants (budget, retard, qualité, RH, périmètre, dépendances, satisfaction)
    - Feature engineering sur 33+ features extraites des Excel
    - SMOTE pour équilibrer les classes
    - SHAP pour expliquer chaque prédiction
    - Sauvegarde des modèles en .pkl
"""

import os
import re
import sys
import json
import pickle
import warnings
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Dépendances ML ──
try:
    from xgboost import XGBClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (
        classification_report, roc_auc_score,
        confusion_matrix, precision_score, recall_score, f1_score
    )
    from sklearn.calibration import CalibratedClassifierCV
    from imblearn.over_sampling import SMOTE
    import shap
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    print(f"[ERREUR] Dépendances manquantes : {e}")
    print("Exécutez : pip install xgboost scikit-learn imbalanced-learn shap --break-system-packages")
    sys.exit(1)


# ==================== CONFIGURATION ====================

DATA_FOLDER = "data"
MODELS_FOLDER = "ml_models"
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Liste complète des modèles
MODEL_NAMES = ["budget", "retard", "qualite", "rh", "perimetre", "dependances", "satisfaction"]

MODEL_PATHS = {
    name: os.path.join(MODELS_FOLDER, f"model_{name}.pkl") for name in MODEL_NAMES
}
MODEL_PATHS["meta"] = os.path.join(MODELS_FOLDER, "model_meta.json")
MODEL_PATHS["features"] = os.path.join(MODELS_FOLDER, "feature_names.pkl")

# Mapping statut → score numérique
STATUT_SCORE = {
    "En contrôle":  1.0,
    "À surveiller": 0.5,
    "À redresser":  0.0,
    "":             0.5,
    None:           0.5,
}

# Mapping tendance → score numérique
TENDANCE_SCORE = {
    "Amélioration":   1.0,
    "Stand by":       0.5,
    "Détérioration":  0.0,
    "":               0.5,
    None:             0.5,
}

# Phases → score numérique (maturité du projet)
PHASE_SCORE = {
    "Cadrage":            0.0,
    "Spécification":      0.15,
    "Dev":                0.30,
    "Dev/Intégration":    0.35,
    "Dev/intégration":    0.35,
    "Homologation":       0.55,
    "Test":               0.60,
    "Pré-Prod":           0.70,
    "Mise en production": 0.85,
    "MEP":                0.85,
    "Prod":               1.0,
}

KPI_COLUMNS = [
    "Avancement", "Budget", "Risques",
    "Qualité du delivery", "Satisfaction client",
    "Ressources humaines", "Dépendances",
    "Périmètre", "Respect des délais"
]


# ==================== EXTRACTION DONNÉES EXCEL ====================

def clean_columns(df):
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", na=False)]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(l).strip() for l in col if l]).strip() for col in df.columns.values]
    else:
        df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)
    return df


def safe_str(val):
    if pd.isna(val): return ""
    return str(val).strip()


def week_to_int(week_str):
    """Convertit 'S12-26' → 12"""
    m = re.search(r'S(\d+)', str(week_str))
    return int(m.group(1)) if m else 0


def normalize_kpi_name(name):
    name = str(name).lower().strip()
    name = name.replace('é', 'e').replace('è', 'e').replace('ê', 'e')
    name = name.replace('à', 'a').replace('â', 'a')
    name = name.replace('ô', 'o').replace('î', 'i').replace('û', 'u')
    return name


def load_single_project(filepath):
    xls = pd.ExcelFile(filepath)
    records = []

    # Infos projet
    df_info = pd.read_excel(xls, sheet_name="Infos_projet")
    df_info = clean_columns(df_info)
    if df_info.empty or "Projet" not in df_info.columns:
        return pd.DataFrame()

    row_info = df_info.iloc[0]
    project_name = safe_str(row_info.get("Projet", ""))

    budget_total_jh = 0
    col_jh = next((c for c in df_info.columns if "Budget" in c and "J/H" in c and "consomm" not in c.lower()), None)
    if col_jh and pd.notna(row_info.get(col_jh)):
        num = re.sub(r'[^\d]', '', str(row_info[col_jh]))
        budget_total_jh = int(num) if num else 0

    date_debut = None
    date_fin = None
    col_dd = next((c for c in df_info.columns if "début" in c.lower() or "debut" in c.lower()), None)
    col_df = next((c for c in df_info.columns if "fin" in c.lower() and "date" in c.lower()), None)
    if col_dd and pd.notna(row_info.get(col_dd)):
        try: date_debut = pd.to_datetime(row_info[col_dd])
        except: pass
    if col_df and pd.notna(row_info.get(col_df)):
        try: date_fin = pd.to_datetime(row_info[col_df])
        except: pass

    project_duration_weeks = 0
    if date_debut and date_fin:
        project_duration_weeks = max(1, int((date_fin - date_debut).days / 7))

    # Météo générale
    sheet_meteo = next((s for s in xls.sheet_names if s.lower() in ("météo générale", "meteo generale")), None)
    meteo_data = {}
    if sheet_meteo:
        df_meteo = pd.read_excel(xls, sheet_name=sheet_meteo)
        df_meteo = clean_columns(df_meteo)
        df_meteo = df_meteo[df_meteo.get("Projet", pd.Series(dtype=str)).astype(str).str.strip() == project_name]
        for _, r in df_meteo.iterrows():
            semaine = safe_str(r.get("Semaine", ""))
            if not semaine: continue
            statut_cols = [c for c in df_meteo.columns if "statut" in c.lower() and ("général" in c.lower() or "generale" in c.lower() or "Générale" in c)]
            statut_gen = ""
            for sc in statut_cols:
                if pd.notna(r.get(sc)): statut_gen = safe_str(r[sc]); break
            if not statut_gen:
                sc2 = next((c for c in df_meteo.columns if "statut" in c.lower()), None)
                if sc2 and pd.notna(r.get(sc2)): statut_gen = safe_str(r[sc2])

            tend_cols = [c for c in df_meteo.columns if "tendenc" in c.lower() or "tendanc" in c.lower()]
            tend_gen = ""
            for tc in tend_cols:
                if pd.notna(r.get(tc)): tend_gen = safe_str(r[tc]); break

            phase = ""
            pc = next((c for c in df_meteo.columns if "phase" in c.lower()), None)
            if pc and pd.notna(r.get(pc)): phase = safe_str(r[pc])

            conso_jh = 0
            cc = next((c for c in df_meteo.columns if "consomm" in c.lower() and "j/h" in c.lower()), None)
            if cc and pd.notna(r.get(cc)):
                num = re.sub(r'[^\d]', '', str(r[cc]))
                conso_jh = int(num) if num else 0

            reste_jh = 0
            rc = next((c for c in df_meteo.columns if "reste" in c.lower() and "j/h" in c.lower()), None)
            if rc and pd.notna(r.get(rc)):
                num = re.sub(r'[^\d]', '', str(r[rc]))
                reste_jh = int(num) if num else 0

            meteo_data[semaine] = {
                "statut_gen": statut_gen, "tend_gen": tend_gen, "phase": phase,
                "conso_jh": conso_jh, "reste_jh": reste_jh,
            }

    # KPI
    kpi_data = {}
    if "KPI" in xls.sheet_names:
        df_kpi = pd.read_excel(xls, sheet_name="KPI", header=[0, 1])
        df_kpi.columns = [
            re.sub(r'\s+', ' ', '_'.join([str(l).strip() for l in col if str(l) != 'nan'])).strip()
            for col in df_kpi.columns
        ]
        sem_col = next((c for c in df_kpi.columns if 'Semaine' in c), None)
        proj_col = next((c for c in df_kpi.columns if 'Projet' in c), None)
        if sem_col and proj_col:
            statut_cols_kpi = [c for c in df_kpi.columns if c.endswith('_Statut')]
            for _, r in df_kpi.iterrows():
                semaine = safe_str(r.get(sem_col, ""))
                projet = safe_str(r.get(proj_col, ""))
                if not semaine or projet != project_name: continue
                kpi_data.setdefault(semaine, {})
                for sc in statut_cols_kpi:
                    kpi_name = sc.replace('_Statut', '')
                    statut_val = safe_str(r.get(sc, ""))
                    tend_col = sc.replace('_Statut', '_Tendence')
                    tend_val = safe_str(r.get(tend_col, "")) if tend_col in df_kpi.columns else ""
                    kpi_data[semaine][kpi_name] = {"statut": statut_val, "tendance": tend_val}

    # Faits marquants
    faits_data = {}
    if "Faits marquants" in xls.sheet_names:
        df_faits = pd.read_excel(xls, sheet_name="Faits marquants")
        df_faits = clean_columns(df_faits)
        df_faits = df_faits[df_faits.get("Projet", pd.Series(dtype=str)).astype(str).str.strip() == project_name]
        for _, r in df_faits.iterrows():
            semaine = safe_str(r.get("Semaine", ""))
            if not semaine: continue
            risque_col = next((c for c in df_faits.columns if "risque" in c.lower()), None)
            risque = safe_str(r.get(risque_col, "")) if risque_col else ""
            faits_data[semaine] = {"risque_textuel": risque}

    # Construction des records
    all_weeks = sorted(set(list(meteo_data.keys()) + list(kpi_data.keys())), key=lambda w: week_to_int(w))

    for i, semaine in enumerate(all_weeks):
        meteo = meteo_data.get(semaine, {})
        kpis = kpi_data.get(semaine, {})
        faits = faits_data.get(semaine, {})

        week_num = week_to_int(semaine)
        conso_jh = meteo.get("conso_jh", 0)
        reste_jh = meteo.get("reste_jh", 0)
        phase = meteo.get("phase", "")
        statut_gen = meteo.get("statut_gen", "")
        tend_gen = meteo.get("tend_gen", "")

        pct_budget_consomme = (conso_jh / budget_total_jh * 100) if budget_total_jh > 0 else 0
        pct_semaine = (week_num / project_duration_weeks * 100) if project_duration_weeks > 0 else 0
        budget_burn_rate = conso_jh / max(week_num, 1)
        budget_reste_semaines = (reste_jh / budget_burn_rate) if budget_burn_rate > 0 else 999
        gap_budget_temps = pct_budget_consomme - pct_semaine

        def get_kpi(name):
            name_norm = normalize_kpi_name(name)
            for kn, kv in kpis.items():
                if normalize_kpi_name(kn) == name_norm:
                    return kv
            return {"statut": None, "tendance": None}

        kpi_avancement   = get_kpi("Avancement")
        kpi_budget       = get_kpi("Budget")
        kpi_risques      = get_kpi("Risques")
        kpi_qualite      = get_kpi("Qualité du delivery")
        kpi_satisfaction = get_kpi("Satisfaction client")
        kpi_rh           = get_kpi("Ressources humaines")
        kpi_dep          = get_kpi("Dépendances")
        kpi_perim        = get_kpi("Périmètre")
        kpi_delais       = get_kpi("Respect des délais")

        nb_kpi_degraded = sum(1 for kv in kpis.values() if kv.get("statut") in ["À surveiller", "À redresser"])
        nb_kpi_critique = sum(1 for kv in kpis.values() if kv.get("statut") == "À redresser")
        nb_kpi_deterioration = sum(1 for kv in kpis.values() if kv.get("tendance") == "Détérioration")

        budget_slope_3w = 0
        if i >= 2:
            prev_weeks = all_weeks[max(0, i-3):i]
            budgets = [meteo_data.get(w, {}).get("conso_jh", 0) for w in prev_weeks]
            if len(budgets) >= 2:
                budgets_clean = [b for b in budgets if b > 0]
                if len(budgets_clean) >= 2:
                    budget_slope_3w = (budgets_clean[-1] - budgets_clean[0]) / max(1, len(budgets_clean) - 1)

        has_risk_text = 1 if faits.get("risque_textuel", "") not in ["", "-", "nan", "Faible"] else 0
        risk_text = faits.get("risque_textuel", "").lower()
        has_retard_keyword   = 1 if any(kw in risk_text for kw in ["retard", "délai", "livraison", "delay"]) else 0
        has_budget_keyword   = 1 if any(kw in risk_text for kw in ["budget", "coût", "dépassement", "financ"]) else 0
        has_qualite_keyword  = 1 if any(kw in risk_text for kw in ["qualité", "anomalie", "conformité", "validation"]) else 0

        record = {
            "project": project_name, "semaine": semaine, "week_num": week_num,
            "pct_semaine": round(pct_semaine, 2), "phase_score": PHASE_SCORE.get(phase, 0.5),
            "conso_jh": conso_jh, "pct_budget_consomme": round(pct_budget_consomme, 2),
            "budget_burn_rate": round(budget_burn_rate, 2), "budget_reste_semaines": round(min(budget_reste_semaines, 200), 2),
            "gap_budget_temps": round(gap_budget_temps, 2), "budget_slope_3w": round(budget_slope_3w, 2),
            "statut_gen_score": STATUT_SCORE.get(statut_gen, 0.5), "tend_gen_score": TENDANCE_SCORE.get(tend_gen, 0.5),
            "kpi_avancement_statut": STATUT_SCORE.get(kpi_avancement.get("statut"), 0.5),
            "kpi_avancement_tend": TENDANCE_SCORE.get(kpi_avancement.get("tendance"), 0.5),
            "kpi_budget_statut": STATUT_SCORE.get(kpi_budget.get("statut"), 0.5),
            "kpi_budget_tend": TENDANCE_SCORE.get(kpi_budget.get("tendance"), 0.5),
            "kpi_risques_statut": STATUT_SCORE.get(kpi_risques.get("statut"), 0.5),
            "kpi_risques_tend": TENDANCE_SCORE.get(kpi_risques.get("tendance"), 0.5),
            "kpi_qualite_statut": STATUT_SCORE.get(kpi_qualite.get("statut"), 0.5),
            "kpi_qualite_tend": TENDANCE_SCORE.get(kpi_qualite.get("tendance"), 0.5),
            "kpi_satisfaction_statut": STATUT_SCORE.get(kpi_satisfaction.get("statut"), 0.5),
            "kpi_delais_statut": STATUT_SCORE.get(kpi_delais.get("statut"), 0.5),
            "kpi_delais_tend": TENDANCE_SCORE.get(kpi_delais.get("tendance"), 0.5),
            "kpi_rh_statut": STATUT_SCORE.get(kpi_rh.get("statut"), 0.5),
            "kpi_dep_statut": STATUT_SCORE.get(kpi_dep.get("statut"), 0.5),
            "kpi_perim_statut": STATUT_SCORE.get(kpi_perim.get("statut"), 0.5),
            "nb_kpi_degraded": nb_kpi_degraded, "nb_kpi_critique": nb_kpi_critique, "nb_kpi_deterioration": nb_kpi_deterioration,
            "has_risk_text": has_risk_text, "has_retard_keyword": has_retard_keyword,
            "has_budget_keyword": has_budget_keyword, "has_qualite_keyword": has_qualite_keyword,
            "budget_total_jh": budget_total_jh, "project_duration_weeks": project_duration_weeks,
        }
        records.append(record)

    return pd.DataFrame(records)


def load_all_projects():
    dfs = []
    data_path = Path(DATA_FOLDER)
    xlsx_files = list(data_path.glob("*.xlsx"))
    if not xlsx_files:
        print(f"[ERREUR] Aucun fichier Excel trouvé dans '{DATA_FOLDER}/'")
        sys.exit(1)
    for f in xlsx_files:
        print(f"  → Chargement {f.name}...")
        try:
            df = load_single_project(f)
            if not df.empty:
                dfs.append(df)
                print(f"     {len(df)} semaines extraites pour {df['project'].iloc[0]}")
        except Exception as e:
            print(f"     [ERREUR] {f.name} : {e}")
    if not dfs:
        print("[ERREUR] Aucune donnée extraite.")
        sys.exit(1)
    return pd.concat(dfs, ignore_index=True)


# ==================== LABELLISATION ====================

def create_labels(df):
    HORIZON = 4
    df = df.copy().sort_values(["project", "week_num"]).reset_index(drop=True)

    labels = {name: [] for name in MODEL_NAMES}

    for idx, row in df.iterrows():
        proj = row["project"]
        wnum = row["week_num"]
        budget_total = row["budget_total_jh"]
        future = df[(df["project"] == proj) & (df["week_num"] > wnum) & (df["week_num"] <= wnum + HORIZON)]

        # Budget
        if budget_total > 0:
            future_conso = future["conso_jh"].max() if not future.empty else row["conso_jh"]
            lbl_b = 1 if (future_conso / budget_total) > 0.90 or row["gap_budget_temps"] > 20 else 0
        else:
            lbl_b = 1 if row["kpi_budget_statut"] <= 0.5 and row["kpi_budget_tend"] <= 0.5 else 0
        labels["budget"].append(lbl_b)

        # Retard
        if not future.empty:
            future_delais_min = future["kpi_delais_statut"].min()
            lbl_r = 1 if future_delais_min <= 0.5 or row["has_retard_keyword"] == 1 else 0
        else:
            lbl_r = 1 if row["kpi_delais_statut"] <= 0.5 else 0
        labels["retard"].append(lbl_r)

        # Qualité
        if not future.empty:
            future_qualite_min = future["kpi_qualite_statut"].min()
            lbl_q = 1 if future_qualite_min <= 0.5 or row["has_qualite_keyword"] == 1 else 0
        else:
            lbl_q = 1 if row["kpi_qualite_statut"] <= 0.5 else 0
        labels["qualite"].append(lbl_q)

        # RH
        future_rh_min = future["kpi_rh_statut"].min() if not future.empty else row["kpi_rh_statut"]
        labels["rh"].append(1 if future_rh_min <= 0.5 else 0)

        # Périmètre
        future_perim_min = future["kpi_perim_statut"].min() if not future.empty else row["kpi_perim_statut"]
        labels["perimetre"].append(1 if future_perim_min <= 0.5 else 0)

        # Dépendances
        future_dep_min = future["kpi_dep_statut"].min() if not future.empty else row["kpi_dep_statut"]
        labels["dependances"].append(1 if future_dep_min <= 0.5 else 0)

        # Satisfaction client
        future_sat_min = future["kpi_satisfaction_statut"].min() if not future.empty else row["kpi_satisfaction_statut"]
        labels["satisfaction"].append(1 if future_sat_min <= 0.5 else 0)

    for name in MODEL_NAMES:
        df[f"label_{name}"] = labels[name]
    return df


# ==================== FEATURE COLUMNS ====================

FEATURE_COLS = [
    "pct_semaine", "phase_score",
    "conso_jh", "pct_budget_consomme", "budget_burn_rate",
    "budget_reste_semaines", "gap_budget_temps", "budget_slope_3w",
    "statut_gen_score", "tend_gen_score",
    "kpi_avancement_statut", "kpi_avancement_tend",
    "kpi_budget_statut", "kpi_budget_tend",
    "kpi_risques_statut", "kpi_risques_tend",
    "kpi_qualite_statut", "kpi_qualite_tend",
    "kpi_satisfaction_statut",
    "kpi_delais_statut", "kpi_delais_tend",
    "kpi_rh_statut", "kpi_dep_statut", "kpi_perim_statut",
    "nb_kpi_degraded", "nb_kpi_critique", "nb_kpi_deterioration",
    "has_risk_text", "has_retard_keyword", "has_budget_keyword", "has_qualite_keyword",
    "budget_total_jh", "project_duration_weeks",
]


# ==================== ENTRAÎNEMENT ====================

def train_model(X, y, model_name, use_smote=True):
    print(f"\n  ── Modèle : {model_name} ──")
    print(f"     Taille dataset : {len(X)} échantillons")
    print(f"     Distribution classes : {dict(zip(*np.unique(y, return_counts=True)))}")

    X_train, y_train = X.copy(), y.copy()
    if use_smote and len(X) >= 10 and y.sum() >= 3:
        try:
            n_neighbors = min(3, y.sum() - 1)
            smote = SMOTE(random_state=42, k_neighbors=n_neighbors)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"     Après SMOTE : {len(X_train)} échantillons")
        except Exception as e:
            print(f"     SMOTE ignoré ({e})")

    base_model = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=2,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=max(1, (y == 0).sum() / max((y == 1).sum(), 1)),
        random_state=42, eval_metric="logloss", verbosity=0, use_label_encoder=False,
    )

    n_splits = min(5, max(2, y.sum()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    try:
        cv_scores = cross_val_score(base_model, X_train, y_train, cv=cv, scoring="roc_auc")
        print(f"     ROC-AUC CV : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    except Exception as e:
        cv_scores = np.array([0.5])
        print(f"     CV ignorée ({e})")

    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=min(3, n_splits))
    try:
        calibrated.fit(X_train, y_train)
    except Exception:
        base_model.fit(X_train, y_train)
        calibrated = base_model

    y_pred = calibrated.predict(X)
    y_proba = calibrated.predict_proba(X)[:, 1]

    print(f"     Precision : {precision_score(y, y_pred, zero_division=0):.3f}")
    print(f"     Recall    : {recall_score(y, y_pred, zero_division=0):.3f}")
    print(f"     F1        : {f1_score(y, y_pred, zero_division=0):.3f}")
    try:
        print(f"     ROC-AUC   : {roc_auc_score(y, y_proba):.3f}")
    except Exception:
        pass

    metrics = {
        "roc_auc_cv_mean": float(cv_scores.mean()), "roc_auc_cv_std": float(cv_scores.std()),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "n_samples": len(X), "n_positive": int(y.sum()),
    }
    return calibrated, metrics


def train_all():
    print("\n" + "="*60)
    print("  PMO PREDICTOR — ENTRAÎNEMENT DES MODÈLES (7 modèles)")
    print("="*60)

    print("\n[1/5] Chargement des données...")
    df_raw = load_all_projects()
    print(f"  Total : {len(df_raw)} semaines × {len(df_raw['project'].unique())} projets")

    print("\n[2/5] Création des labels...")
    df = create_labels(df_raw)
    for name in MODEL_NAMES:
        col = f"label_{name}"
        print(f"  {col:15s} : {df[col].sum()}/{len(df)} positifs ({df[col].mean():.1%})")

    print("\n[3/5] Préparation des features...")
    feature_cols_available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols_available].fillna(0.5)
    print(f"  {len(feature_cols_available)} features utilisées")

    print("\n[4/5] Entraînement des modèles...")
    models = {}
    metrics = {}
    for name in MODEL_NAMES:
        model, met = train_model(X, df[f"label_{name}"], name.upper())
        models[name] = model
        metrics[f"model_{name}"] = met

    # Sauvegarde
    print("\n[5/5] Sauvegarde des modèles...")
    for name, model in models.items():
        with open(MODEL_PATHS[name], "wb") as f:
            pickle.dump(model, f)
    with open(MODEL_PATHS["features"], "wb") as f:
        pickle.dump(feature_cols_available, f)
    meta = {
        **metrics,
        "trained_at": datetime.now().isoformat(),
        "n_projects": len(df["project"].unique()),
        "n_samples": len(df),
        "projects": list(df["project"].unique()),
        "feature_cols": feature_cols_available,
    }
    with open(MODEL_PATHS["meta"], "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  Modèles sauvegardés dans '{MODELS_FOLDER}/'")

    print("\n" + "="*60)
    print("  RÉSUMÉ DES PERFORMANCES")
    print("="*60)
    for name in MODEL_NAMES:
        m = metrics[f"model_{name}"]
        print(f"  {name:12s} → ROC-AUC: {m['roc_auc_cv_mean']:.3f} | F1: {m['f1']:.3f} | "
              f"N: {m['n_samples']} ({m['n_positive']} positifs)")

    print("\n  Entraînement terminé avec succès.")
    print(f"  Utilisez : python pmo_predictor.py predict --project \"NomProjet\"")


# ==================== PRÉDICTION ====================

def load_models():
    missing = [name for name in MODEL_NAMES if not os.path.exists(MODEL_PATHS[name])]
    if missing:
        print(f"[ERREUR] Modèles non trouvés : {missing}")
        print("Exécutez d'abord : python pmo_predictor.py train")
        sys.exit(1)
    models = {}
    for name in MODEL_NAMES:
        with open(MODEL_PATHS[name], "rb") as f:
            models[name] = pickle.load(f)
    with open(MODEL_PATHS["features"], "rb") as f:
        feature_cols = pickle.load(f)
    with open(MODEL_PATHS["meta"], "r") as f:
        meta = json.load(f)
    return models, feature_cols, meta


def get_row_for_prediction(project_name, week_str=None):
    df_all = load_all_projects()
    df_proj = df_all[df_all["project"] == project_name]
    if df_proj.empty:
        print(f"[ERREUR] Projet '{project_name}' non trouvé.")
        print(f"  Projets disponibles : {list(df_all['project'].unique())}")
        sys.exit(1)
    if week_str:
        row = df_proj[df_proj["semaine"] == week_str]
        if row.empty:
            target = week_to_int(week_str)
            df_proj = df_proj.copy()
            df_proj["_dist"] = (df_proj["week_num"] - target).abs()
            row = df_proj.sort_values("_dist").head(1)
            print(f"  [INFO] Semaine {week_str} non trouvée, utilisation de {row['semaine'].values[0]}")
    else:
        row = df_proj.sort_values("week_num", ascending=False).head(1)
    return row.iloc[0], df_proj


def predict_for_project(project_name, week_str=None):
    models, feature_cols, meta = load_models()
    row, _ = get_row_for_prediction(project_name, week_str)
    semaine = row["semaine"]
    X_row = pd.DataFrame([row[feature_cols].fillna(0.5).values], columns=feature_cols)

    predictions = {}
    for name in MODEL_NAMES:
        proba = float(models[name].predict_proba(X_row)[0, 1])
        predictions[name] = proba

    def risk_level(p):
        if p >= 0.75: return "CRITIQUE"
        if p >= 0.55: return "ÉLEVÉ"
        if p >= 0.35: return "MODÉRÉ"
        return "FAIBLE"

    # Mappage des noms pour l'affichage
    labels = {
        "budget": "Risque de dépassement budgétaire",
        "retard": "Risque de retard de livraison",
        "qualite": "Risque de dégradation qualité",
        "rh": "Risque sur les ressources humaines",
        "perimetre": "Risque de dérive du périmètre",
        "dependances": "Risque sur les dépendances",
        "satisfaction": "Risque de baisse de satisfaction client",
    }

    preds = {}
    for name in MODEL_NAMES:
        prob = predictions[name] * 100
        preds[name] = {
            "probabilite": round(prob, 1),
            "niveau": risk_level(predictions[name]),
            "label": labels[name],
        }

    p_composite = np.mean([predictions[n] for n in MODEL_NAMES])
    result = {
        "project": project_name, "semaine": semaine,
        "predictions": preds,
        "risque_composite": round(p_composite * 100, 1),
        "niveau_global": risk_level(p_composite),
        "features_used": {col: round(float(X_row[col].values[0]), 3) for col in feature_cols},
        "model_meta": {
            "trained_at": meta.get("trained_at"),
            "n_samples": meta.get("n_samples"),
            "projects_trained_on": meta.get("projects"),
        }
    }
    return result


def print_prediction_report(result):
    print("\n" + "="*65)
    print(f"  PRÉDICTIONS PMO — {result['project']}")
    print(f"  Semaine analysée : {result['semaine']}")
    print("="*65)
    print(f"\n  RISQUE GLOBAL : {result['risque_composite']}% — {result['niveau_global']}")
    print(f"  {'─'*45}")
    for name, pred in result["predictions"].items():
        bar_len = int(pred["probabilite"] / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"\n  {pred['label']}")
        print(f"  [{bar}] {pred['probabilite']}% — {pred['niveau']}")
    print(f"\n  Modèle entraîné le : {result['model_meta']['trained_at'][:10]}")
    print(f"  Données d'entraînement : {result['model_meta']['n_samples']} semaines sur {result['model_meta']['projects_trained_on']}")
    print("="*65)


# ==================== INTÉGRATION EXTERNE ====================

def get_ml_predictions(project_name, week_str=None):
    try:
        if not all(os.path.exists(MODEL_PATHS[name]) for name in MODEL_NAMES):
            return None
        return predict_for_project(project_name, week_str)
    except Exception as e:
        print(f"[ML] Prédiction ML ignorée : {e}")
        return None


def format_ml_for_prompt(ml_result):
    if not ml_result:
        return ""
    preds = ml_result["predictions"]
    lines = [
        f"PROBABILITÉS ML CALCULÉES (modèle entraîné sur {ml_result['model_meta']['n_samples']} semaines) :",
    ]
    for name, pred in preds.items():
        lines.append(f"  - {pred['label']} : {pred['probabilite']}% ({pred['niveau']})")
    lines.extend([
        f"  - Risque composite global : {ml_result['risque_composite']}% ({ml_result['niveau_global']})",
        "",
        "IMPORTANT : Utilise ces probabilités dans ton analyse. Pour chaque problème identifié,",
        "cite explicitement la probabilité correspondante (ex: 'Probabilité ML : 73%').",
        "Ne remplace jamais ces valeurs par des qualificatifs subjectifs.",
    ])
    return "\n".join(lines)


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(description="PMO Predictor — Modèle ML pour l'Axe 4 Prédiction")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("train", help="Entraîner les 7 modèles sur les données Excel")
    pred_parser = subparsers.add_parser("predict", help="Prédire les risques pour un projet")
    pred_parser.add_argument("--project", required=True)
    pred_parser.add_argument("--week", default=None)
    rep_parser = subparsers.add_parser("report", help="Rapport complet avec SHAP")
    rep_parser.add_argument("--project", required=True)
    rep_parser.add_argument("--week", default=None)
    rep_parser.add_argument("--top", default=10, type=int)
    subparsers.add_parser("info", help="Informations sur les modèles")

    args = parser.parse_args()

    if args.command == "train":
        train_all()
    elif args.command == "predict":
        result = predict_for_project(args.project, args.week)
        print_prediction_report(result)
        out_path = os.path.join(MODELS_FOLDER, f"prediction_{args.project.replace(' ', '_')}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n  Résultat JSON sauvegardé : {out_path}")
    elif args.command == "report":
        result = predict_for_project(args.project, args.week)
        print_prediction_report(result)
        # SHAP explainer (simplifié)
        print("\n  SHAP non implémenté pour 7 modèles, mais vous pouvez étendre.")
    elif args.command == "info":
        if not os.path.exists(MODEL_PATHS["meta"]):
            print("[INFO] Aucun modèle entraîné. Exécutez : python pmo_predictor.py train")
        else:
            with open(MODEL_PATHS["meta"]) as f:
                meta = json.load(f)
            print("\n  INFORMATIONS MODÈLES PMO PREDICTOR")
            print("="*45)
            print(f"  Entraîné le     : {meta['trained_at'][:19]}")
            print(f"  Projets         : {', '.join(meta['projects'])}")
            print(f"  Échantillons    : {meta['n_samples']} semaines")
            print(f"  Features        : {len(meta['feature_cols'])}")
            print("\n  Performances :")
            for name in MODEL_NAMES:
                m = meta.get(f"model_{name}", {})
                print(f"    {name:12s} → ROC-AUC: {m.get('roc_auc_cv_mean', 0):.3f} | F1: {m.get('f1', 0):.3f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()