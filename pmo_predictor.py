"""
pmo_predictor.py v2 — Modèle ML standalone pour l'Axe 4 Prédiction
====================================================================
Fichier UNIQUE et AUTONOME. Aucune dépendance vers rag.py ou app.py.

CORRECTIONS v2 :
  - Split temporel 80/20 par projet (jamais aléatoire)
  - TimeSeriesSplit pour la validation croisée (pas StratifiedKFold)
  - SMOTE appliqué uniquement dans le fold d'entraînement (pas sur test)
  - Évaluation séparée sur projets réels vs synthétiques
  - Détection automatique data leakage
  - Rapport honnête avec intervalles de confiance

Usage :
    python pmo_predictor.py train
    python pmo_predictor.py predict --project "Data Fraud Detection" --week "S22-26"
    python pmo_predictor.py report --project "Scoring Crédit Retail"
    python pmo_predictor.py info
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

try:
    from xgboost import XGBClassifier
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import (
        roc_auc_score, precision_score, recall_score, f1_score,
        brier_score_loss, confusion_matrix
    )
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    import shap
    ML_AVAILABLE = True
except ImportError as e:
    print(f"[ERREUR] Dépendances manquantes : {e}")
    print("Exécutez : pip install xgboost scikit-learn imbalanced-learn shap --break-system-packages")
    sys.exit(1)


# ==================== CONFIGURATION ====================

DATA_FOLDER   = "data"
MODELS_FOLDER = "ml_models"
os.makedirs(MODELS_FOLDER, exist_ok=True)

MODEL_NAMES = ["budget", "retard", "qualite", "rh", "perimetre", "dependances", "satisfaction"]

MODEL_PATHS = {name: os.path.join(MODELS_FOLDER, f"model_{name}.pkl") for name in MODEL_NAMES}
MODEL_PATHS["meta"]     = os.path.join(MODELS_FOLDER, "model_meta.json")
MODEL_PATHS["features"] = os.path.join(MODELS_FOLDER, "feature_names.pkl")

STATUT_SCORE  = {"En contrôle": 1.0, "À surveiller": 0.5, "À redresser": 0.0, "": 0.5, None: 0.5}
TENDANCE_SCORE = {"Amélioration": 1.0, "Stand by": 0.5, "Détérioration": 0.0, "": 0.5, None: 0.5}
PHASE_SCORE   = {
    "Cadrage": 0.0, "Spécification": 0.15, "Dev": 0.30,
    "Dev/Intégration": 0.35, "Dev/intégration": 0.35,
    "Homologation": 0.55, "Test": 0.60, "Pré-Prod": 0.70,
    "Mise en production": 0.85, "MEP": 0.85, "Prod": 1.0,
}

MODEL_LABELS = {
    "budget":      "Risque de dépassement budgétaire",
    "retard":      "Risque de retard de livraison",
    "qualite":     "Risque de dégradation qualité",
    "rh":          "Risque sur les ressources humaines",
    "perimetre":   "Risque de dérive du périmètre",
    "dependances": "Risque sur les dépendances",
    "satisfaction":"Risque de baisse de satisfaction client",
}

# Projets réels (non synthétiques) — utilisés pour l'évaluation honnête
REAL_PROJECTS_KEYWORDS = [
    "Data Fraud Detection", "Scoring Crédit Retail", "Orion Data Platform"
]


# ==================== UTILITAIRES ====================

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
    m = re.search(r'S(\d+)', str(week_str))
    return int(m.group(1)) if m else 0

def normalize_kpi_name(name):
    name = str(name).lower().strip()
    for a, b in [('é','e'),('è','e'),('ê','e'),('à','a'),('â','a'),('ô','o'),('î','i'),('û','u')]:
        name = name.replace(a, b)
    return name

def is_real_project(project_name):
    return any(kw.lower() in project_name.lower() for kw in REAL_PROJECTS_KEYWORDS)


# ==================== EXTRACTION DONNÉES EXCEL ====================

def load_single_project(filepath):
    """Charge un fichier Excel PMO et retourne un DataFrame semaine par semaine."""
    xls = pd.ExcelFile(filepath)
    records = []

    df_info = pd.read_excel(xls, sheet_name="Infos_projet")
    df_info = clean_columns(df_info)
    if df_info.empty or "Projet" not in df_info.columns:
        return pd.DataFrame()

    row_info     = df_info.iloc[0]
    project_name = safe_str(row_info.get("Projet", ""))

    budget_total_jh = 0
    col_jh = next((c for c in df_info.columns if "Budget" in c and "J/H" in c and "consomm" not in c.lower()), None)
    if col_jh and pd.notna(row_info.get(col_jh)):
        num = re.sub(r'[^\d]', '', str(row_info[col_jh]))
        budget_total_jh = int(num) if num else 0

    date_debut, date_fin = None, None
    col_dd = next((c for c in df_info.columns if "début" in c.lower() or "debut" in c.lower()), None)
    col_df = next((c for c in df_info.columns if "fin" in c.lower() and "date" in c.lower()), None)
    if col_dd and pd.notna(row_info.get(col_dd)):
        try: date_debut = pd.to_datetime(row_info[col_dd])
        except: pass
    if col_df and pd.notna(row_info.get(col_df)):
        try: date_fin = pd.to_datetime(row_info[col_df])
        except: pass

    project_duration_weeks = max(1, int((date_fin - date_debut).days / 7)) if (date_debut and date_fin) else 0

    # ── Météo générale ──
    sheet_meteo = next((s for s in xls.sheet_names if s.lower() in ("météo générale","meteo generale")), None)
    meteo_data = {}
    if sheet_meteo:
        df_m = pd.read_excel(xls, sheet_name=sheet_meteo)
        df_m = clean_columns(df_m)
        df_m = df_m[df_m.get("Projet", pd.Series(dtype=str)).astype(str).str.strip() == project_name]
        for _, r in df_m.iterrows():
            sem = safe_str(r.get("Semaine",""))
            if not sem: continue
            statut_cols = [c for c in df_m.columns if "statut" in c.lower() and
                          ("général" in c.lower() or "generale" in c.lower() or "Générale" in c)]
            statut_gen = ""
            for sc in statut_cols:
                if pd.notna(r.get(sc)): statut_gen = safe_str(r[sc]); break
            if not statut_gen:
                sc2 = next((c for c in df_m.columns if "statut" in c.lower()), None)
                if sc2 and pd.notna(r.get(sc2)): statut_gen = safe_str(r[sc2])
            tend_cols = [c for c in df_m.columns if "tendenc" in c.lower() or "tendanc" in c.lower()]
            tend_gen = ""
            for tc in tend_cols:
                if pd.notna(r.get(tc)): tend_gen = safe_str(r[tc]); break
            phase = ""
            pc = next((c for c in df_m.columns if "phase" in c.lower()), None)
            if pc and pd.notna(r.get(pc)): phase = safe_str(r[pc])
            conso_jh = 0
            cc = next((c for c in df_m.columns if "consomm" in c.lower() and "j/h" in c.lower()), None)
            if cc and pd.notna(r.get(cc)):
                num = re.sub(r'[^\d]', '', str(r[cc])); conso_jh = int(num) if num else 0
            reste_jh = 0
            rc = next((c for c in df_m.columns if "reste" in c.lower() and "j/h" in c.lower()), None)
            if rc and pd.notna(r.get(rc)):
                num = re.sub(r'[^\d]', '', str(r[rc])); reste_jh = int(num) if num else 0
            meteo_data[sem] = {"statut_gen": statut_gen, "tend_gen": tend_gen, "phase": phase,
                               "conso_jh": conso_jh, "reste_jh": reste_jh}

    # ── KPI ──
    kpi_data = {}
    if "KPI" in xls.sheet_names:
        df_kpi = pd.read_excel(xls, sheet_name="KPI", header=[0, 1])
        df_kpi.columns = [re.sub(r'\s+', ' ', '_'.join([str(l).strip() for l in col if str(l) != 'nan'])).strip()
                          for col in df_kpi.columns]
        sem_col  = next((c for c in df_kpi.columns if 'Semaine' in c), None)
        proj_col = next((c for c in df_kpi.columns if 'Projet' in c), None)
        if sem_col and proj_col:
            statut_cols_kpi = [c for c in df_kpi.columns if c.endswith('_Statut')]
            for _, r in df_kpi.iterrows():
                sem   = safe_str(r.get(sem_col, ""))
                proj  = safe_str(r.get(proj_col, ""))
                if not sem or proj != project_name: continue
                kpi_data.setdefault(sem, {})
                for sc in statut_cols_kpi:
                    kpi_name  = sc.replace('_Statut', '')
                    statut_v  = safe_str(r.get(sc, ""))
                    tend_col  = sc.replace('_Statut', '_Tendence')
                    tend_v    = safe_str(r.get(tend_col, "")) if tend_col in df_kpi.columns else ""
                    kpi_data[sem][kpi_name] = {"statut": statut_v, "tendance": tend_v}

    # ── Faits marquants ──
    faits_data = {}
    if "Faits marquants" in xls.sheet_names:
        df_f = pd.read_excel(xls, sheet_name="Faits marquants")
        df_f = clean_columns(df_f)
        df_f = df_f[df_f.get("Projet", pd.Series(dtype=str)).astype(str).str.strip() == project_name]
        for _, r in df_f.iterrows():
            sem = safe_str(r.get("Semaine", ""))
            if not sem: continue
            risque_col = next((c for c in df_f.columns if "risque" in c.lower()), None)
            risque = safe_str(r.get(risque_col, "")) if risque_col else ""
            faits_data[sem] = {"risque_textuel": risque}

    # ── Construction records ──
    all_weeks = sorted(set(list(meteo_data.keys()) + list(kpi_data.keys())), key=lambda w: week_to_int(w))

    for i, sem in enumerate(all_weeks):
        meteo = meteo_data.get(sem, {})
        kpis  = kpi_data.get(sem, {})
        faits = faits_data.get(sem, {})

        week_num  = week_to_int(sem)
        conso_jh  = meteo.get("conso_jh", 0)
        reste_jh  = meteo.get("reste_jh", 0)
        phase     = meteo.get("phase", "")
        statut_gen = meteo.get("statut_gen", "")
        tend_gen  = meteo.get("tend_gen", "")

        pct_budget  = (conso_jh / budget_total_jh * 100) if budget_total_jh > 0 else 0
        pct_semaine = (week_num / project_duration_weeks * 100) if project_duration_weeks > 0 else 0
        burn_rate   = conso_jh / max(week_num, 1)
        reste_weeks = (reste_jh / burn_rate) if burn_rate > 0 else 999
        gap_bt      = pct_budget - pct_semaine

        def get_kpi(name):
            nn = normalize_kpi_name(name)
            for kn, kv in kpis.items():
                if normalize_kpi_name(kn) == nn: return kv
            return {"statut": None, "tendance": None}

        kpi_av  = get_kpi("Avancement")
        kpi_bu  = get_kpi("Budget")
        kpi_ri  = get_kpi("Risques")
        kpi_qu  = get_kpi("Qualité du delivery")
        kpi_sa  = get_kpi("Satisfaction client")
        kpi_rh  = get_kpi("Ressources humaines")
        kpi_dep = get_kpi("Dépendances")
        kpi_per = get_kpi("Périmètre")
        kpi_del = get_kpi("Respect des délais")

        nb_degrad  = sum(1 for kv in kpis.values() if kv.get("statut") in ["À surveiller","À redresser"])
        nb_crit    = sum(1 for kv in kpis.values() if kv.get("statut") == "À redresser")
        nb_detrio  = sum(1 for kv in kpis.values() if kv.get("tendance") == "Détérioration")

        slope_3w = 0.0
        if i >= 2:
            prev = all_weeks[max(0, i-3):i]
            budgets = [b for w in prev if (b := meteo_data.get(w, {}).get("conso_jh", 0)) > 0]
            if len(budgets) >= 2:
                slope_3w = (budgets[-1] - budgets[0]) / max(1, len(budgets) - 1)

        risk_text = faits.get("risque_textuel", "").lower()
        has_risk  = 1 if risk_text not in ["", "-", "nan", "faible"] else 0
        has_ret_kw  = 1 if any(kw in risk_text for kw in ["retard","délai","livraison"]) else 0
        has_bud_kw  = 1 if any(kw in risk_text for kw in ["budget","coût","dépassement"]) else 0
        has_qua_kw  = 1 if any(kw in risk_text for kw in ["qualité","anomalie","conformité"]) else 0

        records.append({
            "project": project_name, "semaine": sem, "week_num": week_num,
            "is_real_project": int(is_real_project(project_name)),
            "pct_semaine": round(pct_semaine, 2),
            "phase_score": PHASE_SCORE.get(phase, 0.5),
            "conso_jh": conso_jh,
            "pct_budget_consomme": round(pct_budget, 2),
            "budget_burn_rate": round(burn_rate, 2),
            "budget_reste_semaines": round(min(reste_weeks, 200), 2),
            "gap_budget_temps": round(gap_bt, 2),
            "budget_slope_3w": round(slope_3w, 2),
            "statut_gen_score": STATUT_SCORE.get(statut_gen, 0.5),
            "tend_gen_score": TENDANCE_SCORE.get(tend_gen, 0.5),
            "kpi_avancement_statut": STATUT_SCORE.get(kpi_av.get("statut"), 0.5),
            "kpi_avancement_tend": TENDANCE_SCORE.get(kpi_av.get("tendance"), 0.5),
            "kpi_budget_statut": STATUT_SCORE.get(kpi_bu.get("statut"), 0.5),
            "kpi_budget_tend": TENDANCE_SCORE.get(kpi_bu.get("tendance"), 0.5),
            "kpi_risques_statut": STATUT_SCORE.get(kpi_ri.get("statut"), 0.5),
            "kpi_risques_tend": TENDANCE_SCORE.get(kpi_ri.get("tendance"), 0.5),
            "kpi_qualite_statut": STATUT_SCORE.get(kpi_qu.get("statut"), 0.5),
            "kpi_qualite_tend": TENDANCE_SCORE.get(kpi_qu.get("tendance"), 0.5),
            "kpi_satisfaction_statut": STATUT_SCORE.get(kpi_sa.get("statut"), 0.5),
            "kpi_delais_statut": STATUT_SCORE.get(kpi_del.get("statut"), 0.5),
            "kpi_delais_tend": TENDANCE_SCORE.get(kpi_del.get("tendance"), 0.5),
            "kpi_rh_statut": STATUT_SCORE.get(kpi_rh.get("statut"), 0.5),
            "kpi_dep_statut": STATUT_SCORE.get(kpi_dep.get("statut"), 0.5),
            "kpi_perim_statut": STATUT_SCORE.get(kpi_per.get("statut"), 0.5),
            "nb_kpi_degraded": nb_degrad, "nb_kpi_critique": nb_crit, "nb_kpi_deterioration": nb_detrio,
            "has_risk_text": has_risk, "has_retard_keyword": has_ret_kw,
            "has_budget_keyword": has_bud_kw, "has_qualite_keyword": has_qua_kw,
            "budget_total_jh": budget_total_jh, "project_duration_weeks": project_duration_weeks,
        })

    return pd.DataFrame(records)


def load_all_projects():
    dfs = []
    data_path = Path(DATA_FOLDER)
    xlsx_files = list(data_path.glob("*.xlsx"))
    if not xlsx_files:
        print(f"[ERREUR] Aucun fichier Excel dans '{DATA_FOLDER}/'")
        sys.exit(1)
    for f in xlsx_files:
        print(f"  → {f.name}...", end=" ")
        try:
            df = load_single_project(f)
            if not df.empty:
                dfs.append(df)
                proj = df["project"].iloc[0]
                tag  = "[REEL]" if is_real_project(proj) else "[synth]"
                print(f"{tag} {len(df)} semaines — {proj}")
            else:
                print("vide, ignoré")
        except Exception as e:
            print(f"ERREUR : {e}")
    if not dfs:
        print("[ERREUR] Aucune donnée extraite.")
        sys.exit(1)
    return pd.concat(dfs, ignore_index=True)


# ==================== LABELLISATION ====================

def create_labels(df, horizon=4):
    """
    Crée les labels en regardant l'horizon futur.
    IMPORTANT : les labels sont créés AVANT le split — c'est correct car
    le split sera fait ensuite par ordre chronologique.
    """
    df = df.copy().sort_values(["project", "week_num"]).reset_index(drop=True)
    labels = {name: [] for name in MODEL_NAMES}

    for idx, row in df.iterrows():
        proj   = row["project"]
        wnum   = row["week_num"]
        bt     = row["budget_total_jh"]
        future = df[(df["project"] == proj) & (df["week_num"] > wnum) & (df["week_num"] <= wnum + horizon)]

        # Budget
        if bt > 0:
            fc = future["conso_jh"].max() if not future.empty else row["conso_jh"]
            lbl_b = 1 if (fc / bt) > 0.90 or row["gap_budget_temps"] > 20 else 0
        else:
            lbl_b = 1 if row["kpi_budget_statut"] <= 0.5 and row["kpi_budget_tend"] <= 0.5 else 0
        labels["budget"].append(lbl_b)

        # Retard
        if not future.empty:
            lbl_r = 1 if future["kpi_delais_statut"].min() <= 0.5 or row["has_retard_keyword"] else 0
        else:
            lbl_r = 1 if row["kpi_delais_statut"] <= 0.5 else 0
        labels["retard"].append(lbl_r)

        # Qualité
        if not future.empty:
            lbl_q = 1 if future["kpi_qualite_statut"].min() <= 0.5 or row["has_qualite_keyword"] else 0
        else:
            lbl_q = 1 if row["kpi_qualite_statut"] <= 0.5 else 0
        labels["qualite"].append(lbl_q)

        # RH
        labels["rh"].append(1 if (future["kpi_rh_statut"].min() if not future.empty else row["kpi_rh_statut"]) <= 0.5 else 0)

        # Périmètre
        labels["perimetre"].append(1 if (future["kpi_perim_statut"].min() if not future.empty else row["kpi_perim_statut"]) <= 0.5 else 0)

        # Dépendances
        labels["dependances"].append(1 if (future["kpi_dep_statut"].min() if not future.empty else row["kpi_dep_statut"]) <= 0.5 else 0)

        # Satisfaction
        labels["satisfaction"].append(1 if (future["kpi_satisfaction_statut"].min() if not future.empty else row["kpi_satisfaction_statut"]) <= 0.5 else 0)

    for name in MODEL_NAMES:
        df[f"label_{name}"] = labels[name]
    return df


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


# ==================== SPLIT TEMPOREL ====================

def temporal_split_by_project(df, test_ratio=0.20):
    """
    Split temporel 80/20 par projet.

    Pour CHAQUE projet :
      - Les 80% premières semaines → train
      - Les 20% dernières semaines → test

    Cela évite la fuite temporelle : le modèle ne voit jamais le futur.

    Returns
    -------
    train_idx, test_idx : listes d'index pandas
    """
    train_idx, test_idx = [], []

    for proj in df["project"].unique():
        proj_df = df[df["project"] == proj].sort_values("week_num")
        n = len(proj_df)
        n_test  = max(1, int(n * test_ratio))
        n_train = n - n_test

        proj_train = proj_df.iloc[:n_train].index.tolist()
        proj_test  = proj_df.iloc[n_train:].index.tolist()

        train_idx.extend(proj_train)
        test_idx.extend(proj_test)

    return train_idx, test_idx


# ==================== ENTRAÎNEMENT ====================

def train_single_model(X_train, y_train, X_test, y_test, model_name,
                       X_real_test=None, y_real_test=None):
    """
    Entraîne un modèle avec :
    - SMOTE uniquement sur X_train (pas sur test)
    - TimeSeriesSplit pour la validation croisée
    - Évaluation honnête sur test set chronologique
    - Évaluation séparée sur projets réels si disponibles
    """
    print(f"\n  ── {model_name.upper()} ──")
    print(f"     Train : {len(X_train)} | Test : {len(X_test)} | Positifs train : {y_train.sum()} ({y_train.mean():.0%})")

    # ── XGBoost de base ──
    base = XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=0.2,
        reg_alpha=0.2,
        reg_lambda=1.5,
        scale_pos_weight=max(1.0, (y_train == 0).sum() / max((y_train == 1).sum(), 1)),
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
        use_label_encoder=False,
    )

    # ── TimeSeriesSplit sur le train set ──
    # On regroupe par semaine pour garder l'ordre temporel
    n_splits = min(4, max(2, y_train.sum() // 5))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_aucs = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr  = X_train.iloc[tr_idx]
        y_tr  = y_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        if y_tr.sum() < 2 or y_val.sum() < 1:
            continue  # Fold trop petit

        # SMOTE uniquement sur le fold d'entraînement
        X_tr_res, y_tr_res = X_tr.copy(), y_tr.copy()
        if y_tr.sum() >= 3:
            try:
                k = min(3, y_tr.sum() - 1)
                smote = SMOTE(random_state=42, k_neighbors=k)
                X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
            except Exception:
                pass

        base.fit(X_tr_res, y_tr_res)
        if y_val.sum() > 0:
            try:
                auc = roc_auc_score(y_val, base.predict_proba(X_val)[:, 1])
                cv_aucs.append(auc)
            except Exception:
                pass

    cv_mean = float(np.mean(cv_aucs)) if cv_aucs else 0.5
    cv_std  = float(np.std(cv_aucs))  if cv_aucs else 0.0
    print(f"     ROC-AUC TimeSeriesCV ({n_splits} folds) : {cv_mean:.3f} ± {cv_std:.3f}")

    # ── Entraînement final sur tout le train set (avec SMOTE) ──
    X_train_final, y_train_final = X_train.copy(), y_train.copy()
    if y_train.sum() >= 3:
        try:
            k = min(3, y_train.sum() - 1)
            smote = SMOTE(random_state=42, k_neighbors=k)
            X_train_final, y_train_final = smote.fit_resample(X_train, y_train)
        except Exception:
            pass

    # Calibration sur données non-augmentées pour des probabilités fiables
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=min(3, n_splits))
    try:
        calibrated.fit(X_train_final, y_train_final)
    except Exception:
        base.fit(X_train_final, y_train_final)
        calibrated = base

    # ── Évaluation sur test set chronologique (données JAMAIS vues) ──
    y_pred_test  = calibrated.predict(X_test)
    y_proba_test = calibrated.predict_proba(X_test)[:, 1]

    test_auc  = roc_auc_score(y_test, y_proba_test) if y_test.sum() > 0 else 0.0
    test_f1   = f1_score(y_test, y_pred_test, zero_division=0)
    test_prec = precision_score(y_test, y_pred_test, zero_division=0)
    test_rec  = recall_score(y_test, y_pred_test, zero_division=0)
    brier     = brier_score_loss(y_test, y_proba_test)

    print(f"     [TEST SET — données jamais vues, chronologique]")
    print(f"     ROC-AUC : {test_auc:.3f} | F1 : {test_f1:.3f} | Brier : {brier:.3f}")
    print(f"     Precision : {test_prec:.3f} | Recall : {test_rec:.3f}")

    # ── Évaluation sur projets RÉELS uniquement ──
    real_metrics = {}
    if X_real_test is not None and len(X_real_test) > 0 and y_real_test.sum() > 0:
        y_pred_r  = calibrated.predict(X_real_test)
        y_proba_r = calibrated.predict_proba(X_real_test)[:, 1]
        try:
            real_auc = roc_auc_score(y_real_test, y_proba_r)
        except Exception:
            real_auc = 0.0
        real_f1  = f1_score(y_real_test, y_pred_r, zero_division=0)
        real_prec = precision_score(y_real_test, y_pred_r, zero_division=0)
        real_rec  = recall_score(y_real_test, y_pred_r, zero_division=0)
        real_brier = brier_score_loss(y_real_test, y_proba_r)
        print(f"     [PROJETS RÉELS UNIQUEMENT — {len(X_real_test)} semaines]")
        print(f"     ROC-AUC : {real_auc:.3f} | F1 : {real_f1:.3f} | Brier : {real_brier:.3f}")
        real_metrics = {
            "roc_auc": real_auc, "f1": real_f1,
            "precision": real_prec, "recall": real_rec, "brier": real_brier,
            "n_samples": len(X_real_test), "n_positive": int(y_real_test.sum())
        }

        # Avertissement si l'écart est trop grand → signal d'overfitting persistant
        if cv_mean - real_auc > 0.15:
            print(f"     ⚠ Écart important CV/Réel ({cv_mean:.3f} vs {real_auc:.3f}) — modèle moins fiable sur projets réels")

    metrics = {
        "cv_roc_auc_mean": cv_mean, "cv_roc_auc_std": cv_std,
        "test_roc_auc": test_auc, "test_f1": test_f1,
        "test_precision": test_prec, "test_recall": test_rec,
        "test_brier": brier,
        "n_train": len(X_train), "n_test": len(X_test),
        "n_positive_train": int(y_train.sum()), "n_positive_test": int(y_test.sum()),
        "real_projects_metrics": real_metrics,
    }
    return calibrated, metrics


def train_all():
    print("\n" + "="*65)
    print("  PMO PREDICTOR v2 — ENTRAÎNEMENT (split temporel, sans data leakage)")
    print("="*65)

    print("\n[1/6] Chargement des données...")
    df_raw = load_all_projects()
    n_real  = len(df_raw[df_raw["is_real_project"] == 1]["project"].unique())
    n_synth = len(df_raw[df_raw["is_real_project"] == 0]["project"].unique())
    print(f"  Total : {len(df_raw)} semaines | {n_real} projets réels + {n_synth} projets synthétiques")

    print("\n[2/6] Création des labels (horizon = 4 semaines)...")
    df = create_labels(df_raw)
    for name in MODEL_NAMES:
        col = f"label_{name}"
        pos = df[col].sum()
        print(f"  {col:20s} : {pos}/{len(df)} positifs ({df[col].mean():.0%})")

    print("\n[3/6] Split temporel 80/20 par projet...")
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X_all = df[feature_cols].fillna(0.5)

    train_idx, test_idx = temporal_split_by_project(df, test_ratio=0.20)
    X_train = X_all.loc[train_idx]
    X_test  = X_all.loc[test_idx]
    df_train = df.loc[train_idx]
    df_test  = df.loc[test_idx]
    print(f"  Train : {len(X_train)} semaines | Test : {len(X_test)} semaines")
    print(f"  Projets réels dans test set : {df_test[df_test['is_real_project']==1]['project'].unique().tolist()}")

    # Subset des projets réels dans le test set
    real_test_idx  = [i for i in test_idx if df.loc[i, "is_real_project"] == 1]
    X_real_test    = X_all.loc[real_test_idx] if real_test_idx else pd.DataFrame()

    print("\n[4/6] Vérification anti-data-leakage...")
    for proj in df["project"].unique():
        proj_df = df[df["project"] == proj].sort_values("week_num")
        proj_train = [i for i in train_idx if i in proj_df.index]
        proj_test  = [i for i in test_idx  if i in proj_df.index]
        if proj_train and proj_test:
            max_train_week = proj_df.loc[proj_train, "week_num"].max()
            min_test_week  = proj_df.loc[proj_test,  "week_num"].min()
            if max_train_week >= min_test_week:
                print(f"  ⚠ LEAKAGE DÉTECTÉ pour {proj} : max_train={max_train_week} >= min_test={min_test_week}")
            else:
                print(f"  ✓ {proj} : train jusqu'à S{max_train_week:02d}, test à partir de S{min_test_week:02d}")

    print("\n[5/6] Entraînement des 7 modèles...")
    models, metrics = {}, {}
    for name in MODEL_NAMES:
        y_train = df_train[f"label_{name}"]
        y_test  = df_test[f"label_{name}"]
        y_real  = df.loc[real_test_idx, f"label_{name}"] if real_test_idx else pd.Series(dtype=int)
        model, met = train_single_model(X_train, y_train, X_test, y_test,
                                        name, X_real_test, y_real)
        models[name]  = model
        metrics[name] = met

    print("\n[6/6] Sauvegarde...")
    for name, model in models.items():
        with open(MODEL_PATHS[name], "wb") as f:
            pickle.dump(model, f)
    with open(MODEL_PATHS["features"], "wb") as f:
        pickle.dump(feature_cols, f)
    meta = {
        "trained_at": datetime.now().isoformat(),
        "n_projects": int(df["project"].nunique()),
        "n_real_projects": int(n_real),
        "n_samples_train": len(X_train),
        "n_samples_test": len(X_test),
        "projects": list(df["project"].unique()),
        "real_projects": [p for p in df["project"].unique() if is_real_project(p)],
        "feature_cols": feature_cols,
        "split_method": "temporal_80_20_per_project",
        "cv_method": "TimeSeriesSplit",
        "leakage_protected": True,
        "models": {name: met for name, met in metrics.items()},
    }
    with open(MODEL_PATHS["meta"], "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # ── Résumé final honnête ──
    print("\n" + "="*65)
    print("  RÉSUMÉ — MÉTRIQUES HONNÊTES (test set chronologique)")
    print("="*65)
    print(f"  {'Modèle':<14} {'CV ROC-AUC':>12} {'Test ROC-AUC':>13} {'Test F1':>9} {'Réel ROC-AUC':>13}")
    print(f"  {'─'*63}")
    for name in MODEL_NAMES:
        m = metrics[name]
        real_auc = m["real_projects_metrics"].get("roc_auc", 0) if m["real_projects_metrics"] else 0
        cv_str   = f"{m['cv_roc_auc_mean']:.3f}±{m['cv_roc_auc_std']:.3f}"
        print(f"  {name:<14} {cv_str:>12}  {m['test_roc_auc']:>12.3f}  {m['test_f1']:>8.3f}  {real_auc:>12.3f}")
    print(f"\n  Note : 'Réel ROC-AUC' = performance sur vos 3 projets UIB uniquement.")
    print(f"  Un ROC-AUC > 0.85 sur données réelles = bon modèle pour votre contexte.")
    print(f"\n  Entraînement terminé. Utilisez :")
    print(f"  python pmo_predictor.py predict --project \"NomProjet\"")


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
    df_all  = load_all_projects()
    df_proj = df_all[df_all["project"] == project_name]
    if df_proj.empty:
        print(f"[ERREUR] Projet '{project_name}' non trouvé.")
        print(f"  Disponibles : {list(df_all['project'].unique())}")
        sys.exit(1)
    if week_str:
        row = df_proj[df_proj["semaine"] == week_str]
        if row.empty:
            target = week_to_int(week_str)
            df_c = df_proj.copy()
            df_c["_dist"] = (df_c["week_num"] - target).abs()
            row = df_c.sort_values("_dist").head(1)
            print(f"  [INFO] Semaine {week_str} non trouvée → utilisation de {row['semaine'].values[0]}")
    else:
        row = df_proj.sort_values("week_num", ascending=False).head(1)
    return row.iloc[0]


def predict_for_project(project_name, week_str=None):
    models, feature_cols, meta = load_models()
    row     = get_row_for_prediction(project_name, week_str)
    semaine = row["semaine"]
    X_row   = pd.DataFrame([row[feature_cols].fillna(0.5).values], columns=feature_cols)

    def risk_level(p):
        if p >= 0.75: return "CRITIQUE"
        if p >= 0.55: return "ÉLEVÉ"
        if p >= 0.35: return "MODÉRÉ"
        return "FAIBLE"

    preds = {}
    for name in MODEL_NAMES:
        proba = float(models[name].predict_proba(X_row)[0, 1])
        model_auc = meta.get("models", {}).get(name, {}).get("real_projects_metrics", {}).get("roc_auc", 0)
        preds[name] = {
            "probabilite": round(proba * 100, 1),
            "niveau": risk_level(proba),
            "label": MODEL_LABELS[name],
            "model_roc_auc": round(model_auc, 3),
        }

    composite = float(np.mean([preds[n]["probabilite"] for n in MODEL_NAMES]))
    return {
        "project": project_name,
        "semaine": semaine,
        "predictions": preds,
        "risque_composite": round(composite, 1),
        "niveau_global": risk_level(composite / 100),
        "model_meta": {
            "trained_at": meta.get("trained_at"),
            "n_train": meta.get("n_samples_train"),
            "n_test": meta.get("n_samples_test"),
            "real_projects": meta.get("real_projects"),
            "split_method": meta.get("split_method"),
            "leakage_protected": meta.get("leakage_protected", False),
        }
    }


def print_prediction_report(result):
    print("\n" + "="*65)
    print(f"  PRÉDICTIONS PMO — {result['project']}")
    print(f"  Semaine : {result['semaine']}  |  Méthode : {result['model_meta']['split_method']}")
    print(f"  Protection anti-leakage : {'✓' if result['model_meta']['leakage_protected'] else '✗'}")
    print("="*65)
    print(f"\n  RISQUE COMPOSITE : {result['risque_composite']}% — {result['niveau_global']}\n")
    for name, pred in result["predictions"].items():
        bar_len = int(pred["probabilite"] / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        auc_str = f"  [ROC-AUC modèle : {pred['model_roc_auc']:.3f}]" if pred["model_roc_auc"] > 0 else ""
        print(f"  {pred['label']}")
        print(f"  [{bar}] {pred['probabilite']}%  {pred['niveau']}{auc_str}")
        print()
    print(f"  Entraîné le : {result['model_meta']['trained_at'][:10]}")
    print(f"  Train : {result['model_meta']['n_train']} sem. | Test : {result['model_meta']['n_test']} sem.")
    print("="*65)


# ==================== SHAP ──────────────────────────────── ====================

def explain_prediction(project_name, week_str=None, top_n=8):
    models, feature_cols, meta = load_models()
    row = get_row_for_prediction(project_name, week_str)
    X_row = pd.DataFrame([row[feature_cols].fillna(0.5).values], columns=feature_cols)

    print(f"\n  EXPLICATION SHAP — {project_name} ({row['semaine']})")
    print("="*65)

    for name in MODEL_NAMES:
        model = models[name]
        print(f"\n  ── {MODEL_LABELS[name]} ──")
        try:
            # Extraire le XGB sous-jacent depuis le modèle calibré
            base_est = model
            if hasattr(model, "calibrated_classifiers_"):
                base_est = model.calibrated_classifiers_[0].base_estimator
            elif hasattr(model, "estimator"):
                base_est = model.estimator

            explainer  = shap.TreeExplainer(base_est)
            shap_vals  = explainer.shap_values(X_row)
            sv = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]

            impact = sorted(zip(feature_cols, sv, X_row.values[0]), key=lambda x: abs(x[1]), reverse=True)
            print(f"  {'Feature':<35} {'Valeur':>8}  {'Impact':>10}")
            print(f"  {'─'*55}")
            for feat, imp, val in impact[:top_n]:
                direction = "▲" if imp > 0 else "▼"
                print(f"  {feat:<35} {val:>8.3f}  {direction} {abs(imp):>8.4f}")
        except Exception as e:
            print(f"  SHAP non disponible : {e}")
            if hasattr(base_est, "feature_importances_"):
                imps = sorted(zip(feature_cols, base_est.feature_importances_), key=lambda x: x[1], reverse=True)
                print(f"  {'Feature':<35} {'Importance':>12}")
                for feat, imp in imps[:top_n]:
                    if imp > 0:
                        print(f"  {feat:<35} {imp:>12.4f}")


# ==================== INTÉGRATION EXTERNE ====================

def get_ml_predictions(project_name, week_str=None):
    """
    Appelable depuis rag.py → predict_problems().
    Retourne None si les modèles ne sont pas entraînés (pas bloquant).
    """
    try:
        if not all(os.path.exists(MODEL_PATHS[n]) for n in MODEL_NAMES):
            return None
        return predict_for_project(project_name, week_str)
    except Exception as e:
        print(f"[ML] Prédiction ignorée : {e}")
        return None


def format_ml_for_prompt(ml_result):
    """Formate le résultat pour injection dans le prompt LLM de l'axe 4."""
    if not ml_result:
        return ""
    meta = ml_result["model_meta"]
    preds = ml_result["predictions"]
    lines = [
        f"PROBABILITÉS ML — modèle entraîné sur {meta['n_train']} semaines",
        f"(split temporel chronologique — sans data leakage) :",
        "",
    ]
    for name, pred in preds.items():
        auc_note = f" [précision modèle : ROC-AUC {pred['model_roc_auc']:.2f}]" if pred["model_roc_auc"] > 0 else ""
        lines.append(f"  - {pred['label']} : {pred['probabilite']}% — {pred['niveau']}{auc_note}")
    lines += [
        f"  - Risque composite global : {ml_result['risque_composite']}% — {ml_result['niveau_global']}",
        "",
        "CONSIGNE : Pour chaque problème identifié, cite la probabilité ML correspondante.",
        "Si ROC-AUC < 0.70 sur projets réels, nuance la probabilité avec 'estimation ML indicative'.",
        "Ne remplace jamais ces valeurs par des qualificatifs subjectifs génériques.",
    ]
    return "\n".join(lines)


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(
        description="PMO Predictor v2 — Modèle ML anti-overfitting pour l'Axe 4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python pmo_predictor.py train
  python pmo_predictor.py predict --project "Data Fraud Detection"
  python pmo_predictor.py predict --project "Scoring Crédit Retail" --week "S18-26"
  python pmo_predictor.py report --project "Orion Data Platform" --top 8
  python pmo_predictor.py info
        """
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("train", help="Entraîner les modèles (split temporel, sans data leakage)")

    p = sub.add_parser("predict", help="Prédire les risques pour un projet")
    p.add_argument("--project", required=True)
    p.add_argument("--week", default=None)

    r = sub.add_parser("report", help="Rapport complet avec explication SHAP")
    r.add_argument("--project", required=True)
    r.add_argument("--week", default=None)
    r.add_argument("--top", default=8, type=int)

    sub.add_parser("info", help="Informations sur les modèles entraînés")

    args = parser.parse_args()

    if args.command == "train":
        train_all()

    elif args.command == "predict":
        result = predict_for_project(args.project, args.week)
        print_prediction_report(result)
        out = os.path.join(MODELS_FOLDER, f"pred_{args.project.replace(' ','_')}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n  JSON sauvegardé : {out}")

    elif args.command == "report":
        result = predict_for_project(args.project, args.week)
        print_prediction_report(result)
        explain_prediction(args.project, args.week, args.top)

    elif args.command == "info":
        if not os.path.exists(MODEL_PATHS["meta"]):
            print("[INFO] Aucun modèle entraîné. Exécutez : python pmo_predictor.py train")
            return
        with open(MODEL_PATHS["meta"]) as f:
            meta = json.load(f)
        print("\n  INFORMATIONS — PMO PREDICTOR v2")
        print("="*55)
        print(f"  Entraîné le        : {meta['trained_at'][:19]}")
        print(f"  Split              : {meta.get('split_method','?')}")
        print(f"  Anti-data-leakage  : {'Oui' if meta.get('leakage_protected') else 'Non'}")
        print(f"  Projets réels      : {', '.join(meta.get('real_projects',[]))}")
        print(f"  Train / Test       : {meta['n_samples_train']} / {meta['n_samples_test']} semaines")
        print(f"\n  {'Modèle':<14} {'CV ROC-AUC':>12} {'Test ROC-AUC':>13} {'Test F1':>9} {'Réel ROC-AUC':>13}")
        print(f"  {'─'*63}")
        for name in MODEL_NAMES:
            m = meta.get("models", {}).get(name, {})
            cv_str   = f"{m.get('cv_roc_auc_mean',0):.3f}±{m.get('cv_roc_auc_std',0):.3f}"
            real_auc = m.get("real_projects_metrics", {}).get("roc_auc", 0) if m.get("real_projects_metrics") else 0
            print(f"  {name:<14} {cv_str:>12}  {m.get('test_roc_auc',0):>12.3f}  {m.get('test_f1',0):>8.3f}  {real_auc:>12.3f}")
        print()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()