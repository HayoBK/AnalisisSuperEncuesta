
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    precision_recall_fscore_support,
)
from catboost import CatBoostClassifier

# ============================================================
# CONFIG
# ============================================================

INPUT_XLSX = "3-Revision_manual_F_con_Edad.xlsx"
OUTPUT_XLSX = "scores_y_roc_hospital_digital.xlsx"
OUTPUT_FIG_DIR = "figuras_roc_scores"

OUTCOMES = [
    "IA01_MV",
    "IA02_VPPB",
    "IA03_EM",
    "IA04_NV",
    "IA05_VB",
    "IA06_MPPP",
    "IA07_ACVCentral",
]

MIN_POSITIVES_TO_MODEL = 20
N_SPLITS = 3
RANDOM_STATE = 42

# ============================================================
# QUESTION MAP + CODING
# ============================================================

QUESTION_MAP = {
    "Q01": "Patrón temporal general",
    "Q02": "Tiempo de evolución",
    "Q03": "Sensación de giro",
    "Q04": "Sensación de inestabilidad/barco",
    "Q05": "Náuseas",
    "Q06": "Cabeza flotando",
    "Q07": "Dolor de cabeza durante el mareo",
    "Q08": "Presión en la cabeza",
    "Q09": "Sensibilidad a luz/cambios sensoriales",
    "Q10": "Sensibilidad a sonidos",
    "Q11": "Sonidos empeoran el mareo",
    "Q12": "Pérdida de conciencia / desmayo",
    "Q13": "Temor persistente a que aparezca el mareo",
    "Q14": "Antecedentes familiares de cefalea",
    "Q15": "Miedo a caer",
    "Q16": "Ansiedad/angustia frente al mareo",
    "Q17": "Cefalea prolongada con náuseas/fotofobia",
    "Q18": "Hipervigilancia del equilibrio",
    "Q19": "Problemas al caminar en oscuridad",
    "Q20": "Audición peor reciente",
    "Q21": "Cambios auditivos durante la crisis",
    "Q22": "Tinnitus durante el mareo",
    "Q23": "Oído tapado durante el mareo",
    "Q24": "Otalgia durante el mareo",
    "Q25": "Duración típica de la crisis",
    "Q26": "Frecuencia habitual",
    "Q27": "Normalidad entre crisis",
    "Q28": "Mareo sentado o quieto de pie",
    "Q29": "Mareo al girar en la cama",
    "Q30": "Mareo al levantarse de la cama",
    "Q31": "Mareo al acostarse",
    "Q32": "Mareo al girar/mover rápido la cabeza de pie",
    "Q33": "Mareo en espacios con mucha gente",
    "Q34": "Opresión/adormecimiento con mareo",
    "Q35": "Mareo al estar en movimiento",
    "Q36": "Mareo muy corto al levantarse rápido de silla",
    "Q37": "Amanece bien y luego aparece el mareo",
    "Q38": "Cambio del mareo en ciertas posiciones",
    "Q39": "Cambio del mareo en período menstrual",
    "Q40": "Cambio del mareo al quedarse quieto",
    "Q41": "Cambio del mareo al mover rápido la cabeza",
}

YES_SET = {"1", "1.0"}
NO_SET = {"0", "0.0"}
ALT_SET = {"2", "2.0"}  # for Q38-Q41 => more mareo

# ============================================================
# BASIC HELPERS
# ============================================================

def build_q_prefix_map(columns):
    qmap = {}
    for i in range(1, 42):
        pref = f"Q{i:02d}"
        match = [c for c in columns if str(c).startswith(pref)]
        if match:
            qmap[pref] = match[0]
    return qmap

def qcol(qmap, pref):
    return qmap.get(pref)

def clean_question_column(series):
    s = series.copy().replace("-", np.nan).astype("object")
    vc = s.value_counts(dropna=True)
    rare = vc[vc < 3].index
    if len(rare) > 0:
        s.loc[s.isin(rare)] = np.nan
    mask = s.notna()
    s.loc[mask] = s.loc[mask].astype(str).str.strip()
    return s

def safe_binary(series):
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    return (s != 0).astype(int)

def is_yes(x):
    if pd.isna(x): return 0
    return int(str(x).strip() in YES_SET)

def is_no(x):
    if pd.isna(x): return 0
    return int(str(x).strip() in NO_SET)

def is_alt(x):
    if pd.isna(x): return 0
    return int(str(x).strip() in ALT_SET)

def cat_to_num(x):
    if pd.isna(x): return np.nan
    try:
        return float(str(x).strip())
    except:
        return np.nan

def base_feature_name(feature_name, raw_cat_prefixes):
    if feature_name.startswith("num__"):
        return feature_name.replace("num__", "")
    if feature_name.startswith("cat__"):
        rest = feature_name.replace("cat__", "")
        for q in raw_cat_prefixes:
            if rest.startswith(q):
                return q
    return feature_name

# ============================================================
# ENGINEERED FEATURES
# ============================================================

def build_engineered_features(df, qmap):
    out = pd.DataFrame(index=df.index)

    for i in range(1, 42):
        q = f"Q{i:02d}"
        col = qcol(qmap, q)
        out[f"{q}_num"] = df[col].apply(cat_to_num) if col else np.nan

    out["Edad"] = pd.to_numeric(df["Edad"], errors="coerce")
    out["Edad_ge_60"] = (out["Edad"] >= 60).astype(float)
    out["Edad_ge_70"] = (out["Edad"] >= 70).astype(float)
    out["Edad_lt_50"] = (out["Edad"] < 50).astype(float)

    # temporal
    out["temporal_constante"] = out["Q01_num"].isin([0]).astype(float)
    out["temporal_crisis"] = out["Q01_num"].isin([1]).astype(float)
    out["temporal_residual_post_crisis"] = out["Q01_num"].isin([2]).astype(float)

    out["curso_lt_3m"] = out["Q02_num"].isin([0]).astype(float)
    out["curso_3_12m"] = out["Q02_num"].isin([1]).astype(float)
    out["curso_gt_1a"] = out["Q02_num"].isin([2]).astype(float)

    out["duracion_menos_3min"] = out["Q25_num"].isin([1]).astype(float)
    out["duracion_3_15min"] = out["Q25_num"].isin([2]).astype(float)
    out["duracion_15_60min"] = out["Q25_num"].isin([3]).astype(float)
    out["duracion_1_12h"] = out["Q25_num"].isin([4]).astype(float)
    out["duracion_12h_1sem"] = out["Q25_num"].isin([5]).astype(float)
    out["duracion_semanas_meses"] = out["Q25_num"].isin([6]).astype(float)
    out["duracion_variable"] = out["Q25_num"].isin([7]).astype(float)

    out["duracion_muy_corta"] = out["Q25_num"].isin([1, 2]).astype(float)
    out["duracion_intermedia"] = out["Q25_num"].isin([3, 4]).astype(float)
    out["duracion_larga"] = out["Q25_num"].isin([5, 6]).astype(float)

    out["frecuencia_unica"] = out["Q26_num"].isin([1]).astype(float)
    out["frecuencia_constante"] = out["Q26_num"].isin([2]).astype(float)
    out["frecuencia_menos_1_mes"] = out["Q26_num"].isin([3]).astype(float)
    out["frecuencia_mensual"] = out["Q26_num"].isin([4]).astype(float)
    out["frecuencia_semanal"] = out["Q26_num"].isin([5]).astype(float)
    out["frecuencia_diaria"] = out["Q26_num"].isin([6]).astype(float)
    out["frecuencia_variable"] = out["Q26_num"].isin([7]).astype(float)

    out["frecuencia_baja"] = out["Q26_num"].isin([1, 3, 4]).astype(float)
    out["frecuencia_alta"] = out["Q26_num"].isin([5, 6]).astype(float)

    q27 = qcol(qmap, "Q27")
    out["normal_entre_crisis"] = df[q27].apply(is_yes).astype(float) if q27 else 0.0

    q38 = qcol(qmap, "Q38")
    q39 = qcol(qmap, "Q39")
    q40 = qcol(qmap, "Q40")
    q41 = qcol(qmap, "Q41")

    # exact coding: less=1, more=2, no change=0
    out["Q38_menos"] = df[q38].apply(is_yes).astype(float) if q38 else 0.0
    out["Q38_mas"] = df[q38].apply(is_alt).astype(float) if q38 else 0.0
    out["Q40_menos"] = df[q40].apply(is_yes).astype(float) if q40 else 0.0
    out["Q40_mas"] = df[q40].apply(is_alt).astype(float) if q40 else 0.0
    out["Q41_menos"] = df[q41].apply(is_yes).astype(float) if q41 else 0.0
    out["Q41_mas"] = df[q41].apply(is_alt).astype(float) if q41 else 0.0
    out["Q39_mas"] = df[q39].apply(is_alt).astype(float) if q39 else 0.0

    # binary yes/no
    for q in ["Q03","Q04","Q05","Q06","Q07","Q08","Q09","Q10","Q11","Q12","Q13","Q14","Q15","Q16","Q17","Q18","Q19","Q28","Q29","Q30","Q31","Q32","Q33","Q34","Q35","Q36"]:
        col = qcol(qmap, q)
        out[q + "_yes"] = df[col].apply(is_yes).astype(float) if col else 0.0
        out[q + "_no"] = df[col].apply(is_no).astype(float) if col else 0.0

    # blocks
    migraine_items = ["Q07_yes","Q08_yes","Q09_yes","Q10_yes","Q14_yes","Q17_yes"]
    out["bloque_migrañoso_n"] = out[migraine_items].sum(axis=1)
    out["bloque_migrañoso_2mas"] = (out["bloque_migrañoso_n"] >= 2).astype(float)
    out["bloque_migrañoso_3mas"] = (out["bloque_migrañoso_n"] >= 3).astype(float)

    pos_items = ["Q29_yes","Q30_yes","Q31_yes","Q32_yes","Q38_mas","Q41_mas"]
    out["bloque_posicional_n"] = out[pos_items].sum(axis=1)
    out["bloque_posicional_2mas"] = (out["bloque_posicional_n"] >= 2).astype(float)

    func_items = ["Q13_yes","Q15_yes","Q16_yes","Q18_yes","Q33_yes","Q35_yes"]
    out["bloque_funcional_n"] = out[func_items].sum(axis=1)
    out["bloque_funcional_2mas"] = (out["bloque_funcional_n"] >= 2).astype(float)
    out["bloque_funcional_3mas"] = (out["bloque_funcional_n"] >= 3).astype(float)

    aud_items_num = []
    for qn in ["Q20_num","Q21_num","Q22_num","Q23_num","Q24_num"]:
        if qn in out.columns:
            aud_items_num.append(qn)

    out["audicion_peor_reciente"] = out["Q20_num"].isin([1, 2]).astype(float)
    out["cambio_auditivo_crisis"] = out["Q21_num"].isin([1, 2]).astype(float)
    out["tinnitus_crisis"] = out["Q22_num"].isin([1, 2]).astype(float)
    out["oido_tapado_crisis"] = out["Q23_num"].isin([1, 2]).astype(float)
    out["otalgia_crisis"] = out["Q24_num"].isin([1, 2]).astype(float)
    out["audicion_peor_ambos"] = out["Q20_num"].isin([2]).astype(float)
    aud_items = ["audicion_peor_reciente","cambio_auditivo_crisis","tinnitus_crisis","oido_tapado_crisis","otalgia_crisis"]
    out["bloque_auditivo_n"] = out[aud_items].sum(axis=1)
    out["bloque_auditivo_2mas"] = (out["bloque_auditivo_n"] >= 2).astype(float)

    # simplified clinical features
    out["giro"] = out["Q03_yes"]
    out["barco"] = out["Q04_yes"]
    out["nauseas"] = out["Q05_yes"]
    out["cabeza_flotando"] = out["Q06_yes"]
    out["oscuridad"] = out["Q19_yes"]
    out["mov_cabeza_rapido"] = out["Q32_yes"]
    out["quieto_de_pie"] = out["Q28_yes"]
    out["en_movimiento"] = out["Q35_yes"]
    out["espacios_llenos"] = out["Q33_yes"]
    out["desmayo"] = out["Q12_yes"]
    out["opresion_pecho_manos"] = out["Q34_yes"]
    out["levantarse_rapido_silla"] = out["Q36_yes"]
    out["no_giro"] = out["Q03_no"]

    return out

# ============================================================
# SCORE CLÍNICO
# ============================================================

def build_scores(df, eng, qmap):
    scores = pd.DataFrame(index=df.index)

    edad = pd.to_numeric(df["Edad"], errors="coerce")

    # MV
    scores["SCORE_MV"] = (
        2 * eng["Q07_yes"] +
        2 * eng["Q17_yes"] +
        1 * eng["Q09_yes"] +
        1 * eng["Q08_yes"] +
        1 * eng["Q14_yes"] +
        1 * eng["duracion_larga"] +
        1 * (edad < 50).fillna(False).astype(int)
    )

    # VPPB
    scores["SCORE_VPPB"] = (
        2 * eng["Q29_yes"] +
        1 * eng["Q30_yes"] +
        1 * eng["Q31_yes"] +
        1 * eng["Q03_yes"] +
        2 * eng["duracion_muy_corta"] +
        1 * eng["normal_entre_crisis"] +
        1 * eng["Q41_mas"]
    )

    # EM
    scores["SCORE_EM"] = (
        1 * eng["audicion_peor_reciente"] +
        2 * eng["cambio_auditivo_crisis"] +
        1 * eng["tinnitus_crisis"] +
        2 * eng["oido_tapado_crisis"] +
        1 * eng["duracion_intermedia"] +
        1 * eng["duracion_larga"] +
        1 * eng["Q03_yes"]
    )

    # NV
    scores["SCORE_NV"] = (
        2 * eng["temporal_residual_post_crisis"] +
        1 * eng["curso_lt_3m"] +
        2 * eng["duracion_larga"] +
        1 * eng["Q04_yes"] +
        1 * eng["Q05_yes"] +
        1 * eng["Q41_mas"]
    )

    # VB
    scores["SCORE_VB"] = (
        1 * (edad >= 60).fillna(False).astype(int) +
        1 * (edad >= 70).fillna(False).astype(int) +
        2 * eng["Q19_yes"] +
        1 * eng["Q03_no"] +
        1 * eng["curso_gt_1a"] +
        2 * eng["Q41_mas"] +
        1 * eng["Q35_yes"]
    )

    # MPPP
    scores["SCORE_MPPP"] = (
        1 * eng["temporal_constante"] +
        1 * eng["curso_gt_1a"] +
        2 * eng["frecuencia_constante"] +
        2 * eng["Q33_yes"] +
        2 * eng["Q18_yes"] +
        1 * eng["Q13_yes"] +
        1 * eng["Q15_yes"] +
        1 * eng["Q35_yes"] +
        1 * eng["Q40_menos"]
    )

    # Central caution
    scores["FLAG_CENTRAL"] = (
        1 * eng["Q12_yes"] +
        1 * eng["duracion_semanas_meses"] +
        1 * eng["duracion_variable"] +
        1 * eng["Q28_yes"] +
        1 * eng["Q36_yes"] +
        1 * eng["Q40_mas"]
    )

    return scores

# ============================================================
# MODEL BUILDERS
# ============================================================

def build_logit_pipeline(num_cols, cat_cols):
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols))
    return Pipeline([
        ("prep", ColumnTransformer(transformers=transformers)),
        ("clf", LogisticRegression(
            penalty="l1", solver="saga", C=0.6, max_iter=4000,
            class_weight="balanced", random_state=RANDOM_STATE
        ))
    ])

def build_reduced_logit_pipeline(num_cols, cat_cols):
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols))
    return Pipeline([
        ("prep", ColumnTransformer(transformers=transformers)),
        ("clf", LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=4000,
            class_weight="balanced", random_state=RANDOM_STATE
        ))
    ])

def build_catboost():
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        depth=4,
        learning_rate=0.05,
        iterations=150,
        l2_leaf_reg=5,
        random_seed=RANDOM_STATE,
        verbose=False,
        auto_class_weights="Balanced",
    )

# ============================================================
# PLANS FOR REDUCED / FULL MODEL
# ============================================================

def diagnosis_feature_plan():
    return {
        "IA01_MV": {
            "engineered_num": ["Edad","Edad_lt_50","bloque_migrañoso_n","bloque_migrañoso_2mas","bloque_migrañoso_3mas","giro","nauseas","cabeza_flotando","duracion_intermedia","duracion_larga","frecuencia_baja","frecuencia_alta","Q39_mas"],
            "raw_cat": ["Q01","Q02","Q07","Q08","Q09","Q10","Q14","Q17","Q25","Q26","Q39"],
            "reduced_top6": ["Edad","bloque_migrañoso_n","Q07","cabeza_flotando","bloque_migrañoso_3mas","Q25"],
            "score_col": "SCORE_MV",
        },
        "IA02_VPPB": {
            "engineered_num": ["Edad","Edad_ge_60","duracion_muy_corta","frecuencia_baja","normal_entre_crisis","giro","bloque_posicional_n","bloque_posicional_2mas","Q38_mas","Q41_mas"],
            "raw_cat": ["Q03","Q25","Q26","Q27","Q29","Q30","Q31","Q32","Q38","Q41"],
            "reduced_top6": ["Edad","Q29","duracion_muy_corta","Q25","Q30","giro"],
            "score_col": "SCORE_VPPB",
        },
        "IA03_EM": {
            "engineered_num": ["Edad","bloque_auditivo_n","bloque_auditivo_2mas","audicion_peor_ambos","giro","nauseas","duracion_intermedia","duracion_larga"],
            "raw_cat": ["Q20","Q21","Q22","Q23","Q24","Q25","Q26"],
            "reduced_top6": ["Edad","Q20","Q22","bloque_auditivo_n","Q23","bloque_auditivo_2mas"],
            "score_col": "SCORE_EM",
        },
        "IA04_NV": {
            "engineered_num": ["Edad","curso_lt_3m","temporal_residual_post_crisis","duracion_larga","giro","barco","nauseas","mov_cabeza_rapido","Q41_mas"],
            "raw_cat": ["Q01","Q02","Q03","Q04","Q05","Q25","Q32","Q41"],
            "reduced_top6": ["Edad","Q02","Q25","Q41","Q04","temporal_residual_post_crisis"],
            "score_col": "SCORE_NV",
        },
        "IA05_VB": {
            "engineered_num": ["Edad","Edad_ge_60","Edad_ge_70","curso_gt_1a","oscuridad","mov_cabeza_rapido","Q41_mas","no_giro","en_movimiento","bloque_posicional_n"],
            "raw_cat": ["Q02","Q03","Q19","Q25","Q32","Q35","Q41"],
            "reduced_top6": ["Edad","Q41","Edad_ge_70","bloque_posicional_n","no_giro","Q41_mas"],
            "score_col": "SCORE_VB",
        },
        "IA06_MPPP": {
            "engineered_num": ["Edad","curso_gt_1a","frecuencia_constante","frecuencia_alta","bloque_funcional_n","bloque_funcional_2mas","bloque_funcional_3mas","espacios_llenos","en_movimiento","Q40_menos","quieto_de_pie","barco","no_giro"],
            "raw_cat": ["Q01","Q02","Q13","Q15","Q16","Q18","Q26","Q28","Q33","Q35","Q40"],
            "reduced_top6": ["Edad","Q26","Q33","Q01","Q18","Q02"],
            "score_col": "SCORE_MPPP",
        },
        "IA07_ACVCentral": {
            "engineered_num": ["Edad","Edad_ge_60","Edad_ge_70","quieto_de_pie","Q40_mas","desmayo","opresion_pecho_manos","duracion_variable"],
            "raw_cat": ["Q12","Q25","Q28","Q34","Q36","Q40"],
            "reduced_top6": ["Edad","Q40","Q25","quieto_de_pie","Edad_ge_60","Q36"],
            "score_col": "FLAG_CENTRAL",
        },
    }

# ============================================================
# EVAL
# ============================================================

def compute_basic_metrics(y_true, probs, thr=0.5):
    pred = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    _, _, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    return {
        "AUC": roc_auc_score(y_true, probs),
        "AUPRC": average_precision_score(y_true, probs),
        "Sensibilidad@0.5": sens,
        "Especificidad@0.5": spec,
        "VPP@0.5": ppv,
        "VPN@0.5": npv,
        "F1@0.5": f1,
    }

def normalize_score(series):
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    mx = s.max()
    if mx <= 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return s / mx

# ============================================================
# MAIN
# ============================================================

def main():
    input_path = Path(INPUT_XLSX)
    output_path = Path(OUTPUT_XLSX)
    fig_dir = Path(OUTPUT_FIG_DIR)
    fig_dir.mkdir(exist_ok=True, parents=True)

    if not input_path.exists():
        raise FileNotFoundError(f"No encontré el archivo: {input_path.resolve()}")

    print(f"Leyendo: {input_path.resolve()}")
    df = pd.read_excel(input_path)

    all_cols = df.columns.tolist()
    qmap = build_q_prefix_map(all_cols)
    ia_cols = [c for c in all_cols if c.startswith("IA")]
    q_real_cols = list(qmap.values())

    for c in ia_cols:
        df[c] = safe_binary(df[c])

    for c in q_real_cols:
        df[c] = clean_question_column(df[c])

    df["Edad"] = pd.to_numeric(df["Edad"], errors="coerce")

    cohort = df[df[ia_cols].sum(axis=1) > 0].copy()
    print(f"Cohorte IA positiva: {len(cohort)} pacientes")

    eng = build_engineered_features(cohort, qmap)
    scores = build_scores(cohort, eng, qmap)
    plan_map = diagnosis_feature_plan()

    outcomes = []
    for o in OUTCOMES:
        if o in cohort.columns:
            n_pos = int(cohort[o].sum())
            if n_pos >= MIN_POSITIVES_TO_MODEL:
                outcomes.append(o)
            else:
                print(f"Saltando {o}: solo {n_pos} positivos")
    print("Outcomes analizados:", outcomes)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    results = []
    pred_df = pd.DataFrame({"RUT": cohort["RUT"].values})

    for outcome in outcomes:
        print(f"\nProcesando {outcome} ...")
        y = cohort[outcome].astype(int).values
        plan = plan_map[outcome]

        # -------- Full model: compare logit vs catboost, use better by AUPRC
        num_cols = [c for c in plan["engineered_num"] if c in eng.columns]
        raw_cat_prefixes = [c for c in plan["raw_cat"] if c in qmap]
        raw_cat_real_cols = [qmap[c] for c in raw_cat_prefixes]

        X_num = eng[num_cols].copy() if num_cols else pd.DataFrame(index=cohort.index)
        X_cat = cohort[raw_cat_real_cols].copy() if raw_cat_real_cols else pd.DataFrame(index=cohort.index)

        # Logistic full
        X_log = pd.concat([X_num, X_cat], axis=1)
        logit = build_logit_pipeline(num_cols, raw_cat_real_cols)
        p_log = cross_val_predict(logit, X_log, y, cv=cv, method="predict_proba")[:, 1]
        auc_log = roc_auc_score(y, p_log)
        auprc_log = average_precision_score(y, p_log)

        # CatBoost full
        X_cb = pd.concat([X_num, X_cat], axis=1).copy()
        for c in raw_cat_real_cols:
            X_cb[c] = X_cb[c].where(X_cb[c].notna(), "__MISSING__").astype(str)
        cb_cat_idx = [X_cb.columns.get_loc(c) for c in raw_cat_real_cols]

        p_cat = np.zeros(len(y))
        for tr, te in cv.split(X_cb, y):
            model = build_catboost()
            model.fit(X_cb.iloc[tr], y[tr], cat_features=cb_cat_idx, verbose=False)
            p_cat[te] = model.predict_proba(X_cb.iloc[te])[:, 1]
        auc_cat = roc_auc_score(y, p_cat)
        auprc_cat = average_precision_score(y, p_cat)

        if (auprc_cat > auprc_log) or (np.isclose(auprc_cat, auprc_log) and auc_cat > auc_log):
            p_full = p_cat
            full_model_name = "Full_CatBoost"
        else:
            p_full = p_log
            full_model_name = "Full_LogitL1"

        # -------- Reduced model
        reduced_feats = plan["reduced_top6"]
        red_num_cols = [c for c in reduced_feats if c in eng.columns]
        red_pref_cols = [c for c in reduced_feats if isinstance(c, str) and c.startswith("Q")]
        red_real_cols = [qmap[c] for c in red_pref_cols if c in qmap]

        Xr_num = eng[red_num_cols].copy() if red_num_cols else pd.DataFrame(index=cohort.index)
        Xr_cat = cohort[red_real_cols].copy() if red_real_cols else pd.DataFrame(index=cohort.index)
        Xr = pd.concat([Xr_num, Xr_cat], axis=1)

        red_model = build_reduced_logit_pipeline(red_num_cols, red_real_cols)
        p_red = cross_val_predict(red_model, Xr, y, cv=cv, method="predict_proba")[:, 1]

        # -------- Clinical score
        score_col = plan["score_col"]
        p_score = normalize_score(scores[score_col]).values

        # store predictions
        pred_df[f"{outcome}_FullProb"] = p_full
        pred_df[f"{outcome}_ReducedProb"] = p_red
        pred_df[f"{outcome}_ScoreNorm"] = p_score
        pred_df[f"{outcome}_ScoreRaw"] = scores[score_col].values

        # metrics
        for model_name, probs in [
            (full_model_name, p_full),
            ("Reduced_Logit_Top6", p_red),
            ("Clinical_Score", p_score),
        ]:
            row = {"Diagnostico": outcome, "Modelo": model_name, "n_positivos": int(y.sum())}
            row.update(compute_basic_metrics(y, probs))
            results.append(row)

        # ROC figure
        plt.figure(figsize=(6, 6))
        for label, probs in [
            (full_model_name, p_full),
            ("Reduced_Logit_Top6", p_red),
            ("Clinical_Score", p_score),
        ]:
            fpr, tpr, _ = roc_curve(y, probs)
            auc_val = roc_auc_score(y, probs)
            plt.plot(fpr, tpr, label=f"{label} (AUC={auc_val:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("1 - Especificidad")
        plt.ylabel("Sensibilidad")
        plt.title(f"ROC - {outcome}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(fig_dir / f"ROC_{outcome}.png", dpi=200)
        plt.close()

        # PR figure
        plt.figure(figsize=(6, 6))
        for label, probs in [
            (full_model_name, p_full),
            ("Reduced_Logit_Top6", p_red),
            ("Clinical_Score", p_score),
        ]:
            precision, recall, _ = precision_recall_curve(y, probs)
            auprc_val = average_precision_score(y, probs)
            plt.plot(recall, precision, label=f"{label} (AUPRC={auprc_val:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall - {outcome}")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(fig_dir / f"PR_{outcome}.png", dpi=200)
        plt.close()

    results_df = pd.DataFrame(results).sort_values(["Diagnostico", "Modelo"])

    score_manual = pd.DataFrame([
        {"Diagnostico": "IA01_MV", "Score": "SCORE_MV", "Interpretacion": "0-2 baja; 3-5 intermedia; >=6 alta"},
        {"Diagnostico": "IA02_VPPB", "Score": "SCORE_VPPB", "Interpretacion": "0-2 baja; 3-5 intermedia; >=6 alta"},
        {"Diagnostico": "IA03_EM", "Score": "SCORE_EM", "Interpretacion": "0-2 baja; 3-5 intermedia; >=6 alta"},
        {"Diagnostico": "IA04_NV", "Score": "SCORE_NV", "Interpretacion": "0-2 baja; 3-5 intermedia; >=6 alta"},
        {"Diagnostico": "IA05_VB", "Score": "SCORE_VB", "Interpretacion": "0-2 baja; 3-5 intermedia; >=6 alta"},
        {"Diagnostico": "IA06_MPPP", "Score": "SCORE_MPPP", "Interpretacion": "0-3 baja; 4-6 intermedia; >=7 alta"},
        {"Diagnostico": "IA07_ACVCentral", "Score": "FLAG_CENTRAL", "Interpretacion": "0-1 sin bandera; 2+ cautela; 3+ priorizar evaluación"},
    ])

    methodology = pd.DataFrame({
        "Paso": [
            "Base analítica",
            "Patrón de referencia",
            "Predictores",
            "Score clínico",
            "Modelo completo",
            "Modelo reducido",
            "Comparación",
            "Curvas",
        ],
        "Descripcion": [
            "Pacientes con al menos un IA positivo",
            "Diagnósticos IA",
            "Edad + preguntas recodificadas + variables derivadas",
            "Score clínico por diagnóstico basado en reglas interpretables",
            "Mejor entre Logística L1 y CatBoost según AUPRC y AUC",
            "Logística con top-6 predictores por diagnóstico",
            "Comparación entre modelo completo, modelo reducido y score",
            "Generación de curvas ROC y Precision-Recall por diagnóstico",
        ],
    })

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        methodology.to_excel(writer, sheet_name="Metodologia", index=False)
        results_df.to_excel(writer, sheet_name="Comparacion_modelos_y_score", index=False)
        score_manual.to_excel(writer, sheet_name="Manual_score", index=False)
        scores.to_excel(writer, sheet_name="Scores_raw", index=False)
        pred_df.to_excel(writer, sheet_name="Predicciones", index=False)

    print(f"\nArchivo Excel guardado en: {output_path.resolve()}")
    print(f"Figuras guardadas en: {fig_dir.resolve()}")
    print("\nResumen:")
    disp = results_df.copy()
    for c in ["AUC", "AUPRC", "F1@0.5"]:
        disp[c] = disp[c].round(3)
    print(disp.to_string(index=False))

if __name__ == "__main__":
    main()
