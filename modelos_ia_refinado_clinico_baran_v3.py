
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    confusion_matrix,
)
from catboost import CatBoostClassifier

# ============================================================
# CONFIG
# ============================================================

INPUT_XLSX = "3-Revision_manual_F_con_Edad.xlsx"
OUTPUT_XLSX = "Modelos_IA_refinado_clinico_baran_y_respuestas.xlsx"

DEFAULT_OUTCOMES = [
    "IA01_MV",
    "IA02_VPPB",
    "IA03_EM",
    "IA04_NV",
    "IA05_VB",
    "IA06_MPPP",
    "IA07_ACVCentral",
    "IA08_Orto",
]

MIN_POSITIVES_TO_MODEL = 20
N_SPLITS = 3
RANDOM_STATE = 42

CATBOOST_ITERATIONS = 150
CATBOOST_DEPTH = 4
CATBOOST_LR = 0.05
CATBOOST_L2 = 5

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

ENGINEERED_FEATURE_DESCRIPTIONS = {
    "SVA_like": "Patrón tipo síndrome vestibular agudo",
    "SVE_like": "Patrón tipo síndrome vestibular episódico",
    "SVC_like": "Patrón tipo síndrome vestibular crónico",
    "bloque_migrañoso_n": "Conteo de rasgos migrañosos",
    "bloque_migrañoso_2mas": "Al menos 2 rasgos migrañosos",
    "bloque_migrañoso_3mas": "Al menos 3 rasgos migrañosos",
    "bloque_funcional_n": "Conteo de rasgos funcionales/PPPD",
    "bloque_funcional_2mas": "Al menos 2 rasgos funcionales/PPPD",
    "bloque_funcional_3mas": "Al menos 3 rasgos funcionales/PPPD",
    "bloque_posicional_n": "Conteo de rasgos posicionales",
    "bloque_posicional_2mas": "Al menos 2 rasgos posicionales",
    "bloque_auditivo_n": "Conteo de rasgos auditivos",
    "bloque_auditivo_2mas": "Al menos 2 rasgos auditivos",
}

# ============================================================
# HELPERS
# ============================================================

YES_SET = {"1", "1.0", "si", "sí", "SI", "Sí"}
NO_SET = {"0", "0.0", "no", "NO", "No"}
ALT_SET = {"2", "2.0"}

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

def safe_numeric_binary(series):
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    return (s != 0).astype(int)

def clean_question_column(series):
    s = series.copy().replace("-", np.nan).astype("object")
    vc = s.value_counts(dropna=True)
    rare = vc[vc < 3].index
    if len(rare) > 0:
        s.loc[s.isin(rare)] = np.nan
    mask = s.notna()
    s.loc[mask] = s.loc[mask].astype(str).str.strip()
    return s

def is_yes(x):
    if pd.isna(x):
        return 0
    return int(str(x).strip() in YES_SET)

def is_no(x):
    if pd.isna(x):
        return 0
    return int(str(x).strip() in NO_SET)

def is_alt(x):
    if pd.isna(x):
        return 0
    return int(str(x).strip() in ALT_SET)

def cat_to_num(x):
    if pd.isna(x):
        return np.nan
    try:
        return float(str(x).strip())
    except:
        return np.nan

def get_metrics(y_true, probs, threshold=0.5):
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    return {
        "AUC": roc_auc_score(y_true, probs),
        "AUPRC": average_precision_score(y_true, probs),
        "Sensibilidad": sens,
        "Especificidad": spec,
        "VPP": ppv,
        "VPN": npv,
        "F1": f1,
        "Brier": brier_score_loss(y_true, probs),
    }

def build_logit_pipeline(num_cols, cat_cols):
    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler())
            ]),
            num_cols
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]),
            cat_cols
        ))
    return Pipeline([
        ("prep", ColumnTransformer(transformers=transformers)),
        ("clf", LogisticRegression(
            penalty="l1",
            solver="saga",
            C=0.6,
            max_iter=4000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

def build_reduced_logit_pipeline(num_cols, cat_cols):
    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler())
            ]),
            num_cols
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]),
            cat_cols
        ))
    return Pipeline([
        ("prep", ColumnTransformer(transformers=transformers)),
        ("clf", LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=4000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

def build_catboost():
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        depth=CATBOOST_DEPTH,
        learning_rate=CATBOOST_LR,
        iterations=CATBOOST_ITERATIONS,
        l2_leaf_reg=CATBOOST_L2,
        random_seed=RANDOM_STATE,
        verbose=False,
        auto_class_weights="Balanced",
    )

def base_feature_name(feature_name, raw_cat_prefixes):
    if feature_name.startswith("num__"):
        return feature_name.replace("num__", "")
    if feature_name.startswith("cat__"):
        rest = feature_name.replace("cat__", "")
        for q in raw_cat_prefixes:
            if rest.startswith(q):
                return q
    return feature_name

def make_question_map_df(qmap):
    rows = []
    for q, desc in QUESTION_MAP.items():
        rows.append({
            "Pregunta_prefijo": q,
            "Columna_real": qmap.get(q, ""),
            "Descripcion": desc
        })
    return pd.DataFrame(rows)

# ============================================================
# ENGINEERED FEATURES
# ============================================================

def build_engineered_features(df, qmap):
    out = pd.DataFrame(index=df.index)

    # numeric versions
    for i in range(1, 42):
        q = f"Q{i:02d}"
        col = qcol(qmap, q)
        out[f"{q}_num"] = df[col].apply(cat_to_num) if col else np.nan

    # age
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

    out["duracion_muy_corta"] = out["Q25_num"].isin([0, 1]).astype(float)
    out["duracion_intermedia"] = out["Q25_num"].isin([2, 3]).astype(float)
    out["duracion_larga"] = out["Q25_num"].isin([4, 5]).astype(float)
    out["duracion_variable"] = out["Q25_num"].isin([6]).astype(float)

    out["frecuencia_unica"] = out["Q26_num"].isin([0]).astype(float)
    out["frecuencia_constante"] = out["Q26_num"].isin([1]).astype(float)
    out["frecuencia_baja"] = out["Q26_num"].isin([2, 3]).astype(float)
    out["frecuencia_alta"] = out["Q26_num"].isin([4, 5]).astype(float)
    out["frecuencia_variable"] = out["Q26_num"].isin([6]).astype(float)

    q27 = qcol(qmap, "Q27")
    q37 = qcol(qmap, "Q37")
    out["normal_entre_crisis"] = df[q27].apply(is_yes).astype(float) if q27 else 0.0
    out["amanece_bien_luego_aparece"] = df[q37].apply(is_yes).astype(float) if q37 else 0.0

    out["SVA_like"] = (
        (out["temporal_constante"] == 1)
        | (out["temporal_residual_post_crisis"] == 1)
        | ((out["curso_lt_3m"] == 1) & (out["duracion_larga"] == 1))
    ).astype(float)

    out["SVE_like"] = (
        (out["temporal_crisis"] == 1)
        & (out["normal_entre_crisis"] == 1)
        & (
            (out["duracion_muy_corta"] == 1)
            | (out["duracion_intermedia"] == 1)
            | (out["frecuencia_baja"] == 1)
            | (out["frecuencia_alta"] == 1)
        )
    ).astype(float)

    out["SVC_like"] = (
        ((out["temporal_constante"] == 1) | (out["frecuencia_constante"] == 1))
        & ((out["curso_3_12m"] == 1) | (out["curso_gt_1a"] == 1))
    ).astype(float)

    # simple yes/no helpers
    for q in ["Q03","Q04","Q05","Q06","Q07","Q08","Q09","Q10","Q11","Q12","Q13","Q14","Q15","Q16","Q17","Q18","Q19","Q27","Q28","Q29","Q30","Q31","Q32","Q33","Q34","Q35","Q36","Q37"]:
        col = qcol(qmap, q)
        out[q + "_yes"] = df[col].apply(is_yes).astype(float) if col else 0.0
        out[q + "_no"] = df[col].apply(is_no).astype(float) if col else 0.0

    # migraine block
    migraine_items = ["Q07_yes","Q08_yes","Q09_yes","Q10_yes","Q14_yes","Q17_yes"]
    out["bloque_migrañoso_n"] = out[migraine_items].sum(axis=1)
    out["bloque_migrañoso_2mas"] = (out["bloque_migrañoso_n"] >= 2).astype(float)
    out["bloque_migrañoso_3mas"] = (out["bloque_migrañoso_n"] >= 3).astype(float)

    # functional block
    func_items = ["Q13_yes","Q15_yes","Q16_yes","Q18_yes","Q33_yes","Q35_yes"]
    out["bloque_funcional_n"] = out[func_items].sum(axis=1)
    out["bloque_funcional_2mas"] = (out["bloque_funcional_n"] >= 2).astype(float)
    out["bloque_funcional_3mas"] = (out["bloque_funcional_n"] >= 3).astype(float)

    # positional and change items
    q38 = qcol(qmap, "Q38")
    q39 = qcol(qmap, "Q39")
    q40 = qcol(qmap, "Q40")
    q41 = qcol(qmap, "Q41")

    out["Q38_mas"] = df[q38].apply(is_yes).astype(float) if q38 else 0.0
    out["Q38_menos"] = df[q38].apply(is_alt).astype(float) if q38 else 0.0
    out["Q38_sin_cambio"] = df[q38].apply(is_no).astype(float) if q38 else 0.0

    out["Q39_mas"] = df[q39].apply(is_yes).astype(float) if q39 else 0.0
    out["Q39_menos"] = df[q39].apply(is_alt).astype(float) if q39 else 0.0
    out["Q39_sin_cambio"] = df[q39].apply(is_no).astype(float) if q39 else 0.0

    out["Q40_mas"] = df[q40].apply(is_yes).astype(float) if q40 else 0.0
    out["Q40_menos"] = df[q40].apply(is_alt).astype(float) if q40 else 0.0
    out["Q40_sin_cambio"] = df[q40].apply(is_no).astype(float) if q40 else 0.0

    out["Q41_mas"] = df[q41].apply(is_yes).astype(float) if q41 else 0.0
    out["Q41_menos"] = df[q41].apply(is_alt).astype(float) if q41 else 0.0
    out["Q41_sin_cambio"] = df[q41].apply(is_no).astype(float) if q41 else 0.0

    pos_items = ["Q29_yes","Q30_yes","Q31_yes","Q32_yes","Q38_mas","Q41_mas"]
    out["bloque_posicional_n"] = out[pos_items].sum(axis=1)
    out["bloque_posicional_2mas"] = (out["bloque_posicional_n"] >= 2).astype(float)

    # auditory block using any-ear logic
    out["audicion_peor_reciente"] = out["Q20_num"].isin([0, 1]).astype(float)
    out["cambio_auditivo_crisis"] = out["Q21_num"].isin([0, 1]).astype(float)
    out["tinnitus_crisis"] = out["Q22_num"].isin([0, 1]).astype(float)
    out["oido_tapado_crisis"] = out["Q23_num"].isin([0, 1]).astype(float)
    out["otalgia_crisis"] = out["Q24_num"].isin([0, 1]).astype(float)
    aud_items = ["audicion_peor_reciente","cambio_auditivo_crisis","tinnitus_crisis","oido_tapado_crisis","otalgia_crisis"]
    out["bloque_auditivo_n"] = out[aud_items].sum(axis=1)
    out["bloque_auditivo_2mas"] = (out["bloque_auditivo_n"] >= 2).astype(float)

    # selected interpretable derived features
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
# DIAGNOSIS PLANS
# ============================================================

def diagnosis_feature_plan():
    return {
        "IA01_MV": {
            "engineered_num": ["Edad","Edad_lt_50","SVE_like","bloque_migrañoso_n","bloque_migrañoso_2mas","bloque_migrañoso_3mas","giro","nauseas","cabeza_flotando","duracion_intermedia","duracion_larga","frecuencia_baja","frecuencia_alta"],
            "raw_cat": ["Q01","Q02","Q07","Q08","Q09","Q10","Q14","Q17","Q25","Q26","Q39"],
            "clinical_comment": "Migrañoso + episódico + edad más joven."
        },
        "IA02_VPPB": {
            "engineered_num": ["Edad","Edad_ge_60","SVE_like","duracion_muy_corta","frecuencia_baja","normal_entre_crisis","giro","bloque_posicional_n","bloque_posicional_2mas","Q38_mas","Q41_mas"],
            "raw_cat": ["Q03","Q25","Q26","Q27","Q29","Q30","Q31","Q32","Q38","Q41"],
            "clinical_comment": "Giro + posicional + crisis breves."
        },
        "IA03_EM": {
            "engineered_num": ["Edad","SVE_like","bloque_auditivo_n","bloque_auditivo_2mas","giro","nauseas","duracion_intermedia","duracion_larga"],
            "raw_cat": ["Q20","Q21","Q22","Q23","Q24","Q25","Q26"],
            "clinical_comment": "Auditivo + episódico + duración intermedia/larga."
        },
        "IA04_NV": {
            "engineered_num": ["Edad","SVA_like","curso_lt_3m","temporal_residual_post_crisis","duracion_larga","giro","barco","nauseas","mov_cabeza_rapido","Q41_mas"],
            "raw_cat": ["Q01","Q02","Q03","Q04","Q05","Q25","Q32","Q41"],
            "clinical_comment": "Agudo/subagudo + vértigo/instabilidad."
        },
        "IA05_VB": {
            "engineered_num": ["Edad","Edad_ge_60","Edad_ge_70","SVC_like","curso_gt_1a","oscuridad","mov_cabeza_rapido","Q41_mas","no_giro","en_movimiento","bloque_posicional_n"],
            "raw_cat": ["Q02","Q03","Q19","Q25","Q32","Q35","Q41"],
            "clinical_comment": "Edad alta + cronicidad + oscuridad + oscilopsia/cabeza."
        },
        "IA06_MPPP": {
            "engineered_num": ["Edad","SVC_like","curso_gt_1a","frecuencia_constante","frecuencia_alta","bloque_funcional_n","bloque_funcional_2mas","bloque_funcional_3mas","espacios_llenos","en_movimiento","Q40_menos","quieto_de_pie","barco","no_giro"],
            "raw_cat": ["Q01","Q02","Q13","Q15","Q16","Q18","Q26","Q28","Q33","Q35","Q40"],
            "clinical_comment": "Cronicidad + rasgos funcionales + gatillos visuales/posturales."
        },
        "IA07_ACVCentral": {
            "engineered_num": ["Edad","Edad_ge_60","Edad_ge_70","quieto_de_pie","Q40_mas","Q40_sin_cambio","desmayo","opresion_pecho_manos","duracion_variable","SVA_like","SVC_like"],
            "raw_cat": ["Q12","Q25","Q28","Q34","Q36","Q40"],
            "clinical_comment": "Salida cautelosa para señales no periféricas."
        },
        "IA08_Orto": {
            "engineered_num": ["Edad","desmayo","opresion_pecho_manos","levantarse_rapido_silla","quieto_de_pie","duracion_muy_corta","frecuencia_alta"],
            "raw_cat": ["Q12","Q28","Q34","Q36","Q25","Q26"],
            "clinical_comment": "Ortostático/no vestibular."
        },
    }

# ============================================================
# RESPONSE ORIENTATION
# ============================================================

def make_response_orientation_table(df_raw, outcomes, feature_plan, qmap):
    rows = []
    for outcome in outcomes:
        y = pd.to_numeric(df_raw[outcome], errors="coerce").fillna(0)
        y = (y != 0).astype(int)
        base_prev = y.mean()

        for pref in feature_plan[outcome]["raw_cat"]:
            col = qcol(qmap, pref)
            if not col:
                continue
            s = df_raw[col].copy().replace("-", np.nan).astype("object")
            m = s.notna()
            s.loc[m] = s.loc[m].astype(str).str.strip()
            vc = s.value_counts(dropna=True)
            for resp, n in vc.items():
                if n < 10:
                    continue
                mask = (s == resp)
                pos_rate = y[mask].mean() if mask.sum() else np.nan
                lift = pos_rate / base_prev if base_prev > 0 else np.nan
                rows.append({
                    "Diagnostico": outcome,
                    "Pregunta_prefijo": pref,
                    "Columna_real": col,
                    "Pregunta_desc": QUESTION_MAP.get(pref, ""),
                    "Respuesta": resp,
                    "n_respuesta": int(mask.sum()),
                    "Prevalencia_diag_global": base_prev,
                    "Tasa_diag_en_esa_respuesta": pos_rate,
                    "Lift_vs_prevalencia": lift,
                    "Diferencia_absoluta": pos_rate - base_prev,
                })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["Diagnostico","Lift_vs_prevalencia","n_respuesta"], ascending=[True,False,False])
    return out

# ============================================================
# MAIN
# ============================================================

def main():
    input_path = Path(INPUT_XLSX)
    output_path = Path(OUTPUT_XLSX)

    if not input_path.exists():
        raise FileNotFoundError(f"No encontré el archivo de entrada: {input_path.resolve()}")

    print(f"Leyendo: {input_path.resolve()}")
    df = pd.read_excel(input_path)
    all_cols = df.columns.tolist()
    qmap = build_q_prefix_map(all_cols)
    ia_cols = [c for c in all_cols if c.startswith("IA")]
    q_real_cols = list(qmap.values())

    for c in ia_cols:
        df[c] = safe_numeric_binary(df[c])

    for c in q_real_cols:
        df[c] = clean_question_column(df[c])

    df["Edad"] = pd.to_numeric(df["Edad"], errors="coerce")
    cohort = df[df[ia_cols].sum(axis=1) > 0].copy()
    print(f"Cohorte IA positiva: {len(cohort)} pacientes")

    eng = build_engineered_features(cohort, qmap)

    outcomes = []
    for o in DEFAULT_OUTCOMES:
        if o in cohort.columns:
            n_pos = int(cohort[o].sum())
            if n_pos >= MIN_POSITIVES_TO_MODEL:
                outcomes.append(o)
            else:
                print(f"Saltando {o}: solo {n_pos} positivos")
    print("Outcomes modelados:", outcomes)

    plan_map = diagnosis_feature_plan()
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    results = []
    winners = []
    importance_frames = []
    reduced_rows = []
    algorithm_rows = []
    predictions = pd.DataFrame({"RUT": cohort["RUT"].values})
    reduced_predictions = pd.DataFrame({"RUT": cohort["RUT"].values})

    for outcome in outcomes:
        print(f"\nProcesando {outcome} ...")
        y = cohort[outcome].astype(int).values
        plan = plan_map[outcome]

        num_cols = [c for c in plan["engineered_num"] if c in eng.columns]
        raw_cat_prefixes = [c for c in plan["raw_cat"] if c in qmap]
        raw_cat_real_cols = [qmap[c] for c in raw_cat_prefixes]

        X_num = eng[num_cols].copy() if num_cols else pd.DataFrame(index=cohort.index)
        X_cat = cohort[raw_cat_real_cols].copy() if raw_cat_real_cols else pd.DataFrame(index=cohort.index)

        # logistic
        X_log = pd.concat([X_num, X_cat], axis=1)
        logit = build_logit_pipeline(num_cols, raw_cat_real_cols)
        p_log = cross_val_predict(logit, X_log, y, cv=cv, method="predict_proba")[:, 1]
        d = get_metrics(y, p_log)
        d.update({
            "Diagnostico": outcome,
            "Modelo": "Logistica_L1",
            "n": len(y),
            "n_positivos": int(y.sum()),
            "Comentario_clinico": plan["clinical_comment"],
        })
        results.append(d)
        predictions[f"{outcome}_LogitProb"] = p_log

        logit.fit(X_log, y)
        prep = logit.named_steps["prep"]
        clf = logit.named_steps["clf"]
        feat_names = prep.get_feature_names_out()
        coef_df = pd.DataFrame({"feature": feat_names, "coef": clf.coef_[0]})
        coef_df["score"] = coef_df["coef"].abs()
        coef_df["base_feature"] = coef_df["feature"].apply(lambda z: base_feature_name(z, raw_cat_prefixes))
        imp = coef_df.groupby("base_feature", as_index=False)["score"].max().sort_values("score", ascending=False)
        imp["Diagnostico"] = outcome
        imp["Modelo"] = "Logistica_L1"
        importance_frames.append(imp)

        # catboost
        X_cb = pd.concat([X_num, X_cat], axis=1).copy()
        for c in raw_cat_real_cols:
            X_cb[c] = X_cb[c].where(X_cb[c].notna(), "__MISSING__").astype(str)
        cb_cat_idx = [X_cb.columns.get_loc(c) for c in raw_cat_real_cols]

        p_cat = np.zeros(len(y))
        for tr, te in cv.split(X_cb, y):
            model = build_catboost()
            model.fit(X_cb.iloc[tr], y[tr], cat_features=cb_cat_idx, verbose=False)
            p_cat[te] = model.predict_proba(X_cb.iloc[te])[:, 1]

        d = get_metrics(y, p_cat)
        d.update({
            "Diagnostico": outcome,
            "Modelo": "CatBoost",
            "n": len(y),
            "n_positivos": int(y.sum()),
            "Comentario_clinico": plan["clinical_comment"],
        })
        results.append(d)
        predictions[f"{outcome}_CatBoostProb"] = p_cat

        full = build_catboost()
        full.fit(X_cb, y, cat_features=cb_cat_idx, verbose=False)
        imp = pd.DataFrame({"base_feature": list(X_cb.columns), "score": full.get_feature_importance()})
        # rename real raw q col names back to prefixes where possible
        rename_back = {v: k for k, v in qmap.items()}
        imp["base_feature"] = imp["base_feature"].replace(rename_back)
        imp["Diagnostico"] = outcome
        imp["Modelo"] = "CatBoost"
        importance_frames.append(imp)

    results_df = pd.DataFrame(results)[[
        "Diagnostico","Modelo","n","n_positivos","AUC","AUPRC","Sensibilidad",
        "Especificidad","VPP","VPN","F1","Brier","Comentario_clinico"
    ]].sort_values(["Diagnostico","Modelo"])

    importance_df = pd.concat(importance_frames, ignore_index=True)

    for outcome in outcomes:
        sub = results_df[results_df["Diagnostico"] == outcome].sort_values(["AUPRC","AUC"], ascending=False)
        winners.append({
            "Diagnostico": outcome,
            "Modelo_ganador": sub.iloc[0]["Modelo"],
            "AUC_ganador": sub.iloc[0]["AUC"],
            "AUPRC_ganador": sub.iloc[0]["AUPRC"],
            "F1_ganador": sub.iloc[0]["F1"],
        })
    winners_df = pd.DataFrame(winners)

    for outcome in outcomes:
        y = cohort[outcome].astype(int).values
        winner = winners_df.loc[winners_df["Diagnostico"] == outcome, "Modelo_ganador"].iloc[0]
        top = (
            importance_df[(importance_df["Diagnostico"] == outcome) & (importance_df["Modelo"] == winner)]
            .sort_values("score", ascending=False)["base_feature"]
            .drop_duplicates()
            .tolist()
        )

        chosen = []
        if "Edad" in top:
            chosen.append("Edad")
        for feat in top:
            if feat != "Edad":
                chosen.append(feat)
            if len(chosen) >= 6:
                break
        chosen = chosen[:6]

        algorithm_rows.append({
            "Diagnostico": outcome,
            "Modelo_ganador": winner,
            "Preguntas_clave_top6": " | ".join(chosen),
            "Comentario_clinico": plan_map[outcome]["clinical_comment"],
        })

        num_cols = [c for c in chosen if c in eng.columns]
        raw_pref_cols = [c for c in chosen if isinstance(c, str) and c.startswith("Q")]
        raw_real_cols = [qmap[c] for c in raw_pref_cols if c in qmap]

        Xr_num = eng[num_cols].copy() if num_cols else pd.DataFrame(index=cohort.index)
        Xr_cat = cohort[raw_real_cols].copy() if raw_real_cols else pd.DataFrame(index=cohort.index)
        Xr = pd.concat([Xr_num, Xr_cat], axis=1)

        red_model = build_reduced_logit_pipeline(num_cols, raw_real_cols)
        p = cross_val_predict(red_model, Xr, y, cv=cv, method="predict_proba")[:, 1]
        reduced_predictions[f"{outcome}_ReducedProb"] = p

        d = get_metrics(y, p)
        d.update({
            "Diagnostico": outcome,
            "Modelo_reducido": "Logistica_reducida_top6",
            "Top6": " | ".join(chosen),
        })
        reduced_rows.append(d)

    reduced_df = pd.DataFrame(reduced_rows)[[
        "Diagnostico","Modelo_reducido","AUC","AUPRC","Sensibilidad","Especificidad",
        "VPP","VPN","F1","Brier","Top6"
    ]].sort_values("Diagnostico")

    algorithm_df = pd.DataFrame(algorithm_rows)
    response_orientation_df = make_response_orientation_table(cohort, outcomes, plan_map, qmap)
    feat_desc_df = pd.DataFrame([{"Feature": k, "Descripcion": v} for k, v in ENGINEERED_FEATURE_DESCRIPTIONS.items()])
    methodology_df = pd.DataFrame({
        "Paso": [
            "Cohorte","Patron de oro","Base clinica","Recodificacion","Temporalidad/Bárány",
            "Bloques sintomaticos","Modelos","Seleccion de modelo","Version reducida","Respuestas orientadoras"
        ],
        "Descripcion": [
            "Pacientes con al menos un IA positivo",
            "IA como referencia de trabajo",
            "Preguntas reinterpretadas según encuesta original en papel",
            "Limpieza de faltantes y categorías raras",
            "Variables derivadas usando Q01, Q02, Q25, Q26, Q27, Q37",
            "Bloques migrañoso, funcional, posicional y auditivo",
            "Comparación entre Logística penalizada L1 y CatBoost",
            "Elección por AUPRC y luego AUC",
            "Versión top-6 para aplicación clínica",
            "Tabla de respuesta específica y lift por diagnóstico"
        ]
    })
    question_map_df = make_question_map_df(qmap)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        methodology_df.to_excel(writer, sheet_name="Metodologia", index=False)
        question_map_df.to_excel(writer, sheet_name="Mapa_preguntas", index=False)
        feat_desc_df.to_excel(writer, sheet_name="Features_derivadas", index=False)
        results_df.to_excel(writer, sheet_name="Rendimiento_modelos", index=False)
        winners_df.to_excel(writer, sheet_name="Modelo_ganador", index=False)
        reduced_df.to_excel(writer, sheet_name="Modelos_reducidos_top6", index=False)
        algorithm_df.to_excel(writer, sheet_name="Algoritmo_propuesto", index=False)
        importance_df.sort_values(["Diagnostico","Modelo","score"], ascending=[True,True,False]).to_excel(writer, sheet_name="Importancia_variables", index=False)
        response_orientation_df.to_excel(writer, sheet_name="Respuestas_orientadoras", index=False)
        predictions.to_excel(writer, sheet_name="Predicciones_full", index=False)
        reduced_predictions.to_excel(writer, sheet_name="Predicciones_reducidas", index=False)

    print(f"\nArchivo guardado en: {output_path.resolve()}")

    disp = results_df.copy()
    for c in ["AUC","AUPRC","F1","Sensibilidad","Especificidad"]:
        disp[c] = disp[c].round(3)
    print("\n=== Rendimiento de modelos ===")
    print(disp.to_string(index=False))
    print("\n=== Modelos ganadores ===")
    print(winners_df.to_string(index=False))
    tmp = reduced_df.copy()
    for c in ["AUC","AUPRC","F1","Sensibilidad","Especificidad"]:
        tmp[c] = tmp[c].round(3)
    print("\n=== Modelos reducidos top6 ===")
    print(tmp.to_string(index=False))

if __name__ == "__main__":
    main()
