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

INPUT_XLSX = "3-Revision_manual_F_con_Edad.xlsx"
OUTPUT_XLSX = "New_Modelos_IA_refinado_clinico_baran_y_respuestas.xlsx"

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
    "Q01": "Patrón temporal general: constante / en crisis / crisis reciente con síntomas residuales",
    "Q02": "Tiempo de evolución del problema (<3m / 3-12m / >1a)",
    "Q03": "Sensación de giro",
    "Q04": "Inestabilidad / estar en un barco",
    "Q05": "Náuseas",
    "Q06": "Cabeza flotando",
    "Q07": "Dolor de cabeza durante el mareo",
    "Q08": "Presión en la cabeza",
    "Q09": "Sensibilidad a luz / luces / cambios sonoros del entorno",
    "Q10": "Sensibilidad a sonidos / ruidos",
    "Q11": "Sonidos empeoran el mareo o hacen que el mundo se mueva",
    "Q12": "Pérdida de conciencia / desmayo",
    "Q13": "Temor gran parte del día a que aparezca mareo",
    "Q14": "Antecedentes familiares de cefalea recurrente",
    "Q15": "Miedo a caer",
    "Q16": "Gran ansiedad y angustia frente al mareo",
    "Q17": "Cefaleas prolongadas con náuseas o fotofobia",
    "Q18": "Hipervigilancia del equilibrio / del mareo",
    "Q19": "Dificultad para caminar en la oscuridad",
    "Q20": "Cambio auditivo reciente para peor",
    "Q21": "Cambios auditivos durante una crisis",
    "Q22": "Tinnitus durante el mareo",
    "Q23": "Oídos tapados durante el mareo",
    "Q24": "Dolor de oído durante el mareo",
    "Q25": "Duración típica de la crisis",
    "Q26": "Frecuencia habitual de las crisis",
    "Q27": "Crisis súbitas con normalidad entre crisis",
    "Q28": "Mareo sentado o quieto de pie",
    "Q29": "Mareo al girar en la cama",
    "Q30": "Mareo al levantarse de la cama",
    "Q31": "Mareo al acostarse en la cama",
    "Q32": "Mareo al girar o mover rápidamente la cabeza estando de pie",
    "Q33": "Mareo en espacios llenos de mucha gente",
    "Q34": "Adormecimiento de manos / opresión en pecho cuando se marea",
    "Q35": "Mareo al estar en movimiento",
    "Q36": "Mareo muy corto al levantarse rápido de una silla",
    "Q37": "Amanece bien y luego aparece el mareo",
    "Q38": "Cambio del mareo en ciertas posiciones (más/menos/no cambio)",
    "Q39": "Cambio del mareo en período menstrual",
    "Q40": "Cambio del mareo al quedarse quieto",
    "Q41": "Cambio del mareo al mover rápidamente la cabeza",
}

YES_SET = {"1", "1.0", "SI", "Sí", "si", "SI ", "Sí "}
NO_SET  = {"0", "0.0", "NO", "No", "no", "NO "}
ALT_SET = {"2", "2.0"}

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
    s = str(x).strip()
    try:
        return float(s)
    except:
        return np.nan

def safe_numeric_binary(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    return (s != 0).astype(int)

def clean_question_column(series: pd.Series) -> pd.Series:
    s = series.copy().replace("-", np.nan).astype("object")
    vc = s.value_counts(dropna=True)
    rare = vc[vc < 3].index
    if len(rare) > 0:
        s.loc[s.isin(rare)] = np.nan
    m = s.notna()
    s.loc[m] = s.loc[m].astype(str).str.strip()
    return s

def get_metrics(y_true, probs, threshold=0.5):
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    _, _, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
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
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols))
    return Pipeline([
        ("prep", ColumnTransformer(transformers=transformers)),
        ("clf", LogisticRegression(penalty="l1", solver="saga", C=0.6, max_iter=4000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])

def build_reduced_logit_pipeline(num_cols, cat_cols):
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols))
    return Pipeline([
        ("prep", ColumnTransformer(transformers=transformers)),
        ("clf", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=4000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])

def build_catboost():
    return CatBoostClassifier(
        loss_function="Logloss", eval_metric="AUC", depth=CATBOOST_DEPTH,
        learning_rate=CATBOOST_LR, iterations=CATBOOST_ITERATIONS,
        l2_leaf_reg=CATBOOST_L2, random_seed=RANDOM_STATE,
        verbose=False, auto_class_weights="Balanced",
    )

def base_feature_name(feat, raw_cat_cols):
    if feat.startswith("num__"):
        return feat.replace("num__", "")
    if feat.startswith("cat__"):
        rest = feat.replace("cat__", "")
        for q in raw_cat_cols:
            if rest.startswith(q):
                return q
    return feat

def build_engineered_features(df):
    out = pd.DataFrame(index=df.index)
    for q in [f"Q{i:02d}" for i in range(1, 42)]:
        if q in df.columns:
            out[f"{q}_num"] = df[q].apply(cat_to_num)
    out["Edad"] = pd.to_numeric(df["Edad"], errors="coerce")
    out["Edad_ge_60"] = (out["Edad"] >= 60).astype(float)
    out["Edad_ge_70"] = (out["Edad"] >= 70).astype(float)
    out["Edad_lt_50"] = (out["Edad"] < 50).astype(float)
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
    out["normal_entre_crisis"] = df["Q27"].apply(is_yes).astype(float)
    out["amanece_bien_luego_aparece"] = df["Q37"].apply(is_yes).astype(float)
    out["SVA_like"] = (((out["temporal_constante"] == 1) | (out["temporal_residual_post_crisis"] == 1) | ((out["curso_lt_3m"] == 1) & (out["duracion_larga"] == 1)))).astype(float)
    out["SVE_like"] = (((out["temporal_crisis"] == 1) & (out["normal_entre_crisis"] == 1) & ((out["duracion_muy_corta"] == 1) | (out["duracion_intermedia"] == 1) | (out["frecuencia_baja"] == 1) | (out["frecuencia_alta"] == 1)))).astype(float)
    out["SVC_like"] = ((((out["temporal_constante"] == 1) | (out["frecuencia_constante"] == 1)) & ((out["curso_3_12m"] == 1) | (out["curso_gt_1a"] == 1)))).astype(float)
    out["giro"] = df["Q03"].apply(is_yes).astype(float)
    out["barco"] = df["Q04"].apply(is_yes).astype(float)
    out["nauseas"] = df["Q05"].apply(is_yes).astype(float)
    out["cabeza_flotando"] = df["Q06"].apply(is_yes).astype(float)
    migraine_items = ["Q07", "Q08", "Q09", "Q10", "Q14", "Q17"]
    for q in migraine_items:
        out[f"{q}_yes"] = df[q].apply(is_yes).astype(float)
    out["bloque_migrañoso_n"] = out[[f"{q}_yes" for q in migraine_items]].sum(axis=1)
    out["bloque_migrañoso_2mas"] = (out["bloque_migrañoso_n"] >= 2).astype(float)
    out["bloque_migrañoso_3mas"] = (out["bloque_migrañoso_n"] >= 3).astype(float)
    pppd_items = ["Q13", "Q15", "Q16", "Q18", "Q33", "Q35"]
    for q in pppd_items:
        out[f"{q}_yes"] = df[q].apply(is_yes).astype(float)
    out["bloque_funcional_n"] = out[[f"{q}_yes" for q in pppd_items]].sum(axis=1)
    out["bloque_funcional_2mas"] = (out["bloque_funcional_n"] >= 2).astype(float)
    out["bloque_funcional_3mas"] = (out["bloque_funcional_n"] >= 3).astype(float)
    positional_items = ["Q29", "Q30", "Q31", "Q32"]
    for q in positional_items:
        out[f"{q}_yes"] = df[q].apply(is_yes).astype(float)
    out["Q38_mas"] = df["Q38"].apply(is_yes).astype(float)
    out["Q38_menos"] = df["Q38"].apply(is_alt).astype(float)
    out["Q38_sin_cambio"] = df["Q38"].apply(is_no).astype(float)
    out["Q41_mas"] = df["Q41"].apply(is_yes).astype(float)
    out["Q41_menos"] = df["Q41"].apply(is_alt).astype(float)
    out["Q41_sin_cambio"] = df["Q41"].apply(is_no).astype(float)
    out["bloque_posicional_n"] = out[[f"{q}_yes" for q in positional_items] + ["Q38_mas", "Q41_mas"]].sum(axis=1)
    out["bloque_posicional_2mas"] = (out["bloque_posicional_n"] >= 2).astype(float)
    out["audicion_peor_reciente"] = (out["Q20_num"].isin([0, 1])).astype(float)
    out["cambio_auditivo_crisis"] = (out["Q21_num"].isin([0, 1])).astype(float)
    out["tinnitus_crisis"] = (out["Q22_num"].isin([0, 1])).astype(float)
    out["oido_tapado_crisis"] = (out["Q23_num"].isin([0, 1])).astype(float)
    out["otalgia_crisis"] = (out["Q24_num"].isin([0, 1])).astype(float)
    out["bloque_auditivo_n"] = out[["audicion_peor_reciente", "cambio_auditivo_crisis", "tinnitus_crisis", "oido_tapado_crisis", "otalgia_crisis"]].sum(axis=1)
    out["bloque_auditivo_2mas"] = (out["bloque_auditivo_n"] >= 2).astype(float)
    out["oscuridad"] = df["Q19"].apply(is_yes).astype(float)
    out["mov_cabeza_rapido"] = df["Q32"].apply(is_yes).astype(float)
    out["cambia_cabeza_rapido_mas"] = out["Q41_mas"]
    out["no_giro"] = df["Q03"].apply(is_no).astype(float)
    out["desmayo"] = df["Q12"].apply(is_yes).astype(float)
    out["opresion_pecho_manos"] = df["Q34"].apply(is_yes).astype(float)
    out["levantarse_rapido_silla"] = df["Q36"].apply(is_yes).astype(float)
    out["quieto_de_pie"] = df["Q28"].apply(is_yes).astype(float)
    out["espacios_llenos"] = df["Q33"].apply(is_yes).astype(float)
    out["en_movimiento"] = df["Q35"].apply(is_yes).astype(float)
    out["quedarse_quieto_menos"] = df["Q40"].apply(is_alt).astype(float)
    out["quedarse_quieto_mas"] = df["Q40"].apply(is_yes).astype(float)
    out["quedarse_quieto_sin_cambio"] = df["Q40"].apply(is_no).astype(float)
    out["periodo_menstrual_mas"] = df["Q39"].apply(is_yes).astype(float)
    out["periodo_menstrual_menos"] = df["Q39"].apply(is_alt).astype(float)
    return out

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
    "bloque_posicional_n": "Conteo de rasgos posicionales / cambios con posición-cabeza",
    "bloque_posicional_2mas": "Al menos 2 rasgos posicionales",
    "bloque_auditivo_n": "Conteo de rasgos auditivos concomitantes",
    "bloque_auditivo_2mas": "Al menos 2 rasgos auditivos",
    "oscuridad": "Dificultad para caminar en oscuridad",
    "mov_cabeza_rapido": "Mareo al mover/girar rápido la cabeza de pie",
    "no_giro": "Ausencia de sensación de giro",
}

def diagnosis_feature_plan():
    return {
        "IA01_MV": {
            "engineered_num": ["Edad", "Edad_lt_50", "SVE_like", "bloque_migrañoso_n", "bloque_migrañoso_2mas", "bloque_migrañoso_3mas", "giro", "nauseas", "cabeza_flotando", "duracion_intermedia", "duracion_larga", "frecuencia_baja", "frecuencia_alta"],
            "raw_cat": ["Q01", "Q02", "Q07", "Q08", "Q09", "Q10", "Q14", "Q17", "Q25", "Q26", "Q39"],
            "clinical_comment": "Prioriza bloque migrañoso + temporalidad episódica y edad menor."
        },
        "IA02_VPPB": {
            "engineered_num": ["Edad", "Edad_ge_60", "SVE_like", "duracion_muy_corta", "frecuencia_baja", "normal_entre_crisis", "giro", "bloque_posicional_n", "bloque_posicional_2mas", "Q38_mas", "Q41_mas"],
            "raw_cat": ["Q03", "Q25", "Q26", "Q27", "Q29", "Q30", "Q31", "Q32", "Q38", "Q41"],
            "clinical_comment": "Prioriza giro + desencadenantes posicionales + crisis breves."
        },
        "IA03_EM": {
            "engineered_num": ["Edad", "SVE_like", "bloque_auditivo_n", "bloque_auditivo_2mas", "giro", "nauseas", "duracion_intermedia", "duracion_larga"],
            "raw_cat": ["Q20", "Q21", "Q22", "Q23", "Q24", "Q25", "Q26"],
            "clinical_comment": "Prioriza rasgos auditivos + crisis episódicas de duración intermedia/larga."
        },
        "IA04_NV": {
            "engineered_num": ["Edad", "SVA_like", "curso_lt_3m", "temporal_residual_post_crisis", "duracion_larga", "giro", "barco", "nauseas", "mov_cabeza_rapido", "Q41_mas"],
            "raw_cat": ["Q01", "Q02", "Q03", "Q04", "Q05", "Q25", "Q32", "Q41"],
            "clinical_comment": "Prioriza patrón agudo/subagudo, vértigo/instabilidad y relación con cabeza."
        },
        "IA05_VB": {
            "engineered_num": ["Edad", "Edad_ge_60", "Edad_ge_70", "SVC_like", "curso_gt_1a", "oscuridad", "mov_cabeza_rapido", "Q41_mas", "no_giro", "en_movimiento", "bloque_posicional_n"],
            "raw_cat": ["Q02", "Q03", "Q19", "Q25", "Q32", "Q35", "Q41"],
            "clinical_comment": "Prioriza edad alta, cronicidad, oscuridad, oscilopsia/cabeza y menos giro clásico."
        },
        "IA06_MPPP": {
            "engineered_num": ["Edad", "SVC_like", "curso_gt_1a", "frecuencia_constante", "frecuencia_alta", "bloque_funcional_n", "bloque_funcional_2mas", "bloque_funcional_3mas", "espacios_llenos", "en_movimiento", "quedarse_quieto_menos", "quieto_de_pie", "barco", "no_giro"],
            "raw_cat": ["Q01", "Q02", "Q13", "Q15", "Q16", "Q18", "Q26", "Q28", "Q33", "Q35", "Q40"],
            "clinical_comment": "Prioriza cronicidad + hipervigilancia/ansiedad + gatillos visuales/posturales."
        },
        "IA07_ACVCentral": {
            "engineered_num": ["Edad", "Edad_ge_60", "Edad_ge_70", "quieto_de_pie", "quedarse_quieto_mas", "quedarse_quieto_sin_cambio", "desmayo", "opresion_pecho_manos", "duracion_variable", "SVA_like", "SVC_like"],
            "raw_cat": ["Q12", "Q25", "Q28", "Q34", "Q36", "Q40"],
            "clinical_comment": "Se restringe a señales de alarma/no periféricas. Mantenerlo como salida cautelosa."
        },
        "IA08_Orto": {
            "engineered_num": ["Edad", "desmayo", "opresion_pecho_manos", "levantarse_rapido_silla", "quieto_de_pie", "duracion_muy_corta", "frecuencia_alta"],
            "raw_cat": ["Q12", "Q28", "Q34", "Q36", "Q25", "Q26"],
            "clinical_comment": "Prioriza ortostatismo/no vestibular: levantarse rápido, síncope, síntomas vegetativos."
        },
    }

def make_response_orientation_table(df_raw, outcomes, feature_plan):
    rows = []
    for outcome in outcomes:
        y = pd.to_numeric(df_raw[outcome], errors="coerce").fillna(0)
        y = (y != 0).astype(int)
        base_prev = y.mean()
        for q in feature_plan[outcome]["raw_cat"]:
            if q not in df_raw.columns:
                continue
            s = df_raw[q].copy().replace("-", np.nan).astype("object")
            m = s.notna()
            s.loc[m] = s.loc[m].astype(str).str.strip()
            vc = s.value_counts(dropna=True)
            for resp, n in vc.items():
                if n < 10:
                    continue
                mask = (s == resp)
                pos_rate = y[mask].mean() if mask.sum() > 0 else np.nan
                lift = pos_rate / base_prev if base_prev > 0 else np.nan
                rows.append({
                    "Diagnostico": outcome,
                    "Pregunta": q,
                    "Pregunta_desc": QUESTION_MAP.get(q, ""),
                    "Respuesta": resp,
                    "n_respuesta": int(mask.sum()),
                    "Prevalencia_diag_global": base_prev,
                    "Tasa_diag_en_esa_respuesta": pos_rate,
                    "Lift_vs_prevalencia": lift,
                    "Diferencia_absoluta": pos_rate - base_prev,
                })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["Diagnostico", "Lift_vs_prevalencia", "n_respuesta"], ascending=[True, False, False])
    return out

def main():
    input_path = Path(INPUT_XLSX)
    output_path = Path(OUTPUT_XLSX)
    if not input_path.exists():
        raise FileNotFoundError(f"No encontré el archivo de entrada: {input_path.resolve()}")
    print(f"Leyendo: {input_path.resolve()}")
    df = pd.read_excel(input_path)
    ia_cols = [c for c in df.columns if c.startswith("IA")]
    q_cols = [c for c in df.columns if c.startswith("Q")]
    for c in ia_cols:
        df[c] = safe_numeric_binary(df[c])
    for c in q_cols:
        df[c] = clean_question_column(df[c])
    df["Edad"] = pd.to_numeric(df["Edad"], errors="coerce")
    cohort = df[df[ia_cols].sum(axis=1) > 0].copy()
    print(f"Cohorte IA positiva: {len(cohort)} pacientes")
    eng = build_engineered_features(cohort)
    outcomes = []
    for o in DEFAULT_OUTCOMES:
        if o in cohort.columns:
            n_pos = int(cohort[o].sum())
            if n_pos >= MIN_POSITIVES_TO_MODEL:
                outcomes.append(o)
            else:
                print(f"Saltando {o}: solo {n_pos} positivos")
    print(f"Outcomes modelados: {outcomes}")
    feature_plan = diagnosis_feature_plan()
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results, winners, importance_frames, reduced_rows, algorithm_rows = [], [], [], [], []
    predictions = pd.DataFrame({"RUT": cohort["RUT"].values})
    reduced_predictions = pd.DataFrame({"RUT": cohort["RUT"].values})
    for outcome in outcomes:
        print(f"Procesando {outcome} ...")
        y = cohort[outcome].astype(int).values
        plan = feature_plan[outcome]
        num_cols = [c for c in plan["engineered_num"] if c in eng.columns]
        raw_cat_cols = [c for c in plan["raw_cat"] if c in cohort.columns]
        X_num = eng[num_cols].copy()
        X_cat = cohort[raw_cat_cols].copy()
        X_cb = pd.concat([X_num, X_cat], axis=1).copy()
        for c in raw_cat_cols:
            X_cb[c] = X_cb[c].where(X_cb[c].notna(), "__MISSING__").astype(str)
        cb_cat_idx = [X_cb.columns.get_loc(c) for c in raw_cat_cols]
        logit = build_logit_pipeline(num_cols, raw_cat_cols)
        X_log = pd.concat([X_num, X_cat], axis=1).copy()
        p_log = cross_val_predict(logit, X_log, y, cv=cv, method="predict_proba")[:, 1]
        d = get_metrics(y, p_log)
        d.update({"Diagnostico": outcome, "Modelo": "Logistica_L1", "n": len(y), "n_positivos": int(y.sum()), "Comentario_clinico": plan["clinical_comment"]})
        results.append(d)
        predictions[f"{outcome}_LogitProb"] = p_log
        logit.fit(X_log, y)
        prep, clf = logit.named_steps["prep"], logit.named_steps["clf"]
        feat_names = prep.get_feature_names_out()
        coef_df = pd.DataFrame({"feature": feat_names, "coef": clf.coef_[0]})
        coef_df["score"] = coef_df["coef"].abs()
        coef_df["base_feature"] = coef_df["feature"].apply(lambda z: base_feature_name(z, raw_cat_cols))
        imp = coef_df.groupby("base_feature", as_index=False)["score"].max().sort_values("score", ascending=False)
        imp["Diagnostico"] = outcome
        imp["Modelo"] = "Logistica_L1"
        importance_frames.append(imp)
        p_cat = np.zeros(len(y))
        for tr, te in cv.split(X_cb, y):
            model = build_catboost()
            model.fit(X_cb.iloc[tr], y[tr], cat_features=cb_cat_idx, verbose=False)
            p_cat[te] = model.predict_proba(X_cb.iloc[te])[:, 1]
        d = get_metrics(y, p_cat)
        d.update({"Diagnostico": outcome, "Modelo": "CatBoost", "n": len(y), "n_positivos": int(y.sum()), "Comentario_clinico": plan["clinical_comment"]})
        results.append(d)
        predictions[f"{outcome}_CatBoostProb"] = p_cat
        full = build_catboost()
        full.fit(X_cb, y, cat_features=cb_cat_idx, verbose=False)
        imp = pd.DataFrame({"base_feature": X_cb.columns, "score": full.get_feature_importance()})
        imp["Diagnostico"] = outcome
        imp["Modelo"] = "CatBoost"
        importance_frames.append(imp)
    results_df = pd.DataFrame(results)[["Diagnostico", "Modelo", "n", "n_positivos", "AUC", "AUPRC", "Sensibilidad", "Especificidad", "VPP", "VPN", "F1", "Brier", "Comentario_clinico"]].sort_values(["Diagnostico", "Modelo"])
    importance_df = pd.concat(importance_frames, ignore_index=True)
    for outcome in outcomes:
        sub = results_df[results_df["Diagnostico"] == outcome].sort_values(["AUPRC", "AUC"], ascending=False)
        winners.append({"Diagnostico": outcome, "Modelo_ganador": sub.iloc[0]["Modelo"], "AUC_ganador": sub.iloc[0]["AUC"], "AUPRC_ganador": sub.iloc[0]["AUPRC"], "F1_ganador": sub.iloc[0]["F1"]})
    winners_df = pd.DataFrame(winners)
    for outcome in outcomes:
        y = cohort[outcome].astype(int).values
        winner = winners_df.loc[winners_df["Diagnostico"] == outcome, "Modelo_ganador"].iloc[0]
        top = importance_df[(importance_df["Diagnostico"] == outcome) & (importance_df["Modelo"] == winner)].sort_values("score", ascending=False)["base_feature"].drop_duplicates().tolist()
        chosen = []
        if "Edad" in top:
            chosen.append("Edad")
        for f in top:
            if f != "Edad":
                chosen.append(f)
            if len(chosen) >= 6:
                break
        chosen = chosen[:6]
        algorithm_rows.append({"Diagnostico": outcome, "Modelo_ganador": winner, "Preguntas_clave_top6": " | ".join(chosen), "Comentario_clinico": feature_plan[outcome]["clinical_comment"]})
        num_cols = [c for c in chosen if c in eng.columns]
        raw_cat_cols = [c for c in chosen if c in cohort.columns and c.startswith("Q")]
        Xr_num = eng[num_cols].copy() if num_cols else pd.DataFrame(index=cohort.index)
        Xr_cat = cohort[raw_cat_cols].copy() if raw_cat_cols else pd.DataFrame(index=cohort.index)
        Xr = pd.concat([Xr_num, Xr_cat], axis=1)
        red_model = build_reduced_logit_pipeline(num_cols, raw_cat_cols)
        p = cross_val_predict(red_model, Xr, y, cv=cv, method="predict_proba")[:, 1]
        reduced_predictions[f"{outcome}_ReducedProb"] = p
        d = get_metrics(y, p)
        d.update({"Diagnostico": outcome, "Modelo_reducido": "Logistica_reducida_top6", "Top6": " | ".join(chosen)})
        reduced_rows.append(d)
    reduced_df = pd.DataFrame(reduced_rows)[["Diagnostico", "Modelo_reducido", "AUC", "AUPRC", "Sensibilidad", "Especificidad", "VPP", "VPN", "F1", "Brier", "Top6"]].sort_values("Diagnostico")
    algorithm_df = pd.DataFrame(algorithm_rows)
    response_orientation_df = make_response_orientation_table(cohort, outcomes, feature_plan)
    feat_desc_df = pd.DataFrame([{"Feature": k, "Descripcion": v} for k, v in ENGINEERED_FEATURE_DESCRIPTIONS.items()])
    methodology_df = pd.DataFrame({
        "Paso": ["Cohorte", "Patron de oro", "Base clinica", "Recodificacion", "Temporalidad/Bárány", "Bloques sintomaticos", "Modelos", "Seleccion de modelo", "Version reducida", "Respuestas orientadoras"],
        "Descripcion": [
            "Pacientes con al menos un IA positivo",
            "IA como referencia de trabajo",
            "Se reinterpretaron las Q según la encuesta original en papel",
            "Se limpiaron categorias raras y faltantes; se construyeron variables clínicas derivadas",
            "Se construyeron rasgos tipo SVA/SVE/SVC usando Q01, Q02, Q25, Q26, Q27 y Q37",
            "Se construyeron bloques migrañoso, funcional/PPPD, posicional y auditivo",
            "Comparacion entre Logistica penalizada L1 y CatBoost por diagnostico",
            "Eleccion por AUPRC y luego AUC",
            "Modelo reducido top-6 para aplicacion clinica",
            "Tabla específica de respuestas y lift por diagnostico para interpretar dirección clínica",
        ]})
    question_map_df = pd.DataFrame([{"Pregunta": q, "Descripcion": d} for q, d in QUESTION_MAP.items()])
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        methodology_df.to_excel(writer, sheet_name="Metodologia", index=False)
        question_map_df.to_excel(writer, sheet_name="Mapa_preguntas", index=False)
        feat_desc_df.to_excel(writer, sheet_name="Features_derivadas", index=False)
        results_df.to_excel(writer, sheet_name="Rendimiento_modelos", index=False)
        winners_df.to_excel(writer, sheet_name="Modelo_ganador", index=False)
        reduced_df.to_excel(writer, sheet_name="Modelos_reducidos_top6", index=False)
        algorithm_df.to_excel(writer, sheet_name="Algoritmo_propuesto", index=False)
        importance_df.sort_values(["Diagnostico", "Modelo", "score"], ascending=[True, True, False]).to_excel(writer, sheet_name="Importancia_variables", index=False)
        response_orientation_df.to_excel(writer, sheet_name="Respuestas_orientadoras", index=False)
        predictions.to_excel(writer, sheet_name="Predicciones_full", index=False)
        reduced_predictions.to_excel(writer, sheet_name="Predicciones_reducidas", index=False)
    print(f"Archivo guardado en: {output_path.resolve()}")

if __name__ == "__main__":
    main()
