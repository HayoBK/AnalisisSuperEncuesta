
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
OUTPUT_XLSX = "Modelos_IA_comparacion_logistica_vs_catboost_y_algoritmo_final.xlsx"

OUTCOMES = [
    "IA01_MV",
    "IA02_VPPB",
    "IA04_NV",
    "IA05_VB",
    "IA06_MPPP",
    "IA07_ACVCentral",
]

N_SPLITS = 3
RANDOM_STATE = 42

# Para hacer CatBoost más rápido/estable localmente, puedes subir a 150-300 si quieres
CATBOOST_ITERATIONS = 120
CATBOOST_DEPTH = 4
CATBOOST_LR = 0.07
CATBOOST_L2 = 5

# ============================================================
# HELPERS
# ============================================================

def safe_numeric_binary(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    return (s != 0).astype(int)

def clean_question_column(series: pd.Series) -> pd.Series:
    s = series.copy()
    s = s.replace("-", np.nan)
    # Trabajar como object desde el inicio evita warnings de dtype
    s = s.astype("object")

    vc = s.value_counts(dropna=True)
    rare = vc[vc < 3].index
    if len(rare) > 0:
        s.loc[s.isin(rare)] = np.nan

    mask = s.notna()
    s.loc[mask] = s.loc[mask].astype(str)
    return s

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

def build_logit_pipeline(q_cols):
    return Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline(
                                [
                                    ("imp", SimpleImputer(strategy="median")),
                                    ("sc", StandardScaler()),
                                ]
                            ),
                            ["Edad"],
                        ),
                        (
                            "cat",
                            Pipeline(
                                [
                                    ("imp", SimpleImputer(strategy="most_frequent")),
                                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                                ]
                            ),
                            q_cols,
                        ),
                    ]
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    C=0.5,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

def build_reduced_logit_pipeline(num_feats, cat_feats):
    return Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline(
                                [
                                    ("imp", SimpleImputer(strategy="median")),
                                    ("sc", StandardScaler()),
                                ]
                            ),
                            num_feats,
                        ),
                        (
                            "cat",
                            Pipeline(
                                [
                                    ("imp", SimpleImputer(strategy="most_frequent")),
                                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                                ]
                            ),
                            cat_feats,
                        ),
                    ]
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

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

def base_feature_name(feat, q_cols):
    if feat.startswith("num__"):
        return "Edad"
    if feat.startswith("cat__"):
        rest = feat.replace("cat__", "")
        for q in q_cols:
            if rest.startswith(q):
                return q
    return feat

# ============================================================
# MAIN
# ============================================================

def main():
    input_path = Path(INPUT_XLSX)
    output_path = Path(OUTPUT_XLSX)

    if not input_path.exists():
        raise FileNotFoundError(
            f"No encontré el archivo de entrada: {input_path.resolve()}"
        )

    print(f"Leyendo: {input_path.resolve()}")
    df = pd.read_excel(input_path)

    all_cols = df.columns.tolist()
    ia_cols = [c for c in all_cols if c.startswith("IA")]
    q_cols = [c for c in all_cols if c.startswith("Q")]

    # 1) Limpiar labels IA
    for c in ia_cols:
        df[c] = safe_numeric_binary(df[c])

    # 2) Limpiar preguntas
    for c in q_cols:
        df[c] = clean_question_column(df[c])

    # 3) Edad
    df["Edad"] = pd.to_numeric(df["Edad"], errors="coerce")

    # 4) Cohorte principal: al menos un IA positivo
    cohort = df[df[ia_cols].sum(axis=1) > 0].copy()
    print(f"Cohorte IA positiva: {len(cohort)} pacientes")

    X = cohort[["Edad"] + q_cols].copy()

    X_cat = X.copy()
    for c in q_cols:
        X_cat[c] = X_cat[c].where(X_cat[c].notna(), "__MISSING__").astype(str)

    cat_features = [X_cat.columns.get_loc(c) for c in q_cols]

    cv = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )

    logit = build_logit_pipeline(q_cols)

    results = []
    predictions = pd.DataFrame({"RUT": cohort["RUT"].values})
    importance_frames = []

    # ========================================================
    # Full models
    # ========================================================
    for outcome in OUTCOMES:
        print(f"\nProcesando {outcome} ...")
        y = cohort[outcome].astype(int).values

        # -------- Logistic L1
        p_log = cross_val_predict(
            logit, X, y, cv=cv, method="predict_proba"
        )[:, 1]

        d = get_metrics(y, p_log)
        d.update(
            {
                "Diagnostico": outcome,
                "Modelo": "Logistica_L1",
                "n": len(y),
                "n_positivos": int(y.sum()),
            }
        )
        results.append(d)
        predictions[f"{outcome}_LogitProb"] = p_log

        # Importancias logística (fit full)
        logit.fit(X, y)
        prep = logit.named_steps["prep"]
        clf = logit.named_steps["clf"]

        feat_names = prep.get_feature_names_out()
        coef_df = pd.DataFrame(
            {"feature": feat_names, "coef": clf.coef_[0]}
        )
        coef_df["score"] = coef_df["coef"].abs()
        coef_df["base_feature"] = coef_df["feature"].apply(
            lambda z: base_feature_name(z, q_cols)
        )

        imp = (
            coef_df.groupby("base_feature", as_index=False)["score"]
            .max()
            .sort_values("score", ascending=False)
        )
        imp["Diagnostico"] = outcome
        imp["Modelo"] = "Logistica_L1"
        importance_frames.append(imp)

        # -------- CatBoost
        p_cat = np.zeros(len(y))
        for tr, te in cv.split(X_cat, y):
            model = build_catboost()
            model.fit(
                X_cat.iloc[tr],
                y[tr],
                cat_features=cat_features,
                verbose=False,
            )
            p_cat[te] = model.predict_proba(X_cat.iloc[te])[:, 1]

        d = get_metrics(y, p_cat)
        d.update(
            {
                "Diagnostico": outcome,
                "Modelo": "CatBoost",
                "n": len(y),
                "n_positivos": int(y.sum()),
            }
        )
        results.append(d)
        predictions[f"{outcome}_CatBoostProb"] = p_cat

        full = build_catboost()
        full.fit(X_cat, y, cat_features=cat_features, verbose=False)
        imp = pd.DataFrame(
            {
                "base_feature": X_cat.columns,
                "score": full.get_feature_importance(),
            }
        ).sort_values("score", ascending=False)
        imp["Diagnostico"] = outcome
        imp["Modelo"] = "CatBoost"
        importance_frames.append(imp)

    results_df = pd.DataFrame(results)[
        [
            "Diagnostico",
            "Modelo",
            "n",
            "n_positivos",
            "AUC",
            "AUPRC",
            "Sensibilidad",
            "Especificidad",
            "VPP",
            "VPN",
            "F1",
            "Brier",
        ]
    ].sort_values(["Diagnostico", "Modelo"])

    importance_df = pd.concat(importance_frames, ignore_index=True)

    # ========================================================
    # Winners
    # ========================================================
    winners = []
    for outcome in OUTCOMES:
        sub = results_df[results_df["Diagnostico"] == outcome].sort_values(
            ["AUPRC", "AUC"], ascending=False
        )
        winners.append(
            {
                "Diagnostico": outcome,
                "Modelo_ganador": sub.iloc[0]["Modelo"],
                "AUC_ganador": sub.iloc[0]["AUC"],
                "AUPRC_ganador": sub.iloc[0]["AUPRC"],
                "F1_ganador": sub.iloc[0]["F1"],
            }
        )
    winners_df = pd.DataFrame(winners)

    # ========================================================
    # Reduced models top-6
    # ========================================================
    reduced_rows = []
    algorithm_rows = []
    reduced_predictions = pd.DataFrame({"RUT": cohort["RUT"].values})

    for outcome in OUTCOMES:
        winner = winners_df.loc[
            winners_df["Diagnostico"] == outcome, "Modelo_ganador"
        ].iloc[0]

        top = (
            importance_df[
                (importance_df["Diagnostico"] == outcome)
                & (importance_df["Modelo"] == winner)
            ]
            .sort_values("score", ascending=False)["base_feature"]
            .drop_duplicates()
            .tolist()
        )

        chosen = []
        if "Edad" in top:
            chosen.append("Edad")
        for f in top:
            if f != "Edad" and f in q_cols:
                chosen.append(f)
            if len(chosen) >= 6:
                break
        chosen = chosen[:6]

        algorithm_rows.append(
            {
                "Diagnostico": outcome,
                "Modelo_ganador": winner,
                "Preguntas_clave_top6": " | ".join(chosen),
            }
        )

        Xr = cohort[chosen].copy()
        num_feats = [c for c in chosen if c == "Edad"]
        cat_feats = [c for c in chosen if c != "Edad"]

        red_model = build_reduced_logit_pipeline(num_feats, cat_feats)

        y = cohort[outcome].astype(int).values
        p = cross_val_predict(
            red_model, Xr, y, cv=cv, method="predict_proba"
        )[:, 1]

        reduced_predictions[f"{outcome}_ReducedProb"] = p

        d = get_metrics(y, p)
        d.update(
            {
                "Diagnostico": outcome,
                "Modelo_reducido": "Logistica_reducida_top6",
                "Top6": " | ".join(chosen),
            }
        )
        reduced_rows.append(d)

    reduced_df = pd.DataFrame(reduced_rows)[
        [
            "Diagnostico",
            "Modelo_reducido",
            "AUC",
            "AUPRC",
            "Sensibilidad",
            "Especificidad",
            "VPP",
            "VPN",
            "F1",
            "Brier",
            "Top6",
        ]
    ].sort_values("Diagnostico")

    algorithm_df = pd.DataFrame(algorithm_rows)

    # ========================================================
    # Save
    # ========================================================
    methodology_df = pd.DataFrame(
        {
            "Paso": [
                "Cohorte",
                "Predictores",
                "Outcomes",
                "Limpieza",
                "Modelo 1",
                "Modelo 2",
                "Validacion",
                "Seleccion",
                "Reduccion",
                "Salida",
            ],
            "Descripcion": [
                "Pacientes con al menos un IA positivo",
                "Edad + 41 preguntas Q",
                ", ".join(OUTCOMES),
                "Se trato '-' y categorias ultra-raras (<3) como faltantes",
                "Regresion logistica penalizada L1",
                "CatBoost",
                f"Validacion cruzada estratificada de {N_SPLITS} particiones",
                "Modelo ganador por AUPRC y luego AUC",
                "Version reducida con 6 variables clave",
                "Probabilidad por diagnostico + set de preguntas clave",
            ],
        }
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        methodology_df.to_excel(writer, sheet_name="Metodologia", index=False)
        results_df.to_excel(writer, sheet_name="Rendimiento_modelos", index=False)
        winners_df.to_excel(writer, sheet_name="Modelo_ganador", index=False)
        reduced_df.to_excel(writer, sheet_name="Modelos_reducidos_top6", index=False)
        algorithm_df.to_excel(writer, sheet_name="Algoritmo_propuesto", index=False)
        importance_df.sort_values(
            ["Diagnostico", "Modelo", "score"],
            ascending=[True, True, False],
        ).to_excel(writer, sheet_name="Importancia_variables", index=False)
        predictions.to_excel(writer, sheet_name="Predicciones_full", index=False)
        reduced_predictions.to_excel(
            writer, sheet_name="Predicciones_reducidas", index=False
        )

    print(f"\nArchivo guardado en: {output_path.resolve()}")

    # resumen corto en consola
    disp = results_df.copy()
    for c in ["AUC", "AUPRC", "F1", "Sensibilidad", "Especificidad"]:
        disp[c] = disp[c].round(3)
    print("\n=== Rendimiento de modelos ===")
    print(disp.to_string(index=False))

    print("\n=== Modelos ganadores ===")
    print(winners_df.to_string(index=False))

    tmp = reduced_df.copy()
    for c in ["AUC", "AUPRC", "F1", "Sensibilidad", "Especificidad"]:
        tmp[c] = tmp[c].round(3)
    print("\n=== Modelos reducidos top6 ===")
    print(tmp.to_string(index=False))


if __name__ == "__main__":
    main()
