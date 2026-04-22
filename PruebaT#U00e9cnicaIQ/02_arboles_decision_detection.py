# 02_arboles_decision_detection.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib, json
from pathlib import Path
from tqdm import tqdm  # <-- barra de progreso

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay
)

# ------------------------
# Configuración general
# ------------------------
DATA_PATH = "LLENAR/tarjetas_fraude_con_ruido_20pct_augmented.csv"
OUTDIR = Path("out_detection")
OUTDIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

# ------------------------
# Cargar dataset
# ------------------------
df = pd.read_csv(DATA_PATH)
print(f"Dataset leído: {DATA_PATH} ({len(df)} filas)")

y = df["fraude"]
X = df.drop(columns=["fraude", "fraude_true", "es_etiqueta_ruidosa", "Numero_tarjeta"])

# Identificar columnas categóricas y numéricas
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# ------------------------
# Pipeline de preprocesamiento
# ------------------------
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# ------------------------
# División train/test
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
)

# ------------------------
# Definición de la búsqueda de hiperparámetros
# ------------------------
param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

param_list = list(ParameterGrid(param_grid))
results = []

print(f"Probando {len(param_list)} combinaciones de hiperparámetros...")

# ------------------------
# Entrenamiento con barra de progreso
# ------------------------
for params in tqdm(param_list, desc="Modelos probados"):
    pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
            **params
        ))
    ])
    pipe.fit(X_train, y_train)
    score = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
    results.append((params, score, pipe))

# seleccionar el mejor modelo
best_params, best_score, best_pipe = max(results, key=lambda x: x[1])
print("Mejores parámetros:", best_params)
print("ROC AUC en test:", best_score)

# ------------------------
# Evaluación en test
# ------------------------
y_pred = best_pipe.predict(X_test)
y_proba = best_pipe.predict_proba(X_test)[:, 1]

report = classification_report(y_test, y_pred, digits=4, output_dict=True)
roc_auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)

print("Reporte clasificación:")
print(classification_report(y_test, y_pred, digits=4))
print("ROC AUC:", roc_auc)
print("PR AUC:", pr_auc)

# Guardar métricas en JSON
metrics = {
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "classification_report": report,
    "best_params": best_params
}
with open(OUTDIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("Métricas guardadas en metrics.json")

# ------------------------
# Gráficas
# ------------------------
plt.figure()
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.savefig(OUTDIR / "roc_curve.png")

plt.figure()
PrecisionRecallDisplay.from_predictions(y_test, y_proba)
plt.savefig(OUTDIR / "pr_curve.png")

plt.figure()
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()
plt.savefig(OUTDIR / "confusion_matrix.png")

print("Gráficas guardadas en carpeta:", OUTDIR)

# ------------------------
# Importancia de variables
# ------------------------
# Extraer columnas transformadas (dummies incluidas)
ohe = best_pipe.named_steps["pre"].named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = ohe.get_feature_names_out(cat_cols)
feature_names = num_cols + list(cat_feature_names)

importances = best_pipe.named_steps["clf"].feature_importances_

feat_imp = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

feat_imp.to_csv(OUTDIR / "feature_importances.csv", index=False)

plt.figure(figsize=(10,6))
feat_imp.head(20).plot(kind="bar", x="feature", y="importance", legend=False)
plt.title("Top 20 Importancia de Variables (RandomForest)")
plt.tight_layout()
plt.savefig(OUTDIR / "feature_importances.png")

# ------------------------
# Guardado del modelo completo (pipeline)
# ------------------------
joblib.dump(best_pipe, OUTDIR / "random_forest_pipeline.joblib")
print("Modelo entrenado guardado en random_forest_pipeline.joblib")
