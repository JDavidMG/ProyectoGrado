"""
================================================================================
COMPARACIÓN EXPERIMENTAL DE TÉCNICAS DE CLASIFICACIÓN
Proyecto de grado — Detección de fraude con tarjeta de crédito
================================================================================

Este script complementa la búsqueda de hiperparámetros ya realizada sobre Random
Forest, ejecutando ahora una comparación entre TRES técnicas distintas de
clasificación bajo condiciones idénticas (mismo pipeline de preprocesamiento,
mismo split de datos, mismo random_state, mismas métricas de evaluación).

TÉCNICAS COMPARADAS:
    1. Random Forest       (árbol de decisión — el modelo elegido en el proyecto)
    2. Regresión Logística (modelo lineal — baseline clásico)
    3. Gradient Boosting   (árbol de decisión secuencial — alternativa moderna)

Opcionalmente, si XGBoost está instalado, se agrega como cuarto modelo.

REQUISITOS:
    pip install pandas numpy scikit-learn matplotlib joblib
    (opcional) pip install xgboost

CÓMO CORRERLO:
    1. Copia este archivo en la misma carpeta donde está tu dataset.
    2. Ajusta la variable RUTA_DATASET más abajo con el nombre real de tu CSV.
    3. Ejecuta:   python comparacion_modelos.py
    4. Los resultados se guardan en:
       - comparacion_modelos_resultados.csv   (tabla comparativa)
       - comparacion_modelos_curvas_roc.png   (gráfica curvas ROC)
       - comparacion_modelos_metricas.png     (gráfica de barras de métricas)
       - comparacion_modelos_reporte.txt      (texto imprimible para el documento)
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, accuracy_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

# XGBoost es opcional
try:
    from xgboost import XGBClassifier
    XGBOOST_DISPONIBLE = True
except ImportError:
    XGBOOST_DISPONIBLE = False
    print("[Aviso] XGBoost no está instalado. Se comparará sin él.")
    print("        Para incluirlo, instala:  pip install xgboost")

# ============================================================================
# CONFIGURACIÓN — AJUSTA AQUÍ
# ============================================================================
RUTA_DATASET = "tarjetas_fraude_base.csv"   # <-- CAMBIA por el nombre real de tu CSV
COLUMNA_OBJETIVO = "fraude"                  # nombre de la columna con la etiqueta 0/1
COLUMNAS_A_EXCLUIR = ["numero_tarjeta", "fraude_true", "es_etiqueta_ruidosa"]
SEMILLA = 42                                 # para reproducibilidad
TEST_SIZE = 0.30                             # 70/30 igual que el proyecto original

# ============================================================================
# 1. CARGA DE DATOS
# ============================================================================
print("=" * 78)
print("COMPARACIÓN EXPERIMENTAL DE TÉCNICAS DE CLASIFICACIÓN")
print("=" * 78)
print(f"\n[1/6] Cargando dataset: {RUTA_DATASET}")

df = pd.read_csv(RUTA_DATASET)
print(f"      Registros cargados: {len(df):,}")
print(f"      Columnas totales:   {df.shape[1]}")

# Eliminar columnas que no aportan al modelo
for col in COLUMNAS_A_EXCLUIR:
    if col in df.columns:
        df = df.drop(columns=[col])

# Separar X e y
if COLUMNA_OBJETIVO not in df.columns:
    raise ValueError(
        f"No se encontró la columna '{COLUMNA_OBJETIVO}' en el dataset. "
        f"Ajusta la variable COLUMNA_OBJETIVO al nombre correcto."
    )

y = df[COLUMNA_OBJETIVO].astype(int)
X = df.drop(columns=[COLUMNA_OBJETIVO])

# Detectar columnas numéricas y categóricas
columnas_numericas = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
columnas_categoricas = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Convertir fechas a números (días desde fecha mínima) si aparecen como texto de fecha
for col in columnas_categoricas.copy():
    if "fecha" in col.lower():
        try:
            X[col] = pd.to_datetime(X[col], errors="coerce")
            X[col] = (X[col] - X[col].min()).dt.days
            columnas_categoricas.remove(col)
            columnas_numericas.append(col)
        except Exception:
            pass

print(f"      Variables numéricas:   {len(columnas_numericas)}")
print(f"      Variables categóricas: {len(columnas_categoricas)}")
print(f"      Distribución de clases: {dict(y.value_counts())}")

# ============================================================================
# 2. PIPELINE DE PREPROCESAMIENTO (idéntico al del proyecto)
# ============================================================================
print("\n[2/6] Construyendo pipeline de preprocesamiento (idéntico al proyecto)...")

pipeline_numericas = Pipeline(steps=[
    ("imputador", SimpleImputer(strategy="median")),
    ("escalador", StandardScaler()),
])

pipeline_categoricas = Pipeline(steps=[
    ("imputador", SimpleImputer(strategy="most_frequent")),
    ("codificador", OneHotEncoder(handle_unknown="ignore")),
])

preprocesador = ColumnTransformer(transformers=[
    ("num", pipeline_numericas, columnas_numericas),
    ("cat", pipeline_categoricas, columnas_categoricas),
])

# ============================================================================
# 3. DIVISIÓN ENTRENAMIENTO / PRUEBA (misma que el proyecto)
# ============================================================================
print(f"\n[3/6] Dividiendo dataset (train={int((1-TEST_SIZE)*100)}%, test={int(TEST_SIZE*100)}%)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=SEMILLA,
    stratify=y
)

# ============================================================================
# 4. DEFINICIÓN DE MODELOS A COMPARAR
# ============================================================================
print("\n[4/6] Definiendo modelos a comparar...")

modelos = {
    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=5,
        random_state=SEMILLA,
        n_jobs=-1,
    ),
    "Regresión Logística": LogisticRegression(
        max_iter=1000,
        random_state=SEMILLA,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=SEMILLA,
    ),
}

if XGBOOST_DISPONIBLE:
    modelos["XGBoost"] = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        random_state=SEMILLA,
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=-1,
    )

print(f"      Modelos a evaluar: {list(modelos.keys())}")

# ============================================================================
# 5. ENTRENAMIENTO Y EVALUACIÓN
# ============================================================================
print("\n[5/6] Entrenando y evaluando cada modelo con el mismo pipeline...")

resultados = []
predicciones_probabilidad = {}

for nombre, modelo in modelos.items():
    print(f"      → {nombre}...", end=" ")

    pipeline_completo = Pipeline(steps=[
        ("preprocesador", preprocesador),
        ("clasificador", modelo),
    ])

    # Entrenar
    pipeline_completo.fit(X_train, y_train)

    # Predecir
    y_pred = pipeline_completo.predict(X_test)
    y_prob = pipeline_completo.predict_proba(X_test)[:, 1]
    predicciones_probabilidad[nombre] = y_prob

    # Métricas
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    tasa_fp = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    resultados.append({
        "Modelo": nombre,
        "ROC_AUC": roc_auc_score(y_test, y_prob),
        "PR_AUC": average_precision_score(y_test, y_prob),
        "Precisión (fraude)": precision_score(y_test, y_pred, pos_label=1),
        "Recall / Sensibilidad": recall_score(y_test, y_pred, pos_label=1),
        "F1_score (fraude)": f1_score(y_test, y_pred, pos_label=1),
        "Exactitud": accuracy_score(y_test, y_pred),
        "Tasa_FP": tasa_fp,
        "VP": tp, "FN": fn, "FP": fp, "VN": tn,
    })
    print("ok")

df_resultados = pd.DataFrame(resultados)

# ============================================================================
# 6. REPORTE, TABLAS Y GRÁFICAS
# ============================================================================
print("\n[6/6] Guardando resultados...")

# --- Tabla CSV ---
df_resultados.to_csv("comparacion_modelos_resultados.csv", index=False)
print("      ✓ comparacion_modelos_resultados.csv")

# --- Gráfica de curvas ROC ---
plt.figure(figsize=(9, 7))
for nombre, y_prob in predicciones_probabilidad.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, linewidth=2, label=f"{nombre} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Clasificador aleatorio")
plt.xlabel("Tasa de falsos positivos", fontsize=12)
plt.ylabel("Tasa de verdaderos positivos (sensibilidad)", fontsize=12)
plt.title("Comparación de curvas ROC entre técnicas de clasificación", fontsize=13)
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("comparacion_modelos_curvas_roc.png", dpi=150)
plt.close()
print("      ✓ comparacion_modelos_curvas_roc.png")

# --- Gráfica de barras comparativa ---
metricas_a_graficar = ["ROC_AUC", "PR_AUC", "Precisión (fraude)",
                        "Recall / Sensibilidad", "F1_score (fraude)", "Exactitud"]
n_modelos = len(df_resultados)
n_metricas = len(metricas_a_graficar)

fig, ax = plt.subplots(figsize=(12, 6))
x_pos = np.arange(n_metricas)
ancho_barra = 0.8 / n_modelos
colores = ["#1E2761", "#C41E3A", "#2C7A3E", "#D4A017"][:n_modelos]

for i, (_, fila) in enumerate(df_resultados.iterrows()):
    valores = [fila[m] for m in metricas_a_graficar]
    ax.bar(x_pos + i * ancho_barra, valores, ancho_barra,
           label=fila["Modelo"], color=colores[i])

ax.set_xlabel("Métrica", fontsize=12)
ax.set_ylabel("Valor", fontsize=12)
ax.set_title("Comparación de desempeño entre técnicas de clasificación",
             fontsize=13)
ax.set_xticks(x_pos + ancho_barra * (n_modelos - 1) / 2)
ax.set_xticklabels(metricas_a_graficar, rotation=15, ha="right")
ax.legend(fontsize=11)
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("comparacion_modelos_metricas.png", dpi=150)
plt.close()
print("      ✓ comparacion_modelos_metricas.png")

# --- Reporte de texto listo para pegar ---
with open("comparacion_modelos_reporte.txt", "w", encoding="utf-8") as f:
    f.write("=" * 78 + "\n")
    f.write("REPORTE DE COMPARACIÓN EXPERIMENTAL DE TÉCNICAS DE CLASIFICACIÓN\n")
    f.write("=" * 78 + "\n\n")
    f.write(f"Registros totales:       {len(df):,}\n")
    f.write(f"Entrenamiento:           {len(X_train):,} ({int((1-TEST_SIZE)*100)}%)\n")
    f.write(f"Prueba:                  {len(X_test):,} ({int(TEST_SIZE*100)}%)\n")
    f.write(f"Semilla aleatoria:       {SEMILLA}\n")
    f.write(f"Modelos comparados:      {len(modelos)}\n\n")

    f.write("-" * 78 + "\n")
    f.write("TABLA COMPARATIVA DE MÉTRICAS\n")
    f.write("-" * 78 + "\n\n")
    f.write(df_resultados.round(4).to_string(index=False))
    f.write("\n\n")

    # Ganador
    ganador = df_resultados.sort_values("ROC_AUC", ascending=False).iloc[0]
    f.write("-" * 78 + "\n")
    f.write("MODELO GANADOR (por ROC AUC)\n")
    f.write("-" * 78 + "\n\n")
    f.write(f"  {ganador['Modelo']}\n")
    f.write(f"  ROC AUC:                 {ganador['ROC_AUC']:.4f}\n")
    f.write(f"  PR AUC:                  {ganador['PR_AUC']:.4f}\n")
    f.write(f"  Precisión (fraude):      {ganador['Precisión (fraude)']:.4f}\n")
    f.write(f"  Recall / Sensibilidad:   {ganador['Recall / Sensibilidad']:.4f}\n")
    f.write(f"  F1-score (fraude):       {ganador['F1_score (fraude)']:.4f}\n")
    f.write(f"  Exactitud:               {ganador['Exactitud']:.4f}\n")
    f.write(f"  Tasa de falsos positivos: {ganador['Tasa_FP']:.4f}\n")

print("      ✓ comparacion_modelos_reporte.txt")

print("\n" + "=" * 78)
print("COMPARACIÓN FINALIZADA")
print("=" * 78)
print("\nResumen de resultados:\n")
print(df_resultados[["Modelo", "ROC_AUC", "PR_AUC", "Precisión (fraude)",
                     "Recall / Sensibilidad", "F1_score (fraude)"]].round(4).to_string(index=False))
print("\nModelo con mayor ROC AUC:", df_resultados.sort_values("ROC_AUC", ascending=False).iloc[0]["Modelo"])
print("\nArchivos generados en la carpeta actual. Úsalos para actualizar el documento y el pptx.\n")
