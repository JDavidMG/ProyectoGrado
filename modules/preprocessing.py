# modules/preprocessing.py
# Preprocesamiento EXACTAMENTE igual al script 02_arboles_decision_detection.py
#
# REGLA DE ORO: el pipeline de sklearn ya hace todo el preprocesamiento internamente
# (imputer + scaler + onehot). Aquí solo debemos:
#   1. Eliminar las columnas que el modelo NO vio durante el entrenamiento.
#   2. Asegurarnos de que las columnas restantes existen y tienen los mismos nombres.
# NO transformar valores — el pipeline lo hace solo.

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from typing import Tuple, Optional

# Columnas que el script de entrenamiento eliminó antes de entrenar:
#   df.drop(columns=["fraude", "fraude_true", "es_etiqueta_ruidosa", "Numero_tarjeta"])
# Más las columnas extra que puede traer el CSV de resultados de la app.
COLUMNAS_EXCLUIR = [
    "fraude",
    "fraude_true",
    "es_etiqueta_ruidosa",
    "Numero_tarjeta",
    "prediccion",
    "probabilidad",
    "clasificacion",
    "marcacion_usuario",
    "es_outlier",
]

# Columnas mínimas que el modelo necesita
COLUMNAS_REQUERIDAS = [
    "Tipo_tarjeta",
    "Estado_tarjeta",
    "Fecha_vencimiento",
    "Indicador_repositorio",
    "Localizacion_tarjeta",
    "Acumulado_cupo",
    "Fecha_ult_retiro",
    "Cuotas_mora",
    "Ind_estado",
    "Es_amparada",
    "Es_reexpedicion",
    "Tipo_nomina",
]


def _obtener_columnas_modelo() -> list:
    """
    Lee las columnas exactas que el pipeline del modelo espera,
    inspeccionando el ColumnTransformer guardado en disco.
    """
    base = Path(__file__).parent.parent
    posibles = [
        base / "out_detection" / "random_forest_pipeline.joblib",
        base / "RedNeuronal"   / "random_forest_pipeline.joblib",
    ]
    for ruta in posibles:
        if ruta.exists():
            try:
                pipe = joblib.load(ruta)
                pre  = pipe.named_steps["pre"]
                cols = []
                for _, _, column_names in pre.transformers_:
                    cols.extend(column_names)
                return cols
            except Exception:
                pass
    return []


def preprocesar_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Prepara el DataFrame para ser pasado al pipeline del modelo.

    Solo elimina columnas de control y alinea las columnas al modelo.
    NO transforma valores — el ColumnTransformer del pipeline ya lo hace.
    """
    df = df.copy()

    # 1. Eliminar columnas de control que el modelo nunca vio
    df = df.drop(columns=[c for c in COLUMNAS_EXCLUIR if c in df.columns],
                 errors="ignore")

    # 2. Verificar columnas mínimas
    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltantes:
        return None, f"Columnas faltantes: {', '.join(faltantes)}"

    # 3. Convertir fechas a string YYYY-MM-DD
    #    (el modelo las trata como columnas categóricas / object)
    for col in ["Fecha_vencimiento", "Fecha_ult_retiro"]:
        if col in df.columns:
            df[col] = (pd.to_datetime(df[col], errors="coerce")
                         .dt.strftime("%Y-%m-%d")
                         .fillna("desconocido"))

    # 4. Alinear columnas exactamente con las que vio el modelo
    columnas_modelo = _obtener_columnas_modelo()
    if columnas_modelo:
        # Agregar columnas faltantes con valor neutro
        for col in columnas_modelo:
            if col not in df.columns:
                # Determinar si es numérica o categórica por el nombre
                df[col] = 0 if col not in [
                    "Tipo_tarjeta", "Estado_tarjeta", "Fecha_vencimiento",
                    "Localizacion_tarjeta", "Ind_estado", "Tipo_nomina",
                    "Fecha_ult_retiro", "Ubicacion"
                ] else "desconocido"

        # Mantener solo las columnas que el modelo conoce, en el mismo orden
        cols_presentes = [c for c in columnas_modelo if c in df.columns]
        df = df[cols_presentes]

    return df, None


# ── Funciones auxiliares ─────────────────────────────────────────────────────

def detectar_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df["es_outlier"] = False
    for col in num_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df.loc[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR),
               "es_outlier"] = True
    return df


def obtener_estadisticas(df: pd.DataFrame) -> dict:
    stats = {
        "total_filas":    len(df),
        "total_columnas": len(df.columns),
        "valores_nulos":  int(df.isnull().sum().sum()),
        "columnas":       list(df.columns),
    }
    if "fraude" in df.columns:
        conteo = df["fraude"].value_counts()
        stats["fraudes"]    = int(conteo.get(1, 0))
        stats["legitimas"]  = int(conteo.get(0, 0))
        stats["pct_fraude"] = round(stats["fraudes"] / len(df) * 100, 2)
    return stats
