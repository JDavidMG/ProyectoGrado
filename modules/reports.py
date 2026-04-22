# modules/reports.py
# Módulo de generación de reportes exportables

import pandas as pd
import json
from pathlib import Path
from datetime import datetime


def generar_reporte_fraude(df_fraudes: pd.DataFrame, directorio_salida: Path) -> Path:
    """
    Genera un CSV con las transacciones identificadas como fraudulentas.
    Incluye timestamp de generación del reporte.
    """
    directorio_salida = Path(directorio_salida)
    directorio_salida.mkdir(parents=True, exist_ok=True)

    df_reporte = df_fraudes.copy()

    # Agregar columna de timestamp del reporte
    df_reporte.insert(0, "Timestamp_reporte", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Ordenar por probabilidad de fraude descendente (si existe)
    if "probabilidad" in df_reporte.columns:
        df_reporte = df_reporte.sort_values("probabilidad", ascending=False)

    # Seleccionar columnas relevantes para el reporte
    columnas_preferidas = [
        "Timestamp_reporte", "Numero_tarjeta", "Tipo_tarjeta",
        "Estado_tarjeta", "Acumulado_cupo", "Cuotas_mora",
        "Localizacion_tarjeta", "probabilidad", "clasificacion",
        "marcacion_usuario"
    ]
    columnas_disponibles = [c for c in columnas_preferidas if c in df_reporte.columns]

    if columnas_disponibles:
        df_reporte = df_reporte[columnas_disponibles]

    nombre_archivo = f"reporte_fraude_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    ruta_salida = directorio_salida / nombre_archivo
    df_reporte.to_csv(ruta_salida, index=False, encoding="utf-8-sig")

    return ruta_salida


def generar_reporte_rendimiento(metricas: dict, retroalimentacion: list, directorio_salida: Path) -> Path:
    """
    Genera un CSV con el reporte de rendimiento del modelo,
    incluyendo métricas y resumen de retroalimentación de usuarios.
    """
    directorio_salida = Path(directorio_salida)
    directorio_salida.mkdir(parents=True, exist_ok=True)

    filas = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ─── Métricas del modelo ───────────────────────────────────────
    filas.append({"Sección": "MÉTRICAS DEL MODELO", "Métrica": "", "Valor": ""})

    if "roc_auc" in metricas:
        filas.append({
            "Sección": "Métricas generales",
            "Métrica": "ROC AUC",
            "Valor": round(metricas["roc_auc"], 4)
        })
    if "pr_auc" in metricas:
        filas.append({
            "Sección": "Métricas generales",
            "Métrica": "PR AUC",
            "Valor": round(metricas["pr_auc"], 4)
        })

    # Métricas por clase desde classification_report
    cls_report = metricas.get("classification_report", {})
    for clase, label in [("0", "Clase 0 - Legítima"), ("1", "Clase 1 - Fraude")]:
        if clase in cls_report:
            for metrica in ["precision", "recall", "f1-score", "support"]:
                filas.append({
                    "Sección": label,
                    "Métrica": metrica.capitalize(),
                    "Valor": round(cls_report[clase].get(metrica, 0), 4)
                })

    if "accuracy" in cls_report:
        filas.append({
            "Sección": "Métricas generales",
            "Métrica": "Accuracy",
            "Valor": round(cls_report["accuracy"], 4)
        })

    if "best_params" in metricas:
        filas.append({
            "Sección": "Parámetros del modelo",
            "Métrica": "Mejores hiperparámetros",
            "Valor": str(metricas["best_params"])
        })

    # ─── Retroalimentación de usuarios ────────────────────────────
    filas.append({"Sección": "", "Métrica": "", "Valor": ""})
    filas.append({"Sección": "RETROALIMENTACIÓN DE USUARIOS", "Métrica": "", "Valor": ""})

    if retroalimentacion:
        vp = sum(1 for r in retroalimentacion if r.get("marcacion") == "verdadero_positivo")
        fp = sum(1 for r in retroalimentacion if r.get("marcacion") == "falso_positivo")
        filas.append({"Sección": "Retroalimentación", "Métrica": "Total marcaciones", "Valor": len(retroalimentacion)})
        filas.append({"Sección": "Retroalimentación", "Métrica": "Verdaderos positivos confirmados", "Valor": vp})
        filas.append({"Sección": "Retroalimentación", "Métrica": "Falsos positivos reportados", "Valor": fp})
    else:
        filas.append({"Sección": "Retroalimentación", "Métrica": "Total marcaciones", "Valor": 0})

    # ─── Información del reporte ───────────────────────────────────
    filas.append({"Sección": "", "Métrica": "", "Valor": ""})
    filas.append({"Sección": "INFO REPORTE", "Métrica": "Generado el", "Valor": timestamp})
    filas.append({"Sección": "INFO REPORTE", "Métrica": "Sistema", "Valor": "Detección de Fraude - Universidad Libre"})

    df_reporte = pd.DataFrame(filas)
    nombre_archivo = f"reporte_rendimiento_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    ruta_salida = directorio_salida / nombre_archivo
    df_reporte.to_csv(ruta_salida, index=False, encoding="utf-8-sig")

    return ruta_salida
