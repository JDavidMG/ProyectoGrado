# modules/retraining.py

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, confusion_matrix,
                              RocCurveDisplay, PrecisionRecallDisplay,
                              ConfusionMatrixDisplay)

# ── Rutas base ─────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent
OUTDIR       = BASE_DIR / "out_detection"
REDNEURONAL  = BASE_DIR / "RedNeuronal"
MODELOS_DIR  = REDNEURONAL / "modelos_reentrenados"   # carpeta para reentrenados
DATABASE_DIR = BASE_DIR / "database"
REDNEURONAL.mkdir(exist_ok=True)
MODELOS_DIR.mkdir(exist_ok=True)
DATABASE_DIR.mkdir(exist_ok=True)

_posibles = [
    OUTDIR      / "random_forest_pipeline.joblib",
    REDNEURONAL / "random_forest_pipeline.joblib",
]
# Si no encuentra random_forest_pipeline.joblib, busca CUALQUIER .joblib en RedNeuronal/
# (excluyendo modelos_reentrenados/ y archivos de backup)
def _encontrar_modelo_original() -> Path:
    for p in _posibles:
        if p.exists():
            return p
    # Buscar cualquier .joblib en RedNeuronal/ directamente
    if REDNEURONAL.exists():
        for f in sorted(REDNEURONAL.glob("*.joblib")):
            if "anterior" not in f.name.lower() and "backup" not in f.name.lower():
                return f
    return _posibles[0]   # fallback aunque no exista

MODEL_PATH   = _encontrar_modelo_original()
MODEL_BACKUP = MODELOS_DIR / "modelo_anterior.joblib"
METRICS_PATH = OUTDIR / "metrics.json"
TRAIN_DATA   = BASE_DIR / "LLENAR" / "tarjetas_fraude_con_ruido_20pct_augmented.csv"
ERRORES_ACUM = BASE_DIR / "uploads" / "errores_acumulados.csv"

COLS_EXCLUIR = [
    "fraude", "fraude_true", "es_etiqueta_ruidosa", "Numero_tarjeta",
    "prediccion", "probabilidad", "clasificacion", "marcacion_usuario",
    "es_outlier", "Ubicacion", "resultado"
]

# ── Complejidades disponibles ──────────────────────────────────────────────
GRAFICAS_DIR = OUTDIR / "graficas"
OUTDIR.mkdir(exist_ok=True)
GRAFICAS_DIR.mkdir(exist_ok=True)


def carpeta_graficas(nombre_modelo: str) -> Path:
    """Retorna la carpeta de gráficas para un modelo dado."""
    stem = Path(nombre_modelo).stem
    carpeta = GRAFICAS_DIR / stem
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta


def generar_graficas_modelo(pipe, X_te, y_te, y_pred, y_proba,
                             nombre_modelo: str, log=None) -> dict:
    """
    Genera las 4 gráficas estándar para un modelo y las guarda en
    out_detection/graficas/[nombre_modelo]/

    Retorna dict con rutas de cada imagen.
    """
    def _log(msg):
        if log: log(msg)

    carpeta = carpeta_graficas(nombre_modelo)
    rutas = {}

    try:
        # 1. Curva ROC
        fig, ax = plt.subplots(figsize=(7, 5))
        RocCurveDisplay.from_predictions(y_te, y_proba, ax=ax)
        ax.set_title(f"Curva ROC — {Path(nombre_modelo).stem}")
        plt.tight_layout()
        ruta = carpeta / "roc_curve.png"
        plt.savefig(ruta, dpi=120)
        plt.close()
        rutas["roc_curve"] = str(ruta)

        # 2. Curva Precisión-Recall
        fig, ax = plt.subplots(figsize=(7, 5))
        PrecisionRecallDisplay.from_predictions(y_te, y_proba, ax=ax)
        ax.set_title(f"Curva PR — {Path(nombre_modelo).stem}")
        plt.tight_layout()
        ruta = carpeta / "pr_curve.png"
        plt.savefig(ruta, dpi=120)
        plt.close()
        rutas["pr_curve"] = str(ruta)

        # 3. Matriz de confusión
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(
            confusion_matrix(y_te, y_pred),
            display_labels=["Legítima", "Fraude"]
        ).plot(ax=ax, colorbar=False)
        ax.set_title(f"Matriz de Confusión — {Path(nombre_modelo).stem}")
        plt.tight_layout()
        ruta = carpeta / "confusion_matrix.png"
        plt.savefig(ruta, dpi=120)
        plt.close()
        rutas["confusion_matrix"] = str(ruta)

        # 4. Importancia de variables
        try:
            ohe  = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["onehot"]
            X_df = X_te if hasattr(X_te, "columns") else pd.DataFrame(X_te)
            cat_cols = X_df.select_dtypes(include=["object"]).columns.tolist()
            num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
            cat_feat = list(ohe.get_feature_names_out(cat_cols))
            feat_names = num_cols + cat_feat
            importances = pipe.named_steps["clf"].feature_importances_
            feat_imp = pd.DataFrame({
                "feature":    feat_names[:len(importances)],
                "importance": importances
            }).sort_values("importance", ascending=False).head(20)

            fig, ax = plt.subplots(figsize=(10, 6))
            feat_imp.plot(kind="bar", x="feature", y="importance",
                          legend=False, ax=ax)
            ax.set_title(f"Top 20 Variables — {Path(nombre_modelo).stem}")
            plt.tight_layout()
            ruta = carpeta / "feature_importances.png"
            plt.savefig(ruta, dpi=120)
            plt.close()
            rutas["feature_importances"] = str(ruta)

            feat_imp.to_csv(carpeta / "feature_importances.csv", index=False)
        except Exception as e:
            _log(f"⚠️  Importancia de variables no disponible: {e}")

        _log(f"📊 Gráficas guardadas en: {carpeta}")

    except Exception as e:
        _log(f"⚠️  Error generando gráficas: {e}")

    return rutas


# ── Complejidades disponibles ──────────────────────────────────────────────
COMPLEJIDADES = {
    "rapido": {
        "label":          "⚡ Rápido",
        "descripcion":    "150 árboles · muestra 30k filas · ~1-2 min",
        "n_estimators":   150,
        "max_depth":      20,
        "max_filas_base": 30_000,
    },
    "balanceado": {
        "label":          "⚖️ Balanceado",
        "descripcion":    "250 árboles · muestra 80k filas · ~3-4 min",
        "n_estimators":   250,
        "max_depth":      None,
        "max_filas_base": 80_000,
    },
    "completo": {
        "label":          "🏋️ Completo",
        "descripcion":    "400 árboles · dataset completo · ~5-8 min — igual al original",
        "n_estimators":   400,
        "max_depth":      None,
        "max_filas_base": None,   # sin límite = todas las filas
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────

def _preparar_X_y(df: pd.DataFrame):
    y = df["fraude"].astype(int)
    X = df.drop(columns=[c for c in COLS_EXCLUIR if c in df.columns], errors="ignore")
    for col in ["Fecha_vencimiento", "Fecha_ult_retiro"]:
        if col in X.columns:
            X[col] = (pd.to_datetime(X[col], errors="coerce")
                        .dt.strftime("%Y-%m-%d").fillna("desconocido"))
    return X, y


def _construir_pipeline(X: pd.DataFrame, cfg: dict) -> Pipeline:
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc",  StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore",
                                                sparse_output=False))]), cat_cols),
    ])
    clf = RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


# ── Base de datos de archivos ──────────────────────────────────────────────

def listar_database() -> list:
    """Retorna todos los CSV guardados en database/ con su metadata."""
    index_path = DATABASE_DIR / "index.json"
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)
    return []


def _guardar_index(entradas: list):
    with open(DATABASE_DIR / "index.json", "w") as f:
        json.dump(entradas, f, indent=4, ensure_ascii=False)


def guardar_en_database(df: pd.DataFrame, nombre_original: str,
                        descripcion: str = "") -> dict:
    """
    Guarda un DataFrame en la carpeta database/ con timestamp.
    Retorna la entrada del índice.
    """
    DATABASE_DIR.mkdir(exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(nombre_original).stem
    nombre_archivo = f"{stem}_{ts}.csv"
    ruta = DATABASE_DIR / nombre_archivo
    df.to_csv(ruta, index=False)

    entrada = {
        "nombre":        nombre_archivo,
        "nombre_orig":   nombre_original,
        "descripcion":   descripcion or nombre_original,
        "ruta":          str(ruta),
        "fecha":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filas":         len(df),
        "columnas":      len(df.columns),
        "tiene_fraude":  "fraude" in df.columns,
        "n_fraudes":     int(df["fraude"].sum()) if "fraude" in df.columns else None,
    }

    entradas = listar_database()
    entradas.append(entrada)
    _guardar_index(entradas)
    return entrada


def eliminar_de_database(nombre_archivo: str) -> bool:
    entradas = listar_database()
    nueva = [e for e in entradas if e["nombre"] != nombre_archivo]
    if len(nueva) == len(entradas):
        return False
    ruta = DATABASE_DIR / nombre_archivo
    if ruta.exists():
        ruta.unlink()
    _guardar_index(nueva)
    return True


# ── Modelos disponibles ────────────────────────────────────────────────────

def listar_modelos_disponibles() -> list:
    """
    Modelos originales: cualquier .joblib en REDNEURONAL/ o OUTDIR/ directamente.
    Modelos reentrenados: los que están en MODELOS_DIR/.
    """
    modelos = []

    # 1. Buscar modelos "originales" — cualquier .joblib en RedNeuronal/ o out_detection/
    vistos = set()
    for carpeta in [REDNEURONAL, OUTDIR]:
        if not carpeta.exists():
            continue
        for f in sorted(carpeta.glob("*.joblib")):
            if "anterior" in f.name.lower() or "backup" in f.name.lower():
                continue
            if str(f) in vistos:
                continue
            vistos.add(str(f))
            tam = f.stat().st_size
            tam_str = f"{tam/1e9:.2f} GB" if tam >= 1e9 else f"{tam/1e6:.0f} MB"
            label = f.stem.replace("_", " ").replace("-", " ")
            modelos.append({
                "nombre":      f.name,
                "ruta":        str(f),
                "carpeta":     carpeta.name,
                "tamaño":      tam_str,
                "es_original": True,
                "fecha":       datetime.fromtimestamp(f.stat().st_mtime)
                               .strftime("%Y-%m-%d %H:%M"),
                "label":       label,
            })

    # 2. Modelos reentrenados en modelos_reentrenados/
    if MODELOS_DIR.exists():
        for f in sorted(MODELOS_DIR.glob("*.joblib"),
                        key=lambda x: x.stat().st_mtime, reverse=True):
            if "anterior" in f.name.lower():
                continue
            tam = f.stat().st_size
            tam_str = f"{tam/1e9:.2f} GB" if tam >= 1e9 else f"{tam/1e6:.0f} MB"
            modelos.append({
                "nombre":      f.name,
                "ruta":        str(f),
                "carpeta":     "modelos_reentrenados",
                "tamaño":      tam_str,
                "es_original": False,
                "fecha":       datetime.fromtimestamp(f.stat().st_mtime)
                               .strftime("%Y-%m-%d %H:%M"),
                "label":       f.stem.replace("_", " ").replace("-", " "),
            })

    return modelos


def listar_modelos() -> list:
    """Compatibilidad con app.py."""
    return listar_modelos_disponibles()


# ── Errores acumulados ─────────────────────────────────────────────────────

def extraer_errores(df: pd.DataFrame) -> dict:
    if "fraude" not in df.columns or "prediccion" not in df.columns:
        return {"fp": 0, "fn": 0, "total": 0, "tiene_ground_truth": False}
    y_real = df["fraude"].astype(int)
    y_pred = df["prediccion"].astype(int)
    return {
        "fp": int(((y_real==0)&(y_pred==1)).sum()),
        "fn": int(((y_real==1)&(y_pred==0)).sum()),
        "total": int(((y_real!=y_pred)).sum()),
        "tiene_ground_truth": True,
    }


def acumular_errores(df: pd.DataFrame, peso_repeticiones: int = 5) -> dict:
    ERRORES_ACUM.parent.mkdir(parents=True, exist_ok=True)
    if "fraude" not in df.columns or "prediccion" not in df.columns:
        return {"ok": False,
                "msg": "El CSV necesita columna 'fraude' con el valor real.", "n": 0}
    y_real = df["fraude"].astype(int)
    y_pred = df["prediccion"].astype(int)
    df_fp  = df[(y_real==0)&(y_pred==1)].copy(); df_fp["fraude"] = 0
    df_fn  = df[(y_real==1)&(y_pred==0)].copy(); df_fn["fraude"] = 1
    df_err = pd.concat([df_fp, df_fn], ignore_index=True)
    n, n_fp, n_fn = len(df_err), len(df_fp), len(df_fn)
    if n == 0:
        return {"ok": False, "msg": "No se encontraron errores FP o FN.", "n": 0}
    df_final = pd.concat(
        [pd.read_csv(ERRORES_ACUM), df_err], ignore_index=True
    ) if ERRORES_ACUM.exists() else df_err
    df_final.to_csv(ERRORES_ACUM, index=False)
    return {"ok": True, "n": n, "n_fp": n_fp, "n_fn": n_fn,
            "msg": f"{n} errores acumulados ({n_fp} FP + {n_fn} FN)."}


def acumular_fraudes_detectados(df: pd.DataFrame) -> dict:
    ERRORES_ACUM.parent.mkdir(parents=True, exist_ok=True)
    if "prediccion" not in df.columns:
        return {"ok": False, "msg": "No hay resultados de predicción.", "n": 0}
    df_n = df[df["prediccion"]==1].copy(); df_n["fraude"] = 1
    n = len(df_n)
    if n == 0:
        return {"ok": False, "msg": "No se detectaron fraudes.", "n": 0}
    df_final = pd.concat(
        [pd.read_csv(ERRORES_ACUM), df_n], ignore_index=True
    ) if ERRORES_ACUM.exists() else df_n
    df_final.to_csv(ERRORES_ACUM, index=False)
    return {"ok": True, "n": n, "n_fp": 0, "n_fn": n,
            "msg": f"{n} fraudes acumulados."}


def hay_errores_acumulados() -> dict:
    if not ERRORES_ACUM.exists():
        return {"total": 0, "fn": 0, "fp": 0}
    df = pd.read_csv(ERRORES_ACUM)
    n_fn = int(df["fraude"].sum()) if "fraude" in df.columns else 0
    return {"total": len(df), "fn": n_fn, "fp": len(df)-n_fn}


# ── Reentrenamiento ────────────────────────────────────────────────────────

def reentrenar_modelo(
    complejidad:       str       = "completo",
    datasets_extra:    list      = None,   # rutas de CSVs adicionales del database
    nombre_modelo:     str       = None,   # nombre del .joblib de salida
    callback           = None,
) -> dict:
    """
    Reentrenamiento configurable:
      - complejidad:    'rapido' | 'balanceado' | 'completo'
      - datasets_extra: lista de rutas CSV del database a incluir
      - nombre_modelo:  nombre del archivo .joblib a guardar
                        (si None → sobreescribe el modelo activo)
    """
    def log(msg):
        print(msg)
        if callback: callback(msg)

    cfg = COMPLEJIDADES.get(complejidad, COMPLEJIDADES["completo"])
    resultado = {
        "ok": False, "mensaje": "",
        "metricas_anteriores": {}, "metricas_nuevas": {},
        "n_errores_usados": 0, "n_total_entrenamiento": 0,
        "n_nuevos_fraudes_usados": 0,
        "timestamp": datetime.now().isoformat(),
    }

    # ── 1. Dataset base ───────────────────────────────────────────────────
    if not TRAIN_DATA.exists():
        resultado["mensaje"] = f"No se encontró: {TRAIN_DATA}"
        return resultado

    log(f"📂 Cargando dataset base ({cfg['label']})...")
    df_base = pd.read_csv(TRAIN_DATA)
    col_obj = "fraude_true" if "fraude_true" in df_base.columns else "fraude"
    df_base["fraude"] = df_base[col_obj].fillna(0).astype(int)

    max_filas = cfg["max_filas_base"]
    if max_filas and len(df_base) > max_filas:
        df_base = (df_base
                   .groupby("fraude", group_keys=False)
                   .apply(lambda g: g.sample(
                       n=min(len(g), int(max_filas * len(g) / len(df_base))),
                       random_state=42))
                   .sample(frac=1, random_state=42)
                   .reset_index(drop=True))
        log(f"📉 Muestra: {len(df_base):,} filas del dataset base.")
    else:
        log(f"✅ Dataset completo: {len(df_base):,} filas.")

    # ── 2. Datasets adicionales del database ──────────────────────────────
    dfs_extra = []
    if datasets_extra:
        for ruta_str in datasets_extra:
            ruta = Path(ruta_str)
            if not ruta.exists():
                log(f"⚠️  No encontrado: {ruta.name} — omitido.")
                continue
            try:
                df_extra = pd.read_csv(ruta)
                # Detectar etiqueta
                for col in ["fraude_true", "fraude"]:
                    if col in df_extra.columns:
                        df_extra["fraude"] = df_extra[col].fillna(0).astype(int)
                        break
                if "fraude" not in df_extra.columns:
                    log(f"⚠️  {ruta.name} sin columna 'fraude' — omitido.")
                    continue
                dfs_extra.append(df_extra)
                n_f = int(df_extra["fraude"].sum())
                log(f"➕ {ruta.name}: {len(df_extra):,} filas ({n_f:,} fraudes).")
            except Exception as e:
                log(f"⚠️  Error leyendo {ruta.name}: {e}")

    # ── 3. Errores FP/FN acumulados ───────────────────────────────────────
    n_errores = 0
    dfs_errores = []
    if ERRORES_ACUM.exists():
        df_err   = pd.read_csv(ERRORES_ACUM)
        n_errores = len(df_err)
        if n_errores > 0:
            objetivo = int(len(df_base) * 0.10)
            peso     = max(5, objetivo // n_errores)
            df_err_p = pd.concat([df_err] * peso, ignore_index=True)
            n_fn = int(df_err["fraude"].sum()) if "fraude" in df_err.columns else 0
            log(f"⚡ {n_errores} errores ({n_fn} FN + {n_errores-n_fn} FP) "
                f"× {peso} = {len(df_err_p):,} filas correctivas.")
            dfs_errores.append(df_err_p)

    resultado["n_errores_usados"]        = n_errores
    resultado["n_nuevos_fraudes_usados"] = n_errores

    # ── 4. Combinar todo ──────────────────────────────────────────────────
    partes = [df_base] + dfs_extra + dfs_errores
    df_combinado = (pd.concat(partes, ignore_index=True)
                    .sample(frac=1, random_state=42)
                    .reset_index(drop=True))
    resultado["n_total_entrenamiento"] = len(df_combinado)

    n_f = int(df_combinado["fraude"].sum())
    n_l = len(df_combinado) - n_f
    log(f"📊 Dataset final: {len(df_combinado):,} registros "
        f"({n_f:,} fraudes / {n_l:,} legítimas).")

    # ── 5. Preparar y dividir ─────────────────────────────────────────────
    X, y = _preparar_X_y(df_combinado)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    log(f"🔀 Train: {len(X_tr):,} · Test: {len(X_te):,}")

    # ── 6. Métricas anteriores ────────────────────────────────────────────
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            resultado["metricas_anteriores"] = json.load(f)

    # ── 7. Backup ─────────────────────────────────────────────────────────
    ruta_salida = _resolver_ruta_salida(nombre_modelo)
    if ruta_salida.exists():
        MODEL_BACKUP.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(joblib.load(ruta_salida), MODEL_BACKUP)
        log(f"💾 Backup: {MODEL_BACKUP.name}")

    # ── 8. Entrenar ───────────────────────────────────────────────────────
    log(f"🌲 Entrenando {cfg['n_estimators']} árboles ({cfg['label']})...")
    pipe = _construir_pipeline(X_tr, cfg)
    pipe.fit(X_tr, y_tr)
    log("✅ Entrenamiento completado.")

    # ── 9. Evaluar ────────────────────────────────────────────────────────
    y_pred  = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)[:, 1]
    roc     = roc_auc_score(y_te, y_proba)
    pr      = average_precision_score(y_te, y_proba)
    log(f"📈 ROC AUC: {roc:.4f}  |  PR AUC: {pr:.4f}")

    metricas_nuevas = {
        "roc_auc":               roc,
        "pr_auc":                pr,
        "classification_report": classification_report(
            y_te, y_pred, digits=4, output_dict=True, zero_division=0),
        "reentrenado_el":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_errores_incorporados": n_errores,
        "complejidad":           complejidad,
        "best_params":           {
            "n_estimators": cfg["n_estimators"],
            "max_depth":    str(cfg["max_depth"]),
        },
    }

    # ── 10. Guardar modelo ────────────────────────────────────────────────
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, ruta_salida)
    tam = round(ruta_salida.stat().st_size / 1e6, 0)
    log(f"💾 Modelo guardado: {ruta_salida.name} ({tam} MB)")

    # ── 10b. Generar gráficas del nuevo modelo ────────────────────────────
    log("📊 Generando gráficas del nuevo modelo...")
    rutas_graficas = generar_graficas_modelo(
        pipe, X_te, y_te, y_pred, y_proba,
        nombre_modelo=ruta_salida.name,
        log=log
    )
    metricas_nuevas["graficas_dir"] = str(carpeta_graficas(ruta_salida.name))

    # Guardar métricas en la carpeta del modelo también
    with open(carpeta_graficas(ruta_salida.name) / "metrics.json", "w") as f:
        json.dump(metricas_nuevas, f, indent=4)

    # Guardar métricas globales si se sobreescribió el modelo activo
    if ruta_salida == MODEL_PATH:
        with open(METRICS_PATH, "w") as f:
            json.dump(metricas_nuevas, f, indent=4)

    # ── 11. Limpiar errores ───────────────────────────────────────────────
    if ERRORES_ACUM.exists():
        ERRORES_ACUM.unlink()
        log("🧹 Errores limpiados para el siguiente ciclo.")

    resultado.update({
        "ok": True,
        "metricas_nuevas":   metricas_nuevas,
        "mensaje":           f"Modelo '{ruta_salida.name}' listo.",
        "nombre_modelo":     ruta_salida.name,
        "ruta_modelo":       str(ruta_salida),
        "ensemble_config":   str(ruta_salida),
        "n_modelos_activos": 1,
    })
    return resultado


def _resolver_ruta_salida(nombre_modelo: str | None) -> Path:
    """Determina la ruta de salida del modelo."""
    if not nombre_modelo or nombre_modelo.strip() == "":
        # Sin nombre → sobreescribe el modelo activo (original)
        return MODEL_PATH

    nombre = nombre_modelo.strip()
    if not nombre.endswith(".joblib"):
        nombre += ".joblib"
    for c in r'\/:*?"<>|':
        nombre = nombre.replace(c, "_")

    # Con nombre → va a la carpeta de reentrenados
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELOS_DIR / nombre


def obtener_graficas_modelo(nombre_modelo: str) -> dict:
    """
    Retorna las rutas de las gráficas disponibles para un modelo.
    Busca primero en graficas/[stem]/, luego en out_detection/ como fallback
    (para el modelo original que fue generado por el script de entrenamiento).
    """
    stem = Path(nombre_modelo).stem
    carpeta = GRAFICAS_DIR / stem
    graficas = {}

    nombres = {
        "roc_curve":           "roc_curve.png",
        "pr_curve":            "pr_curve.png",
        "confusion_matrix":    "confusion_matrix.png",
        "feature_importances": "feature_importances.png",
    }

    for key, archivo in nombres.items():
        # Buscar en carpeta propia del modelo
        ruta = carpeta / archivo
        if ruta.exists():
            graficas[key] = str(ruta)
            continue
        # Fallback: buscar en out_detection/ (modelo original)
        ruta_fallback = OUTDIR / archivo
        if ruta_fallback.exists():
            graficas[key] = str(ruta_fallback)

    # Métricas del modelo
    ruta_metrics = carpeta / "metrics.json"
    if ruta_metrics.exists():
        graficas["metrics_path"] = str(ruta_metrics)
    else:
        graficas["metrics_path"] = str(METRICS_PATH)

    return graficas


# ── Compatibilidad ─────────────────────────────────────────────────────────

def restaurar_modelo_backup() -> bool:
    if not MODEL_BACKUP.exists():
        return False
    joblib.dump(joblib.load(MODEL_BACKUP), MODEL_PATH)
    return True


def predecir_ensemble(X):
    pipe = joblib.load(MODEL_PATH)
    return pipe.predict(X), pipe.predict_proba(X)[:, 1]


def predecir_cascade(X):
    return predecir_ensemble(X)


def guardar_nuevos_fraudes(df, solo_confirmados=False):
    if solo_confirmados and "marcacion_usuario" in df.columns:
        df = df[df["marcacion_usuario"] == "verdadero_positivo"].copy()
    return acumular_fraudes_detectados(df)


def hay_fraudes_acumulados():
    return hay_errores_acumulados()["total"]
