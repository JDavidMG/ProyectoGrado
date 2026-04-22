# app.py

from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify, send_file)
import pandas as pd
import numpy as np
import joblib
import json
import threading
import time
from pathlib import Path
from datetime import datetime
import io
import base64
import matplotlib
matplotlib.use('Agg')

from modules.auth import verificar_credenciales
from modules.preprocessing import preprocesar_dataframe, detectar_outliers, obtener_estadisticas
from modules.reports import generar_reporte_fraude, generar_reporte_rendimiento
from modules.feedback import guardar_retroalimentacion, cargar_retroalimentacion
from modules import retraining
from modules.retraining import (
    restaurar_modelo_backup,
    MODEL_PATH      as RETRAIN_MODEL_PATH,
    MODEL_BACKUP    as RETRAIN_MODEL_BACKUP,
    DATABASE_DIR,
    COMPLEJIDADES,
)

# ── Rutas ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

def _detectar_modelo_activo():
    """Busca el modelo principal en out_detection/ y RedNeuronal/."""
    candidatos = [
        BASE_DIR / "out_detection" / "random_forest_pipeline.joblib",
        BASE_DIR / "RedNeuronal"   / "random_forest_pipeline.joblib",
    ]
    for p in candidatos:
        if p.exists():
            return p
    # Buscar cualquier .joblib directamente en RedNeuronal/ (no en subcarpetas)
    rn = BASE_DIR / "RedNeuronal"
    if rn.exists():
        for f in sorted(rn.glob("*.joblib")):
            if "anterior" not in f.name.lower() and "backup" not in f.name.lower():
                return f
    return candidatos[0]

MODEL_PATH   = _detectar_modelo_activo()
METRICS_PATH = BASE_DIR / "out_detection" / "metrics.json"
OUTDIR       = BASE_DIR / "out_detection"
UPLOADS_DIR  = BASE_DIR / "uploads"
HISTORIAL_DIR = BASE_DIR / "historial"
UPLOADS_DIR.mkdir(exist_ok=True)
HISTORIAL_DIR.mkdir(exist_ok=True)


def _guardar_historial(df: pd.DataFrame, nombre: str,
                       modelo_nombre: str, db_nombre: str) -> dict:
    """Guarda los resultados en el historial con metadata."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = nombre.strip().replace(" ", "_") or "analisis"
    archivo = f"{stem}_{ts}.csv"
    ruta = HISTORIAL_DIR / archivo
    df.to_csv(ruta, index=False)

    n_fraudes   = int((df["prediccion"] == 1).sum()) if "prediccion" in df.columns else 0
    n_legitimas = int((df["prediccion"] == 0).sum()) if "prediccion" in df.columns else 0

    # Calcular estado de revisión manual al guardar
    n_errores = 0
    n_revisadas = 0
    tiene_gt = "fraude" in df.columns and "prediccion" in df.columns
    if tiene_gt:
        y_real = df["fraude"].astype(int)
        y_pred = df["prediccion"].astype(int)
        n_errores   = int((y_real != y_pred).sum())
        n_revisadas = int(df["etiqueta_usuario"].notna().sum())                       if "etiqueta_usuario" in df.columns else 0

    entrada = {
        "nombre":       nombre.strip() or "Sin nombre",
        "archivo":      archivo,
        "fecha":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modelo":       modelo_nombre,
        "database":     db_nombre or "—",
        "n_filas":      len(df),
        "n_fraudes":    n_fraudes,
        "n_legitimas":  n_legitimas,
        "pct_fraude":   round(n_fraudes / len(df) * 100, 1) if len(df) > 0 else 0,
        "tiene_gt":     tiene_gt,
        "n_errores":    n_errores,
        "n_revisadas":  n_revisadas,
        "pct_revision": round(n_revisadas / n_errores * 100, 1) if n_errores > 0 else 0,
        "revision_completa": n_errores > 0 and n_revisadas >= n_errores,
    }

    idx_path = HISTORIAL_DIR / "index.json"
    if idx_path.exists():
        with open(idx_path) as f:
            idx = json.load(f)
    else:
        idx = []
    idx.append(entrada)
    with open(idx_path, "w") as f:
        json.dump(idx, f, indent=4, ensure_ascii=False)

    return entrada

app = Flask(__name__)
app.secret_key = "clave_secreta_proyecto_fraude_2025"

_retrain_jobs = {}
modelo_cache  = {}   # ruta_str → pipeline (caché por ruta)


# ── Helpers ────────────────────────────────────────────────────────────────

def cargar_modelo(ruta: Path = None):
    ruta = ruta or MODEL_PATH
    key  = str(ruta)
    if key not in modelo_cache:
        if ruta.exists():
            modelo_cache[key] = joblib.load(ruta)
            print(f"✅ Modelo cargado: {ruta.name}")
        else:
            print(f"⚠️  No encontrado: {ruta}")
            return None
    return modelo_cache[key]


def login_requerido(f):
    from functools import wraps
    @wraps(f)
    def deco(*args, **kwargs):
        if "usuario" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return deco


# ── Autenticación ──────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        u = request.form.get("usuario", "").strip()
        p = request.form.get("password", "").strip()
        if verificar_credenciales(u, p):
            session["usuario"] = u
            return redirect(url_for("dashboard"))
        error = "Usuario o contraseña incorrectos."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ── Dashboard ──────────────────────────────────────────────────────────────

@app.route("/dashboard")
@login_requerido
def dashboard():
    metricas = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            metricas = json.load(f)
    return render_template("dashboard.html",
                           usuario=session["usuario"], metricas=metricas)


# ── Database de archivos ───────────────────────────────────────────────────

@app.route("/database")
@login_requerido
def ver_database():
    archivos = retraining.listar_database()
    return render_template("database.html",
                           usuario=session["usuario"],
                           archivos=archivos)


@app.route("/database/usar/<nombre>")
@login_requerido
def usar_archivo_db(nombre):
    """Carga un archivo del database como archivo activo."""
    ruta = DATABASE_DIR / nombre
    if not ruta.exists():
        return redirect(url_for("ver_database"))
    df = pd.read_csv(ruta)
    dest = UPLOADS_DIR / "ultimo_archivo.csv"
    df.to_csv(dest, index=False)
    session["archivo_cargado"]  = str(dest)
    session["n_filas"]          = len(df)
    session["n_cols"]           = len(df.columns)
    session["db_nombre_activo"] = nombre
    session["carga_fresca"]     = True
    session["resultados_path"]  = None
    return redirect(url_for("visualizar_datos"))


@app.route("/database/descargar/<nombre>")
@login_requerido
def descargar_archivo_db(nombre):
    ruta = DATABASE_DIR / nombre
    if not ruta.exists():
        return redirect(url_for("ver_database"))
    return send_file(ruta, as_attachment=True, download_name=nombre)


@app.route("/database/json")
@login_requerido
def database_json():
    """
    Devuelve archivos para el modal de reentrenamiento.
    Si ?solo_revisados=1, devuelve los análisis del HISTORIAL con revisión completa.
    Sin ese parámetro, devuelve los archivos del database normal.
    """
    solo_revisados = request.args.get("solo_revisados", "0") == "1"

    if solo_revisados:
        # Buscar en el HISTORIAL los análisis con revisión 100% completa
        idx_path = HISTORIAL_DIR / "index.json"
        revisados = []
        if idx_path.exists():
            with open(idx_path) as f:
                entradas = json.load(f)
            for e in entradas:
                ruta = HISTORIAL_DIR / e["archivo"]
                if not ruta.exists():
                    continue
                try:
                    df = pd.read_csv(ruta)
                    tiene_gt = "fraude" in df.columns and "prediccion" in df.columns
                    if not tiene_gt:
                        continue
                    y_real = df["fraude"].astype(int)
                    y_pred = df["prediccion"].astype(int)
                    n_err  = int((y_real != y_pred).sum())
                    n_rev  = int(df["etiqueta_usuario"].notna().sum())                              if "etiqueta_usuario" in df.columns else 0
                    if n_err > 0 and n_rev >= n_err:
                        revisados.append({
                            "nombre":     e["archivo"],
                            "nombre_orig": e["nombre"],
                            "filas":      e.get("n_filas", len(df)),
                            "fecha":      e.get("fecha", ""),
                            "ruta":       str(ruta),   # ruta real en historial/
                        })
                except Exception:
                    continue
        return jsonify({"archivos": revisados})

    return jsonify({"archivos": retraining.listar_database()})


@app.route("/database/eliminar/<nombre>", methods=["POST"])
@login_requerido
def eliminar_archivo_db(nombre):
    ok = retraining.eliminar_de_database(nombre)
    return jsonify({"ok": ok})


# ── Cargar archivos ────────────────────────────────────────────────────────

@app.route("/cargar", methods=["GET", "POST"])
@login_requerido
def cargar_archivo():
    from modules.column_detector import analizar_columnas, aplicar_mapeo

    mensaje     = None
    error       = None
    mapeo_manual = None   # solo se llena si el estado es parcial/vacio

    if request.method == "POST":
        archivo     = request.files.get("archivo")
        descripcion = request.form.get("descripcion", "").strip()

        if not archivo or archivo.filename == "":
            error = "No se seleccionó ningún archivo."
        elif not archivo.filename.endswith(".csv"):
            error = "Solo se aceptan archivos .csv"
        else:
            try:
                df = pd.read_csv(archivo)
                resultado = analizar_columnas(list(df.columns))
                estado    = resultado["estado"]

                if estado == "ok":
                    # Carga directa sin cambios
                    _finalizar_carga(df, archivo.filename, descripcion)
                    mensaje = f"✅ Archivo cargado: {len(df):,} filas."

                elif estado == "auto":
                    # Renombrado automático — informar cambios
                    df = aplicar_mapeo(df, resultado["mapeo"])
                    _finalizar_carga(df, archivo.filename, descripcion)
                    cambios = "<br>".join(
                        f"&nbsp;&nbsp;• <code>{k}</code> → <code>{v}</code>"
                        for k, v in resultado["renombrados"].items()
                    )
                    mensaje = (f"✅ Archivo cargado: {len(df):,} filas.<br>"
                               f"Se renombraron automáticamente estas columnas:<br>{cambios}")

                elif estado in ("parcial", "vacio"):
                    # Guardar CSV temporal y pedir asignación manual
                    tmp = UPLOADS_DIR / "_tmp_carga.csv"
                    df.to_csv(tmp, index=False)
                    session["_tmp_carga"]        = str(tmp)
                    session["_tmp_nombre"]       = archivo.filename
                    session["_tmp_descripcion"]  = descripcion
                    session["_tmp_mapeo"]        = resultado["mapeo"]
                    mapeo_manual = {
                        "faltantes":       resultado["faltantes"],
                        "encontradas":     resultado["mapeo"],
                        "renombrados":     resultado["renombrados"],
                        "columnas_csv":    list(df.columns),
                        "mensaje":         resultado["mensaje"],
                    }

                else:  # imposible
                    error = resultado["mensaje"]

            except Exception as e:
                error = f"Error al leer el archivo: {str(e)}"

    return render_template("cargar.html", usuario=session["usuario"],
                           mensaje=mensaje, error=error,
                           mapeo_manual=mapeo_manual,
                           alerta_balanceo=session.pop("alerta_balanceo", None))


def _finalizar_carga(df, nombre_archivo: str, descripcion: str):
    """Guarda el DataFrame como activo y en la database."""
    # Estadísticas ANTES de cualquier transformación
    stats_antes = obtener_estadisticas(df.copy())

    # Detectar outliers (IQR) y marcarlos
    df_con_outliers = detectar_outliers(df.copy())
    n_outliers = int(df_con_outliers["es_outlier"].sum())

    # Guardar stats en sesión para mostrar en datos.html
    session["stats_carga"] = {
        "n_filas":        stats_antes["total_filas"],
        "n_columnas":     stats_antes["total_columnas"],
        "valores_nulos":  stats_antes["valores_nulos"],
        "n_outliers":     n_outliers,
        "n_fraudes":      stats_antes.get("fraudes", None),
        "n_legitimas":    stats_antes.get("legitimas", None),
        "pct_fraude":     stats_antes.get("pct_fraude", None),
    }

    ruta = UPLOADS_DIR / "ultimo_archivo.csv"
    df.to_csv(ruta, index=False)
    session["archivo_cargado"]   = str(ruta)
    session["n_filas"]           = len(df)
    session["n_cols"]            = len(df.columns)
    session["carga_fresca"]      = True
    session["resultados_path"]   = None
    entrada = retraining.guardar_en_database(
        df, nombre_archivo, descripcion or nombre_archivo
    )
    session["db_nombre_activo"] = entrada["nombre"]

    # Advertencia de desbalance de clases (RF-06 / RNF-06)
    alerta_balanceo = None
    if stats_antes.get("pct_fraude") is not None:
        pct = stats_antes["pct_fraude"]
        if pct < 5:
            alerta_balanceo = (f"⚠️ Dataset muy desbalanceado: solo {pct}% son fraudes "
                               f"({stats_antes.get('fraudes',0):,} de {stats_antes['total_filas']:,} registros). "
                               "El modelo puede tener sesgo hacia predecir legítimas.")
        elif pct > 50:
            alerta_balanceo = (f"⚠️ Dataset inusual: {pct}% son fraudes "
                               f"({stats_antes.get('fraudes',0):,} de {stats_antes['total_filas']:,} registros). "
                               "En datos reales el fraude no supera el 10-15%.")
    session["alerta_balanceo"] = alerta_balanceo


@app.route("/cargar/confirmar_mapeo", methods=["POST"])
@login_requerido
def confirmar_mapeo():
    """Recibe el mapeo manual del usuario y finaliza la carga."""
    from modules.column_detector import COLUMNAS_REQUERIDAS

    tmp       = session.get("_tmp_carga")
    nombre    = session.get("_tmp_nombre", "archivo.csv")
    desc      = session.get("_tmp_descripcion", "")
    mapeo_ant = session.get("_tmp_mapeo", {})

    if not tmp or not Path(tmp).exists():
        return redirect(url_for("cargar_archivo"))

    df = pd.read_csv(tmp)

    # Construir mapeo completo: automático + manual
    mapeo_completo = dict(mapeo_ant)   # col_req → col_csv

    for col_req in COLUMNAS_REQUERIDAS:
        if col_req not in mapeo_completo:
            asignada = request.form.get(f"mapeo_{col_req}", "").strip()
            if asignada and asignada in df.columns:
                mapeo_completo[col_req] = asignada

    # Verificar que todas las columnas requeridas estén asignadas
    from modules.column_detector import aplicar_mapeo
    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in mapeo_completo]
    if faltantes:
        return render_template("cargar.html",
                               usuario=session["usuario"],
                               mensaje=None,
                               error=f"Aún faltan columnas sin asignar: {', '.join(faltantes)}",
                               mapeo_manual=None)

    df = aplicar_mapeo(df, mapeo_completo)
    _finalizar_carga(df, nombre, desc)

    # Limpiar temporales
    Path(tmp).unlink(missing_ok=True)
    for k in ["_tmp_carga","_tmp_nombre","_tmp_descripcion","_tmp_mapeo"]:
        session.pop(k, None)

    return redirect(url_for("cargar_archivo"))


# ── Visualizar datos ───────────────────────────────────────────────────────

@app.route("/datos")
@login_requerido
def visualizar_datos():
    archivo = session.get("archivo_cargado")
    carga_fresca = session.get("carga_fresca", False)
    if not archivo or not Path(archivo).exists() or not carga_fresca:
        return redirect(url_for("cargar_archivo"))
    df       = pd.read_csv(archivo)
    busqueda = request.args.get("busqueda", "").strip()
    pagina   = int(request.args.get("pagina", 1))
    pp       = 20
    if busqueda:
        mask = df.apply(lambda r: r.astype(str).str.contains(busqueda, case=False).any(), axis=1)
        df   = df[mask]
    total = len(df)
    df_page = df.iloc[(pagina-1)*pp : pagina*pp]
    total_p = max(1, (total + pp - 1) // pp)
    return render_template("datos.html", usuario=session["usuario"],
                           tabla=df_page.to_html(classes="tabla-datos", index=False, border=0),
                           total=total, pagina=pagina, total_paginas=total_p,
                           busqueda=busqueda,
                           db_nombre=session.get("db_nombre_activo", ""),
                           stats=session.get("stats_carga", {}))


# ── Ejecutar modelo ────────────────────────────────────────────────────────

@app.route("/ejecutar", methods=["GET", "POST"])
@login_requerido
def ejecutar_modelo():
    archivo      = session.get("archivo_cargado")
    carga_fresca = session.get("carga_fresca", False)
    modelos_disponibles = retraining.listar_modelos_disponibles()

    if not archivo or not Path(archivo).exists() or not carga_fresca:
        return redirect(url_for("cargar_archivo"))

    if request.method == "POST":
        modelo_sel  = request.form.get("modelo_sel", str(MODEL_PATH))
        nombre_anal = request.form.get("nombre_analisis", "").strip()
        ruta_modelo = Path(modelo_sel)

        if not nombre_anal:
            return render_template("ejecutar.html", usuario=session["usuario"],
                                   error="Escribe un nombre para este análisis antes de ejecutar.",
                                   n_filas=session.get("n_filas", 0),
                                   modelos=modelos_disponibles,
                                   modelo_activo=str(MODEL_PATH))

        pipe = cargar_modelo(ruta_modelo)
        if pipe is None:
            return render_template("ejecutar.html", usuario=session["usuario"],
                                   error=f"Modelo no encontrado: {ruta_modelo.name}",
                                   n_filas=0, modelos=modelos_disponibles,
                                   modelo_activo=str(MODEL_PATH))

        df = pd.read_csv(archivo)
        df_proc, err = preprocesar_dataframe(df.copy())
        if err:
            return render_template("ejecutar.html", usuario=session["usuario"],
                                   error=err, n_filas=session.get("n_filas", 0),
                                   modelos=modelos_disponibles,
                                   modelo_activo=str(MODEL_PATH))
        try:
            probabilidades = pipe.predict_proba(df_proc)[:, 1]
            predicciones   = pipe.predict(df_proc)
        except Exception as e:
            return render_template("ejecutar.html", usuario=session["usuario"],
                                   error=f"Error en predicción: {str(e)}",
                                   n_filas=session.get("n_filas", 0),
                                   modelos=modelos_disponibles,
                                   modelo_activo=str(MODEL_PATH))

        df["prediccion"]    = predicciones
        df["probabilidad"]  = np.round(probabilidades, 4)
        df["clasificacion"] = df["prediccion"].map({0: "Legítima", 1: "⚠️ Fraudulenta"})

        # Guardar resultado activo
        ruta_res = UPLOADS_DIR / "resultados.csv"
        df.to_csv(ruta_res, index=False)
        session["resultados_path"] = str(ruta_res)
        session["modelo_usado"]    = ruta_modelo.name

        # Guardar en historial
        entrada = _guardar_historial(
            df,
            nombre=nombre_anal,
            modelo_nombre=ruta_modelo.name,
            db_nombre=session.get("db_nombre_activo", "")
        )
        session["historial_nombre_activo"] = entrada["nombre"]

        return redirect(url_for("ver_resultados"))

    return render_template("ejecutar.html", usuario=session["usuario"],
                           n_filas=session.get("n_filas", 0), error=None,
                           modelos=modelos_disponibles,
                           modelo_activo=str(MODEL_PATH))


# ── Resultados ─────────────────────────────────────────────────────────────

@app.route("/resultados")
@login_requerido
def ver_resultados():
    ruta = session.get("resultados_path")
    # resultados_path = None significa que fue limpiado al cargar nuevo archivo
    if ruta is None or not Path(str(ruta)).exists():
        return redirect(url_for("ejecutar_modelo"))

    df     = pd.read_csv(ruta)
    pagina = int(request.args.get("pagina", 1))
    tab    = request.args.get("tab", "fraudes")
    pp_raw = request.args.get("pp", "20")
    pp     = int(pp_raw) if pp_raw in ("20","50","100") else None  # None = todos

    tiene_gt = "fraude" in df.columns

    if tiene_gt:
        y_real = df["fraude"].astype(int)
        y_pred = df["prediccion"].astype(int)
        # Con ground truth: Fraudes = TP, Legítimas = TN (excluir FP y FN)
        n_fraudes   = int(((y_real==1) & (y_pred==1)).sum())   # TP
        n_legitimas = int(((y_real==0) & (y_pred==0)).sum())   # TN
    else:
        n_fraudes   = int((df["prediccion"] == 1).sum())
        n_legitimas = int((df["prediccion"] == 0).sum())

    if tab == "legitimas":
        if tiene_gt:
            # Solo TN (era legítima Y el modelo lo dijo bien)
            y_real = df["fraude"].astype(int)
            y_pred = df["prediccion"].astype(int)
            df_tab = df[(y_real==0) & (y_pred==0)].copy()
        else:
            df_tab = df[df["prediccion"] == 0].copy()
    elif tab == "fraudes":
        if tiene_gt:
            # Solo TP (era fraude Y el modelo lo detectó)
            y_real = df["fraude"].astype(int)
            y_pred = df["prediccion"].astype(int)
            df_tab = df[(y_real==1) & (y_pred==1)].copy()
        else:
            df_tab = df[df["prediccion"] == 1].copy()
    elif tab == "analisis":
        df_tab = df.copy()
    elif tab == "manual":
        filtro_manual = request.args.get("filtro_manual", "todos")
        if tiene_gt:
            y_real = df["fraude"].astype(int)
            y_pred = df["prediccion"].astype(int)
            if filtro_manual == "fn":
                df_tab = df[(y_real==1) & (y_pred==0)].copy()
            elif filtro_manual == "fp":
                df_tab = df[(y_real==0) & (y_pred==1)].copy()
            else:  # todos los errores
                df_tab = df[(y_real!=y_pred)].copy()
        else:
            df_tab = pd.DataFrame()  # sin ground truth → tabla vacía
    else:
        df_tab = df[df["prediccion"] == 1].copy()

    total   = len(df_tab)
    if pp is None:
        # Mostrar todos
        df_page  = df_tab.copy()
        total_p  = 1
        pagina   = 1
    else:
        inicio   = (pagina-1)*pp
        df_page  = df_tab.iloc[inicio : inicio + pp].copy()
        total_p  = max(1, (total + pp - 1) // pp)
    # _idx = índice real en el DataFrame completo, lo usa el JS para colorear filas
    df_page.insert(0, "_idx", df_page.index)

    # Mostrar etiqueta_usuario con valor legible
    if "etiqueta_usuario" in df_page.columns:
        df_page["etiqueta_usuario"] = df_page["etiqueta_usuario"].map(
            {1: "✔ Fraude (1)", 0: "✔ Legítima (0)"}
        ).fillna("— sin revisar")
    else:
        df_page["etiqueta_usuario"] = "— sin revisar"

    analisis = None
    if tiene_gt:
        y_real = df["fraude"].astype(int)
        y_pred = df["prediccion"].astype(int)
        tp = int(((y_real==1)&(y_pred==1)).sum())
        fp = int(((y_real==0)&(y_pred==1)).sum())
        tn = int(((y_real==0)&(y_pred==0)).sum())
        fn = int(((y_real==1)&(y_pred==0)).sum())
        analisis = {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(tp/(tp+fp),4) if (tp+fp)>0 else 0,
            "recall":    round(tp/(tp+fn),4) if (tp+fn)>0 else 0,
        }
        if tab == "analisis":
            df_page["resultado"] = "—"
            df_page.loc[(df_page["fraude"]==1)&(df_page["prediccion"]==1),"resultado"] = "✅ Verdadero Positivo"
            df_page.loc[(df_page["fraude"]==0)&(df_page["prediccion"]==1),"resultado"] = "❌ Falso Positivo"
            df_page.loc[(df_page["fraude"]==0)&(df_page["prediccion"]==0),"resultado"] = "✅ Verdadero Negativo"
            df_page.loc[(df_page["fraude"]==1)&(df_page["prediccion"]==0),"resultado"] = "❌ Falso Negativo"

    n_confirmados = 0
    if "marcacion_usuario" in df.columns:
        n_confirmados = int((df["marcacion_usuario"]=="verdadero_positivo").sum())

    # Mapa de índices ya marcados para colorear filas sin abrir el modal
    marcados_idx = {}
    if "marcacion_usuario" in df.columns:
        for i, val in df["marcacion_usuario"].items():
            if pd.notna(val) and val in ("verdadero_positivo", "falso_positivo"):
                marcados_idx[int(i)] = str(val)

    return render_template("resultados.html",
                           usuario=session["usuario"],
                           tabla=df_page.to_html(classes="tabla-datos", index=False, border=0),
                           n_fraudes=n_fraudes, n_legitimas=n_legitimas,
                           total=total, pagina=pagina, total_paginas=total_p,
                           tab=tab, tiene_ground_truth=tiene_gt,
                           analisis=analisis, n_confirmados=n_confirmados,
                           modelo_usado=session.get("modelo_usado","—"),
                           filtro_manual=request.args.get("filtro_manual","todos"),
                           pp=pp_raw,
                           nombre_analisis=session.get("historial_nombre_activo",""),
                           marcados_idx=marcados_idx)


# ── Marcar transacción ─────────────────────────────────────────────────────

@app.route("/marcar", methods=["POST"])
@login_requerido
def marcar_transaccion():
    data      = request.get_json()
    indice    = data.get("indice")
    marcacion = data.get("marcacion")
    ruta      = session.get("resultados_path")
    if not ruta or not Path(ruta).exists():
        return jsonify({"ok": False, "msg": "No hay resultados cargados."})
    df = pd.read_csv(ruta)
    if indice is None or indice >= len(df):
        return jsonify({"ok": False, "msg": "Índice inválido."})
    df.loc[indice, "marcacion_usuario"] = marcacion
    if "etiqueta_usuario" not in df.columns:
        df["etiqueta_usuario"] = None
    df.loc[indice, "etiqueta_usuario"] = 1 if marcacion == "verdadero_positivo" else 0
    df.to_csv(ruta, index=False)
    guardar_retroalimentacion({"timestamp": datetime.now().isoformat(),
                               "usuario": session["usuario"],
                               "indice": indice, "marcacion": marcacion})
    return jsonify({"ok": True, "msg": f"Marcado como {marcacion}."})



@app.route("/marcar/similares", methods=["POST"])
@login_requerido
def buscar_similares():
    data   = request.get_json()
    indice = data.get("indice")
    ruta   = session.get("resultados_path")
    if not ruta or not Path(ruta).exists():
        return jsonify({"ok": False, "msg": "No hay resultados."})
    df = pd.read_csv(ruta)
    if indice is None or indice >= len(df):
        return jsonify({"ok": False, "msg": "Índice inválido."})
    ref      = df.iloc[indice]
    prob_ref = float(ref.get("probabilidad", 0.5)) if "probabilidad" in df.columns else 0.5

    # Criterio A: rango de probabilidad ±0.10
    if "probabilidad" in df.columns:
        mask_prob = ((df["probabilidad"] >= prob_ref - 0.10) &
                     (df["probabilidad"] <= prob_ref + 0.10))
    else:
        mask_prob = pd.Series([True] * len(df))

    # Criterio B: al menos 2 de 3 columnas clave coinciden
    puntos = pd.Series(0, index=df.index)
    if "Estado_tarjeta" in df.columns and pd.notna(ref.get("Estado_tarjeta")):
        puntos += (df["Estado_tarjeta"] == ref["Estado_tarjeta"]).astype(int)
    if "Localizacion_tarjeta" in df.columns and pd.notna(ref.get("Localizacion_tarjeta")):
        puntos += (df["Localizacion_tarjeta"] == ref["Localizacion_tarjeta"]).astype(int)
    if "Cuotas_mora" in df.columns and pd.notna(ref.get("Cuotas_mora")):
        puntos += (abs(df["Cuotas_mora"] - float(ref["Cuotas_mora"])) <= 1).astype(int)

    mask = mask_prob & (puntos >= 2)
    mask.iloc[indice] = False  # excluir la propia fila
    indices = df.index[mask].tolist()

    cols = [c for c in ["Estado_tarjeta","Localizacion_tarjeta","Cuotas_mora",
                        "probabilidad","clasificacion"] if c in df.columns]
    preview = df.loc[indices[:5], cols].to_html(
        classes="tabla-datos tabla-mini", index=False, border=0
    ) if indices else ""

    return jsonify({"ok": True, "n": len(indices),
                    "indices": indices, "preview": preview})


@app.route("/marcar/lote", methods=["POST"])
@login_requerido
def marcar_lote():
    data      = request.get_json()
    indices   = data.get("indices", [])
    marcacion = data.get("marcacion")
    ruta      = session.get("resultados_path")
    if not ruta or not Path(ruta).exists():
        return jsonify({"ok": False, "msg": "No hay resultados."})
    df = pd.read_csv(ruta)
    if "etiqueta_usuario" not in df.columns:
        df["etiqueta_usuario"] = None
    if "marcacion_usuario" not in df.columns:
        df["marcacion_usuario"] = None
    for idx in indices:
        if 0 <= idx < len(df):
            df.loc[idx, "marcacion_usuario"] = marcacion
            df.loc[idx, "etiqueta_usuario"]  = 1 if marcacion == "verdadero_positivo" else 0
    df.to_csv(ruta, index=False)
    return jsonify({"ok": True, "n": len(indices),
                    "msg": f"{len(indices)} transacciones marcadas."})


@app.route("/marcar/estado_revision")
@login_requerido
def estado_revision():
    ruta = session.get("resultados_path")
    if not ruta or not Path(ruta).exists():
        return jsonify({"revisadas": 0, "total_errores": 0,
                        "puede_reentrenar": False, "pct_revision": 0})
    df = pd.read_csv(ruta)
    revisadas     = int(df["etiqueta_usuario"].notna().sum())                     if "etiqueta_usuario" in df.columns else 0
    total_errores = 0
    if "fraude" in df.columns and "prediccion" in df.columns:
        y_real = df["fraude"].astype(int)
        y_pred = df["prediccion"].astype(int)
        total_errores = int((y_real != y_pred).sum())
    pct = round(revisadas / total_errores * 100, 1) if total_errores > 0 else 0
    return jsonify({"revisadas": revisadas, "total_errores": total_errores,
                    "puede_reentrenar": revisadas > 0, "pct_revision": pct})


# ── Historial ──────────────────────────────────────────────────────────────

def _recalcular_revision(entrada: dict) -> dict:
    """Recalcula el estado de revisión leyendo el CSV actual del historial."""
    ruta = HISTORIAL_DIR / entrada["archivo"]
    if not ruta.exists():
        return entrada
    try:
        df = pd.read_csv(ruta)
        tiene_gt = "fraude" in df.columns and "prediccion" in df.columns
        if tiene_gt:
            y_real = df["fraude"].astype(int)
            y_pred = df["prediccion"].astype(int)
            n_err  = int((y_real != y_pred).sum())
            n_rev  = int(df["etiqueta_usuario"].notna().sum())                      if "etiqueta_usuario" in df.columns else 0
            entrada["tiene_gt"]         = True
            entrada["n_errores"]        = n_err
            entrada["n_revisadas"]      = n_rev
            entrada["pct_revision"]     = round(n_rev / n_err * 100, 1) if n_err > 0 else 0
            entrada["revision_completa"] = n_err > 0 and n_rev >= n_err
        else:
            entrada["tiene_gt"]          = False
            entrada["revision_completa"] = False
    except Exception:
        pass
    return entrada


@app.route("/historial")
@login_requerido
def ver_historial():
    idx_path = HISTORIAL_DIR / "index.json"
    entradas = []
    if idx_path.exists():
        with open(idx_path) as f:
            entradas = json.load(f)
    entradas = [_recalcular_revision(e) for e in entradas]
    entradas = list(reversed(entradas))   # más recientes primero

    # Filtros
    filtro_estado = request.args.get("estado", "todos")
    busqueda      = request.args.get("q", "").strip().lower()

    if filtro_estado == "completo":
        entradas = [e for e in entradas if e.get("revision_completa")]
    elif filtro_estado == "pendiente":
        entradas = [e for e in entradas if e.get("tiene_gt") and not e.get("revision_completa")]
    elif filtro_estado == "sin_gt":
        entradas = [e for e in entradas if not e.get("tiene_gt")]

    if busqueda:
        entradas = [e for e in entradas
                    if busqueda in e.get("nombre","").lower()
                    or busqueda in e.get("modelo","").lower()
                    or busqueda in e.get("database","").lower()]

    # Paginación
    pp      = 10
    pagina  = int(request.args.get("pagina", 1))
    total   = len(entradas)
    total_p = max(1, (total + pp - 1) // pp)
    pagina  = max(1, min(pagina, total_p))
    entradas_pag = entradas[(pagina-1)*pp : pagina*pp]

    return render_template("historial.html",
                           usuario=session["usuario"],
                           entradas=entradas_pag,
                           total=total, pagina=pagina, total_paginas=total_p,
                           filtro_estado=filtro_estado, busqueda=busqueda)


@app.route("/historial/cargar/<archivo>")
@login_requerido
def cargar_historial(archivo):
    """Carga un resultado histórico como resultado activo."""
    ruta = HISTORIAL_DIR / archivo
    if not ruta.exists():
        return redirect(url_for("ver_historial"))

    # Leer metadata del índice
    idx_path = HISTORIAL_DIR / "index.json"
    entrada  = {}
    if idx_path.exists():
        with open(idx_path) as f:
            for e in json.load(f):
                if e["archivo"] == archivo:
                    entrada = e
                    break

    session["resultados_path"]          = str(ruta)
    session["modelo_usado"]             = entrada.get("modelo", "—")
    session["historial_nombre_activo"]  = entrada.get("nombre", archivo)
    session["db_nombre_activo"]         = entrada.get("database", "")
    return redirect(url_for("ver_resultados"))


@app.route("/historial/descargar/<archivo>")
@login_requerido
def descargar_historial(archivo):
    ruta = HISTORIAL_DIR / archivo
    if not ruta.exists():
        return redirect(url_for("ver_historial"))
    return send_file(ruta, as_attachment=True, download_name=archivo)


@app.route("/historial/eliminar/<archivo>", methods=["POST"])
@login_requerido
def eliminar_historial(archivo):
    ruta     = HISTORIAL_DIR / archivo
    idx_path = HISTORIAL_DIR / "index.json"
    if ruta.exists():
        ruta.unlink()
    if idx_path.exists():
        with open(idx_path) as f:
            idx = json.load(f)
        idx = [e for e in idx if e["archivo"] != archivo]
        with open(idx_path, "w") as f:
            json.dump(idx, f, indent=4)
    return jsonify({"ok": True})


@app.route("/resultados/errores_json")
@login_requerido
def errores_json():
    """Retorna FP y FN del resultado actual como JSON para la pestaña Manual."""
    ruta = session.get("resultados_path")
    if not ruta or not Path(ruta).exists():
        return jsonify({"ok": False})
    df = pd.read_csv(ruta)
    if "fraude" not in df.columns or "prediccion" not in df.columns:
        return jsonify({"ok": False, "msg": "Sin ground truth"})
    y_real = df["fraude"].astype(int)
    y_pred = df["prediccion"].astype(int)
    df_fn = df[(y_real==1)&(y_pred==0)].copy()
    df_fp = df[(y_real==0)&(y_pred==1)].copy()
    df_fn["tipo_error"] = "⚠️ Falso Negativo"
    df_fp["tipo_error"] = "❌ Falso Positivo"
    return jsonify({
        "ok": True,
        "fn": df_fn.head(50).to_html(classes="tabla-datos", index=False, border=0),
        "fp": df_fp.head(50).to_html(classes="tabla-datos", index=False, border=0),
        "n_fn": len(df_fn), "n_fp": len(df_fp),
    })


# ── Métricas ───────────────────────────────────────────────────────────────

@app.route("/metricas")
@login_requerido
def ver_metricas():
    from modules.retraining import (listar_modelos_disponibles,
                                     obtener_graficas_modelo)
    modelos    = listar_modelos_disponibles()
    modelo_sel = request.args.get("modelo", "")

    # Modelo por defecto: el primero (original)
    if not modelo_sel and modelos:
        modelo_sel = modelos[0]["nombre"]

    # Cargar métricas e imágenes del modelo seleccionado
    graficas = obtener_graficas_modelo(modelo_sel) if modelo_sel else {}

    metricas = {}
    mp = graficas.get("metrics_path", "")
    if mp and Path(mp).exists():
        with open(mp) as f:
            metricas = json.load(f)

    # Convertir imágenes a base64
    imagenes = {}
    for key in ["roc_curve", "pr_curve", "confusion_matrix", "feature_importances"]:
        ruta_img = graficas.get(key, "")
        if ruta_img and Path(ruta_img).exists():
            with open(ruta_img, "rb") as img_f:
                imagenes[key] = base64.b64encode(img_f.read()).decode("utf-8")

    # Calcular especificidad = TN / (TN + FP)  (RF-07, mencionada en el Word)
    especificidad = None
    cr = metricas.get("classification_report", {})
    if cr:
        soporte_0 = cr.get("0", {}).get("support", 0)
        recall_0  = cr.get("0", {}).get("recall", 0)
        tn = soporte_0 * recall_0
        fp = soporte_0 * (1 - recall_0)
        if (tn + fp) > 0:
            especificidad = round(tn / (tn + fp), 4)

    return render_template("metricas.html",
                           usuario=session["usuario"],
                           metricas=metricas,
                           imagenes=imagenes,
                           modelos=modelos,
                           modelo_sel=modelo_sel,
                           especificidad=especificidad)


# ── Matriz de confusión ────────────────────────────────────────────────────

@app.route("/confusion")
@login_requerido
def ver_confusion():
    """Redirige a la página de métricas que ya incluye la matriz."""
    return redirect(url_for("ver_metricas"))


# ── Reportes ───────────────────────────────────────────────────────────────

@app.route("/reporte/fraude")
@login_requerido
def reporte_fraude():
    ruta = session.get("resultados_path")
    if not ruta or not Path(ruta).exists():
        return redirect(url_for("ver_resultados"))
    df = pd.read_csv(ruta)
    ruta_rep = generar_reporte_fraude(df[df["prediccion"]==1], UPLOADS_DIR)
    return send_file(ruta_rep, as_attachment=True,
                     download_name=f"reporte_fraude_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")


@app.route("/reporte/rendimiento")
@login_requerido
def reporte_rendimiento():
    metricas = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            metricas = json.load(f)
    ruta_rep = generar_reporte_rendimiento(metricas, cargar_retroalimentacion(), UPLOADS_DIR)
    return send_file(ruta_rep, as_attachment=True,
                     download_name=f"reporte_rendimiento_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")


# ── Plantilla CSV ──────────────────────────────────────────────────────────

@app.route("/plantilla")
@login_requerido
def descargar_plantilla():
    datos = [
        {"Numero_tarjeta":"A1B2C3","Tipo_tarjeta":"clasica","Estado_tarjeta":"activa",
         "Fecha_vencimiento":"2026-08-15","Indicador_repositorio":0,"Localizacion_tarjeta":"nacional",
         "Acumulado_cupo":0.35,"Fecha_ult_retiro":"2025-08-01","Cuotas_mora":0,
         "Ind_estado":"al dia","Es_amparada":0,"Es_reexpedicion":0,"Tipo_nomina":"privada"},
        {"Numero_tarjeta":"D4E5F6","Tipo_tarjeta":"oro","Estado_tarjeta":"bloqueada",
         "Fecha_vencimiento":"2024-03-10","Indicador_repositorio":1,"Localizacion_tarjeta":"internacional",
         "Acumulado_cupo":0.95,"Fecha_ult_retiro":"","Cuotas_mora":4,
         "Ind_estado":"en mora","Es_amparada":1,"Es_reexpedicion":1,"Tipo_nomina":"independiente"},
    ]
    buf = io.StringIO()
    pd.DataFrame(datos).to_csv(buf, index=False)
    buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode("utf-8-sig")),
                     as_attachment=True, download_name="plantilla_transacciones.csv",
                     mimetype="text/csv")


# ── Modelos disponibles ────────────────────────────────────────────────────

@app.route("/modelos")
@login_requerido
def ver_modelos():
    from modules.retraining import listar_modelos_disponibles, obtener_graficas_modelo
    modelos = listar_modelos_disponibles()

    # Cargar métricas de cada modelo para el ranking (RF-10 / RNF-10)
    for m in modelos:
        graficas = obtener_graficas_modelo(m["nombre"])
        mp = graficas.get("metrics_path", "")
        if mp and Path(mp).exists():
            with open(mp) as f:
                met = json.load(f)
            cr = met.get("classification_report", {})
            m["roc_auc"]   = round(met.get("roc_auc", 0), 4)
            m["pr_auc"]    = round(met.get("pr_auc",  0), 4)
            m["accuracy"]  = round(cr.get("accuracy", 0) * 100, 2)
            m["f1_fraude"] = round(cr.get("1", {}).get("f1-score", 0), 4)
            m["precision_fraude"] = round(cr.get("1", {}).get("precision", 0), 4)
            m["recall_fraude"]    = round(cr.get("1", {}).get("recall",    0), 4)
            tn = cr.get("0", {}).get("support", 0) * cr.get("0", {}).get("recall", 0)
            fp = cr.get("0", {}).get("support", 0) * (1 - cr.get("0", {}).get("recall", 0))
            m["especificidad"] = round(tn / (tn + fp), 4) if (tn + fp) > 0 else None
            m["tiene_metricas"] = True
        else:
            m["roc_auc"] = m["pr_auc"] = m["accuracy"] = m["f1_fraude"] = None
            m["precision_fraude"] = m["recall_fraude"] = m["especificidad"] = None
            m["tiene_metricas"] = False

    # Ordenar: modelos con métricas primero, por ROC AUC desc
    modelos.sort(key=lambda x: (x["tiene_metricas"], x["roc_auc"] or 0), reverse=True)

    return render_template("modelos.html",
                           usuario=session["usuario"], modelos=modelos)


@app.route("/modelos/exportar_ranking")
@login_requerido
def exportar_ranking_modelos():
    """Exporta el ranking de modelos como CSV (RNF-10)."""
    from modules.retraining import listar_modelos_disponibles, obtener_graficas_modelo
    modelos = listar_modelos_disponibles()
    filas = []
    for m in modelos:
        graficas = obtener_graficas_modelo(m["nombre"])
        mp = graficas.get("metrics_path", "")
        fila = {"Modelo": m["nombre"], "Tipo": "Original" if m["es_original"] else "Reentrenado",
                "Tamaño": m["tamaño"], "Fecha": m["fecha"],
                "ROC_AUC": None, "PR_AUC": None, "Accuracy": None,
                "F1_Fraude": None, "Precision_Fraude": None, "Recall_Fraude": None}
        if mp and Path(mp).exists():
            with open(mp) as f:
                met = json.load(f)
            cr = met.get("classification_report", {})
            fila["ROC_AUC"]          = round(met.get("roc_auc", 0), 4)
            fila["PR_AUC"]           = round(met.get("pr_auc",  0), 4)
            fila["Accuracy"]         = round(cr.get("accuracy", 0) * 100, 2)
            fila["F1_Fraude"]        = round(cr.get("1", {}).get("f1-score",  0), 4)
            fila["Precision_Fraude"] = round(cr.get("1", {}).get("precision", 0), 4)
            fila["Recall_Fraude"]    = round(cr.get("1", {}).get("recall",    0), 4)
        filas.append(fila)

    buf = io.StringIO()
    pd.DataFrame(filas).to_csv(buf, index=False)
    buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode("utf-8-sig")),
                     as_attachment=True,
                     download_name=f"ranking_modelos_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                     mimetype="text/csv")




@app.route("/modelos/eliminar", methods=["POST"])
@login_requerido
def eliminar_modelo():
    nombre = request.form.get("nombre", "").strip()
    if not nombre:
        return jsonify({"ok": False, "msg": "Nombre vacío."})
    from modules.retraining import MODELOS_DIR, MODEL_PATH
    ruta = MODELOS_DIR / nombre
    # Solo se puede eliminar modelos de la carpeta reentrenados, nunca el original
    if not ruta.exists() or ruta == MODEL_PATH:
        return jsonify({"ok": False, "msg": "No se puede eliminar ese modelo."})
    try:
        ruta.unlink()
        modelo_cache.pop(str(ruta), None)
        return jsonify({"ok": True, "msg": f"Modelo '{nombre}' eliminado."})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})

# ── Diagnóstico ────────────────────────────────────────────────────────────

@app.route("/diagnostico")
@login_requerido
def diagnostico_modelo():
    info = {"model_path": str(MODEL_PATH), "model_exists": MODEL_PATH.exists()}
    pipe = cargar_modelo()
    if pipe:
        try:
            cols = []
            for _, _, c in pipe.named_steps["pre"].transformers_:
                cols.extend(c)
            info["columnas_modelo"] = cols
        except Exception as e:
            info["error"] = str(e)
    ruta = session.get("archivo_cargado")
    if ruta and Path(ruta).exists():
        df_tmp = pd.read_csv(ruta, nrows=1)
        info["columnas_csv"] = list(df_tmp.columns)
        if "columnas_modelo" in info:
            info["faltantes"] = [c for c in info["columnas_modelo"] if c not in info["columnas_csv"]]
            info["extras"]    = [c for c in info["columnas_csv"] if c not in info["columnas_modelo"]]
    return jsonify(info)


# ── Reentrenamiento ────────────────────────────────────────────────────────

@app.route("/reentrenar/iniciar", methods=["POST"])
@login_requerido
def iniciar_reentrenamiento():
    import uuid

    modo           = request.form.get("modo", "errores")
    complejidad    = request.form.get("complejidad", "completo")
    nombre_modelo  = request.form.get("nombre_modelo", "").strip()
    datasets_sel   = request.form.getlist("datasets")   # nombres de archivos del db

    ruta = session.get("resultados_path")

    # Verificar revisión manual COMPLETA (100%) ANTES de reentrenar
    if ruta and Path(ruta).exists():
        df_chk   = pd.read_csv(ruta)
        tiene_gt = "fraude" in df_chk.columns and "prediccion" in df_chk.columns
        if tiene_gt:
            y_real          = df_chk["fraude"].astype(int)
            y_pred          = df_chk["prediccion"].astype(int)
            n_errores_total = int((y_real != y_pred).sum())
            n_revisadas     = int(df_chk["etiqueta_usuario"].notna().sum())                               if "etiqueta_usuario" in df_chk.columns else 0
            if n_errores_total > 0 and n_revisadas < n_errores_total:
                faltan = n_errores_total - n_revisadas
                pct    = round(n_revisadas / n_errores_total * 100, 1)
                return jsonify({
                    "ok": False,
                    "msg": (f"La revisión manual no está completa ({pct}% — "
                            f"faltan {faltan} de {n_errores_total} errores). "
                            "Ve a la pestaña Manual y etiqueta todas las transacciones "
                            "FP y FN antes de reentrenar.")
                })

    # Acumular errores según modo
    if ruta and Path(ruta).exists():
        df_res = pd.read_csv(ruta)
        if modo == "errores":
            info = retraining.acumular_errores(df_res)
        else:
            info = retraining.acumular_fraudes_detectados(df_res)
        if not info["ok"]:
            return jsonify({"ok": False, "msg": info["msg"]})
        n_err = info["n"]
    else:
        n_err = retraining.hay_errores_acumulados()["total"]
        if n_err == 0 and not datasets_sel:
            return jsonify({"ok": False, "msg": "No hay errores ni datos adicionales."})

    # Resolver rutas de datasets adicionales
    # Los datasets revisados vienen del historial/, no del database/
    rutas_extra = []
    for n in datasets_sel:
        ruta_hist = HISTORIAL_DIR / n
        ruta_db   = DATABASE_DIR  / n
        if ruta_hist.exists():
            rutas_extra.append(str(ruta_hist))
        elif ruta_db.exists():
            rutas_extra.append(str(ruta_db))

    cfg = COMPLEJIDADES.get(complejidad, COMPLEJIDADES["completo"])
    job_id = str(uuid.uuid4())[:8]
    _retrain_jobs[job_id] = {
        "progreso": 0, "mensajes": [], "resultado": None, "terminado": False
    }

    def run(jid):
        job = _retrain_jobs[jid]

        # Pasos simulados según complejidad
        pasos_base = [
            (5,  "📂 Cargando dataset base..."),
            (12, f"⚡ Incorporando {n_err} errores FP/FN con peso..."),
        ]
        if rutas_extra:
            pasos_base.append((18, f"➕ Agregando {len(rutas_extra)} dataset(s) del database..."))

        pasos_base += [
            (22, "🔀 Combinando y mezclando todos los datos..."),
            (26, "💾 Guardando backup del modelo anterior..."),
            (30, f"🌲 Iniciando entrenamiento ({cfg['n_estimators']} árboles)..."),
        ]

        if complejidad == "rapido":
            pasos_base += [
                (60, "⏳ Entrenando... (~1-2 min)"),
                (88, "📊 Evaluando métricas..."),
                (95, "💾 Guardando modelo..."),
            ]
        elif complejidad == "balanceado":
            pasos_base += [
                (45, "⏳ Entrenando... (~3-4 min)"),
                (65, "⏳ Construyendo árboles..."),
                (85, "📊 Evaluando métricas..."),
                (95, "💾 Guardando modelo..."),
            ]
        else:
            pasos_base += [
                (40, "⏳ Entrenando 400 árboles (~5-8 min)..."),
                (55, "⏳ A mitad del proceso..."),
                (75, "⏳ Casi listo..."),
                (88, "📊 Evaluando métricas..."),
                (95, "💾 Guardando modelo (~1.3 GB)..."),
            ]

        for pct, msg in pasos_base:
            job["progreso"] = pct
            job["mensajes"].append(msg)
            time.sleep(0.4)

        resultado = retraining.reentrenar_modelo(
            complejidad=complejidad,
            datasets_extra=rutas_extra,
            nombre_modelo=nombre_modelo or None,
        )
        job["resultado"] = resultado
        job["progreso"]  = 100
        job["terminado"] = True
        job["mensajes"].append(
            f"✅ Modelo '{resultado.get('nombre_modelo','?')}' listo." if resultado["ok"]
            else f"❌ Error: {resultado['mensaje']}"
        )
        # Refrescar caché
        modelo_cache.clear()
        cargar_modelo()

    threading.Thread(target=run, args=(job_id,), daemon=True).start()

    return jsonify({
        "ok":         True,
        "job_id":     job_id,
        "n":          n_err,
        "msg":        f"Iniciando reentrenamiento ({cfg['label']})...",
        "complejidad": cfg["label"],
    })


@app.route("/reentrenar/estado/<job_id>")
@login_requerido
def estado_reentrenamiento(job_id):
    job = _retrain_jobs.get(job_id)
    if not job:
        return jsonify({"ok": False, "msg": "Job no encontrado."})
    return jsonify({
        "ok":          True,
        "progreso":    job["progreso"],
        "mensajes":    job["mensajes"],
        "terminado":   job["terminado"],
        "resultado":   job["resultado"],
        "ruta_nuevo":  str(RETRAIN_MODEL_PATH),
        "ruta_backup": str(RETRAIN_MODEL_BACKUP),
    })


@app.route("/reentrenar/restaurar", methods=["POST"])
@login_requerido
def restaurar_backup():
    ok = restaurar_modelo_backup()
    modelo_cache.clear()
    cargar_modelo()
    return jsonify({"ok": ok,
                    "msg": "Modelo anterior restaurado." if ok
                           else "No se encontró backup."})


# ── Iniciar ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cargar_modelo()
    print("─" * 50)
    print("🚀 Servidor: http://127.0.0.1:5000")
    print("─" * 50)
    app.run(debug=True, port=5000)
