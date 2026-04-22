# modules/column_detector.py
#
# Detección inteligente de columnas para archivos CSV no estándar.
#
# FLUJO:
#   1. Coincidencia exacta         → carga normal
#   2. Fuzzy match automático      → renombra + informa
#   3. Parcial (algunas encontradas) → pregunta al usuario las faltantes
#   4. Sin coincidencias           → pregunta todas al usuario
#   5. Imposible                   → mensaje con recomendación

import re
import unicodedata

# ── Columnas requeridas con sinónimos conocidos ────────────────────────────
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

# Columnas opcionales que el modelo puede ignorar si no están
COLUMNAS_OPCIONALES = ["Numero_tarjeta", "fraude", "fraude_true",
                       "es_etiqueta_ruidosa", "Ubicacion"]

# Sinónimos predefinidos para casos comunes
SINONIMOS = {
    "Tipo_tarjeta": [
        "tipo_tarjeta", "tipotarjeta", "tipo tarjeta", "type_card",
        "card_type", "tipo de tarjeta", "tipo_de_tarjeta",
    ],
    "Estado_tarjeta": [
        "estado_tarjeta", "estadotarjeta", "estado tarjeta", "card_status",
        "status_card", "estado", "status", "estado_de_tarjeta",
    ],
    "Fecha_vencimiento": [
        "fecha_vencimiento", "fechavencimiento", "fecha vencimiento",
        "expiry_date", "expiration_date", "vencimiento", "fecha_exp",
        "fecha_de_vencimiento", "expiry",
    ],
    "Indicador_repositorio": [
        "indicador_repositorio", "indicadorrepositorio",
        "indicador repositorio", "repositorio", "ind_repo",
        "indicator_repo", "repositorio_indicator",
    ],
    "Localizacion_tarjeta": [
        "localizacion_tarjeta", "localizacion", "localizacion tarjeta",
        "location", "card_location", "ubicacion_tarjeta",
        "localización_tarjeta", "localizacion_de_tarjeta",
    ],
    "Acumulado_cupo": [
        "acumulado_cupo", "acumuladocupo", "acumulado cupo",
        "cupo_acumulado", "accumulated_quota", "cupo", "quota",
        "limite_usado", "used_limit",
    ],
    "Fecha_ult_retiro": [
        "fecha_ult_retiro", "fechaultretiro", "fecha ult retiro",
        "last_withdrawal_date", "ultimo_retiro", "fecha_ultimo_retiro",
        "last_withdrawal", "fecha_ultimo_movimiento",
    ],
    "Cuotas_mora": [
        "cuotas_mora", "cuotasmora", "cuotas mora", "past_due_installments",
        "mora", "overdue_payments", "cuotas_vencidas", "installments_overdue",
    ],
    "Ind_estado": [
        "ind_estado", "indestado", "ind estado", "status_indicator",
        "indicador_estado", "estado_indicador", "payment_status",
    ],
    "Es_amparada": [
        "es_amparada", "esamparada", "es amparada", "is_insured",
        "amparada", "insured", "covered", "is_covered",
    ],
    "Es_reexpedicion": [
        "es_reexpedicion", "esreexpedicion", "es reexpedicion",
        "is_reissued", "reexpedicion", "reissued", "reexpedida",
        "es_reexpedida",
    ],
    "Tipo_nomina": [
        "tipo_nomina", "tiponómina", "tipo nomina", "payroll_type",
        "nomina", "payroll", "tipo_de_nomina", "tipo de nómina",
    ],
}


def _normalizar(texto: str) -> str:
    """Normaliza texto: minúsculas, sin tildes, sin caracteres especiales."""
    texto = texto.lower().strip()
    # Quitar tildes
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    # Reemplazar separadores por _
    texto = re.sub(r'[\s\-\.]+', '_', texto)
    # Solo alfanuméricos y _
    texto = re.sub(r'[^a-z0-9_]', '', texto)
    return texto


def _similitud(a: str, b: str) -> float:
    """
    Similitud simple entre dos strings normalizados.
    Usa coincidencia de substrings y Jaccard de caracteres.
    """
    a, b = _normalizar(a), _normalizar(b)
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.85
    # Jaccard de bigramas
    def bigramas(s):
        return set(s[i:i+2] for i in range(len(s)-1))
    bg_a, bg_b = bigramas(a), bigramas(b)
    if not bg_a or not bg_b:
        return 0.0
    return len(bg_a & bg_b) / len(bg_a | bg_b)


def _buscar_columna(col_req: str, columnas_csv: list,
                    umbral: float = 0.60) -> tuple[str | None, float]:
    """
    Busca la columna requerida en la lista de columnas del CSV.
    Primero por sinónimos predefinidos, luego por similitud.
    Retorna (nombre_columna_csv, score) o (None, 0).
    """
    # 1. Coincidencia exacta
    if col_req in columnas_csv:
        return col_req, 1.0

    # 2. Sinónimos predefinidos
    sinonimos = [_normalizar(s) for s in SINONIMOS.get(col_req, [])]
    for col_csv in columnas_csv:
        if _normalizar(col_csv) in sinonimos:
            return col_csv, 0.95

    # 3. Similitud fuzzy
    mejor_col   = None
    mejor_score = 0.0
    for col_csv in columnas_csv:
        score = _similitud(col_req, col_csv)
        if score > mejor_score:
            mejor_score = score
            mejor_col   = col_csv

    if mejor_score >= umbral:
        return mejor_col, mejor_score
    return None, 0.0


# ── Función principal ──────────────────────────────────────────────────────

def analizar_columnas(columnas_csv: list) -> dict:
    """
    Analiza las columnas del CSV y determina qué acción tomar.

    Retorna un dict con:
        estado:      "ok" | "auto" | "parcial" | "vacio" | "imposible"
        mapeo:       {col_requerida: col_csv}  — columnas encontradas
        faltantes:   [col_requerida]           — columnas no encontradas
        renombrados: {col_csv: col_requerida}  — cambios automáticos
        mensaje:     str                       — descripción del resultado
    """
    mapeo       = {}   # col_requerida → col_csv_original
    renombrados = {}   # col_csv_original → col_requerida (solo los auto)
    faltantes   = []

    for col_req in COLUMNAS_REQUERIDAS:
        col_csv, score = _buscar_columna(col_req, columnas_csv)
        if col_csv is not None:
            mapeo[col_req] = col_csv
            if col_csv != col_req:
                renombrados[col_csv] = col_req
        else:
            faltantes.append(col_req)

    n_encontradas = len(mapeo)
    n_requeridas  = len(COLUMNAS_REQUERIDAS)

    if not faltantes:
        if renombrados:
            return {
                "estado":      "auto",
                "mapeo":       mapeo,
                "faltantes":   [],
                "renombrados": renombrados,
                "mensaje":     (f"Se detectaron {len(renombrados)} columnas con "
                                f"nombres diferentes y se renombraron automáticamente."),
            }
        return {
            "estado":      "ok",
            "mapeo":       mapeo,
            "faltantes":   [],
            "renombrados": {},
            "mensaje":     "Todas las columnas coinciden.",
        }

    if n_encontradas >= n_requeridas * 0.5:
        return {
            "estado":      "parcial",
            "mapeo":       mapeo,
            "faltantes":   faltantes,
            "renombrados": renombrados,
            "mensaje":     (f"Se encontraron {n_encontradas} de {n_requeridas} columnas. "
                            f"Faltan {len(faltantes)}: necesitamos que las asignes manualmente."),
        }

    if n_encontradas > 0:
        return {
            "estado":      "vacio",
            "mapeo":       mapeo,
            "faltantes":   faltantes,
            "renombrados": renombrados,
            "mensaje":     (f"Solo se encontraron {n_encontradas} columnas. "
                            f"Necesitamos asignar {len(faltantes)} manualmente."),
        }

    return {
        "estado":      "imposible",
        "mapeo":       {},
        "faltantes":   COLUMNAS_REQUERIDAS[:],
        "renombrados": {},
        "mensaje":     "No se pudo detectar ninguna columna requerida en el archivo.",
    }


def aplicar_mapeo(df, mapeo: dict):
    """
    Renombra las columnas del DataFrame según el mapeo detectado.
    Retorna el DataFrame con las columnas renombradas.
    """
    import pandas as pd
    # mapeo: {col_requerida: col_csv}  → invertir para rename
    rename_dict = {v: k for k, v in mapeo.items() if v != k}
    if rename_dict:
        df = df.rename(columns=rename_dict)
    return df


def nombre_legible(col: str) -> str:
    """Convierte Tipo_tarjeta → Tipo de tarjeta."""
    return col.replace("_", " ").title()
