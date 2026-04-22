# modules/feedback.py
# Módulo para guardar y cargar retroalimentación de usuarios

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Ruta al archivo de retroalimentación
FEEDBACK_FILE = Path(__file__).parent.parent / "uploads" / "retroalimentacion.json"

def guardar_retroalimentacion(entrada: Dict) -> bool:
    """
    Guarda una nueva entrada de retroalimentación.
    
    Parámetros:
        entrada: dict con campos timestamp, usuario, indice, marcacion
    
    Retorna:
        True si se guardó exitosamente, False si hubo error.
    """
    try:
        FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Cargar retroalimentación existente
        if FEEDBACK_FILE.exists():
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                datos = json.load(f)
        else:
            datos = []

        # Agregar nueva entrada con timestamp si no tiene
        if "timestamp" not in entrada:
            entrada["timestamp"] = datetime.now().isoformat()

        datos.append(entrada)

        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=4, ensure_ascii=False)

        return True

    except Exception as e:
        print(f"Error guardando retroalimentación: {e}")
        return False


def cargar_retroalimentacion() -> List[Dict]:
    """
    Carga toda la retroalimentación guardada.
    
    Retorna:
        Lista de entradas de retroalimentación.
    """
    try:
        if not FEEDBACK_FILE.exists():
            return []
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error cargando retroalimentación: {e}")
        return []


def obtener_resumen_retroalimentacion() -> Dict:
    """
    Retorna un resumen estadístico de la retroalimentación.
    """
    datos = cargar_retroalimentacion()

    resumen = {
        "total": len(datos),
        "verdaderos_positivos": 0,
        "falsos_positivos":     0,
        "por_usuario":          {}
    }

    for entrada in datos:
        marcacion = entrada.get("marcacion", "")
        usuario   = entrada.get("usuario", "desconocido")

        if marcacion == "verdadero_positivo":
            resumen["verdaderos_positivos"] += 1
        elif marcacion == "falso_positivo":
            resumen["falsos_positivos"] += 1

        resumen["por_usuario"][usuario] = resumen["por_usuario"].get(usuario, 0) + 1

    return resumen


def limpiar_retroalimentacion() -> bool:
    """
    Elimina toda la retroalimentación guardada (usar con precaución).
    """
    try:
        if FEEDBACK_FILE.exists():
            FEEDBACK_FILE.unlink()
        return True
    except Exception as e:
        print(f"Error limpiando retroalimentación: {e}")
        return False
