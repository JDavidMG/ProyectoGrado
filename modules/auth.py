# modules/auth.py
# Módulo de autenticación de usuarios

import json
from pathlib import Path

# Ruta al archivo de usuarios
USERS_FILE = Path(__file__).parent.parent / "users.json"

# Usuarios por defecto (se crean si no existe el archivo)
USUARIOS_DEFAULT = {
    "admin": "admin123",
    "analista": "fraude2025"
}

def _cargar_usuarios():
    """Carga usuarios desde archivo JSON. Si no existe, lo crea con defaults."""
    if not USERS_FILE.exists():
        with open(USERS_FILE, "w") as f:
            json.dump(USUARIOS_DEFAULT, f, indent=4)
        return USUARIOS_DEFAULT
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def verificar_credenciales(usuario: str, password: str) -> bool:
    """
    Verifica si el usuario y contraseña son válidos.
    Retorna True si son correctos, False en caso contrario.
    """
    if not usuario or not password:
        return False
    usuarios = _cargar_usuarios()
    return usuarios.get(usuario) == password

def agregar_usuario(usuario: str, password: str) -> bool:
    """Agrega un nuevo usuario al sistema."""
    usuarios = _cargar_usuarios()
    if usuario in usuarios:
        return False  # Ya existe
    usuarios[usuario] = password
    with open(USERS_FILE, "w") as f:
        json.dump(usuarios, f, indent=4)
    return True

def cambiar_password(usuario: str, password_actual: str, nueva_password: str) -> bool:
    """Cambia la contraseña de un usuario."""
    usuarios = _cargar_usuarios()
    if usuarios.get(usuario) != password_actual:
        return False
    usuarios[usuario] = nueva_password
    with open(USERS_FILE, "w") as f:
        json.dump(usuarios, f, indent=4)
    return True
