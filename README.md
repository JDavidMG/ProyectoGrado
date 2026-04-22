# 🛡️ FraudeDetect — Instrucciones de instalación y uso

## Estructura de archivos (agregar a tu proyecto)

```
Codigo proyecto de grado/
│
├── app.py                         ← Servidor Flask principal  ✅ NUEVO
│
├── modules/
│   ├── __init__.py                ← ✅ NUEVO
│   ├── auth.py                    ← Autenticación             ✅ NUEVO
│   ├── preprocessing.py           ← Preprocesamiento          ✅ NUEVO
│   ├── reports.py                 ← Reportes exportables      ✅ NUEVO
│   └── feedback.py                ← Retroalimentación         ✅ NUEVO
│
├── templates/
│   ├── base.html                  ← Layout base               ✅ NUEVO
│   ├── login.html                 ← Inicio de sesión          ✅ NUEVO
│   ├── dashboard.html             ← Dashboard                 ✅ NUEVO
│   ├── cargar.html                ← Carga de archivos         ✅ NUEVO
│   ├── datos.html                 ← Visualizar datos          ✅ NUEVO
│   ├── ejecutar.html              ← Ejecutar modelo           ✅ NUEVO
│   ├── resultados.html            ← Ver resultados            ✅ NUEVO
│   ├── metricas.html              ← Métricas                  ✅ NUEVO
│   └── confusion.html             ← Matriz de confusión       ✅ NUEVO
│
├── static/
│   ├── css/style.css              ← Estilos                   ✅ NUEVO
│   └── js/main.js                 ← JavaScript                ✅ NUEVO
│
├── requirements.txt               ← Dependencias              ✅ NUEVO
│
├── LLENAR/                        ← Ya existía
│   ├── 01_llenar_tablas_with_gan.py
│   └── tarjetas_fraude_*.csv
│
├── PruebaTécnicaIQ/               ← Ya existía
│   └── 02_arboles_decision_detection.py
│
├── RedNeuronal/                   ← Ya existía
│   └── random_forest_pipeline.joblib
│
├── out_detection/                 ← Ya existía
│   ├── metrics.json
│   └── *.png
│
└── uploads/                       ← Se crea automáticamente
```

---

## ⚙️ Pasos de instalación

### 1. Instalar Flask y dependencias
Abre la terminal en la carpeta del proyecto y ejecuta:

```bash
pip install -r requirements.txt
```

### 2. Iniciar el servidor
```bash
python app.py
```

### 3. Abrir en el navegador
```
http://127.0.0.1:5000
```

---

## 🔐 Credenciales de acceso

| Usuario   | Contraseña   |
|-----------|-------------|
| admin     | admin123    |
| analista  | fraude2025  |

> Los usuarios se guardan en `users.json` (se crea automáticamente).

---

## 🗺️ Flujo de uso

1. **Login** → Inicia sesión
2. **Cargar Archivo** → Sube un CSV con transacciones
3. **Ver Datos** → Revisa los datos cargados
4. **Ejecutar Modelo** → Clasifica las transacciones
5. **Resultados** → Ve las transacciones fraudulentas detectadas
6. **Métricas** → Revisa ROC AUC, PR AUC, F1-Score
7. **Matriz** → Visualiza la matriz de confusión
8. **Reportes** → Descarga reportes CSV

---

## ⚠️ Notas importantes

- El modelo `random_forest_pipeline.joblib` debe estar en `RedNeuronal/`
- Las métricas `metrics.json` deben estar en `out_detection/`
- Los archivos CSV deben contener las columnas requeridas del modelo
