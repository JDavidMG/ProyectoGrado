// static/js/main.js

// ─── Drag & Drop de archivos ───────────────────────────────────────────────
const dropZone = document.getElementById("drop-zone");
const inputFile = document.getElementById("archivo");
const nombreDiv = document.getElementById("archivo-nombre");

if (dropZone && inputFile) {
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });
    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("drag-over");
    });
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            inputFile.files = files;
            mostrarNombreArchivo(files[0].name);
        }
    });
    inputFile.addEventListener("change", () => {
        if (inputFile.files.length > 0) {
            mostrarNombreArchivo(inputFile.files[0].name);
        }
    });
}

function mostrarNombreArchivo(nombre) {
    if (nombreDiv) {
        nombreDiv.textContent = "📄 Archivo seleccionado: " + nombre;
        nombreDiv.style.display = "block";
    }
}

// ─── Marcación de transacciones (modal) ────────────────────────────────────
let indiceSeleccionado = null;

// Agregar click a filas de la tabla de resultados
const tablaResultados = document.getElementById("tabla-resultados");
if (tablaResultados) {
    tablaResultados.querySelectorAll("tbody tr").forEach((fila, idx) => {
        fila.addEventListener("click", () => {
            indiceSeleccionado = idx;
            const modal = document.getElementById("modal-indice");
            if (modal) modal.textContent = idx;
            document.getElementById("modal-marcar").style.display = "flex";
        });
    });
}

function cerrarModal() {
    document.getElementById("modal-marcar").style.display = "none";
    indiceSeleccionado = null;
}

function marcar(tipo) {
    if (indiceSeleccionado === null) return;

    fetch("/marcar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ indice: indiceSeleccionado, marcacion: tipo })
    })
    .then(res => res.json())
    .then(data => {
        cerrarModal();
        if (data.ok) {
            mostrarToast(data.msg, "exito");
            // Actualizar color de fila visualmente
            const filas = document.querySelectorAll("#tabla-resultados tbody tr");
            if (filas[indiceSeleccionado]) {
                filas[indiceSeleccionado].style.background =
                    tipo === "verdadero_positivo" ? "#fef2f2" : "#f0fdf4";
            }
        } else {
            mostrarToast(data.msg, "error");
        }
    })
    .catch(() => mostrarToast("Error de conexión.", "error"));
}

// ─── Toast de notificaciones ───────────────────────────────────────────────
function mostrarToast(mensaje, tipo) {
    const toast = document.createElement("div");
    toast.textContent = mensaje;
    toast.style.cssText = `
        position: fixed; bottom: 1.5rem; right: 1.5rem;
        padding: .75rem 1.25rem; border-radius: 10px; font-size: .9rem;
        background: ${tipo === "exito" ? "#16a34a" : "#dc2626"};
        color: white; box-shadow: 0 4px 16px rgba(0,0,0,.25);
        z-index: 9999; animation: fadeIn .3s ease;
    `;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

// ─── Cerrar modal al hacer clic fuera ──────────────────────────────────────
const modal = document.getElementById("modal-marcar");
if (modal) {
    modal.addEventListener("click", (e) => {
        if (e.target === modal) cerrarModal();
    });
}
