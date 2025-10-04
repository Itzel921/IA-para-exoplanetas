// --- LÓGICA DE BARRA LATERAL ---

const sidebarWidth = "350px"; // Debe coincidir con --sidebar-width en CSS

/* Función para abrir el sidebar y empujar el contenido principal */
function openNav() {
    const sidebar = document.getElementById("mySidebar");
    const mainContent = document.getElementById("main");
    if (sidebar && mainContent) {
        sidebar.style.width = sidebarWidth;
        mainContent.style.marginLeft = sidebarWidth;
        sidebar.classList.add('active'); // 🆕 Añade la clase 'active' para animar los enlaces
    }
}

/* Función para cerrar el sidebar y restaurar el contenido principal */
function closeNav() {
    const sidebar = document.getElementById("mySidebar");
    const mainContent = document.getElementById("main");
    if (sidebar && mainContent) {
        sidebar.style.width = "0";
        mainContent.style.marginLeft = "0";
        sidebar.classList.remove('active'); // 🆕 Elimina la clase 'active' para ocultar los enlaces
    }
}

// Para que las funciones openNav y closeNav sean accesibles desde el onclick del HTML
window.openNav = openNav;
window.closeNav = closeNav;


// --- LÓGICA DE SIMULACIÓN DE IA ---

// Función para simular la activación y respuesta de la IA
function activarExploracion() {
    const mensajeElemento = document.getElementById('mensaje-ia');
    if (!mensajeElemento) return;
    const mensajes = [
        "IA: 'Análisis de datos iniciado. 12 nuevos candidatos detectados en el sector TESS.'",
        "IA: 'Módulo de aprendizaje profundo cargado. Preparando curva de luz.'",
        "IA: '¡Bienvenido/a al panel de control! Tu primera tarea: verificar la señal Kepler-1647b.'",
        "IA: 'Procesando tránsitos. La probabilidad de falso positivo es del 1.5%.'",
    ];

    // Selecciona un mensaje aleatorio
    const mensajeAleatorio = mensajes[Math.floor(Math.random() * mensajes.length)];

    // Usamos la clase CSS 'ia-response-text' para visibilidad
    mensajeElemento.innerHTML = `<p class="ia-response-text"><strong>${mensajeAleatorio}</strong></p>`;
}

// 1. Obtener los elementos del DOM
const botonExplorar = document.getElementById('btn-explorar');
const mensajeInicialDiv = document.getElementById('mensaje-ia');

// 2. Asigna la función al evento 'click' del botón si el elemento existe
if (botonExplorar && typeof activarExploracion === 'function') {
    botonExplorar.addEventListener('click', activarExploracion);
}

// Opcional: Mostrar un mensaje inicial al cargar la página (se ejecuta al inicio)
document.addEventListener('DOMContentLoaded', () => {
    const mensajeDiv = document.getElementById('mensaje-ia');
    if (mensajeDiv) {
        mensajeDiv.innerHTML = '<p>Pulsa <strong>Activar Modelo de IA</strong> para comenzar la simulación.</p>';
    }
});

// Efecto de desvanecimiento del header al hacer scroll
window.addEventListener('scroll', () => {
  const header = document.querySelector('.video-header');
  const opacity = 1 - window.scrollY / 600;
  header.style.opacity = opacity > 0 ? opacity : 0;
});

// Simulación básica de IA
document.getElementById('btn-explorar').addEventListener('click', () => {
  const mensaje = document.getElementById('mensaje-ia');
  mensaje.innerHTML = '<p class="ia-response-text">🌌 Bienvenido/a al sistema de detección inteligente de exoplanetas.</p>';
});
