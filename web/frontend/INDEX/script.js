// --- LÓGICA DE BARRA LATERAL ---

const sidebarWidth = "350px"; // Debe coincidir con --sidebar-width en CSS

/* Función para abrir el sidebar y empujar el contenido principal */
function openNav() {
    const sidebar = document.getElementById("mySidebar");
    const mainContent = document.getElementById("main");
    if (!sidebar) {
        console.error("Sidebar element with id 'mySidebar' not found.");
        return;
    }
    if (!mainContent) {
        console.error("Main content element with id 'main' not found.");
        return;
    }
    sidebar.style.width = sidebarWidth;
    mainContent.style.marginLeft = sidebarWidth;
    sidebar.classList.add('active'); // Añade la clase 'active' para animar los enlaces
}

/* Función para cerrar el sidebar y restaurar el contenido principal */
function closeNav() {
    const sidebar = document.getElementById("mySidebar");
    const mainContent = document.getElementById("main");
    if (sidebar && mainContent) {
        sidebar.style.width = "0";
        mainContent.style.marginLeft = "0";
        sidebar.classList.remove('active'); // Elimina la clase 'active' para ocultar los enlaces
    }
}

// Hacemos que las funciones de navegación sean globales para el HTML (onclick)
window.openNav = openNav;
window.closeNav = closeNav;


// --- LÓGICA DE SIMULACIÓN DE IA ---

/* Función para simular la activación y respuesta de la IA (mensajes aleatorios) */
    function activarExploracion() {
        const mensajeElemento = document.getElementById('mensaje-ia');
        if (!mensajeElemento) return;
        const mensajes = [
            "IA: 'Data analysis started. 12 new candidates detected in the TESS sector.'",
            "IA: 'Deep learning module loaded. Preparing light curve.'",
            "IA: 'Welcome to the control panel! Your first task: verify the Kepler-1647b signal.'",
            "IA: 'Processing transits. The false positive probability is 1.5%.'",
        ];

    // Selecciona un mensaje aleatorio
    const mensajeAleatorio = mensajes[Math.floor(Math.random() * mensajes.length)];

    // Usamos la clase CSS 'ia-response-text' para visibilidad y estilo
    mensajeElemento.innerHTML = `<p class="ia-response-text"><strong>${mensajeAleatorio}</strong></p>`;
}

// --- ASIGNACIÓN DE EVENTOS ---
document.addEventListener('DOMContentLoaded', () => {
    const botonExplorar = document.getElementById('btn-explorar');
    const botonPrueba = document.getElementById("btn-prueba");
    const mensajeDiv = document.getElementById('mensaje-ia');

    // 1. Button "Activate AI Model"
    if (botonExplorar) {
        botonExplorar.addEventListener('click', activarExploracion);

        // Muestra el mensaje inicial de espera
        if (mensajeDiv) {
            mensajeDiv.innerHTML = '<p>Press <strong>Activate AI Model</strong> to start the simulation.</p>';
        }
    }

    // 2. Botón "Prueba" del Header
    if (botonPrueba) {
        botonPrueba.addEventListener("click", () => {
            alert("Iniciando simulación del programa de detección de exoplanetas...");
        });
    }
});


// --- EFECTO DE SCROLL DEL HEADER ---
window.addEventListener('scroll', () => {
    const header = document.querySelector('.video-header');
    if (header) {
        const opacity = 1 - window.scrollY / 600;
        header.style.opacity = opacity > 0 ? opacity : 0;
    }
});