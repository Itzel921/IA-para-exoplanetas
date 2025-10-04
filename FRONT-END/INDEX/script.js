// --- LÓGICA DE BARRA LATERAL ---

const sidebarWidth = "350px"; // Debe coincidir con --sidebar-width en CSS

/* Función para abrir el sidebar y empujar el contenido principal */
function openNav() {
  const sidebar = document.getElementById("mySidebar");
  const mainContent = document.getElementById("main");
  if (sidebar && mainContent) {
    sidebar.style.width = sidebarWidth;
    mainContent.style.marginLeft = sidebarWidth;
    sidebar.classList.add('active'); // Añade la clase 'active' para animar los enlaces
  }
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
    "IA: 'Análisis de datos iniciado. 12 nuevos candidatos detectados en el sector TESS.'",
    "IA: 'Módulo de aprendizaje profundo cargado. Preparando curva de luz.'",
    "IA: '¡Bienvenido/a al panel de control! Tu primera tarea: verificar la señal Kepler-1647b.'",
    "IA: 'Procesando tránsitos. La probabilidad de falso positivo es del 1.5%.'",
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

  // 1. Botón "Activar Modelo de IA"
  if (botonExplorar) {
    // Enlaza la función de mensajes aleatorios (la que has estado usando)
    botonExplorar.addEventListener('click', activarExploracion);

    // Muestra el mensaje inicial de espera
    if (mensajeDiv) {
      mensajeDiv.innerHTML = '<p>Pulsa <strong>Activar Modelo de IA</strong> para comenzar la simulación.</p>';
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