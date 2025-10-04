// Función para simular la activación y respuesta de la IA
function activarExploracion() {
    const mensajeElemento = document.getElementById('mensaje-ia');
    const mensajes = [
        "IA: 'Análisis de datos iniciado. 12 nuevos candidatos detectados en el sector TESS.'",
        "IA: 'Módulo de aprendizaje profundo cargado. Preparando curva de luz.'",
        "IA: '¡Bienvenido/a al panel de control! Tu primera tarea: verificar la señal Kepler-1647b.'",
        "IA: 'Procesando tránsitos. La probabilidad de falso positivo es del 1.5%.'",
    ];
    
    // Selecciona un mensaje aleatorio
    const mensajeAleatorio = mensajes[Math.floor(Math.random() * mensajes.length)];

    // Actualiza el contenido del div. Nota: El estilo (color de texto) se define en el CSS (ver #mensaje-ia p)
    mensajeElemento.innerHTML = `<p><strong>${mensajeAleatorio}</strong></p>`;
}

// 1. Obtener los elementos del DOM
const botonExplorar = document.getElementById('btn-explorar');
const mensajeInicialDiv = document.getElementById('mensaje-ia');

// 2. Asigna la función al evento 'click' del botón
botonExplorar.addEventListener('click', activarExploracion);

// Opcional: Mostrar un mensaje inicial al cargar la página (se ejecuta al inicio)
document.addEventListener('DOMContentLoaded', () => {
    // Usamos el color de texto definido en el CSS para esta sección (color-base)
    mensajeInicialDiv.innerHTML = '<p>Pulsa <strong>Activar Modelo de IA</strong> para comenzar la simulación.</p>';
});