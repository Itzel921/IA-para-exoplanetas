// --- LÓGICA DE BARRA LATERAL (Mantenida del index.js) ---

const sidebarWidth = "350px";

function openNav() {
    const sidebar = document.getElementById("mySidebar");
    const mainContent = document.getElementById("main");
    if (sidebar && mainContent) {
        sidebar.style.width = sidebarWidth;
        mainContent.style.marginLeft = sidebarWidth;
        sidebar.classList.add('active');
    }
}

function closeNav() {
    const sidebar = document.getElementById("mySidebar");
    const mainContent = document.getElementById("main");
    if (sidebar && mainContent) {
        sidebar.style.width = "0";
        mainContent.style.marginLeft = "0";
        sidebar.classList.remove('active');
    }
}

window.openNav = openNav;
window.closeNav = closeNav;


// --- LÓGICA ESPECÍFICA DE EXPLORE.HTML (NUEVA) ---

document.addEventListener('DOMContentLoaded', () => {

    // --- LÓGICA DE PESTAÑAS (TABS) ---
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.dataset.tab;

            // 1. Deactivate all
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.add('hidden'));

            // 2. Activate the selected one
            button.classList.add('active');
            document.getElementById(targetTab).classList.remove('hidden');

            // Note: This is where the function to render the chart would run if targetTab is 'graficas'
            if (targetTab === 'graficas') {
                // TODO: Llamar a una función de renderizado de gráficos (e.g., renderCharts();)
                console.log("Gráficas seleccionadas. Se necesita biblioteca (Chart.js/D3) para renderizar.");
            }
        });
    });

    // --- LÓGICA DE FILTROS (DEMOSTRACIÓN) ---
    const radioMin = document.getElementById('radio-rango-min');
    const radioMax = document.getElementById('radio-rango-max');
    const radioValor = document.getElementById('radio-valor');
    const periodoRango = document.getElementById('periodo-rango');
    const periodoValor = document.getElementById('periodo-valor');

    // Función para actualizar el valor visible del radio
    function updateRadioRange() {
        const min = parseFloat(radioMin.value);
        const max = parseFloat(radioMax.value);
        radioValor.textContent = `${min} a ${max} RE`;
    }

    // Función para actualizar el valor visible del período
    function updatePeriodoRange() {
        periodoValor.textContent = `Máx ${periodoRango.value} Días`;
    }

    radioMin.addEventListener('input', updateRadioRange);
    radioMax.addEventListener('input', updateRadioRange);
    periodoRango.addEventListener('input', updatePeriodoRange);

    // Inicializar valores
    updateRadioRange();
    updatePeriodoRange();

    // Simulación de carga de lista inicial
    // (En un proyecto real, esto haría una llamada a una API de la NASA)
    function loadExoplanetList() {
        const container = document.getElementById('exoplanet-table-container');
        // Simulación de datos (solo la primera columna se anima por el CSS de la pestaña)
        const dummyData = [
            { nombre: "Kepler-186f", mision: "Kepler", estado: "Confirmado" },
            { nombre: "TOI-700 d", mision: "TESS", estado: "Confirmado" },
            { nombre: "K2-18 b", mision: "K2", estado: "Confirmado" },
            { nombre: "KOI-7923.01", mision: "Kepler", estado: "Candidato" },
            { nombre: "K2-FAKE-X1", mision: "K2", estado: "Falso Positivo" },
            // ... (más datos)
        ];

        let tableHTML = `
            <table class="data-table">
                <thead>
                    <tr><th>Nombre</th><th>Misión</th><th>Clasificación</th></tr>
                </thead>
                <tbody>
        `;
        dummyData.forEach(p => {
            tableHTML += `
                <tr>
                    <td>${p.nombre}</td>
                    <td>${p.mision}</td>
                    <td>${p.estado}</td>
                </tr>
            `;
        });
        tableHTML += `</tbody></table>`;

        container.innerHTML = tableHTML;
    }

    // Llamada inicial a cargar la lista
    setTimeout(loadExoplanetList, 1000); // Simula un retraso de carga


    // --- EFECTO DE SCROLL DEL HEADER (Se mantiene, pero se aplica al body) ---
    // (Asegúrate de que este bloque no interfiera con tu lógica de index.html si usas el mismo script)
    // En esta página, este efecto no es estrictamente necesario ya que el header es fijo, pero lo dejamos por consistencia
    window.addEventListener('scroll', () => {
        // La lógica de scroll de opacidad se aplicaba al video header, que no está aquí.
        // Se elimina la lógica de opacidad de scroll del header para esta página, si se usa este script.js.
    });
});

// Note: The AI simulation logic (activarExploracion) and buttons (btn-explorar, btn-prueba)
// se ha omitido aquí ya que esta es la página de exploración. Si copias y pegas todo el JS anterior,
// you can remove those functions.