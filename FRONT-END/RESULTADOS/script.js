// --- LÓGICA DE BARRA LATERAL ---

const sidebarWidth = "350px"; 

function openNav() {
    const sidebar = document.getElementById("mySidebar");
    const mainContent = document.getElementById("main");
    const openBtn = document.querySelector('.openbtn');
    
    if (sidebar && mainContent && openBtn) {
        sidebar.style.width = sidebarWidth;
        mainContent.style.marginLeft = sidebarWidth;
        sidebar.classList.add('active'); 
        
        // OCULTAR BOTÓN AL ABRIR EL MENÚ
        openBtn.style.opacity = '0';
        openBtn.style.pointerEvents = 'none';
    }
}

function closeNav() {
    const sidebar = document.getElementById("mySidebar");
    const mainContent = document.getElementById("main");
    const openBtn = document.querySelector('.openbtn');

    if (sidebar && mainContent && openBtn) {
        sidebar.style.width = "0";
        mainContent.style.marginLeft = "0";
        sidebar.classList.remove('active'); 
        
        // MOSTRAR BOTÓN AL CERRAR EL MENÚ
        openBtn.style.opacity = '1';
        openBtn.style.pointerEvents = 'auto';
    }
}

window.openNav = openNav;
window.closeNav = closeNav;


// --- LÓGICA ESPECÍFICA DE RESULTADOS.HTML ---

document.addEventListener('DOMContentLoaded', () => {
    
    const historyContainer = document.getElementById('prediction-history');
    const curveChartContainer = document.getElementById('light-curve-chart');
    const comparisonDetailsContainer = document.getElementById('comparison-details');
    
    // Datos simulados para el historial
    const historyData = [
        { id: 1, name: "Análisis 2025-01-15", result: "Exoplaneta confirmado", prob: 92, radius: 1.4, period: 3.5, similar: ["WASP-96b", "Kepler-186f"] },
        { id: 2, name: "Análisis 2025-01-14", result: "Falso positivo", prob: 88, radius: 0.9, period: 1.1, similar: ["Ninguno cercano"] },
        { id: 3, name: "Análisis 2025-01-13", result: "Candidato planetario", prob: 65, radius: 2.1, period: 10.2, similar: ["TOI-700 d"] },
        { id: 4, name: "Análisis 2025-01-12", result: "Exoplaneta confirmado", prob: 78, radius: 3.0, period: 55.0, similar: ["K2-18 b"] },
        { id: 5, name: "Análisis 2025-01-11", result: "Falso positivo", prob: 95, radius: 0.6, period: 0.9, similar: ["Ninguno cercano"] },
    ];

    // 1. Cargar la galería de historial
    function loadHistoryGallery() {
        if (!historyContainer) return;
        
        if (historyData.length === 0) {
            historyContainer.innerHTML = '<p class="initial-message">Historial de predicciones vacío.</p>';
            return;
        }

        historyContainer.innerHTML = '';
        historyData.forEach(data => {
            const statusClass = data.result.includes('confirmado') ? 'status-confirmed' : 
                                data.result.includes('candidato') ? 'status-candidate' : 'status-false';
            
            const item = document.createElement('div');
            item.className = 'prediction-item';
            item.dataset.id = data.id;
            item.innerHTML = `
                <p class="item-title">${data.name}</p>
                <p class="item-status">
                    Clasificación: <span class="${statusClass}">${data.result}</span>
                </p>
            `;
            item.addEventListener('click', () => handleItemClick(data.id));
            historyContainer.appendChild(item);
        });

        // Seleccionar el primer elemento por defecto al cargar
        if (historyData.length > 0) {
            handleItemClick(historyData[0].id);
        }
    }

    // 2. Manejar la selección de un elemento del historial
    function handleItemClick(id) {
        const selectedData = historyData.find(d => d.id === id);
        
        // Remover clase 'active' de todos y añadirla al seleccionado
        document.querySelectorAll('.prediction-item').forEach(item => {
            item.classList.remove('active');
            if (parseInt(item.dataset.id) === id) {
                item.classList.add('active');
            }
        });
        
        if (selectedData) {
            renderLightCurve(selectedData);
            renderComparison(selectedData);
        }
    }

    // 3. Simular la renderización de la curva de luz (Placeholder)
    function renderLightCurve(data) {
        if (!curveChartContainer) return;
        
        curveChartContainer.innerHTML = `
            <h3>Curva de Luz: ${data.name}</h3>
            <p>Predicción: ${data.result} (${data.prob}%)</p>
            <div style="width: 90%; height: 250px; background-color: ${data.result.includes('false') ? '#dd361c30' : '#2c7be530'}; border-radius: 4px; display: flex; align-items: center; justify-content: center; margin-top: 10px;">
                <p style="color: var(--color-acento-alt);">[Gráfico de la Curva de Luz - Tránsitos Simulados]</p>
            </div>
        `;
    }

    // 4. Renderizar la sección de comparación
    function renderComparison(data) {
        if (!comparisonDetailsContainer) return;
        
        const similarListHTML = data.similar.map(planet => `
            <div class="comparison-planet">
                <span>${planet}</span>
                <span>Radio: ${(data.radius + (Math.random() * 0.2 - 0.1)).toFixed(2)} RE</span>
            </div>
        `).join('');

        comparisonDetailsContainer.innerHTML = `
            <p>Tu predicción (Radio: <b>${data.radius} RE</b>, Periodo: <b>${data.period} días</b>) es similar a:</p>
            <div id="similar-planets-list">
                ${similarListHTML || '<p class="item-status">No se encontraron planetas similares en el catálogo.</p>'}
            </div>
        `;
    }


    // Ejecutar al inicio
    loadHistoryGallery();
});