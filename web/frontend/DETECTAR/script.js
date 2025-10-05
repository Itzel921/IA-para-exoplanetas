// --- LÓGICA DE BARRA LATERAL ---

const sidebarWidth = "350px"; // Debe coincidir con --sidebar-width en CSS

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

// --- LÓGICA ESPECÍFICA DE DETECT.HTML (NUEVA) ---

document.addEventListener('DOMContentLoaded', () => {
    
    const fileUpload = document.getElementById('file-upload');
    const fileNameSpan = document.getElementById('file-name');
    const runAnalysisBtn = document.getElementById('run-analysis-btn');
    const resultBox = document.getElementById('result-box');
    
    let selectedFile = null;

    // 1. Manejar la selección de archivos
    fileUpload.addEventListener('change', (event) => {
        if (event.target.files.length > 0) {
            selectedFile = event.target.files[0];
            fileNameSpan.textContent = `Archivo cargado: ${selectedFile.name}`;
            runAnalysisBtn.disabled = false; // Habilita el botón de análisis
        } else {
            selectedFile = null;
            fileNameSpan.textContent = 'Ningún archivo seleccionado.';
            runAnalysisBtn.disabled = true;
        }
    });

    // 2. Simulación de la predicción de IA
    runAnalysisBtn.addEventListener('click', () => {
        if (!selectedFile) return;

        // Mostrar estado de carga
        resultBox.innerHTML = `
            <p class="initial-message">Analizando ${selectedFile.name}...</p>
            <div class="spinner"></div> `;
        runAnalysisBtn.disabled = true; // Deshabilita el botón durante el análisis

        // Simular un retraso de procesamiento de la IA
        setTimeout(simulatePrediction, 2500); 
    });

    function simulatePrediction() {
        // Generar resultados aleatorios
        const results = [
            { label: "Exoplaneta confirmado", statusClass: "status-confirmed", minProb: 75, maxProb: 98 },
            { label: "Candidato planetario", statusClass: "status-candidate", minProb: 50, maxProb: 85 },
            { label: "Falso positivo", statusClass: "status-false", minProb: 65, maxProb: 99 }
        ];

        const predictionIndex = Math.floor(Math.random() * results.length);
        const prediction = results[predictionIndex];
        
        // Generar una probabilidad aleatoria dentro del rango definido
        const probability = Math.floor(Math.random() * (prediction.maxProb - prediction.minProb + 1)) + prediction.minProb;
        
        const resultHTML = `
            <div class="prediction-result">
                <h3>Clasificación: <span class="${prediction.statusClass}">${prediction.label}</span></h3>
                <p class="probability">${probability}%</p>
                <p>Probabilidad de ser un ${prediction.label}</p>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${probability}%;"></div>
                </div>
            </div>
        `;

        resultBox.innerHTML = resultHTML;
        runAnalysisBtn.disabled = false; // Vuelve a habilitar el botón
    }
});