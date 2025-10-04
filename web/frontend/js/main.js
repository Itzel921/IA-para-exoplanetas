/**
 * NASA Space Apps Challenge 2025 - Exoplanet Detection System
 * Main JavaScript application logic
 * 
 * Handles navigation, form submissions, and UI interactions
 */

// Global application state
const ExoplanetApp = {
    currentSection: 'dashboard',
    apiBaseUrl: 'http://localhost:8000/api',
    statistics: {
        predictionsToday: 0,
        confirmedPlanets: 0,
        falsePositives: 0
    },
    analysisHistory: []
};

// Initialize application on DOM load
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Exoplanet Detection System - Initializing...');
    
    initializeApplication();
    bindEventListeners();
    loadModelInfo();
    initializeCharts();
    
    console.log('✅ Application initialized successfully');
});

/**
 * Initialize application components
 */
function initializeApplication() {
    // Show default section
    showSection('dashboard');
    
    // Load dashboard statistics
    updateDashboardStats();
    
    // Set up periodic updates
    setInterval(updateDashboardStats, 30000); // Update every 30 seconds
}

/**
 * Bind event listeners to form elements
 */
function bindEventListeners() {
    // Single analysis form
    const singleForm = document.getElementById('singleAnalysisForm');
    if (singleForm) {
        singleForm.addEventListener('submit', handleSingleAnalysis);
    }
    
    // Batch analysis form
    const batchForm = document.getElementById('batchAnalysisForm');
    if (batchForm) {
        batchForm.addEventListener('submit', handleBatchAnalysis);
    }
    
    // File input validation
    const csvFileInput = document.getElementById('csvFile');
    if (csvFileInput) {
        csvFileInput.addEventListener('change', validateCsvFile);
    }
    
    // Add real-time form validation
    addFormValidation();
}

/**
 * Show specific section and hide others
 */
function showSection(sectionId) {
    // Hide all sections
    const sections = document.querySelectorAll('.content-section');
    sections.forEach(section => {
        section.style.display = 'none';
        section.classList.remove('fade-in');
    });
    
    // Show target section
    const targetSection = document.getElementById(sectionId);
    if (targetSection) {
        targetSection.style.display = 'block';
        targetSection.classList.add('fade-in');
    }
    
    // Update navigation
    updateNavigation(sectionId);
    
    // Update current section
    ExoplanetApp.currentSection = sectionId;
    
    // Section-specific initialization
    switch(sectionId) {
        case 'dashboard':
            refreshDashboard();
            break;
        case 'model-info':
            loadModelInfo();
            break;
        case 'history':
            loadAnalysisHistory();
            break;
    }
    
    console.log(`📄 Switched to section: ${sectionId}`);
}

/**
 * Update navigation active state
 */
function updateNavigation(activeSectionId) {
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.classList.remove('active');
        
        // Check if this link corresponds to the active section
        const href = link.getAttribute('href');
        if (href === `#${activeSectionId}`) {
            link.classList.add('active');
        }
    });
}

/**
 * Handle single exoplanet analysis
 */
async function handleSingleAnalysis(event) {
    event.preventDefault();
    
    console.log('🔍 Starting single analysis...');
    
    // Show loading state
    showLoadingState();
    
    try {
        // Collect form data
        const formData = collectSingleAnalysisData();
        
        // Validate data
        if (!validateSingleAnalysisData(formData)) {
            throw new Error('Invalid form data');
        }
        
        // Make API request
        const result = await ApiClient.predictSingle(formData);
        
        // Display results
        displaySingleAnalysisResults(result);
        
        // Update statistics
        updateStatisticsAfterPrediction(result);
        
        // Add to history
        addToAnalysisHistory('individual', result);
        
        console.log('✅ Single analysis completed successfully');
        
    } catch (error) {
        console.error('❌ Error in single analysis:', error);
        showErrorMessage('Error en el análisis: ' + error.message);
    } finally {
        hideLoadingState();
    }
}

/**
 * Handle batch analysis
 */
async function handleBatchAnalysis(event) {
    event.preventDefault();
    
    console.log('📊 Starting batch analysis...');
    
    const fileInput = document.getElementById('csvFile');
    const file = fileInput.files[0];
    
    if (!file) {
        showErrorMessage('Por favor selecciona un archivo CSV');
        return;
    }
    
    // Show batch loading state
    showBatchLoadingState();
    
    try {
        // Upload and process file
        const result = await ApiClient.predictBatch(file);
        
        // Display batch results
        displayBatchResults(result);
        
        // Update statistics
        updateStatisticsAfterBatchPrediction(result);
        
        // Add to history
        addToAnalysisHistory('batch', result);
        
        console.log('✅ Batch analysis completed successfully');
        
    } catch (error) {
        console.error('❌ Error in batch analysis:', error);
        showErrorMessage('Error en el análisis por lotes: ' + error.message);
    } finally {
        hideBatchLoadingState();
    }
}

/**
 * Collect single analysis form data
 */
function collectSingleAnalysisData() {
    return {
        period: parseFloat(document.getElementById('period').value),
        radius: parseFloat(document.getElementById('radius').value),
        temp: parseFloat(document.getElementById('temp').value),
        starRadius: parseFloat(document.getElementById('starRadius').value),
        starMass: parseFloat(document.getElementById('starMass').value),
        starTemp: parseFloat(document.getElementById('starTemp').value),
        depth: parseFloat(document.getElementById('depth').value),
        duration: parseFloat(document.getElementById('duration').value),
        snr: parseFloat(document.getElementById('snr').value)
    };
}

/**
 * Validate single analysis data
 */
function validateSingleAnalysisData(data) {
    // Check for NaN values
    for (const [key, value] of Object.entries(data)) {
        if (isNaN(value) || value <= 0) {
            showErrorMessage(`Valor inválido para ${key}: ${value}`);
            return false;
        }
    }
    
    // Astronomical validation
    if (data.period < 0.1 || data.period > 10000) {
        showErrorMessage('Período orbital debe estar entre 0.1 y 10,000 días');
        return false;
    }
    
    if (data.radius < 0.1 || data.radius > 100) {
        showErrorMessage('Radio planetario debe estar entre 0.1 y 100 radios terrestres');
        return false;
    }
    
    if (data.starTemp < 1000 || data.starTemp > 50000) {
        showErrorMessage('Temperatura estelar debe estar entre 1,000 y 50,000 K');
        return false;
    }
    
    return true;
}

/**
 * Display single analysis results
 */
function displaySingleAnalysisResults(result) {
    const resultsPanel = document.getElementById('predictionResults');
    const resultsContent = document.getElementById('predictionContent');
    
    if (!resultsPanel || !resultsContent) return;
    
    // Determine result styling
    const isConfirmed = result.prediction === 'CONFIRMED';
    const confidenceLevel = getConfidenceLevel(result.confidence);
    
    // Create results HTML
    const resultsHtml = `
        <div class="prediction-result ${isConfirmed ? 'confirmed' : 'false-positive'} fade-in">
            <div class="d-flex align-items-center mb-3">
                <i class="fas ${isConfirmed ? 'fa-check-circle text-success' : 'fa-times-circle text-danger'} fa-2x me-3"></i>
                <div>
                    <h5 class="mb-1">${isConfirmed ? 'EXOPLANETA CONFIRMADO' : 'FALSO POSITIVO'}</h5>
                    <small class="text-muted">Análisis completado: ${new Date(result.analysis_timestamp).toLocaleString()}</small>
                </div>
            </div>
            
            <div class="mb-3">
                <label class="form-label">Confianza del Modelo: ${(result.confidence * 100).toFixed(1)}%</label>
                <div class="confidence-bar">
                    <div class="confidence-fill ${confidenceLevel}" style="width: ${result.confidence * 100}%"></div>
                </div>
            </div>
            
            <div class="row mb-3">
                <div class="col-6">
                    <strong>Prob. Confirmado:</strong><br>
                    <span class="text-success">${(result.probabilities.CONFIRMED * 100).toFixed(1)}%</span>
                </div>
                <div class="col-6">
                    <strong>Prob. Falso Positivo:</strong><br>
                    <span class="text-danger">${(result.probabilities.FALSE_POSITIVE * 100).toFixed(1)}%</span>
                </div>
            </div>
            
            <div class="interpretation-box p-3 bg-light rounded">
                <h6><i class="fas fa-lightbulb text-warning"></i> Interpretación</h6>
                <p class="mb-0">
                    ${getInterpretationText(result)}
                </p>
            </div>
            
            ${result.feature_importance ? createFeatureImportanceHtml(result.feature_importance) : ''}
        </div>
    `;
    
    resultsContent.innerHTML = resultsHtml;
    resultsPanel.style.display = 'block';
    resultsPanel.classList.add('slide-in-right');
}

/**
 * Display batch analysis results
 */
function displayBatchResults(result) {
    const batchResults = document.getElementById('batchResults');
    const batchSummary = document.getElementById('batchSummary');
    const batchResultsTable = document.getElementById('batchResultsTable');
    const tableBody = document.getElementById('batchResultsTableBody');
    
    if (!batchResults || !batchSummary) return;
    
    // Create summary HTML
    const summaryHtml = `
        <div class="row text-center">
            <div class="col-md-3">
                <div class="stat-item">
                    <h4 class="text-primary">${result.total_processed}</h4>
                    <p class="mb-0">Total Procesados</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-item">
                    <h4 class="text-success">${result.confirmed_planets}</h4>
                    <p class="mb-0">Planetas Confirmados</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-item">
                    <h4 class="text-warning">${result.false_positives}</h4>
                    <p class="mb-0">Falsos Positivos</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-item">
                    <h4 class="text-info">${result.processing_time.toFixed(2)}s</h4>
                    <p class="mb-0">Tiempo de Procesamiento</p>
                </div>
            </div>
        </div>
        
        <div class="mt-3 text-center">
            <small class="text-muted">
                Procesado: ${new Date(result.analysis_timestamp).toLocaleString()}
            </small>
        </div>
    `;
    
    batchSummary.innerHTML = summaryHtml;
    batchResults.style.display = 'block';
    
    // Populate results table
    if (tableBody && result.results) {
        const tableRows = result.results.slice(0, 100).map(item => `
            <tr>
                <td>${item.index + 1}</td>
                <td>
                    <span class="badge ${item.prediction === 'CONFIRMED' ? 'bg-success' : 'bg-danger'}">
                        ${item.prediction === 'CONFIRMED' ? 'CONFIRMADO' : 'FALSO POSITIVO'}
                    </span>
                </td>
                <td>${(item.confidence * 100).toFixed(1)}%</td>
                <td>${(item.prob_confirmed * 100).toFixed(1)}%</td>
                <td>${(item.prob_false_positive * 100).toFixed(1)}%</td>
            </tr>
        `).join('');
        
        tableBody.innerHTML = tableRows;
        batchResultsTable.style.display = 'block';
        
        // Show message if results were truncated
        if (result.results.length > 100) {
            const truncateMessage = document.createElement('p');
            truncateMessage.className = 'text-muted text-center mt-2';
            truncateMessage.innerHTML = `<i class="fas fa-info-circle"></i> Mostrando los primeros 100 resultados de ${result.results.length} total`;
            batchResultsTable.appendChild(truncateMessage);
        }
    }
}

/**
 * Update dashboard statistics
 */
function updateDashboardStats() {
    document.getElementById('model-accuracy').textContent = '83.08%';
    document.getElementById('predictions-today').textContent = ExoplanetApp.statistics.predictionsToday;
    document.getElementById('confirmed-planets').textContent = ExoplanetApp.statistics.confirmedPlanets;
    document.getElementById('false-positives').textContent = ExoplanetApp.statistics.falsePositives;
}

/**
 * Update statistics after prediction
 */
function updateStatisticsAfterPrediction(result) {
    ExoplanetApp.statistics.predictionsToday++;
    
    if (result.prediction === 'CONFIRMED') {
        ExoplanetApp.statistics.confirmedPlanets++;
    } else {
        ExoplanetApp.statistics.falsePositives++;
    }
    
    updateDashboardStats();
}

/**
 * Update statistics after batch prediction
 */
function updateStatisticsAfterBatchPrediction(result) {
    ExoplanetApp.statistics.predictionsToday += result.total_processed;
    ExoplanetApp.statistics.confirmedPlanets += result.confirmed_planets;
    ExoplanetApp.statistics.falsePositives += result.false_positives;
    
    updateDashboardStats();
}

/**
 * Add analysis to history
 */
function addToAnalysisHistory(type, result) {
    const historyItem = {
        timestamp: new Date().toISOString(),
        type: type,
        result: result
    };
    
    ExoplanetApp.analysisHistory.unshift(historyItem);
    
    // Keep only last 50 items
    if (ExoplanetApp.analysisHistory.length > 50) {
        ExoplanetApp.analysisHistory = ExoplanetApp.analysisHistory.slice(0, 50);
    }
    
    // Save to localStorage
    localStorage.setItem('exoplanet_history', JSON.stringify(ExoplanetApp.analysisHistory));
}

/**
 * Load analysis history
 */
function loadAnalysisHistory() {
    const stored = localStorage.getItem('exoplanet_history');
    if (stored) {
        ExoplanetApp.analysisHistory = JSON.parse(stored);
    }
    
    displayAnalysisHistory();
}

/**
 * Display analysis history
 */
function displayAnalysisHistory() {
    const historyTableBody = document.getElementById('historyTableBody');
    if (!historyTableBody) return;
    
    if (ExoplanetApp.analysisHistory.length === 0) {
        historyTableBody.innerHTML = `
            <tr>
                <td colspan="5" class="text-center text-muted">
                    No hay análisis previos
                </td>
            </tr>
        `;
        return;
    }
    
    const historyRows = ExoplanetApp.analysisHistory.map(item => {
        const date = new Date(item.timestamp).toLocaleString();
        const typeLabel = item.type === 'individual' ? 'Individual' : 'Por Lotes';
        
        let results, confirmed;
        if (item.type === 'individual') {
            results = '1 objeto';
            confirmed = item.result.prediction === 'CONFIRMED' ? 1 : 0;
        } else {
            results = `${item.result.total_processed} objetos`;
            confirmed = item.result.confirmed_planets;
        }
        
        return `
            <tr>
                <td>${date}</td>
                <td>
                    <span class="badge ${item.type === 'individual' ? 'bg-primary' : 'bg-info'}">
                        ${typeLabel}
                    </span>
                </td>
                <td>${results}</td>
                <td>${confirmed}</td>
                <td>
                    <button class="btn btn-sm btn-outline-primary" onclick="viewHistoryDetails('${item.timestamp}')">
                        <i class="fas fa-eye"></i> Ver
                    </button>
                </td>
            </tr>
        `;
    }).join('');
    
    historyTableBody.innerHTML = historyRows;
}

/**
 * View history details
 */
function viewHistoryDetails(timestamp) {
    const item = ExoplanetApp.analysisHistory.find(h => h.timestamp === timestamp);
    if (!item) return;
    
    // Create modal or detailed view
    // For now, just log to console
    console.log('History item details:', item);
    
    // You could implement a modal here to show detailed results
    alert('Funcionalidad de detalles en desarrollo. Ver consola para más información.');
}

/**
 * Show loading state
 */
function showLoadingState() {
    const loadingPanel = document.getElementById('loadingPanel');
    const resultsPanel = document.getElementById('predictionResults');
    
    if (loadingPanel) loadingPanel.style.display = 'block';
    if (resultsPanel) resultsPanel.style.display = 'none';
}

/**
 * Hide loading state
 */
function hideLoadingState() {
    const loadingPanel = document.getElementById('loadingPanel');
    if (loadingPanel) loadingPanel.style.display = 'none';
}

/**
 * Show batch loading state
 */
function showBatchLoadingState() {
    const batchLoading = document.getElementById('batchLoading');
    const batchResults = document.getElementById('batchResults');
    
    if (batchLoading) batchLoading.style.display = 'block';
    if (batchResults) batchResults.style.display = 'none';
}

/**
 * Hide batch loading state
 */
function hideBatchLoadingState() {
    const batchLoading = document.getElementById('batchLoading');
    if (batchLoading) batchLoading.style.display = 'none';
}

/**
 * Show error message
 */
function showErrorMessage(message) {
    // Create error alert
    const alert = document.createElement('div');
    alert.className = 'alert alert-danger alert-dismissible fade show error-message';
    alert.innerHTML = `
        <i class="fas fa-exclamation-triangle"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at top of current section
    const currentSection = document.getElementById(ExoplanetApp.currentSection);
    if (currentSection) {
        currentSection.insertBefore(alert, currentSection.firstChild);
    }
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (alert.parentNode) {
            alert.remove();
        }
    }, 5000);
}

/**
 * Show success message
 */
function showSuccessMessage(message) {
    const alert = document.createElement('div');
    alert.className = 'alert alert-success alert-dismissible fade show success-message';
    alert.innerHTML = `
        <i class="fas fa-check-circle"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const currentSection = document.getElementById(ExoplanetApp.currentSection);
    if (currentSection) {
        currentSection.insertBefore(alert, currentSection.firstChild);
    }
    
    setTimeout(() => {
        if (alert.parentNode) {
            alert.remove();
        }
    }, 3000);
}

/**
 * Get confidence level for styling
 */
function getConfidenceLevel(confidence) {
    if (confidence >= 0.8) return 'high';
    if (confidence >= 0.6) return 'medium';
    return 'low';
}

/**
 * Get interpretation text for results
 */
function getInterpretationText(result) {
    const isConfirmed = result.prediction === 'CONFIRMED';
    const confidence = result.confidence;
    
    if (isConfirmed) {
        if (confidence > 0.9) {
            return 'Este candidato tiene una probabilidad muy alta de ser un exoplaneta real. Se recomienda fuertemente el seguimiento observacional para confirmación definitiva.';
        } else if (confidence > 0.7) {
            return 'Este candidato tiene una buena probabilidad de ser un exoplaneta. Se recomienda seguimiento observacional adicional.';
        } else {
            return 'Este candidato podría ser un exoplaneta, pero la confianza es moderada. Requiere análisis adicional.';
        }
    } else {
        if (confidence > 0.9) {
            return 'Este candidato es muy probablemente un falso positivo. La señal puede deberse a eclipses binarios, ruido instrumental, o contaminación de fuentes cercanas.';
        } else if (confidence > 0.7) {
            return 'Este candidato probablemente es un falso positivo, pero se recomienda análisis adicional para confirmar.';
        } else {
            return 'La clasificación de este candidato es incierta. Se requiere análisis manual adicional.';
        }
    }
}

/**
 * Create feature importance HTML
 */
function createFeatureImportanceHtml(importance) {
    if (!importance || Object.keys(importance).length === 0) {
        return '';
    }
    
    // Sort features by importance
    const sortedFeatures = Object.entries(importance)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 5); // Top 5 features
    
    const featuresHtml = sortedFeatures.map(([feature, value]) => `
        <div class="feature-item">
            <span class="feature-name">${getFeatureDisplayName(feature)}</span>
            <div class="feature-bar">
                <div class="feature-bar-fill" style="width: ${value * 100}%"></div>
            </div>
            <span class="feature-value">${(value * 100).toFixed(1)}%</span>
        </div>
    `).join('');
    
    return `
        <div class="feature-importance mt-3">
            <h6><i class="fas fa-chart-bar text-info"></i> Características Más Importantes</h6>
            ${featuresHtml}
        </div>
    `;
}

/**
 * Get display name for features
 */
function getFeatureDisplayName(feature) {
    const displayNames = {
        'period': 'Período Orbital',
        'radius': 'Radio Planetario',
        'temp': 'Temperatura',
        'starRadius': 'Radio Estelar',
        'starMass': 'Masa Estelar',
        'starTemp': 'Temperatura Estelar',
        'depth': 'Profundidad Tránsito',
        'duration': 'Duración Tránsito',
        'snr': 'Relación S/N',
        'planet_star_radius_ratio': 'Ratio Radio Planeta/Estrella',
        'equilibrium_temp_ratio': 'Ratio Temperaturas',
        'transit_depth_expected': 'Profundidad Esperada',
        'orbital_velocity': 'Velocidad Orbital'
    };
    
    return displayNames[feature] || feature;
}

/**
 * Validate CSV file
 */
function validateCsvFile(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    // Check file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showErrorMessage('El archivo es demasiado grande. Máximo 10MB.');
        event.target.value = '';
        return;
    }
    
    // Check file type
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showErrorMessage('Solo se permiten archivos CSV.');
        event.target.value = '';
        return;
    }
    
    console.log(`📄 CSV file selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`);
}

/**
 * Add real-time form validation
 */
function addFormValidation() {
    const numberInputs = document.querySelectorAll('input[type="number"]');
    
    numberInputs.forEach(input => {
        input.addEventListener('input', function() {
            const value = parseFloat(this.value);
            
            // Remove previous validation classes
            this.classList.remove('is-valid', 'is-invalid');
            
            if (!isNaN(value) && value > 0) {
                this.classList.add('is-valid');
            } else if (this.value !== '') {
                this.classList.add('is-invalid');
            }
        });
        
        input.addEventListener('blur', function() {
            // Additional validation on blur
            const value = parseFloat(this.value);
            const fieldName = this.id;
            
            // Specific field validations
            if (fieldName === 'period' && (value < 0.1 || value > 10000)) {
                this.classList.add('is-invalid');
                this.setCustomValidity('Período debe estar entre 0.1 y 10,000 días');
            } else if (fieldName === 'radius' && (value < 0.1 || value > 100)) {
                this.classList.add('is-invalid');
                this.setCustomValidity('Radio debe estar entre 0.1 y 100');
            } else if (fieldName === 'starTemp' && (value < 1000 || value > 50000)) {
                this.classList.add('is-invalid');
                this.setCustomValidity('Temperatura debe estar entre 1,000 y 50,000 K');
            } else {
                this.setCustomValidity('');
            }
        });
    });
}

/**
 * Refresh dashboard data
 */
function refreshDashboard() {
    console.log('📊 Refreshing dashboard...');
    updateDashboardStats();
    
    // Refresh charts if they exist
    if (window.Charts) {
        window.Charts.updatePerformanceChart();
        window.Charts.updateDistributionChart();
    }
}

/**
 * Refresh model statistics
 */
async function refreshModelStats() {
    console.log('🔄 Refreshing model statistics...');
    
    try {
        await loadModelInfo();
        showSuccessMessage('Estadísticas del modelo actualizadas');
    } catch (error) {
        console.error('Error refreshing model stats:', error);
        showErrorMessage('Error al actualizar estadísticas');
    }
}

/**
 * Load model information
 */
async function loadModelInfo() {
    try {
        const modelInfo = await ApiClient.getModelInfo();
        displayModelInfo(modelInfo);
    } catch (error) {
        console.error('Error loading model info:', error);
        // Use fallback data
        const fallbackInfo = {
            model_type: 'Ensemble (Stacking)',
            base_models: ['Random Forest', 'AdaBoost', 'Extra Trees', 'LightGBM'],
            training_accuracy: 0.8308,
            features_used: 13,
            last_updated: '2025-01-01',
            target_accuracy: 0.83
        };
        displayModelInfo(fallbackInfo);
    }
}

/**
 * Display model information
 */
function displayModelInfo(modelInfo) {
    const specificationsDiv = document.getElementById('modelSpecifications');
    if (!specificationsDiv) return;
    
    const specificationsHtml = `
        <div class="row">
            <div class="col-md-6">
                <h6 class="text-primary">Tipo de Modelo</h6>
                <p>${modelInfo.model_type}</p>
                
                <h6 class="text-primary">Modelos Base</h6>
                <ul class="list-unstyled">
                    ${modelInfo.base_models.map(model => `<li><i class="fas fa-cog text-secondary"></i> ${model}</li>`).join('')}
                </ul>
            </div>
            <div class="col-md-6">
                <h6 class="text-primary">Accuracy de Entrenamiento</h6>
                <p class="text-success"><strong>${(modelInfo.training_accuracy * 100).toFixed(2)}%</strong></p>
                
                <h6 class="text-primary">Características Utilizadas</h6>
                <p>${modelInfo.features_used}</p>
                
                <h6 class="text-primary">Última Actualización</h6>
                <p>${modelInfo.last_updated}</p>
            </div>
        </div>
        
        <div class="mt-3 p-3 bg-light rounded">
            <h6 class="text-info"><i class="fas fa-info-circle"></i> Objetivo de Rendimiento</h6>
            <p class="mb-0">
                Basado en investigación publicada, el objetivo es alcanzar <strong>${(modelInfo.target_accuracy * 100).toFixed(0)}% de accuracy</strong> 
                usando algoritmos ensemble optimizados para la detección de exoplanetas.
            </p>
        </div>
    `;
    
    specificationsDiv.innerHTML = specificationsHtml;
    
    // Update metrics chart if it exists
    if (window.Charts) {
        window.Charts.updateMetricsChart(modelInfo);
    }
}

// Export for use in other modules
window.ExoplanetApp = ExoplanetApp;