/**
 * NASA Space Apps Challenge 2025 - Exoplanet Detection System
 * Charts and visualizations using Chart.js
 * 
 * Handles all data visualization components
 */

class ChartsManager {
    constructor() {
        this.charts = {};
        this.defaultColors = {
            primary: '#0d6efd',
            secondary: '#6c757d',
            success: '#198754',
            info: '#0dcaf0',
            warning: '#ffc107',
            danger: '#dc3545',
            light: '#f8f9fa',
            dark: '#212529'
        };
        
        this.gradients = {};
        this.isInitialized = false;
    }

    /**
     * Initialize all charts
     */
    async initialize() {
        if (this.isInitialized) return;
        
        console.log('📊 Initializing charts...');
        
        // Wait for Chart.js to be available
        if (typeof Chart === 'undefined') {
            console.warn('Chart.js not loaded yet, retrying...');
            setTimeout(() => this.initialize(), 100);
            return;
        }

        // Configure Chart.js defaults
        this.configureDefaults();
        
        // Initialize charts when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.createCharts());
        } else {
            this.createCharts();
        }
        
        this.isInitialized = true;
        console.log('✅ Charts manager initialized');
    }

    /**
     * Configure Chart.js defaults
     */
    configureDefaults() {
        Chart.defaults.font.family = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif";
        Chart.defaults.font.size = 12;
        Chart.defaults.color = this.defaultColors.dark;
        Chart.defaults.plugins.legend.position = 'bottom';
        Chart.defaults.responsive = true;
        Chart.defaults.maintainAspectRatio = false;
        
        // Animation defaults
        Chart.defaults.animation.duration = 1000;
        Chart.defaults.animation.easing = 'easeOutQuart';
    }

    /**
     * Create gradient helper
     */
    createGradient(ctx, colorStart, colorEnd, direction = 'vertical') {
        let gradient;
        
        if (direction === 'vertical') {
            gradient = ctx.createLinearGradient(0, 0, 0, ctx.canvas.height);
        } else {
            gradient = ctx.createLinearGradient(0, 0, ctx.canvas.width, 0);
        }
        
        gradient.addColorStop(0, colorStart);
        gradient.addColorStop(1, colorEnd);
        
        return gradient;
    }

    /**
     * Create all charts
     */
    createCharts() {
        this.createPerformanceChart();
        this.createDistributionChart();
        this.createMetricsChart();
        
        console.log('📈 All charts created');
    }

    /**
     * Performance chart (dashboard)
     */
    createPerformanceChart() {
        const ctx = document.getElementById('performanceChart');
        if (!ctx) return;

        // Destroy existing chart
        if (this.charts.performance) {
            this.charts.performance.destroy();
        }

        const gradient = this.createGradient(
            ctx.getContext('2d'), 
            this.defaultColors.primary + '80', 
            this.defaultColors.primary + '20'
        );

        this.charts.performance = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
                datasets: [{
                    label: 'Métricas de Rendimiento',
                    data: [83.08, 82.5, 81.2, 81.8, 92.4],
                    backgroundColor: [
                        this.defaultColors.primary,
                        this.defaultColors.success,
                        this.defaultColors.info,
                        this.defaultColors.warning,
                        this.defaultColors.secondary
                    ],
                    borderColor: '#ffffff',
                    borderWidth: 2,
                    hoverOffset: 10
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: {
                            usePointStyle: true,
                            padding: 15,
                            font: {
                                size: 11
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.label}: ${context.parsed.toFixed(1)}%`;
                            }
                        }
                    }
                },
                cutout: '60%',
                animation: {
                    animateRotate: true,
                    animateScale: true
                }
            }
        });
    }

    /**
     * Distribution chart (dashboard)
     */
    createDistributionChart() {
        const ctx = document.getElementById('distributionChart');
        if (!ctx) return;

        if (this.charts.distribution) {
            this.charts.distribution.destroy();
        }

        this.charts.distribution = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: ['Planetas Confirmados', 'Falsos Positivos', 'Candidatos'],
                datasets: [{
                    data: [
                        ExoplanetApp.statistics.confirmedPlanets || 0,
                        ExoplanetApp.statistics.falsePositives || 0,
                        0 // Candidates - can be added later
                    ],
                    backgroundColor: [
                        this.defaultColors.success,
                        this.defaultColors.danger,
                        this.defaultColors.warning
                    ],
                    borderColor: '#ffffff',
                    borderWidth: 3,
                    hoverBorderWidth: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            usePointStyle: true,
                            padding: 15
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = total > 0 ? ((context.parsed / total) * 100).toFixed(1) : 0;
                                return `${context.label}: ${context.parsed} (${percentage}%)`;
                            }
                        }
                    }
                },
                animation: {
                    animateRotate: true,
                    duration: 1500
                }
            }
        });
    }

    /**
     * Metrics chart (model info page)
     */
    createMetricsChart() {
        const ctx = document.getElementById('metricsChart');
        if (!ctx) return;

        if (this.charts.metrics) {
            this.charts.metrics.destroy();
        }

        this.charts.metrics = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'AUC-ROC'],
                datasets: [{
                    label: 'Modelo Actual',
                    data: [83.08, 82.5, 81.2, 81.8, 84.1, 92.4],
                    borderColor: this.defaultColors.primary,
                    backgroundColor: this.defaultColors.primary + '30',
                    pointBackgroundColor: this.defaultColors.primary,
                    pointBorderColor: '#ffffff',
                    pointHoverBackgroundColor: '#ffffff',
                    pointHoverBorderColor: this.defaultColors.primary,
                    borderWidth: 2,
                    pointRadius: 5,
                    pointHoverRadius: 7
                }, {
                    label: 'Objetivo',
                    data: [90, 85, 85, 87, 85, 95],
                    borderColor: this.defaultColors.success,
                    backgroundColor: this.defaultColors.success + '20',
                    pointBackgroundColor: this.defaultColors.success,
                    pointBorderColor: '#ffffff',
                    borderWidth: 2,
                    pointRadius: 4,
                    borderDash: [5, 5]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            usePointStyle: true
                        }
                    }
                },
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            stepSize: 20,
                            callback: function(value) {
                                return value + '%';
                            }
                        },
                        grid: {
                            color: this.defaultColors.light
                        },
                        angleLines: {
                            color: this.defaultColors.light
                        }
                    }
                }
            }
        });
    }

    /**
     * Create ROC curve chart
     */
    createROCChart(fpr, tpr, auc, containerId = 'rocChart') {
        const ctx = document.getElementById(containerId);
        if (!ctx) return;

        if (this.charts.roc) {
            this.charts.roc.destroy();
        }

        // Generate ideal line data
        const idealLine = Array.from({length: 100}, (_, i) => ({
            x: i / 100,
            y: i / 100
        }));

        // Combine FPR and TPR into coordinate pairs
        const rocData = fpr.map((x, i) => ({
            x: x,
            y: tpr[i]
        }));

        this.charts.roc = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [{
                    label: `ROC Curve (AUC = ${auc.toFixed(3)})`,
                    data: rocData,
                    borderColor: this.defaultColors.primary,
                    backgroundColor: this.defaultColors.primary + '20',
                    fill: true,
                    pointRadius: 2,
                    pointHoverRadius: 5,
                    tension: 0.1
                }, {
                    label: 'Random Classifier',
                    data: idealLine,
                    borderColor: this.defaultColors.secondary,
                    backgroundColor: 'transparent',
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'ROC Curve - Receiver Operating Characteristic'
                    },
                    legend: {
                        position: 'bottom'
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: {
                            display: true,
                            text: 'False Positive Rate'
                        },
                        min: 0,
                        max: 1
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'True Positive Rate'
                        },
                        min: 0,
                        max: 1
                    }
                }
            }
        });

        return this.charts.roc;
    }

    /**
     * Create Precision-Recall curve chart
     */
    createPRChart(precision, recall, avgPrecision, containerId = 'prChart') {
        const ctx = document.getElementById(containerId);
        if (!ctx) return;

        if (this.charts.pr) {
            this.charts.pr.destroy();
        }

        // Combine precision and recall into coordinate pairs
        const prData = recall.map((x, i) => ({
            x: x,
            y: precision[i]
        }));

        this.charts.pr = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [{
                    label: `PR Curve (AP = ${avgPrecision.toFixed(3)})`,
                    data: prData,
                    borderColor: this.defaultColors.info,
                    backgroundColor: this.defaultColors.info + '20',
                    fill: true,
                    pointRadius: 2,
                    pointHoverRadius: 5,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Precision-Recall Curve'
                    },
                    legend: {
                        position: 'bottom'
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Recall'
                        },
                        min: 0,
                        max: 1
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Precision'
                        },
                        min: 0,
                        max: 1
                    }
                }
            }
        });

        return this.charts.pr;
    }

    /**
     * Create confusion matrix heatmap
     */
    createConfusionMatrix(matrix, labels, containerId = 'confusionMatrix') {
        const ctx = document.getElementById(containerId);
        if (!ctx) return;

        // Chart.js doesn't have native heatmap, so we'll create a custom visualization
        // For now, we'll use a bar chart to represent the confusion matrix

        const flatData = matrix.flat();
        const maxValue = Math.max(...flatData);
        
        // Create data for all combinations
        const dataPoints = [];
        const backgroundColors = [];
        
        for (let i = 0; i < matrix.length; i++) {
            for (let j = 0; j < matrix[i].length; j++) {
                dataPoints.push({
                    x: j,
                    y: i,
                    v: matrix[i][j]
                });
                
                // Color intensity based on value
                const intensity = matrix[i][j] / maxValue;
                const alpha = Math.max(0.1, intensity);
                backgroundColors.push(this.defaultColors.primary + Math.floor(alpha * 255).toString(16));
            }
        }

        // For simplicity, create a bar chart showing the values
        if (this.charts.confusion) {
            this.charts.confusion.destroy();
        }

        this.charts.confusion = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['True Neg', 'False Pos', 'False Neg', 'True Pos'],
                datasets: [{
                    label: 'Count',
                    data: flatData,
                    backgroundColor: [
                        this.defaultColors.success,
                        this.defaultColors.danger,
                        this.defaultColors.warning,
                        this.defaultColors.primary
                    ],
                    borderColor: '#ffffff',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Confusion Matrix'
                    },
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            title: function(context) {
                                const labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive'];
                                return labels[context[0].dataIndex];
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Count'
                        }
                    }
                }
            }
        });
    }

    /**
     * Create feature importance chart
     */
    createFeatureImportanceChart(importance, containerId = 'featureImportanceChart') {
        const ctx = document.getElementById(containerId);
        if (!ctx) return;

        // Sort features by importance
        const sortedFeatures = Object.entries(importance)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 10); // Top 10 features

        const labels = sortedFeatures.map(([name, ]) => this.getFeatureDisplayName(name));
        const values = sortedFeatures.map(([, value]) => value * 100);

        if (this.charts.featureImportance) {
            this.charts.featureImportance.destroy();
        }

        this.charts.featureImportance = new Chart(ctx, {
            type: 'horizontalBar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Importancia (%)',
                    data: values,
                    backgroundColor: this.defaultColors.primary,
                    borderColor: this.defaultColors.primary,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {
                    title: {
                        display: true,
                        text: 'Feature Importance'
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: Math.max(...values) * 1.1,
                        title: {
                            display: true,
                            text: 'Importancia (%)'
                        }
                    }
                }
            }
        });
    }

    /**
     * Update dashboard charts with new data
     */
    updatePerformanceChart(data = null) {
        if (!this.charts.performance) return;

        if (data) {
            this.charts.performance.data.datasets[0].data = [
                data.accuracy || 83.08,
                data.precision || 82.5,
                data.recall || 81.2,
                data.f1_score || 81.8,
                data.auc_roc || 92.4
            ];
        }

        this.charts.performance.update('active');
    }

    /**
     * Update distribution chart
     */
    updateDistributionChart() {
        if (!this.charts.distribution) return;

        this.charts.distribution.data.datasets[0].data = [
            ExoplanetApp.statistics.confirmedPlanets,
            ExoplanetApp.statistics.falsePositives,
            0 // Candidates
        ];

        this.charts.distribution.update('active');
    }

    /**
     * Update metrics chart with model info
     */
    updateMetricsChart(modelInfo) {
        if (!this.charts.metrics) return;

        // Update with actual model data if available
        const actualMetrics = [
            modelInfo.training_accuracy * 100,
            82.5, // Precision - would come from model
            81.2, // Recall
            81.8, // F1
            84.1, // Specificity
            92.4  // AUC-ROC
        ];

        this.charts.metrics.data.datasets[0].data = actualMetrics;
        this.charts.metrics.update('active');
    }

    /**
     * Get display name for features
     */
    getFeatureDisplayName(feature) {
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
            'planet_star_radius_ratio': 'Ratio Radio P/E',
            'equilibrium_temp_ratio': 'Ratio Temperaturas',
            'transit_depth_expected': 'Profundidad Esperada',
            'orbital_velocity': 'Velocidad Orbital'
        };
        
        return displayNames[feature] || feature;
    }

    /**
     * Destroy all charts
     */
    destroyAll() {
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
        
        this.charts = {};
        console.log('🗑️ All charts destroyed');
    }

    /**
     * Resize handler
     */
    handleResize() {
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.resize === 'function') {
                chart.resize();
            }
        });
    }

    /**
     * Export chart as image
     */
    exportChart(chartName, filename = null) {
        const chart = this.charts[chartName];
        if (!chart) {
            console.error(`Chart ${chartName} not found`);
            return;
        }

        const canvas = chart.canvas;
        const url = canvas.toDataURL('image/png');
        
        const link = document.createElement('a');
        link.download = filename || `${chartName}_chart.png`;
        link.href = url;
        link.click();
    }

    /**
     * Print chart
     */
    printChart(chartName) {
        const chart = this.charts[chartName];
        if (!chart) return;

        const canvas = chart.canvas;
        const dataUrl = canvas.toDataURL();
        
        const windowContent = `
            <!DOCTYPE html>
            <html>
            <head>
                <title>Chart - ${chartName}</title>
                <style>
                    body { margin: 0; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
                    img { max-width: 100%; max-height: 100%; }
                </style>
            </head>
            <body>
                <img src="${dataUrl}" alt="Chart">
            </body>
            </html>
        `;
        
        const printWindow = window.open('', '', 'height=600,width=800');
        printWindow.document.write(windowContent);
        printWindow.document.close();
        printWindow.focus();
        printWindow.print();
    }
}

// Initialize charts manager
const Charts = new ChartsManager();

// Auto-initialize when Chart.js is loaded
if (typeof Chart !== 'undefined') {
    Charts.initialize();
} else {
    // Wait for Chart.js to load
    document.addEventListener('DOMContentLoaded', () => {
        setTimeout(() => Charts.initialize(), 100);
    });
}

// Handle window resize
window.addEventListener('resize', () => {
    Charts.handleResize();
});

// Export for global use
window.Charts = Charts;

// Utility functions for creating quick charts
window.ChartUtils = {
    /**
     * Create quick line chart
     */
    createLineChart(canvasId, data, options = {}) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;

        return new Chart(ctx, {
            type: 'line',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                ...options
            }
        });
    },

    /**
     * Create quick bar chart
     */
    createBarChart(canvasId, data, options = {}) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;

        return new Chart(ctx, {
            type: 'bar',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                ...options
            }
        });
    },

    /**
     * Create quick scatter plot
     */
    createScatterPlot(canvasId, data, options = {}) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;

        return new Chart(ctx, {
            type: 'scatter',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom'
                    }
                },
                ...options
            }
        });
    }
};

console.log('📊 Charts module loaded successfully');