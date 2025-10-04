/**
 * NASA Space Apps Challenge 2025 - Exoplanet Detection System
 * API Client for backend communication
 * 
 * Handles all HTTP requests to the FastAPI backend
 */

class ApiClient {
    constructor() {
        this.baseUrl = 'http://localhost:8000/api';
        this.timeout = 30000; // 30 seconds
    }

    /**
     * Make HTTP request with error handling
     */
    async makeRequest(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            timeout: this.timeout
        };

        const requestOptions = { ...defaultOptions, ...options };
        
        // Handle FormData (for file uploads)
        if (options.body instanceof FormData) {
            delete requestOptions.headers['Content-Type'];
        }

        console.log(`🌐 API Request: ${options.method || 'GET'} ${url}`);

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), this.timeout);
            
            const response = await fetch(url, {
                ...requestOptions,
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(
                    errorData.detail || 
                    `HTTP ${response.status}: ${response.statusText}`
                );
            }

            const data = await response.json();
            console.log(`✅ API Response: ${response.status}`, data);
            
            return data;

        } catch (error) {
            if (error.name === 'AbortError') {
                throw new Error('La solicitud tardó demasiado tiempo. Inténtalo de nuevo.');
            }
            
            console.error(`❌ API Error:`, error);
            
            // Network errors
            if (error.message.includes('fetch')) {
                throw new Error('Error de conexión. Verifica que el servidor esté funcionando.');
            }
            
            throw error;
        }
    }

    /**
     * Single exoplanet prediction
     */
    async predictSingle(features) {
        return await this.makeRequest('/predict', {
            method: 'POST',
            body: JSON.stringify(features)
        });
    }

    /**
     * Batch prediction from CSV file
     */
    async predictBatch(file) {
        const formData = new FormData();
        formData.append('file', file);

        return await this.makeRequest('/batch-predict', {
            method: 'POST',
            body: formData,
            timeout: 120000 // 2 minutes for batch processing
        });
    }

    /**
     * Get model information
     */
    async getModelInfo() {
        return await this.makeRequest('/model-info');
    }

    /**
     * Health check
     */
    async healthCheck() {
        return await this.makeRequest('/health');
    }

    /**
     * Test API connection
     */
    async testConnection() {
        try {
            const health = await this.healthCheck();
            return {
                connected: true,
                status: health.status,
                timestamp: health.timestamp
            };
        } catch (error) {
            return {
                connected: false,
                error: error.message
            };
        }
    }
}

// Create global instance
window.ApiClient = new ApiClient();

// Connection monitoring
class ConnectionMonitor {
    constructor() {
        this.isOnline = navigator.onLine;
        this.lastCheck = null;
        this.checkInterval = 30000; // 30 seconds
        
        this.bindEvents();
        this.startMonitoring();
    }

    bindEvents() {
        window.addEventListener('online', () => {
            this.isOnline = true;
            this.showConnectionStatus('Conexión restaurada', 'success');
            console.log('🌐 Connection restored');
        });

        window.addEventListener('offline', () => {
            this.isOnline = false;
            this.showConnectionStatus('Sin conexión a internet', 'warning');
            console.log('📶 Connection lost');
        });
    }

    startMonitoring() {
        setInterval(async () => {
            await this.checkApiConnection();
        }, this.checkInterval);

        // Initial check
        setTimeout(() => this.checkApiConnection(), 1000);
    }

    async checkApiConnection() {
        try {
            const result = await window.ApiClient.testConnection();
            this.lastCheck = new Date();

            if (!result.connected) {
                this.showConnectionStatus('Servidor no disponible', 'danger');
                console.warn('⚠️ API server not available:', result.error);
            } else {
                // Remove any existing error messages
                this.hideConnectionStatus();
                console.log('✅ API server is healthy');
            }

        } catch (error) {
            console.error('❌ Connection check failed:', error);
        }
    }

    showConnectionStatus(message, type) {
        // Remove existing status
        this.hideConnectionStatus();

        const statusDiv = document.createElement('div');
        statusDiv.id = 'connection-status';
        statusDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        statusDiv.style.cssText = `
            top: 70px;
            right: 20px;
            z-index: 9999;
            min-width: 300px;
        `;
        
        statusDiv.innerHTML = `
            <i class="fas ${this.getStatusIcon(type)}"></i>
            ${message}
            <button type="button" class="btn-close" onclick="this.hideConnectionStatus()"></button>
        `;

        document.body.appendChild(statusDiv);

        // Auto-hide success messages
        if (type === 'success') {
            setTimeout(() => this.hideConnectionStatus(), 3000);
        }
    }

    hideConnectionStatus() {
        const existing = document.getElementById('connection-status');
        if (existing) {
            existing.remove();
        }
    }

    getStatusIcon(type) {
        const icons = {
            'success': 'fa-check-circle',
            'warning': 'fa-exclamation-triangle',
            'danger': 'fa-times-circle',
            'info': 'fa-info-circle'
        };
        return icons[type] || 'fa-info-circle';
    }
}

// Enhanced API client with retry logic
class RobustApiClient extends ApiClient {
    constructor() {
        super();
        this.maxRetries = 3;
        this.retryDelay = 1000; // 1 second
        this.retryMultiplier = 2;
    }

    /**
     * Make request with retry logic
     */
    async makeRequestWithRetry(endpoint, options = {}, retryCount = 0) {
        try {
            return await this.makeRequest(endpoint, options);
        } catch (error) {
            // Don't retry client errors (4xx)
            if (error.message.includes('HTTP 4')) {
                throw error;
            }

            // Retry for network errors and server errors (5xx)
            if (retryCount < this.maxRetries) {
                const delay = this.retryDelay * Math.pow(this.retryMultiplier, retryCount);
                
                console.log(`🔄 Retrying request in ${delay}ms (attempt ${retryCount + 1}/${this.maxRetries})`);
                
                await this.sleep(delay);
                return await this.makeRequestWithRetry(endpoint, options, retryCount + 1);
            }

            throw error;
        }
    }

    /**
     * Sleep utility
     */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Override methods to use retry logic
     */
    async predictSingle(features) {
        return await this.makeRequestWithRetry('/predict', {
            method: 'POST',
            body: JSON.stringify(features)
        });
    }

    async predictBatch(file) {
        const formData = new FormData();
        formData.append('file', file);

        return await this.makeRequestWithRetry('/batch-predict', {
            method: 'POST',
            body: formData,
            timeout: 120000
        });
    }
}

// Request caching for model info
class CachedApiClient extends RobustApiClient {
    constructor() {
        super();
        this.cache = new Map();
        this.cacheTimeout = 5 * 60 * 1000; // 5 minutes
    }

    async getModelInfo() {
        const cacheKey = 'model-info';
        const cached = this.cache.get(cacheKey);

        if (cached && (Date.now() - cached.timestamp) < this.cacheTimeout) {
            console.log('📋 Using cached model info');
            return cached.data;
        }

        const data = await this.makeRequestWithRetry('/model-info');
        
        this.cache.set(cacheKey, {
            data: data,
            timestamp: Date.now()
        });

        return data;
    }

    clearCache() {
        this.cache.clear();
        console.log('🧹 API cache cleared');
    }
}

// Replace global instance with enhanced version
window.ApiClient = new CachedApiClient();

// Request interceptor for common headers
class ApiInterceptor {
    constructor(apiClient) {
        this.apiClient = apiClient;
        this.originalMakeRequest = apiClient.makeRequest.bind(apiClient);
    }

    install() {
        this.apiClient.makeRequest = this.interceptedMakeRequest.bind(this);
    }

    async interceptedMakeRequest(endpoint, options = {}) {
        // Add common headers
        const enhancedOptions = {
            ...options,
            headers: {
                'X-Requested-With': 'XMLHttpRequest',
                'X-Client-Version': '1.0.0',
                'X-Client-Timestamp': new Date().toISOString(),
                ...options.headers
            }
        };

        // Add request ID for tracking
        const requestId = this.generateRequestId();
        enhancedOptions.headers['X-Request-ID'] = requestId;

        console.log(`📋 Request ID: ${requestId}`);

        try {
            const result = await this.originalMakeRequest(endpoint, enhancedOptions);
            return result;
        } catch (error) {
            console.error(`❌ Request ${requestId} failed:`, error.message);
            throw error;
        }
    }

    generateRequestId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }
}

// Install interceptor
const interceptor = new ApiInterceptor(window.ApiClient);
interceptor.install();

// Initialize connection monitoring
document.addEventListener('DOMContentLoaded', () => {
    window.ConnectionMonitor = new ConnectionMonitor();
    console.log('🔍 Connection monitoring initialized');
});

// API utilities
const ApiUtils = {
    /**
     * Format file size
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    /**
     * Validate CSV structure
     */
    async validateCsvStructure(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = function(e) {
                try {
                    const csv = e.target.result;
                    const lines = csv.split('\n');
                    
                    if (lines.length < 2) {
                        reject(new Error('El archivo CSV debe tener al menos una fila de encabezados y una fila de datos'));
                        return;
                    }

                    const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
                    const requiredColumns = [
                        'period', 'radius', 'temp', 'starradius', 'starmass', 
                        'startemp', 'depth', 'duration', 'snr'
                    ];

                    const missingColumns = requiredColumns.filter(col => 
                        !headers.some(h => h.includes(col.toLowerCase()))
                    );

                    if (missingColumns.length > 0) {
                        reject(new Error(`Columnas faltantes: ${missingColumns.join(', ')}`));
                        return;
                    }

                    resolve({
                        valid: true,
                        headers: headers,
                        rowCount: lines.length - 1,
                        fileSize: file.size
                    });

                } catch (error) {
                    reject(new Error('Error al parsear el archivo CSV'));
                }
            };

            reader.onerror = () => reject(new Error('Error al leer el archivo'));
            reader.readAsText(file);
        });
    },

    /**
     * Download results as CSV
     */
    downloadResultsAsCsv(results, filename = 'exoplanet_results.csv') {
        const headers = ['Index', 'Prediction', 'Confidence', 'Prob_Confirmed', 'Prob_False_Positive'];
        const csvContent = [
            headers.join(','),
            ...results.map(result => [
                result.index + 1,
                result.prediction,
                result.confidence.toFixed(4),
                result.prob_confirmed.toFixed(4),
                result.prob_false_positive.toFixed(4)
            ].join(','))
        ].join('\n');

        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        
        if (link.download !== undefined) {
            const url = URL.createObjectURL(blob);
            link.setAttribute('href', url);
            link.setAttribute('download', filename);
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }
};

// Export utilities
window.ApiUtils = ApiUtils;

// Error tracking
class ErrorTracker {
    constructor() {
        this.errors = [];
        this.maxErrors = 100;
    }

    logError(error, context = {}) {
        const errorEntry = {
            timestamp: new Date().toISOString(),
            message: error.message,
            stack: error.stack,
            context: context
        };

        this.errors.unshift(errorEntry);

        // Keep only recent errors
        if (this.errors.length > this.maxErrors) {
            this.errors = this.errors.slice(0, this.maxErrors);
        }

        console.error('📝 Error logged:', errorEntry);
    }

    getErrors(limit = 10) {
        return this.errors.slice(0, limit);
    }

    clearErrors() {
        this.errors = [];
        console.log('🧹 Error log cleared');
    }

    exportErrorLog() {
        const logContent = JSON.stringify(this.errors, null, 2);
        const blob = new Blob([logContent], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = `error_log_${new Date().toISOString().split('T')[0]}.json`;
        link.click();
        
        URL.revokeObjectURL(url);
    }
}

// Global error tracker
window.ErrorTracker = new ErrorTracker();

// Global error handler
window.addEventListener('error', (event) => {
    window.ErrorTracker.logError(event.error, {
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno
    });
});

window.addEventListener('unhandledrejection', (event) => {
    window.ErrorTracker.logError(new Error(event.reason), {
        type: 'unhandled_promise_rejection'
    });
});

console.log('🔌 API Client initialized with enhanced features');