/**
 * Initialize charts when DOM is ready
 * This file ensures charts are loaded after all dependencies
 */

// Wait for all dependencies to load
function initializeCharts() {
    // Check if Chart.js is loaded
    if (typeof Chart === 'undefined') {
        console.log('⏳ Waiting for Chart.js to load...');
        setTimeout(initializeCharts, 100);
        return;
    }

    // Check if main Charts manager is loaded
    if (typeof Charts === 'undefined') {
        console.log('⏳ Waiting for Charts manager to load...');
        setTimeout(initializeCharts, 100);
        return;
    }

    // Initialize the charts
    Charts.initialize();
    console.log('✅ Charts initialization completed');
}

// Start initialization when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeCharts);
} else {
    initializeCharts();
}