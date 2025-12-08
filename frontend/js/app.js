/**
 * WakeBuilder - Main Application Controller
 */

/**
 * Main application state and controller
 */
class App {
    constructor() {
        this.currentPage = 'home';
        this.models = [];
        this.filter = 'all';
        
        this.elements = {};
        this.bindElements();
        this.bindEvents();
    }

    /**
     * Bind DOM elements
     */
    bindElements() {
        this.elements.navLinks = document.querySelectorAll('.nav-link[data-page]');
        this.elements.pages = document.querySelectorAll('.page');
        this.elements.modelsGrid = document.getElementById('models-grid');
        this.elements.filterBtns = document.querySelectorAll('.filter-btn');
        this.elements.btnNewModel = document.getElementById('btn-new-model');
    }

    /**
     * Bind event handlers
     */
    bindEvents() {
        // Navigation
        this.elements.navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const page = link.dataset.page;
                this.navigateTo(page);
            });
        });

        // Filter buttons
        this.elements.filterBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                this.setFilter(btn.dataset.filter);
            });
        });

        // New model button
        this.elements.btnNewModel.addEventListener('click', () => {
            this.navigateTo('train');
        });
    }

    /**
     * Initialize the application
     */
    async initialize() {
        console.log('Initializing WakeBuilder...');
        
        // Check API health
        try {
            const health = await api.getHealth();
            console.log('API Status:', health.status, 'Version:', health.version);
        } catch (error) {
            console.error('API not available:', error);
            showToast('API not available. Please ensure the server is running.', 'error');
        }

        // Initialize sub-modules
        initTrainer();
        initTester();

        // Load initial data
        await this.loadModels();

        console.log('WakeBuilder initialized');
    }

    /**
     * Navigate to a page
     * @param {string} page - Page name (home, train, test)
     */
    navigateTo(page) {
        // Update navigation
        this.elements.navLinks.forEach(link => {
            link.classList.toggle('active', link.dataset.page === page);
        });

        // Update pages
        this.elements.pages.forEach(p => {
            p.classList.toggle('active', p.id === `page-${page}`);
        });

        this.currentPage = page;

        // Page-specific actions
        if (page === 'home') {
            this.loadModels();
        } else if (page === 'test') {
            if (window.tester) {
                window.tester.loadModels();
            }
        }
    }

    /**
     * Set model filter
     * @param {string} filter - Filter value (all, custom, default)
     */
    setFilter(filter) {
        this.filter = filter;
        
        this.elements.filterBtns.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.filter === filter);
        });

        this.renderModels();
    }

    /**
     * Load models from API
     */
    async loadModels() {
        try {
            this.elements.modelsGrid.innerHTML = '<div class="loading-spinner">Loading models...</div>';
            
            const result = await api.listModels();
            this.models = result.models || [];
            
            this.renderModels();

        } catch (error) {
            console.error('Failed to load models:', error);
            this.elements.modelsGrid.innerHTML = `
                <div class="loading-spinner">
                    Failed to load models. 
                    <button class="btn btn-secondary" onclick="app.loadModels()">Retry</button>
                </div>
            `;
        }
    }

    /**
     * Render models grid
     */
    renderModels() {
        // Filter models
        let filtered = this.models;
        if (this.filter !== 'all') {
            filtered = this.models.filter(m => m.category === this.filter);
        }

        // Check if empty
        if (filtered.length === 0) {
            this.elements.modelsGrid.innerHTML = `
                <div class="loading-spinner">
                    No models found. 
                    <button class="btn btn-primary" onclick="app.navigateTo('train')">Train your first model</button>
                </div>
            `;
            return;
        }

        // Render model cards
        this.elements.modelsGrid.innerHTML = filtered.map(model => this.renderModelCard(model)).join('');

        // Bind card events
        this.bindModelCardEvents();
    }

    /**
     * Render a single model card
     * @param {object} model - Model data
     * @returns {string} HTML string
     */
    renderModelCard(model) {
        const accuracy = model.metrics?.val_accuracy 
            ? formatPercent(model.metrics.val_accuracy) 
            : '-';
        const size = model.size_kb ? formatSize(model.size_kb) : '-';
        const date = model.created_at ? new Date(model.created_at).toLocaleDateString() : '-';

        return `
            <div class="model-card" data-model-id="${model.model_id}">
                <div class="model-card-header">
                    <span class="model-name">${model.wake_word}</span>
                    <span class="model-badge ${model.category}">${model.category}</span>
                </div>
                <div class="model-meta">
                    <span class="model-meta-item">
                        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                            <polyline points="22 4 12 14.01 9 11.01"/>
                        </svg>
                        ${accuracy}
                    </span>
                    <span class="model-meta-item">
                        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="2" y="2" width="20" height="8" rx="2" ry="2"/>
                            <rect x="2" y="14" width="20" height="8" rx="2" ry="2"/>
                        </svg>
                        ${model.model_type === 'ast' ? 'AST' : (model.model_type || 'AST')}
                    </span>
                    <span class="model-meta-item">
                        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                            <polyline points="17 8 12 3 7 8"/>
                            <line x1="12" y1="3" x2="12" y2="15"/>
                        </svg>
                        ${size}
                    </span>
                    <span class="model-meta-item">
                        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="3" y="4" width="18" height="18" rx="2" ry="2"/>
                            <line x1="16" y1="2" x2="16" y2="6"/>
                            <line x1="8" y1="2" x2="8" y2="6"/>
                            <line x1="3" y1="10" x2="21" y2="10"/>
                        </svg>
                        ${date}
                    </span>
                </div>
                <div class="model-actions">
                    <button class="btn btn-secondary btn-test" data-model-id="${model.model_id}">
                        Test
                    </button>
                    <button class="btn btn-secondary btn-download" data-model-id="${model.model_id}">
                        Download
                    </button>
                    ${model.category === 'custom' ? `
                        <button class="btn btn-secondary btn-delete" data-model-id="${model.model_id}">
                            Delete
                        </button>
                    ` : ''}
                </div>
            </div>
        `;
    }

    /**
     * Bind events to model cards
     */
    bindModelCardEvents() {
        // Click on card to view details
        document.querySelectorAll('.model-card').forEach(card => {
            card.addEventListener('click', () => {
                const modelId = card.dataset.modelId;
                this.showModelDetails(modelId);
            });
            card.style.cursor = 'pointer';
        });

        // Test buttons
        document.querySelectorAll('.btn-test').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const modelId = btn.dataset.modelId;
                this.testModel(modelId);
            });
        });

        // Download buttons
        document.querySelectorAll('.btn-download').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const modelId = btn.dataset.modelId;
                this.downloadModel(modelId);
            });
        });

        // Delete buttons
        document.querySelectorAll('.btn-delete').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const modelId = btn.dataset.modelId;
                this.deleteModel(modelId);
            });
        });
    }

    /**
     * Show model details in a modal
     * @param {string} modelId - Model ID
     */
    async showModelDetails(modelId) {
        try {
            const metadata = await api.getModelMetadata(modelId);
            
            // Create modal HTML
            const modalHtml = `
                <div class="modal-overlay" id="model-details-modal">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h2>${metadata.wake_word}</h2>
                            <button class="modal-close" onclick="app.closeModelDetails()">&times;</button>
                        </div>
                        <div class="modal-body">
                            <div class="model-details-grid">
                                <div class="detail-section">
                                    <h3>Performance</h3>
                                    <div class="detail-row">
                                        <span>Accuracy</span>
                                        <span>${formatPercent(metadata.metrics?.val_accuracy || 0)}</span>
                                    </div>
                                    <div class="detail-row">
                                        <span>F1 Score</span>
                                        <span>${formatNumber(metadata.metrics?.val_f1 || 0, 3)}</span>
                                    </div>
                                    <div class="detail-row">
                                        <span>Optimal Threshold</span>
                                        <span>${formatNumber(metadata.threshold || 0.5, 2)}</span>
                                    </div>
                                </div>
                                <div class="detail-section">
                                    <h3>Model Info</h3>
                                    <div class="detail-row">
                                        <span>Type</span>
                                        <span>${metadata.model_type === 'ast' ? 'AST' : (metadata.model_type || 'AST')}</span>
                                    </div>
                                    <div class="detail-row">
                                        <span>Parameters</span>
                                        <span>${metadata.parameters ? (metadata.parameters / 1000).toFixed(0) + 'K' : '-'}</span>
                                    </div>
                                    <div class="detail-row">
                                        <span>Size</span>
                                        <span>${metadata.size_kb ? formatSize(metadata.size_kb) : '-'}</span>
                                    </div>
                                    <div class="detail-row">
                                        <span>Created</span>
                                        <span>${metadata.created_at ? new Date(metadata.created_at).toLocaleString() : '-'}</span>
                                    </div>
                                </div>
                            </div>
                            ${metadata.training_config ? `
                                <div class="detail-section">
                                    <h3>Training Configuration</h3>
                                    <div class="config-grid">
                                        <div class="detail-row">
                                            <span>Batch Size</span>
                                            <span>${metadata.training_config.batch_size || '-'}</span>
                                        </div>
                                        <div class="detail-row">
                                            <span>Learning Rate</span>
                                            <span>${metadata.training_config.learning_rate || '-'}</span>
                                        </div>
                                        <div class="detail-row">
                                            <span>Dropout</span>
                                            <span>${metadata.training_config.dropout || '-'}</span>
                                        </div>
                                        <div class="detail-row">
                                            <span>Label Smoothing</span>
                                            <span>${metadata.training_config.label_smoothing || '-'}</span>
                                        </div>
                                        <div class="detail-row">
                                            <span>Mixup Alpha</span>
                                            <span>${metadata.training_config.mixup_alpha || '-'}</span>
                                        </div>
                                        <div class="detail-row">
                                            <span>Focal Loss</span>
                                            <span>${metadata.training_config.use_focal_loss ? 'Yes' : 'No'}</span>
                                        </div>
                                        <div class="detail-row">
                                            <span>Focal Gamma</span>
                                            <span>${metadata.training_config.use_focal_loss ? (metadata.training_config.focal_gamma || '2.0') : 'N/A'}</span>
                                        </div>
                                        <div class="detail-row">
                                            <span>Self-Attention</span>
                                            <span>${metadata.training_config.use_attention ? 'Yes' : 'No'}</span>
                                        </div>
                                        <div class="detail-row">
                                            <span>Classifier Layers</span>
                                            <span>${metadata.training_config.classifier_hidden_dims ? metadata.training_config.classifier_hidden_dims.join(', ') : '-'}</span>
                                        </div>
                                    </div>
                                </div>
                            ` : ''}
                            ${metadata.data_stats ? `
                                <div class="detail-section">
                                    <h3>Training Data</h3>
                                    <div class="config-grid">
                                        <div class="detail-row">
                                            <span>Positive Samples</span>
                                            <span>${metadata.data_stats.num_positive_samples || metadata.data_stats.num_train_samples || '-'}</span>
                                        </div>
                                        <div class="detail-row">
                                            <span>Negative Samples</span>
                                            <span>${metadata.data_stats.num_negative_samples || '-'}</span>
                                        </div>
                                        <div class="detail-row">
                                            <span>Training Time</span>
                                            <span>${metadata.training_time_seconds ? formatDuration(metadata.training_time_seconds) : '-'}</span>
                                        </div>
                                    </div>
                                </div>
                            ` : ''}
                            ${metadata.threshold_analysis ? `
                                <div class="detail-section">
                                    <h3>Threshold Analysis</h3>
                                    <canvas id="modal-threshold-chart" width="500" height="200"></canvas>
                                </div>
                            ` : ''}
                        </div>
                        <div class="modal-footer">
                            <button class="btn btn-secondary" onclick="app.closeModelDetails()">Close</button>
                            <button class="btn btn-primary" onclick="app.testModel('${modelId}'); app.closeModelDetails();">Test Model</button>
                        </div>
                    </div>
                </div>
            `;
            
            // Add modal to DOM
            document.body.insertAdjacentHTML('beforeend', modalHtml);
            
            // Draw threshold chart if available
            if (metadata.threshold_analysis) {
                const canvas = document.getElementById('modal-threshold-chart');
                const chart = new ThresholdChart(canvas);
                chart.setData(metadata.threshold_analysis, metadata.threshold);
            }
            
        } catch (error) {
            showToast('Failed to load model details: ' + error.message, 'error');
        }
    }

    /**
     * Close model details modal
     */
    closeModelDetails() {
        const modal = document.getElementById('model-details-modal');
        if (modal) {
            modal.remove();
        }
    }

    /**
     * Test a model
     * @param {string} modelId - Model ID
     */
    testModel(modelId) {
        this.navigateTo('test');
        setTimeout(() => {
            if (window.tester) {
                window.tester.selectModel(modelId);
            }
        }, 100);
    }

    /**
     * Download a model
     * @param {string} modelId - Model ID
     */
    async downloadModel(modelId) {
        try {
            const blob = await api.downloadModel(modelId);
            downloadBlob(blob, `${modelId}_model.zip`);
            showToast('Model downloaded!', 'success');
        } catch (error) {
            showToast('Failed to download model: ' + error.message, 'error');
        }
    }

    /**
     * Delete a model
     * @param {string} modelId - Model ID
     */
    async deleteModel(modelId) {
        if (!confirm(`Are you sure you want to delete this model?`)) {
            return;
        }

        try {
            await api.deleteModel(modelId);
            showToast('Model deleted', 'success');
            await this.loadModels();
        } catch (error) {
            showToast('Failed to delete model: ' + error.message, 'error');
        }
    }
}

// Global app instance
let app = null;

/**
 * Initialize tooltip system
 */
function initTooltips() {
    // Create overlay element
    const overlay = document.createElement('div');
    overlay.className = 'tooltip-overlay';
    document.body.appendChild(overlay);
    
    // Helper to close all tooltips
    const closeAllTooltips = () => {
        document.querySelectorAll('.tooltip-content.active').forEach(t => {
            t.classList.remove('active');
        });
        overlay.classList.remove('active');
    };
    
    // Handle tooltip icon clicks
    document.querySelectorAll('.tooltip-icon').forEach(icon => {
        icon.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            
            const tooltipId = `tooltip-${icon.dataset.tooltip}`;
            const tooltipContent = document.getElementById(tooltipId);
            
            if (tooltipContent) {
                // Close any open tooltips
                closeAllTooltips();
                
                // Show this tooltip
                tooltipContent.classList.add('active');
                overlay.classList.add('active');
            }
        });
    });
    
    // Handle tooltip close button clicks
    document.querySelectorAll('.tooltip-close').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            closeAllTooltips();
        });
    });
    
    // Close tooltip on overlay click
    overlay.addEventListener('click', closeAllTooltips);
    
    // Close tooltip on escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeAllTooltips();
        }
    });
}

/**
 * Initialize application on DOM ready
 */
document.addEventListener('DOMContentLoaded', () => {
    app = new App();
    app.initialize();
    window.app = app;
    
    // Initialize tooltips
    initTooltips();
});
