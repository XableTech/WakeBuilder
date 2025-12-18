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
        
        // Home page cache elements
        this.elements.homeCachePanel = document.getElementById('cache-panel-home');
        this.elements.homeCacheStatusText = document.getElementById('home-cache-status-text');
        this.elements.homeCacheChunkCount = document.getElementById('home-cache-chunk-count');
        this.elements.homeBuildCacheBtn = document.getElementById('home-build-cache-btn');
        this.elements.homeClearCacheBtn = document.getElementById('home-clear-cache-btn');
        this.elements.cacheRequiredBadge = document.getElementById('cache-required-badge');
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
        
        // Home page cache buttons
        if (this.elements.homeBuildCacheBtn) {
            this.elements.homeBuildCacheBtn.addEventListener('click', () => this.buildCache());
        }
        if (this.elements.homeClearCacheBtn) {
            this.elements.homeClearCacheBtn.addEventListener('click', () => this.clearCache());
        }
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
        
        // Check cache status and warn if missing
        await this.checkCacheStatus();
        
        // Check for orphaned recordings on startup
        this.checkOrphanedRecordings();

        console.log('WakeBuilder initialized');
    }
    
    /**
     * Check cache status and update home page cache panel
     */
    async checkCacheStatus() {
        try {
            const status = await api.getCacheStatus();
            console.log('Cache status:', status);
            
            this.updateHomeCacheDisplay(status);
        } catch (error) {
            console.error('Failed to check cache status:', error);
            if (this.elements.homeCacheStatusText) {
                this.elements.homeCacheStatusText.textContent = 'Failed to load cache info';
            }
        }
    }
    
    /**
     * Update home page cache display
     * @param {object} status - Cache status from API
     */
    updateHomeCacheDisplay(status) {
        const sourceFiles = status.source_files?.total || 0;
        const chunkCount = status.audio_cache?.chunk_count || 0;
        const specCount = status.spectrogram_cache?.count || 0;
        const isReady = status.training_ready;
        
        // Update panel styling
        if (this.elements.homeCachePanel) {
            this.elements.homeCachePanel.classList.remove('cache-missing', 'cache-ready');
            this.elements.homeCachePanel.classList.add(isReady ? 'cache-ready' : 'cache-missing');
        }
        
        // Update badge
        if (this.elements.cacheRequiredBadge) {
            this.elements.cacheRequiredBadge.textContent = isReady ? 'Ready' : 'Required';
        }
        
        // Update status text
        if (this.elements.homeCacheStatusText) {
            this.elements.homeCacheStatusText.textContent = `${sourceFiles.toLocaleString()} source files`;
        }
        
        // Update chunk count
        if (this.elements.homeCacheChunkCount) {
            if (isReady) {
                if (specCount > 0) {
                    this.elements.homeCacheChunkCount.textContent = `${chunkCount.toLocaleString()} chunks, ${specCount.toLocaleString()} spectrograms ✓`;
                } else {
                    this.elements.homeCacheChunkCount.textContent = `${chunkCount.toLocaleString()} chunks ready ✓`;
                }
                this.elements.homeCacheChunkCount.style.color = 'var(--color-success)';
            } else {
                this.elements.homeCacheChunkCount.textContent = '⚠️ No cache - Build required before training!';
                this.elements.homeCacheChunkCount.style.color = 'var(--color-warning)';
            }
        }
        
        // Toggle button states based on cache existence
        if (this.elements.homeBuildCacheBtn) {
            this.elements.homeBuildCacheBtn.disabled = isReady;
        }
        if (this.elements.homeClearCacheBtn) {
            this.elements.homeClearCacheBtn.disabled = !isReady;
        }
    }
    
    /**
     * Build cache from home page
     */
    async buildCache() {
        if (!this.elements.homeBuildCacheBtn) return;
        
        const originalText = this.elements.homeBuildCacheBtn.textContent;
        this.elements.homeBuildCacheBtn.textContent = 'Building...';
        this.elements.homeBuildCacheBtn.disabled = true;
        if (this.elements.homeClearCacheBtn) {
            this.elements.homeClearCacheBtn.disabled = true;
        }
        
        try {
            await api.buildAllCaches((event) => {
                if (event.type === 'start') {
                    const phase = event.phase || 'audio';
                    if (this.elements.homeCacheStatusText) {
                        this.elements.homeCacheStatusText.textContent = `Building ${phase} cache...`;
                    }
                } else if (event.type === 'progress') {
                    const phase = event.phase || 'audio';
                    const percent = event.percent || 0;
                    if (this.elements.homeCacheStatusText) {
                        this.elements.homeCacheStatusText.textContent = `Building ${phase} cache... ${percent}%`;
                    }
                    if (event.processed && event.total && this.elements.homeCacheChunkCount) {
                        this.elements.homeCacheChunkCount.textContent = `${event.processed}/${event.total} files`;
                    }
                } else if (event.type === 'complete') {
                    showToast('Cache built successfully!', 'success');
                }
            });
            
            // Refresh cache status
            await this.checkCacheStatus();
            
        } catch (error) {
            console.error('Failed to build cache:', error);
            showToast('Failed to build cache: ' + error.message, 'error');
        } finally {
            this.elements.homeBuildCacheBtn.textContent = originalText;
            this.elements.homeBuildCacheBtn.disabled = false;
            if (this.elements.homeClearCacheBtn) {
                this.elements.homeClearCacheBtn.disabled = false;
            }
        }
    }
    
    /**
     * Clear cache from home page
     */
    async clearCache() {
        if (!confirm('Clear all cached data (audio chunks and spectrograms)?')) return;
        
        try {
            await api.clearAllCaches();
            await this.checkCacheStatus();
            showToast('Cache cleared', 'success');
        } catch (error) {
            console.error('Failed to clear cache:', error);
            showToast('Failed to clear cache: ' + error.message, 'error');
        }
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
                        <button class="btn btn-secondary btn-move-default" data-model-id="${model.model_id}" title="Move to Default">
                            ⭐
                        </button>
                    ` : ''}
                    <button class="btn btn-secondary btn-delete" data-model-id="${model.model_id}">
                        Delete
                    </button>
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

        // Download buttons (only those with data-model-id attribute)
        document.querySelectorAll('.btn-download[data-model-id]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const modelId = btn.dataset.modelId;
                this.downloadModel(modelId, btn);
            });
        });

        // Move to default buttons
        document.querySelectorAll('.btn-move-default').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const modelId = btn.dataset.modelId;
                this.moveModelToDefault(modelId);
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
                            <div class="detail-section" id="training-charts-section">
                                <h3>Training History</h3>
                                <div class="modal-charts-grid">
                                    <div class="modal-chart-container">
                                        <h4>Loss History</h4>
                                        <canvas id="modal-loss-chart" width="400" height="180"></canvas>
                                    </div>
                                    <div class="modal-chart-container">
                                        <h4>Accuracy & F1 History</h4>
                                        <canvas id="modal-accuracy-chart" width="400" height="180"></canvas>
                                    </div>
                                </div>
                            </div>
                            ${metadata.threshold_analysis ? `
                                <div class="detail-section">
                                    <h3>Threshold Analysis</h3>
                                    <canvas id="modal-threshold-chart" width="500" height="200"></canvas>
                                </div>
                            ` : ''}
                            <div class="detail-section" id="recordings-section">
                                <h3>Training Recordings</h3>
                                <div id="model-recordings-list" class="recordings-list-compact">
                                    <span class="loading-text">Loading recordings...</span>
                                </div>
                            </div>
                            <div class="detail-section">
                                <h3>ONNX Export</h3>
                                <div id="onnx-status-container" class="onnx-status">
                                    <span class="loading-text">Checking ONNX status...</span>
                                </div>
                            </div>
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
            
            // Load and draw training history charts
            this.loadTrainingHistoryCharts(modelId);
            
            // Load recordings
            this.loadModelRecordings(modelId);
            
            // Load ONNX status
            this.loadOnnxStatus(modelId);
            
        } catch (error) {
            showToast('Failed to load model details: ' + error.message, 'error');
        }
    }
    
    /**
     * Load and draw training history charts for a model
     * @param {string} modelId - Model ID
     */
    async loadTrainingHistoryCharts(modelId) {
        const chartsSection = document.getElementById('training-charts-section');
        if (!chartsSection) return;
        
        try {
            const history = await api.getTrainingHistory(modelId);
            
            // Draw loss chart
            const lossCanvas = document.getElementById('modal-loss-chart');
            if (lossCanvas && history.train_loss && history.val_loss) {
                const lossChart = new TrainingChart(lossCanvas);
                lossChart.setData(history.train_loss, history.val_loss);
            }
            
            // Draw accuracy/F1 chart
            const accCanvas = document.getElementById('modal-accuracy-chart');
            if (accCanvas && history.val_accuracy) {
                const accChart = new AccuracyChart(accCanvas);
                accChart.setData(history.val_accuracy, history.val_f1 || []);
            }
            
        } catch (error) {
            // Training history not available - hide the section
            chartsSection.innerHTML = `
                <h3>Training History</h3>
                <p class="no-data-text">Training history not available for this model</p>
            `;
        }
    }
    
    /**
     * Load recordings for a model in the details modal
     * @param {string} modelId - Model ID
     */
    async loadModelRecordings(modelId) {
        const container = document.getElementById('model-recordings-list');
        if (!container) return;
        
        try {
            const result = await api.getModelRecordings(modelId);
            
            if (result.count === 0) {
                container.innerHTML = '<span class="no-recordings">No recordings found</span>';
                return;
            }
            
            container.innerHTML = result.recordings.map(rec => `
                <div class="recording-item-compact">
                    <span class="recording-name">${rec.filename}</span>
                    <span class="recording-size">${rec.size_kb.toFixed(1)} KB</span>
                    <audio controls src="${rec.url}" preload="none"></audio>
                </div>
            `).join('');
            
        } catch (error) {
            container.innerHTML = '<span class="error-text">Failed to load recordings</span>';
        }
    }
    
    /**
     * Load ONNX status for a model in the details modal
     * @param {string} modelId - Model ID
     */
    async loadOnnxStatus(modelId) {
        const container = document.getElementById('onnx-status-container');
        if (!container) return;
        
        try {
            const status = await api.getOnnxStatus(modelId);
            
            if (status.onnx_available) {
                container.innerHTML = `
                    <div class="onnx-available">
                        <span class="onnx-badge success">ONNX Available</span>
                        <span class="onnx-size">${status.onnx_size_mb} MB</span>
                        <button class="btn btn-sm btn-danger" onclick="app.deleteOnnxExport('${modelId}')">Delete ONNX</button>
                    </div>
                `;
            } else {
                container.innerHTML = `
                    <div class="onnx-not-available">
                        <span class="onnx-badge">Not Exported</span>
                        <button class="btn btn-sm btn-primary" onclick="app.exportToOnnx('${modelId}')">Export to ONNX</button>
                    </div>
                `;
            }
            
        } catch (error) {
            container.innerHTML = '<span class="error-text">Failed to check ONNX status</span>';
        }
    }
    
    /**
     * Export model to ONNX format
     * @param {string} modelId - Model ID
     */
    async exportToOnnx(modelId) {
        const container = document.getElementById('onnx-status-container');
        if (container) {
            container.innerHTML = '<span class="loading-text">Exporting to ONNX... (this may take a minute)</span>';
        }
        
        try {
            const result = await api.exportToOnnx(modelId);
            showToast(result.message, 'success');
            this.loadOnnxStatus(modelId);
            
            // Refresh model list to update size display
            await this.loadModels();
        } catch (error) {
            showToast('Failed to export to ONNX: ' + error.message, 'error');
            this.loadOnnxStatus(modelId);
        }
    }
    
    /**
     * Delete ONNX export for a model
     * @param {string} modelId - Model ID
     */
    async deleteOnnxExport(modelId) {
        if (!confirm('Delete ONNX export?')) return;
        
        try {
            await api.deleteOnnxExport(modelId);
            showToast('ONNX export deleted', 'success');
            this.loadOnnxStatus(modelId);
        } catch (error) {
            showToast('Failed to delete ONNX export: ' + error.message, 'error');
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
     * @param {HTMLElement} buttonElement - Optional button element to show loading state
     */
    async downloadModel(modelId, buttonElement = null) {
        // Guard against undefined or empty model ID
        if (!modelId || modelId === 'undefined') {
            console.error('Download called with invalid modelId:', modelId);
            showToast('Error: No model selected', 'error');
            return;
        }
        
        console.log('Downloading model:', modelId);
        
        // Use the provided button element directly
        const btn = buttonElement;
        const originalContent = btn ? btn.innerHTML : null;
        
        // Show loading state with size info
        if (btn) {
            btn.innerHTML = '<span class="spinner-small"></span> Getting info...';
            btn.disabled = true;
        }
        
        try {
            // Get model metadata to show size
            let sizeInfo = '';
            try {
                const metadata = await api.getModelMetadata(modelId);
                if (metadata && metadata.size_kb) {
                    const sizeMB = (metadata.size_kb / 1024).toFixed(1);
                    sizeInfo = ` (${sizeMB} MB)`;
                }
            } catch (e) {
                // Ignore metadata errors, proceed with download
            }
            
            if (btn) {
                btn.innerHTML = `<span class="spinner-small"></span> Preparing${sizeInfo}...`;
            }
            
            const blob = await api.downloadModel(modelId);
            downloadBlob(blob, `${modelId}_model.zip`);
            showToast('Model downloaded!', 'success');
        } catch (error) {
            showToast('Failed to download model: ' + error.message, 'error');
        } finally {
            // Restore button state
            if (btn && originalContent) {
                btn.innerHTML = originalContent;
                btn.disabled = false;
            }
        }
    }

    /**
     * Move a model to the default folder
     * @param {string} modelId - Model ID
     */
    async moveModelToDefault(modelId) {
        if (!confirm(`Move this model to the default folder? It will no longer be deletable from the UI.`)) {
            return;
        }

        try {
            await api.moveModelToDefault(modelId);
            showToast('Model moved to default folder', 'success');
            await this.loadModels();
        } catch (error) {
            showToast('Failed to move model: ' + error.message, 'error');
        }
    }

    /**
     * Delete a model
     * @param {string} modelId - Model ID
     */
    async deleteModel(modelId) {
        if (!confirm(`Are you sure you want to delete this model?\nThis will also delete associated recordings.`)) {
            return;
        }

        try {
            const result = await api.deleteModel(modelId);
            showToast(result.message || 'Model deleted', 'success');
            await this.loadModels();
            // Check for orphaned recordings after deletion
            this.checkOrphanedRecordings();
        } catch (error) {
            showToast('Failed to delete model: ' + error.message, 'error');
        }
    }

    /**
     * Check for orphaned recordings and show cleanup option if any exist
     */
    async checkOrphanedRecordings() {
        try {
            const result = await api.listOrphanedRecordings();
            if (result.count > 0) {
                const orphanedInfo = result.orphaned.map(r => `${r.name} (${r.file_count} files)`).join(', ');
                if (confirm(`Found ${result.count} orphaned recording(s): ${orphanedInfo}\n\nWould you like to clean them up?`)) {
                    await this.cleanOrphanedRecordings();
                }
            }
        } catch (error) {
            // Silently ignore - not critical
        }
    }

    /**
     * Clean up orphaned recordings
     */
    async cleanOrphanedRecordings() {
        try {
            const result = await api.cleanOrphanedRecordings();
            if (result.count > 0) {
                showToast(`Cleaned up ${result.count} orphaned recording(s)`, 'success');
            } else {
                showToast('No orphaned recordings to clean', 'info');
            }
        } catch (error) {
            showToast('Failed to clean orphaned recordings: ' + error.message, 'error');
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
