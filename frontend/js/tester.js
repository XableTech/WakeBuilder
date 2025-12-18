/**
 * WakeBuilder - Real-time Testing Controller
 */

/**
 * Wake word tester state and controller
 */
class WakeWordTester {
    constructor() {
        this.selectedModel = null;
        this.threshold = 0.7;  // Higher default for fewer false positives
        this.isListening = false;
        
        this.websocket = null;
        this.audioStreamer = null;
        this.detectionLog = [];
        this.sessionDetectionCount = 0;  // Counter for detections in current session
        
        this.elements = {};
        this.bindElements();
        this.bindEvents();
    }

    /**
     * Bind DOM elements
     */
    bindElements() {
        this.elements.modelSelect = document.getElementById('test-model-select');
        this.elements.modelInfo = document.getElementById('test-model-info');
        this.elements.thresholdSlider = document.getElementById('threshold-slider');
        this.elements.thresholdValue = document.getElementById('threshold-value');
        this.elements.detectionIndicator = document.getElementById('detection-indicator');
        this.elements.confidenceFill = document.getElementById('confidence-fill');
        this.elements.confidenceValue = document.getElementById('confidence-value');
        this.elements.btnStartTest = document.getElementById('btn-start-test');
        this.elements.detectionLog = document.getElementById('detection-log');
        this.elements.detectionCounter = document.getElementById('detection-counter');
        this.elements.deviceSelect = document.getElementById('device-select');
        this.elements.noiseReduction = document.getElementById('noise-reduction');
        this.elements.useOnnx = document.getElementById('use-onnx');
        this.elements.onnxModelStatus = document.getElementById('onnx-model-status');
    }

    /**
     * Bind event handlers
     */
    bindEvents() {
        this.elements.modelSelect.addEventListener('change', () => this.onModelChange());
        this.elements.thresholdSlider.addEventListener('input', () => this.onThresholdChange());
        this.elements.btnStartTest.addEventListener('click', () => this.toggleListening());
        if (this.elements.deviceSelect) {
            this.elements.deviceSelect.addEventListener('change', () => this.onDeviceChange());
        }
    }

    /**
     * Initialize the tester
     */
    async initialize() {
        await this.loadModels();
        await this.loadDeviceInfo();
    }

    /**
     * Load device info and update UI
     */
    async loadDeviceInfo() {
        try {
            const info = await api.getDeviceInfo();
            console.log('Device info:', info);
            this.updateDeviceUI(info);
        } catch (error) {
            console.error('Failed to load device info:', error);
        }
    }

    /**
     * Update device UI
     */
    updateDeviceUI(info) {
        const deviceSelect = this.elements.deviceSelect;
        if (!deviceSelect) return;

        // Update current selection
        deviceSelect.value = info.current_device;

        // Disable CUDA option if not available
        const cudaOption = deviceSelect.querySelector('option[value="cuda"]');
        if (cudaOption) {
            cudaOption.disabled = !info.cuda_available;
            if (info.cuda_available && info.cuda_device_name) {
                cudaOption.textContent = `GPU (${info.cuda_device_name})`;
            } else {
                cudaOption.textContent = 'GPU (not available)';
            }
        }
    }

    /**
     * Handle device change
     */
    async onDeviceChange() {
        const device = this.elements.deviceSelect.value;
        try {
            const info = await api.setDevice(device);
            this.updateDeviceUI(info);
        } catch (error) {
            console.error('Failed to set device:', error);
        }
    }

    /**
     * Load available models
     */
    async loadModels() {
        try {
            const result = await api.listTestableModels();
            
            // Clear existing options
            this.elements.modelSelect.innerHTML = '<option value="">-- Select a model --</option>';
            
            // Add models
            result.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.model_id;
                option.textContent = `${model.wake_word} (${model.category})`;
                option.dataset.threshold = model.threshold;
                option.dataset.wakeWord = model.wake_word;
                this.elements.modelSelect.appendChild(option);
            });

        } catch (error) {
            console.error('Failed to load models:', error);
            showToast('Failed to load models', 'error');
        }
    }

    /**
     * Handle model selection change
     */
    onModelChange() {
        const select = this.elements.modelSelect;
        const option = select.options[select.selectedIndex];
        
        if (!option.value) {
            this.selectedModel = null;
            this.elements.modelInfo.textContent = 'Select a model to begin testing';
            this.elements.btnStartTest.disabled = true;
            this.updateOnnxStatus(null);
            return;
        }

        this.selectedModel = {
            id: option.value,
            wakeWord: option.dataset.wakeWord,
            threshold: parseFloat(option.dataset.threshold) || 0.5,
        };

        // Update threshold slider
        this.threshold = this.selectedModel.threshold;
        this.elements.thresholdSlider.value = this.threshold * 100;
        this.elements.thresholdValue.textContent = this.threshold.toFixed(2);

        // Update info
        this.elements.modelInfo.textContent = `Wake word: "${this.selectedModel.wakeWord}" | Recommended threshold: ${this.threshold.toFixed(2)}`;
        this.elements.btnStartTest.disabled = false;
        
        // Check ONNX availability
        this.checkOnnxAvailability(this.selectedModel.id);
    }
    
    /**
     * Check ONNX availability for selected model
     * @param {string} modelId - Model ID
     */
    async checkOnnxAvailability(modelId) {
        this.updateOnnxStatus('checking');
        
        try {
            const status = await api.getOnnxStatus(modelId);
            this.selectedModel.onnxAvailable = status.onnx_available;
            this.updateOnnxStatus(status.onnx_available ? 'available' : 'not_available');
        } catch (error) {
            console.error('Failed to check ONNX status:', error);
            this.selectedModel.onnxAvailable = false;
            this.updateOnnxStatus('error');
        }
    }
    
    /**
     * Update ONNX status display
     * @param {string|null} status - Status: 'checking', 'available', 'not_available', 'error', or null
     */
    updateOnnxStatus(status) {
        if (!this.elements.onnxModelStatus || !this.elements.useOnnx) return;
        
        switch (status) {
            case 'checking':
                this.elements.onnxModelStatus.textContent = 'Checking ONNX availability...';
                this.elements.useOnnx.disabled = true;
                break;
            case 'available':
                this.elements.onnxModelStatus.textContent = 'ONNX model available - can use for faster inference';
                this.elements.onnxModelStatus.style.color = 'var(--color-success)';
                this.elements.useOnnx.disabled = false;
                break;
            case 'not_available':
                this.elements.onnxModelStatus.textContent = 'ONNX model not exported - using PyTorch model';
                this.elements.onnxModelStatus.style.color = 'var(--text-muted)';
                this.elements.useOnnx.disabled = true;
                this.elements.useOnnx.checked = false;
                break;
            case 'error':
                this.elements.onnxModelStatus.textContent = 'Failed to check ONNX status';
                this.elements.onnxModelStatus.style.color = 'var(--color-error)';
                this.elements.useOnnx.disabled = true;
                this.elements.useOnnx.checked = false;
                break;
            default:
                this.elements.onnxModelStatus.textContent = 'Select a model to check ONNX availability';
                this.elements.onnxModelStatus.style.color = '';
                this.elements.useOnnx.disabled = true;
                this.elements.useOnnx.checked = false;
        }
    }

    /**
     * Handle threshold slider change
     */
    onThresholdChange() {
        this.threshold = this.elements.thresholdSlider.value / 100;
        this.elements.thresholdValue.textContent = this.threshold.toFixed(2);

        // Update WebSocket if connected
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify({
                type: 'set_threshold',
                threshold: this.threshold,
            }));
        }
    }

    /**
     * Select a specific model
     * @param {string} modelId - Model ID to select
     */
    selectModel(modelId) {
        this.elements.modelSelect.value = modelId;
        this.onModelChange();
    }

    /**
     * Toggle listening state
     */
    async toggleListening() {
        if (this.isListening) {
            this.stopListening();
        } else {
            await this.startListening();
        }
    }

    /**
     * Start listening for wake word
     */
    async startListening() {
        if (!this.selectedModel) {
            showToast('Please select a model first', 'error');
            return;
        }

        try {
            // Initialize audio streamer
            this.audioStreamer = new AudioStreamer();
            await this.audioStreamer.initialize();

            // Connect WebSocket with ONNX option
            const noiseReduction = this.elements.noiseReduction?.checked ?? false;
            const useOnnx = this.elements.useOnnx?.checked ?? false;
            this.websocket = api.createTestWebSocketOnnx(
                this.selectedModel.id,
                this.threshold,
                1000,
                noiseReduction,
                useOnnx
            );

            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.audioStreamer.onAudioData = (buffer) => {
                    if (this.websocket.readyState === WebSocket.OPEN) {
                        this.websocket.send(buffer);
                    }
                };
                this.audioStreamer.start();
            };

            this.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            };

            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                showToast('Connection error', 'error');
                this.stopListening();
            };

            this.websocket.onclose = () => {
                console.log('WebSocket closed');
                if (this.isListening) {
                    this.stopListening();
                }
            };

            // Update UI
            this.isListening = true;
            this.elements.btnStartTest.innerHTML = `
                <svg viewBox="0 0 24 24" fill="currentColor">
                    <rect x="6" y="4" width="4" height="16"/>
                    <rect x="14" y="4" width="4" height="16"/>
                </svg>
                Stop Listening
            `;
            this.elements.detectionIndicator.classList.add('listening');
            this.elements.detectionIndicator.querySelector('.indicator-text').textContent = 
                `Listening for "${this.selectedModel.wakeWord}"...`;

        } catch (error) {
            console.error('Failed to start listening:', error);
            showToast(error.message, 'error');
        }
    }

    /**
     * Stop listening
     */
    stopListening() {
        this.isListening = false;

        // Log session summary if there were detections
        if (this.sessionDetectionCount > 0) {
            this.addSessionSummaryLog(this.sessionDetectionCount);
        }
        
        // Reset session counter
        this.sessionDetectionCount = 0;
        this.updateDetectionCounter();

        // Stop audio streamer
        if (this.audioStreamer) {
            this.audioStreamer.dispose();
            this.audioStreamer = null;
        }

        // Close WebSocket
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }

        // Update UI
        this.elements.btnStartTest.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
            </svg>
            Start Listening
        `;
        this.elements.detectionIndicator.classList.remove('listening', 'detected');
        this.elements.detectionIndicator.querySelector('.indicator-text').textContent = 
            'Say your wake word...';
        this.elements.confidenceFill.style.width = '0%';
        this.elements.confidenceValue.textContent = '0%';
    }
    
    /**
     * Update the detection counter display
     */
    updateDetectionCounter() {
        if (this.elements.detectionCounter) {
            this.elements.detectionCounter.textContent = this.sessionDetectionCount;
        }
    }
    
    /**
     * Pulse the detection counter animation
     */
    pulseDetectionCounter() {
        if (this.elements.detectionCounter) {
            this.elements.detectionCounter.classList.add('pulse');
            setTimeout(() => {
                this.elements.detectionCounter.classList.remove('pulse');
            }, 300);
        }
    }
    
    /**
     * Add session summary to log
     * @param {number} count - Number of detections in the session
     */
    addSessionSummaryLog(count) {
        const now = new Date();
        const timeStr = now.toLocaleTimeString();
        
        const entry = document.createElement('div');
        entry.className = 'log-entry session-summary';
        entry.innerHTML = `
            <span class="log-time">${timeStr}</span>
            <span class="log-message">Session ended: ${count} detection${count !== 1 ? 's' : ''}</span>
        `;
        
        // Remove empty state if present
        const emptyState = this.elements.detectionLog.querySelector('.log-empty');
        if (emptyState) {
            emptyState.remove();
        }
        
        this.elements.detectionLog.insertBefore(entry, this.elements.detectionLog.firstChild);
    }

    /**
     * Handle WebSocket message
     * @param {object} data - Message data
     */
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'ready':
                console.log('WebSocket ready:', data);
                // Store which model format is being used
                this.usingOnnx = data.using_onnx;
                const modelFormat = data.using_onnx ? 'ONNX' : 'PyTorch';
                this.elements.detectionIndicator.querySelector('.indicator-text').textContent = 
                    `Listening for "${data.wake_word}" (${modelFormat})...`;
                // Update ONNX status display if it exists
                if (this.elements.onnxStatusText) {
                    this.elements.onnxStatusText.textContent = data.using_onnx ? 'Using ONNX model ✓' : 'Using PyTorch model';
                    this.elements.onnxStatusText.style.color = data.using_onnx ? 'var(--color-success)' : 'var(--text-secondary)';
                }
                // Show toast to confirm model format
                showToast(`Testing with ${modelFormat} model`, 'info');
                break;

            case 'detection':
                this.handleDetection(data);
                break;

            case 'threshold_updated':
                console.log('Threshold updated:', data.threshold);
                break;

            case 'error':
                console.error('WebSocket error:', data.message);
                showToast(data.message, 'error');
                break;

            case 'pong':
                // Heartbeat response
                break;
        }
    }

    /**
     * Handle detection event
     * @param {object} data - Detection data
     */
    handleDetection(data) {
        // Update confidence display
        const confidence = data.confidence * 100;
        this.elements.confidenceFill.style.width = confidence + '%';
        this.elements.confidenceValue.textContent = confidence.toFixed(1) + '%';

        // Update indicator
        if (data.detected) {
            this.elements.detectionIndicator.classList.add('detected');
            this.elements.detectionIndicator.querySelector('.indicator-text').textContent = 
                'DETECTED!';
            
            // Increment session counter
            this.sessionDetectionCount++;
            this.updateDetectionCounter();
            this.pulseDetectionCounter();
            
            // Log detection
            this.addLogEntry(data);

            // Reset after a short delay
            setTimeout(() => {
                if (this.isListening) {
                    this.elements.detectionIndicator.classList.remove('detected');
                    this.elements.detectionIndicator.querySelector('.indicator-text').textContent = 
                        `Listening for "${this.selectedModel.wakeWord}"...`;
                }
            }, 1000);
        }
    }

    /**
     * Add entry to detection log
     * @param {object} data - Detection data
     */
    addLogEntry(data) {
        // Remove empty message if present
        const empty = this.elements.detectionLog.querySelector('.log-empty');
        if (empty) empty.remove();

        // Create log entry
        const entry = document.createElement('div');
        entry.className = 'log-entry' + (data.detected ? ' detected' : '');
        
        const time = new Date(data.timestamp).toLocaleTimeString();
        entry.innerHTML = `
            <span class="log-time">${time}</span>
            <span class="log-confidence">${(data.confidence * 100).toFixed(1)}%</span>
            <span>${data.detected ? 'DETECTED' : 'below threshold'}</span>
        `;

        // Add to top of log
        this.elements.detectionLog.insertBefore(entry, this.elements.detectionLog.firstChild);

        // Limit log entries
        while (this.elements.detectionLog.children.length > 50) {
            this.elements.detectionLog.removeChild(this.elements.detectionLog.lastChild);
        }
    }

    /**
     * Clear detection log
     */
    clearLog() {
        this.elements.detectionLog.innerHTML = '<div class="log-empty">Detections will appear here...</div>';
    }

    /**
     * Clean up resources
     */
    dispose() {
        this.stopListening();
    }
}

// Global tester instance
let tester = null;

/**
 * Initialize tester
 */
function initTester() {
    tester = new WakeWordTester();
    tester.initialize();
    window.tester = tester;
}
