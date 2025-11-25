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
    }

    /**
     * Bind event handlers
     */
    bindEvents() {
        this.elements.modelSelect.addEventListener('change', () => this.onModelChange());
        this.elements.thresholdSlider.addEventListener('input', () => this.onThresholdChange());
        this.elements.btnStartTest.addEventListener('click', () => this.toggleListening());
    }

    /**
     * Initialize the tester
     */
    async initialize() {
        await this.loadModels();
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

            // Connect WebSocket
            this.websocket = api.createTestWebSocket(
                this.selectedModel.id,
                this.threshold,
                1000
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
     * Handle WebSocket message
     * @param {object} data - Message data
     */
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'ready':
                console.log('WebSocket ready:', data);
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
