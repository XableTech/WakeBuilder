/**
 * WakeBuilder - Training Wizard Controller
 */

/**
 * Training wizard state and controller
 */
class TrainingWizard {
    constructor() {
        this.currentStep = 1;
        this.wakeWord = '';
        this.modelType = 'ast';
        this.recordings = [];
        this.jobId = null;
        this.pollInterval = null;
        
        this.recorder = null;
        this.trainingChart = null;
        this.thresholdChart = null;
        
        this.elements = {};
        this.bindElements();
        this.bindEvents();
    }

    /**
     * Bind DOM elements
     */
    bindElements() {
        // Steps
        this.elements.steps = document.querySelectorAll('.wizard-step');
        this.elements.stepContents = {
            1: document.getElementById('step-1'),
            2: document.getElementById('step-2'),
            3: document.getElementById('step-3'),
            4: document.getElementById('step-4'),
        };

        // Step 1
        this.elements.wakeWordInput = document.getElementById('wake-word-input');
        this.elements.wakeWordError = document.getElementById('wake-word-error');
        this.elements.modelTypeSelect = document.getElementById('model-type-select');
        // Cache elements
        this.elements.cacheStatusText = document.getElementById('cache-status-text');
        this.elements.cacheChunkCount = document.getElementById('cache-chunk-count');
        this.elements.buildCacheBtn = document.getElementById('build-cache-btn');
        this.elements.clearCacheBtn = document.getElementById('clear-cache-btn');
        
        // Data generation options
        this.elements.targetPositiveSamples = document.getElementById('target-positive-samples');
        this.elements.maxRealNegatives = document.getElementById('max-real-negatives');
        this.elements.useTtsPositives = document.getElementById('use-tts-positives');
        this.elements.useRealNegatives = document.getElementById('use-real-negatives');
        this.elements.useHardNegatives = document.getElementById('use-hard-negatives');
        
        // Training parameters
        this.elements.batchSize = document.getElementById('batch-size');
        this.elements.numEpochs = document.getElementById('num-epochs');
        this.elements.earlyStopping = document.getElementById('early-stopping');
        this.elements.learningRate = document.getElementById('learning-rate');
        this.elements.dropout = document.getElementById('dropout');
        this.elements.labelSmoothing = document.getElementById('label-smoothing');
        this.elements.mixupAlpha = document.getElementById('mixup-alpha');
        
        // Model enhancements
        this.elements.useFocalLoss = document.getElementById('use-focal-loss');
        this.elements.focalGamma = document.getElementById('focal-gamma');
        this.elements.useAttention = document.getElementById('use-attention');
        this.elements.classifierDims = document.getElementById('classifier-dims');
        
        this.elements.btnNextStep1 = document.getElementById('btn-next-step1');
        this.elements.btnCancelTrain = document.getElementById('btn-cancel-train');

        // Step 2
        this.elements.displayWakeWord = document.getElementById('display-wake-word');
        this.elements.waveformCanvas = document.getElementById('waveform-canvas');
        this.elements.btnRecord = document.getElementById('btn-record');
        this.elements.recordingsList = document.getElementById('recordings-list');
        this.elements.recordingCount = document.getElementById('recording-count');
        this.elements.btnBackStep2 = document.getElementById('btn-back-step2');
        this.elements.btnNextStep2 = document.getElementById('btn-next-step2');

        // Step 3
        this.elements.trainingPhase = document.getElementById('training-phase');
        this.elements.progressFill = document.getElementById('progress-fill');
        this.elements.progressPercent = document.getElementById('progress-percent');
        this.elements.hpModelType = document.getElementById('hp-model-type');
        this.elements.hpBatchSize = document.getElementById('hp-batch-size');
        this.elements.hpLearningRate = document.getElementById('hp-learning-rate');
        this.elements.hpMaxEpochs = document.getElementById('hp-max-epochs');
        this.elements.hpDropout = document.getElementById('hp-dropout');
        this.elements.hpLabelSmoothing = document.getElementById('hp-label-smoothing');
        this.elements.hpMixupAlpha = document.getElementById('hp-mixup-alpha');
        this.elements.hpFocalLoss = document.getElementById('hp-focal-loss');
        this.elements.hpFocalGamma = document.getElementById('hp-focal-gamma');
        this.elements.hpAttention = document.getElementById('hp-attention');
        this.elements.hpClassifierDims = document.getElementById('hp-classifier-dims');
        this.elements.currentEpoch = document.getElementById('current-epoch');
        this.elements.totalEpochs = document.getElementById('total-epochs');
        this.elements.metricTrainLoss = document.getElementById('metric-train-loss');
        this.elements.metricValLoss = document.getElementById('metric-val-loss');
        this.elements.metricValAcc = document.getElementById('metric-val-acc');
        this.elements.metricValF1 = document.getElementById('metric-val-f1');
        this.elements.trainingChartCanvas = document.getElementById('training-chart');
        this.elements.accuracyChartCanvas = document.getElementById('accuracy-chart');

        // Data stats (Step 3)
        this.elements.dataStatsPanel = document.getElementById('data-stats-panel');
        this.elements.statRecordings = document.getElementById('stat-recordings');
        this.elements.statPositive = document.getElementById('stat-positive');
        this.elements.statNegative = document.getElementById('stat-negative');
        this.elements.statTrain = document.getElementById('stat-train');
        this.elements.statVal = document.getElementById('stat-val');

        // Step 4
        this.elements.resultWakeWord = document.getElementById('result-wake-word');
        this.elements.resultAccuracy = document.getElementById('result-accuracy');
        this.elements.resultF1 = document.getElementById('result-f1');
        this.elements.resultThreshold = document.getElementById('result-threshold');
        this.elements.resultParams = document.getElementById('result-params');
        this.elements.resultRecordings = document.getElementById('result-recordings');
        this.elements.resultPositive = document.getElementById('result-positive');
        this.elements.resultNegative = document.getElementById('result-negative');
        this.elements.resultEpochs = document.getElementById('result-epochs');
        this.elements.thresholdChartCanvas = document.getElementById('threshold-chart');
        this.elements.resultLossChartCanvas = document.getElementById('result-loss-chart');
        this.elements.resultAccuracyChartCanvas = document.getElementById('result-accuracy-chart');
        this.elements.btnDownloadModel = document.getElementById('btn-download-model');
        this.elements.btnTrainAnother = document.getElementById('btn-train-another');
        this.elements.btnTestModel = document.getElementById('btn-test-model');
        
        // Result config elements
        this.elements.resultModelType = document.getElementById('result-model-type');
        this.elements.resultBatchSize = document.getElementById('result-batch-size');
        this.elements.resultLearningRate = document.getElementById('result-learning-rate');
        this.elements.resultDropout = document.getElementById('result-dropout');
        this.elements.resultLabelSmoothing = document.getElementById('result-label-smoothing');
        this.elements.resultMixupAlpha = document.getElementById('result-mixup-alpha');
        this.elements.resultFocalLoss = document.getElementById('result-focal-loss');
        this.elements.resultFocalGamma = document.getElementById('result-focal-gamma');
        this.elements.resultAttention = document.getElementById('result-attention');
        this.elements.resultClassifierDims = document.getElementById('result-classifier-dims');
    }

    /**
     * Bind event handlers
     */
    bindEvents() {
        // Step 1 events
        this.elements.wakeWordInput.addEventListener('input', () => this.validateWakeWord());
        this.elements.btnNextStep1.addEventListener('click', () => this.goToStep(2));
        this.elements.btnCancelTrain.addEventListener('click', () => this.cancel());

        // Step 2 events
        this.elements.btnRecord.addEventListener('mousedown', () => this.startRecording());
        this.elements.btnRecord.addEventListener('mouseup', () => this.stopRecording());
        this.elements.btnRecord.addEventListener('mouseleave', () => this.stopRecording());
        this.elements.btnRecord.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.startRecording();
        });
        this.elements.btnRecord.addEventListener('touchend', () => this.stopRecording());
        this.elements.btnBackStep2.addEventListener('click', () => this.goToStep(1));
        this.elements.btnNextStep2.addEventListener('click', () => this.startTraining());

        // Step 4 events
        this.elements.btnDownloadModel.addEventListener('click', () => this.downloadModel());
        this.elements.btnTrainAnother.addEventListener('click', () => this.reset());
        this.elements.btnTestModel.addEventListener('click', () => this.goToTest());
        
        // Cache events
        if (this.elements.buildCacheBtn) {
            this.elements.buildCacheBtn.addEventListener('click', () => this.buildCache());
        }
        if (this.elements.clearCacheBtn) {
            this.elements.clearCacheBtn.addEventListener('click', () => this.clearCache());
        }
        
        // Show/hide negative ratio based on max negatives value
        if (this.elements.maxRealNegatives) {
            this.elements.maxRealNegatives.addEventListener('input', () => this.updateNegativeRatioVisibility());
            this.updateNegativeRatioVisibility(); // Initial state
        }
    }
    
    /**
     * Show/hide negative ratio dropdown based on max negatives value
     */
    updateNegativeRatioVisibility() {
        const maxNeg = parseInt(this.elements.maxRealNegatives?.value) || 0;
        const ratioGroup = document.getElementById('negative-ratio-group');
        if (ratioGroup) {
            ratioGroup.style.opacity = maxNeg === 0 ? '1' : '0.5';
            const select = ratioGroup.querySelector('select');
            if (select) {
                select.disabled = maxNeg !== 0;
            }
        }
    }

    /**
     * Initialize the wizard
     */
    async initialize() {
        // Initialize training charts
        this.trainingChart = new TrainingChart(this.elements.trainingChartCanvas);
        this.accuracyChart = new AccuracyChart(this.elements.accuracyChartCanvas);
        this.thresholdChart = new ThresholdChart(this.elements.thresholdChartCanvas);
        
        // Initialize result charts
        this.resultLossChart = new TrainingChart(this.elements.resultLossChartCanvas);
        this.resultAccuracyChart = new AccuracyChart(this.elements.resultAccuracyChartCanvas);
        
        // Load cache info
        this.loadCacheInfo();
        
        // Draw idle waveform
        drawIdleWaveform(this.elements.waveformCanvas);
    }

    /**
     * Validate wake word input
     * @returns {boolean} Is valid
     */
    validateWakeWord() {
        const value = this.elements.wakeWordInput.value;
        const result = validateWakeWord(value);
        
        if (result.valid) {
            this.elements.wakeWordError.textContent = '';
            this.elements.btnNextStep1.disabled = false;
            return true;
        } else {
            this.elements.wakeWordError.textContent = result.error;
            this.elements.btnNextStep1.disabled = true;
            return false;
        }
    }

    /**
     * Go to a specific step
     * @param {number} step - Step number (1-4)
     */
    async goToStep(step) {
        // Validate current step before proceeding
        if (step > this.currentStep) {
            if (this.currentStep === 1 && !this.validateWakeWord()) {
                return;
            }
        }

        // Update step indicators
        this.elements.steps.forEach((el, index) => {
            el.classList.remove('active', 'completed');
            if (index + 1 < step) {
                el.classList.add('completed');
            } else if (index + 1 === step) {
                el.classList.add('active');
            }
        });

        // Hide all step contents
        Object.values(this.elements.stepContents).forEach(el => {
            el.classList.add('hidden');
        });

        // Show current step content
        this.elements.stepContents[step].classList.remove('hidden');

        // Step-specific initialization
        if (step === 2) {
            this.wakeWord = this.elements.wakeWordInput.value.trim();
            this.modelType = this.elements.modelTypeSelect.value;
            this.elements.displayWakeWord.textContent = this.wakeWord;
            
            // Initialize recorder if needed
            if (!this.recorder) {
                try {
                    this.recorder = new AudioRecorder();
                    await this.recorder.initialize();
                    this.recorder.onVisualize = (data) => {
                        drawWaveform(this.elements.waveformCanvas, data);
                    };
                } catch (error) {
                    showToast(error.message, 'error');
                    return;
                }
            }
        }

        this.currentStep = step;
    }

    /**
     * Start recording
     */
    startRecording() {
        if (!this.recorder || this.recorder.isRecording) return;
        if (this.recordings.length >= 5) {
            showToast('Maximum 5 recordings allowed', 'error');
            return;
        }

        this.recorder.startRecording();
        this.elements.btnRecord.classList.add('recording');
        this.elements.btnRecord.querySelector('span').textContent = 'Recording...';
    }

    /**
     * Stop recording
     */
    async stopRecording() {
        if (!this.recorder || !this.recorder.isRecording) return;

        const blob = await this.recorder.stopRecording();
        this.elements.btnRecord.classList.remove('recording');
        this.elements.btnRecord.querySelector('span').textContent = 'Hold to Record';
        drawIdleWaveform(this.elements.waveformCanvas);

        if (blob) {
            this.addRecording(blob);
        }
    }

    /**
     * Add a recording to the list
     * @param {Blob} blob - Audio blob
     */
    addRecording(blob) {
        const index = this.recordings.length;
        this.recordings.push(blob);

        // Create recording item
        const item = document.createElement('div');
        item.className = 'recording-item';
        item.innerHTML = `
            <span>Recording ${index + 1}</span>
            <button class="btn-play" data-index="${index}" title="Play">
                <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                    <polygon points="5 3 19 12 5 21 5 3"/>
                </svg>
            </button>
            <button class="btn-delete" data-index="${index}" title="Delete">
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="18" y1="6" x2="6" y2="18"/>
                    <line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
            </button>
        `;

        // Play button
        item.querySelector('.btn-play').addEventListener('click', () => {
            this.playRecording(index);
        });

        // Delete button
        item.querySelector('.btn-delete').addEventListener('click', () => {
            this.deleteRecording(index, item);
        });

        this.elements.recordingsList.appendChild(item);
        this.updateRecordingCount();
    }

    /**
     * Play a recording
     * @param {number} index - Recording index
     */
    playRecording(index) {
        const blob = this.recordings[index];
        if (!blob) return;

        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.onended = () => URL.revokeObjectURL(url);
        audio.play();
    }

    /**
     * Delete a recording
     * @param {number} index - Recording index
     * @param {HTMLElement} element - Recording element
     */
    deleteRecording(index, element) {
        this.recordings.splice(index, 1);
        element.remove();

        // Update indices
        const items = this.elements.recordingsList.querySelectorAll('.recording-item');
        items.forEach((item, i) => {
            item.querySelector('span').textContent = `Recording ${i + 1}`;
            item.querySelector('.btn-play').dataset.index = i;
            item.querySelector('.btn-delete').dataset.index = i;
        });

        this.updateRecordingCount();
    }

    /**
     * Update recording count display
     */
    updateRecordingCount() {
        const count = this.recordings.length;
        this.elements.recordingCount.textContent = count;
        // AST with TTS only needs 1 recording minimum
        this.elements.btnNextStep2.disabled = count < 1;
    }

    /**
     * Start the training process
     */
    async startTraining() {
        try {
            // Get training options with balanced defaults
            // AST-optimized defaults with strong regularization to prevent false positives
            
            // Parse classifier dimensions
            const classifierDimsStr = this.elements.classifierDims?.value || '256,128';
            const classifierHiddenDims = classifierDimsStr.split(',').map(x => parseInt(x.trim()));
            
            const options = {
                modelType: this.modelType,
                // Data generation settings
                targetPositiveSamples: parseInt(this.elements.targetPositiveSamples?.value) || 4000,
                maxRealNegatives: parseInt(this.elements.maxRealNegatives?.value) || 0,
                negativeRatio: parseFloat(document.getElementById('negative-ratio')?.value) || 1.5,
                hardNegativeRatio: parseFloat(document.getElementById('hard-negative-ratio')?.value) || 2.0,
                useTtsPositives: this.elements.useTtsPositives?.checked ?? true,
                useRealNegatives: this.elements.useRealNegatives?.checked ?? true,
                useHardNegatives: this.elements.useHardNegatives?.checked ?? true,
                // Training parameters
                batchSize: parseInt(this.elements.batchSize.value) || 32,
                numEpochs: parseInt(this.elements.numEpochs.value) || 100,
                patience: parseInt(this.elements.earlyStopping?.value) || 8,
                learningRate: parseFloat(this.elements.learningRate.value) || 0.0001,
                dropout: parseFloat(this.elements.dropout?.value) || 0.5,
                labelSmoothing: parseFloat(this.elements.labelSmoothing?.value) || 0.25,
                mixupAlpha: parseFloat(this.elements.mixupAlpha?.value) || 0.5,
                // Model enhancements
                useFocalLoss: this.elements.useFocalLoss?.checked ?? true,
                focalGamma: parseFloat(this.elements.focalGamma?.value) || 2.0,
                useAttention: this.elements.useAttention?.checked ?? false,
                classifierHiddenDims: classifierHiddenDims,
            };

            // Start training
            const result = await api.startTraining(this.wakeWord, this.recordings, options);
            this.jobId = result.job_id;

            // Go to step 3
            await this.goToStep(3);

            // Update hyperparameters display
            this.elements.hpModelType.textContent = this.modelType === 'ast' ? 'AST' : this.modelType;
            this.elements.hpBatchSize.textContent = options.batchSize;
            this.elements.hpLearningRate.textContent = options.learningRate;
            this.elements.hpMaxEpochs.textContent = options.numEpochs;
            this.elements.hpDropout.textContent = options.dropout;
            this.elements.hpLabelSmoothing.textContent = options.labelSmoothing;
            this.elements.hpMixupAlpha.textContent = options.mixupAlpha;
            this.elements.hpFocalLoss.textContent = options.useFocalLoss ? 'Yes' : 'No';
            this.elements.hpFocalGamma.textContent = options.useFocalLoss ? options.focalGamma : 'N/A';
            this.elements.hpAttention.textContent = options.useAttention ? 'Yes' : 'No';
            this.elements.hpClassifierDims.textContent = options.classifierHiddenDims.join(', ');

            // Clear chart
            this.trainingChart.clear();

            // Start polling for status
            this.startPolling();

        } catch (error) {
            showToast(error.message, 'error');
        }
    }

    /**
     * Start polling for training status
     */
    startPolling() {
        this.pollInterval = setInterval(() => this.pollStatus(), 1000);
    }

    /**
     * Stop polling
     */
    stopPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }

    /**
     * Poll training status
     */
    async pollStatus() {
        try {
            const status = await api.getTrainingStatus(this.jobId);
            this.updateProgress(status);

            if (status.status === 'completed') {
                this.stopPolling();
                await this.showResults(status);
            } else if (status.status === 'failed') {
                this.stopPolling();
                showToast('Training failed: ' + (status.error || 'Unknown error'), 'error');
            }
        } catch (error) {
            console.error('Failed to poll status:', error);
        }
    }

    /**
     * Update progress display
     * @param {object} status - Training status
     */
    updateProgress(status) {
        // Update progress bar
        const percent = status.progress_percent || 0;
        this.elements.progressFill.style.width = percent + '%';
        this.elements.progressPercent.textContent = percent.toFixed(1) + '%';

        // Update phase
        this.elements.trainingPhase.textContent = status.current_phase || status.message || 'Processing...';

        // Update training metrics if available
        if (status.training_progress) {
            const tp = status.training_progress;
            this.elements.currentEpoch.textContent = tp.current_epoch || 0;
            this.elements.totalEpochs.textContent = tp.total_epochs || 0;

            // Update data stats if available
            if (tp.data_stats) {
                this.elements.dataStatsPanel.style.display = 'block';
                this.elements.statRecordings.textContent = tp.data_stats.num_recordings || '-';
                this.elements.statPositive.textContent = tp.data_stats.num_positive_samples || '-';
                this.elements.statNegative.textContent = tp.data_stats.num_negative_samples || '-';
                this.elements.statTrain.textContent = tp.data_stats.num_train_samples || '-';
                this.elements.statVal.textContent = tp.data_stats.num_val_samples || '-';
                
                // Store for results page
                this.dataStats = tp.data_stats;
            }

            // Update epoch history charts
            if (tp.epoch_history && tp.epoch_history.length > 0) {
                this.trainingChart.setHistory(tp.epoch_history);
                this.accuracyChart.setHistory(tp.epoch_history);
                
                // Store for results page
                this.epochHistory = tp.epoch_history;

                // Update current metrics
                const latest = tp.epoch_history[tp.epoch_history.length - 1];
                this.elements.metricTrainLoss.textContent = formatNumber(latest.train_loss);
                this.elements.metricValLoss.textContent = formatNumber(latest.val_loss);
                this.elements.metricValAcc.textContent = formatPercent(latest.val_accuracy);
                this.elements.metricValF1.textContent = formatNumber(latest.val_f1, 3);
            }
        }
    }

    /**
     * Show training results
     * @param {object} status - Final training status
     */
    async showResults(status) {
        // Get model metadata for detailed results
        try {
            // Go to step 4
            await this.goToStep(4);

            this.elements.resultWakeWord.textContent = this.wakeWord;

            // Get final metrics from training progress
            if (status.training_progress && status.training_progress.epoch_history) {
                const history = status.training_progress.epoch_history;
                const final = history[history.length - 1];
                
                this.elements.resultAccuracy.textContent = formatPercent(final.val_accuracy);
                this.elements.resultF1.textContent = formatNumber(final.val_f1, 3);
                this.elements.resultEpochs.textContent = history.length;
                
                // Draw result charts with training history
                this.resultLossChart.setHistory(history);
                this.resultAccuracyChart.setHistory(history);
            }

            // Show data stats on results page
            if (this.dataStats) {
                this.elements.resultRecordings.textContent = this.dataStats.num_recordings || '-';
                this.elements.resultPositive.textContent = this.dataStats.num_positive_samples || '-';
                this.elements.resultNegative.textContent = this.dataStats.num_negative_samples || '-';
            }

            // Try to get model metadata for threshold info and training config
            // Note: The model ID is typically the wake word normalized
            const modelId = this.wakeWord.toLowerCase().replace(/\s+/g, '_');
            try {
                const metadata = await api.getModelMetadata(modelId);
                this.elements.resultThreshold.textContent = formatNumber(metadata.threshold, 2);
                this.elements.resultParams.textContent = (metadata.parameters / 1000).toFixed(0) + 'K';

                // Draw threshold chart if available
                if (metadata.threshold_analysis) {
                    this.thresholdChart.setData(metadata.threshold_analysis, metadata.threshold);
                }

                // Show training configuration from saved metadata (actual values used)
                if (metadata.training_config) {
                    const cfg = metadata.training_config;
                    this.elements.resultModelType.textContent = 'AST';
                    this.elements.resultBatchSize.textContent = cfg.batch_size ?? '-';
                    this.elements.resultLearningRate.textContent = cfg.learning_rate ?? '-';
                    this.elements.resultDropout.textContent = cfg.dropout ?? '-';
                    this.elements.resultLabelSmoothing.textContent = cfg.label_smoothing ?? '-';
                    this.elements.resultMixupAlpha.textContent = cfg.mixup_alpha ?? '-';
                    this.elements.resultFocalLoss.textContent = cfg.use_focal_loss ? 'Yes' : 'No';
                    this.elements.resultFocalGamma.textContent = cfg.use_focal_loss ? (cfg.focal_gamma ?? '2.0') : 'N/A';
                    this.elements.resultAttention.textContent = cfg.use_attention ? 'Yes' : 'No';
                    this.elements.resultClassifierDims.textContent = cfg.classifier_hidden_dims ? 
                        cfg.classifier_hidden_dims.join(', ') : '-';
                }
            } catch (e) {
                // Model metadata not available yet, use defaults
                this.elements.resultThreshold.textContent = '0.50';
                this.elements.resultParams.textContent = '-';
            }

            showToast('Training completed successfully!', 'success');

        } catch (error) {
            console.error('Failed to show results:', error);
        }
    }

    /**
     * Download the trained model
     */
    async downloadModel() {
        try {
            const blob = await api.downloadTrainedModel(this.jobId);
            const filename = this.wakeWord.toLowerCase().replace(/\s+/g, '_') + '_model.zip';
            downloadBlob(blob, filename);
            showToast('Model downloaded!', 'success');
        } catch (error) {
            showToast('Failed to download model: ' + error.message, 'error');
        }
    }

    /**
     * Go to test page with current model
     */
    goToTest() {
        const modelId = this.wakeWord.toLowerCase().replace(/\s+/g, '_');
        // Navigate to test page and select this model
        window.app.navigateTo('test');
        // Set model in tester after a short delay
        setTimeout(() => {
            if (window.tester) {
                window.tester.selectModel(modelId);
            }
        }, 100);
    }

    /**
     * Cancel training wizard
     */
    cancel() {
        this.reset();
        window.app.navigateTo('home');
    }

    /**
     * Reset wizard to initial state
     */
    reset() {
        this.stopPolling();
        
        this.currentStep = 1;
        this.wakeWord = '';
        this.recordings = [];
        this.jobId = null;
        this.dataStats = null;
        this.epochHistory = null;

        // Reset form
        this.elements.wakeWordInput.value = '';
        this.elements.wakeWordError.textContent = '';
        this.elements.recordingsList.innerHTML = '';
        this.elements.recordingCount.textContent = '0';
        this.elements.btnNextStep2.disabled = true;

        // Reset progress
        this.elements.progressFill.style.width = '0%';
        this.elements.progressPercent.textContent = '0%';
        this.elements.trainingPhase.textContent = 'Waiting...';

        // Reset metrics display
        this.elements.currentEpoch.textContent = '0';
        this.elements.totalEpochs.textContent = '0';
        this.elements.metricTrainLoss.textContent = '-';
        this.elements.metricValLoss.textContent = '-';
        this.elements.metricValAcc.textContent = '-';
        this.elements.metricValF1.textContent = '-';

        // Reset data stats
        this.elements.dataStatsPanel.style.display = 'none';
        this.elements.statRecordings.textContent = '-';
        this.elements.statPositive.textContent = '-';
        this.elements.statNegative.textContent = '-';
        this.elements.statTrain.textContent = '-';
        this.elements.statVal.textContent = '-';

        // Clear charts
        if (this.trainingChart) this.trainingChart.clear();
        if (this.accuracyChart) this.accuracyChart.clear();
        if (this.resultLossChart) this.resultLossChart.clear();
        if (this.resultAccuracyChart) this.resultAccuracyChart.clear();

        // Go to step 1
        this.goToStep(1);
    }

    /**
     * Clean up resources
     */
    dispose() {
        this.stopPolling();
        if (this.recorder) {
            this.recorder.dispose();
        }
    }
    
    // ========================================================================
    // Cache Management
    // ========================================================================
    
    /**
     * Load and display cache info
     */
    async loadCacheInfo() {
        try {
            const info = await api.getNegativeCacheInfo();
            this.updateCacheDisplay(info);
        } catch (error) {
            console.error('Failed to load cache info:', error);
            if (this.elements.cacheStatusText) {
                this.elements.cacheStatusText.textContent = 'Failed to load cache info';
            }
        }
    }
    
    /**
     * Update cache display
     */
    updateCacheDisplay(info) {
        if (!this.elements.cacheStatusText || !this.elements.cacheChunkCount) return;
        
        if (info.cached && info.chunk_count > 0) {
            this.elements.cacheStatusText.textContent = `${info.source_files.toLocaleString()} source files cached`;
            this.elements.cacheChunkCount.textContent = `${info.chunk_count.toLocaleString()} chunks ready`;
            this.elements.cacheChunkCount.classList.remove('no-cache');
        } else {
            this.elements.cacheStatusText.textContent = `${info.source_files.toLocaleString()} source files available`;
            this.elements.cacheChunkCount.textContent = 'No cache (will be slow)';
            this.elements.cacheChunkCount.classList.add('no-cache');
        }
    }
    
    /**
     * Build the negative data cache
     */
    async buildCache() {
        if (!this.elements.buildCacheBtn) return;
        
        const originalText = this.elements.buildCacheBtn.textContent;
        this.elements.buildCacheBtn.textContent = 'Building...';
        this.elements.buildCacheBtn.disabled = true;
        
        try {
            const result = await api.buildNegativeCache();
            this.elements.cacheStatusText.textContent = 'Building cache in background...';
            this.elements.cacheChunkCount.textContent = `~${result.estimated_chunks.toLocaleString()} chunks`;
            
            // Poll for completion
            this.pollCacheStatus();
        } catch (error) {
            console.error('Failed to build cache:', error);
            alert('Failed to build cache: ' + error.message);
        } finally {
            this.elements.buildCacheBtn.textContent = originalText;
            this.elements.buildCacheBtn.disabled = false;
        }
    }
    
    /**
     * Poll cache status during build
     */
    pollCacheStatus() {
        let pollCount = 0;
        const maxPolls = 120; // 10 minutes max
        
        const poll = async () => {
            pollCount++;
            if (pollCount > maxPolls) return;
            
            try {
                const info = await api.getNegativeCacheInfo();
                this.updateCacheDisplay(info);
                
                // Keep polling if still building (chunk count increasing)
                if (!info.cached || info.chunk_count === 0) {
                    setTimeout(poll, 5000);
                }
            } catch (error) {
                console.error('Cache poll error:', error);
            }
        };
        
        setTimeout(poll, 5000);
    }
    
    /**
     * Clear the negative data cache
     */
    async clearCache() {
        if (!confirm('Clear all cached negative data chunks?')) return;
        
        try {
            await api.clearNegativeCache();
            this.loadCacheInfo();
        } catch (error) {
            console.error('Failed to clear cache:', error);
            alert('Failed to clear cache: ' + error.message);
        }
    }
}

// Global trainer instance
let trainer = null;

/**
 * Initialize trainer
 */
function initTrainer() {
    trainer = new TrainingWizard();
    trainer.initialize();
    window.trainer = trainer;
}
