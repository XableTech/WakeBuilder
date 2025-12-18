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
        this.trainingStartTime = null;
        this.elapsedTimeInterval = null;
        
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
        this.elements.focalAlpha = document.getElementById('focal-alpha');
        this.elements.focalGamma = document.getElementById('focal-gamma');
        this.elements.useAttention = document.getElementById('use-attention');
        this.elements.useSeBlock = document.getElementById('use-se-block');
        this.elements.useTcn = document.getElementById('use-tcn');
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
        this.elements.elapsedTime = document.getElementById('elapsed-time');
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
        this.elements.hpSeBlock = document.getElementById('hp-se-block');
        this.elements.hpTcn = document.getElementById('hp-tcn');
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
        
        // Draw initial placeholder state for all charts
        this.trainingChart.draw();
        this.accuracyChart.draw();
        this.resultLossChart.draw();
        this.resultAccuracyChart.draw();
        
        // Draw idle waveform
        drawIdleWaveform(this.elements.waveformCanvas);
        
        // Fetch cache status and configure dynamic limits
        await this.initializeDataLimits();
    }
    
    /**
     * Initialize data generation limits based on available negative chunks
     */
    async initializeDataLimits() {
        try {
            const cacheStatus = await api.getCacheStatus();
            const availableChunks = cacheStatus.audio_cache?.chunk_count || 0;
            
            // Store for later use
            this.availableNegativeChunks = availableChunks;
            
            // Update positive samples input: min=5000, max=available chunks
            if (this.elements.targetPositiveSamples && availableChunks > 0) {
                this.elements.targetPositiveSamples.min = 5000;
                this.elements.targetPositiveSamples.max = availableChunks;
                // Update hint
                const hint = this.elements.targetPositiveSamples.parentElement?.querySelector('.input-hint');
                if (hint) {
                    hint.textContent = `5000-${availableChunks} samples (based on available negative chunks)`;
                }
            }
            
            // Update max real negatives input
            if (this.elements.maxRealNegatives && availableChunks > 0) {
                this.elements.maxRealNegatives.max = availableChunks;
            }
            
            // Bind event to update negative ratio options dynamically
            if (this.elements.targetPositiveSamples) {
                this.elements.targetPositiveSamples.addEventListener('input', () => this.updateNegativeRatioOptions());
                this.updateNegativeRatioOptions(); // Initial update
            }
        } catch (error) {
            console.warn('Could not fetch cache status for data limits:', error);
        }
    }
    
    /**
     * Update negative ratio dropdown options based on positive samples and available chunks
     */
    updateNegativeRatioOptions() {
        const positiveCount = parseInt(this.elements.targetPositiveSamples?.value) || 4000;
        const availableChunks = this.availableNegativeChunks || 0;
        
        if (availableChunks === 0) return;
        
        // Calculate max ratio: available_chunks / positive_samples
        const maxRatio = Math.floor(availableChunks / positiveCount);
        
        const ratioSelect = document.getElementById('negative-ratio');
        if (!ratioSelect) return;
        
        // Define all possible ratio options
        const allRatios = [
            { value: '1.0', label: '1.0x (minimal)' },
            { value: '1.5', label: '1.5x' },
            { value: '2.0', label: '2.0x (recommended)' },
            { value: '2.5', label: '2.5x' },
            { value: '3.0', label: '3.0x (more robust)' },
        ];
        
        // Get current selection
        const currentValue = ratioSelect.value;
        
        // Rebuild options based on max ratio
        ratioSelect.innerHTML = '';
        let hasSelectedOption = false;
        
        for (const ratio of allRatios) {
            if (parseFloat(ratio.value) <= maxRatio) {
                const option = document.createElement('option');
                option.value = ratio.value;
                option.textContent = ratio.label;
                if (ratio.value === currentValue) {
                    option.selected = true;
                    hasSelectedOption = true;
                }
                ratioSelect.appendChild(option);
            }
        }
        
        // If current selection is no longer valid, select the highest available
        if (!hasSelectedOption && ratioSelect.options.length > 0) {
            ratioSelect.options[ratioSelect.options.length - 1].selected = true;
        }
        
        // Update hint with max info
        const hint = ratioSelect.parentElement?.querySelector('.input-hint');
        if (hint) {
            hint.textContent = `Real negatives from cache. Max ${maxRatio}x based on ${availableChunks.toLocaleString()} chunks / ${positiveCount.toLocaleString()} positives.`;
        }
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
                focalAlpha: parseFloat(this.elements.focalAlpha?.value) || 0.25,
                focalGamma: parseFloat(this.elements.focalGamma?.value) || 2.0,
                useAttention: this.elements.useAttention?.checked ?? false,
                useSeBlock: this.elements.useSeBlock?.checked ?? false,
                useTcn: this.elements.useTcn?.checked ?? false,
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
            this.elements.hpSeBlock.textContent = options.useSeBlock ? 'Yes' : 'No';
            this.elements.hpTcn.textContent = options.useTcn ? 'Yes' : 'No';
            this.elements.hpClassifierDims.textContent = options.classifierHiddenDims.join(', ');

            // Clear chart
            this.trainingChart.clear();

            // Start elapsed time tracking
            this.startElapsedTime();

            // Start polling for status
            this.startPolling();

        } catch (error) {
            showToast(error.message, 'error');
        }
    }

    /**
     * Start elapsed time tracking
     */
    startElapsedTime() {
        this.trainingStartTime = Date.now();
        this.updateElapsedTime();
        this.elapsedTimeInterval = setInterval(() => this.updateElapsedTime(), 1000);
    }

    /**
     * Stop elapsed time tracking
     */
    stopElapsedTime() {
        if (this.elapsedTimeInterval) {
            clearInterval(this.elapsedTimeInterval);
            this.elapsedTimeInterval = null;
        }
    }

    /**
     * Update elapsed time display
     */
    updateElapsedTime() {
        if (!this.trainingStartTime || !this.elements.elapsedTime) return;
        
        const elapsed = Math.floor((Date.now() - this.trainingStartTime) / 1000);
        const hours = Math.floor(elapsed / 3600);
        const minutes = Math.floor((elapsed % 3600) / 60);
        const seconds = elapsed % 60;
        
        let timeStr = '';
        if (hours > 0) {
            timeStr = `${hours}h ${minutes}m ${seconds}s`;
        } else if (minutes > 0) {
            timeStr = `${minutes}m ${seconds}s`;
        } else {
            timeStr = `${seconds}s`;
        }
        
        this.elements.elapsedTime.textContent = `Elapsed: ${timeStr}`;
    }

    /**
     * Start polling for training status
     */
    startPolling() {
        this.pollInterval = setInterval(() => this.pollStatus(), 250);
    }

    /**
     * Stop polling
     */
    stopPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
        this.stopElapsedTime();
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
            this.elements.currentEpoch.textContent = tp.current_epoch + 1 || 0;
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

            // Load recordings for this model
            this.loadResultRecordings(modelId);
            
            // Load ONNX status for this model
            this.loadResultOnnxStatus(modelId);

            showToast('Training completed successfully!', 'success');

        } catch (error) {
            console.error('Failed to show results:', error);
        }
    }
    
    /**
     * Load recordings for the result page
     * @param {string} modelId - Model ID
     */
    async loadResultRecordings(modelId) {
        const container = document.getElementById('result-recordings-list');
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
     * Load ONNX status for the result page
     * @param {string} modelId - Model ID
     */
    async loadResultOnnxStatus(modelId) {
        const container = document.getElementById('result-onnx-status');
        if (!container) return;
        
        try {
            const status = await api.getOnnxStatus(modelId);
            
            if (status.onnx_available) {
                container.innerHTML = `
                    <div class="onnx-available">
                        <span class="onnx-badge success">ONNX Available</span>
                        <span class="onnx-size">${status.onnx_size_mb} MB</span>
                        <button class="btn btn-sm btn-danger" onclick="trainer.deleteResultOnnx('${modelId}')">Delete ONNX</button>
                    </div>
                `;
            } else {
                container.innerHTML = `
                    <div class="onnx-not-available">
                        <span class="onnx-badge">Not Exported</span>
                        <button class="btn btn-sm btn-primary" onclick="trainer.exportResultOnnx('${modelId}')">Export to ONNX</button>
                    </div>
                `;
            }
            
        } catch (error) {
            container.innerHTML = '<span class="error-text">Failed to check ONNX status</span>';
        }
    }
    
    /**
     * Export model to ONNX from results page
     * @param {string} modelId - Model ID
     */
    async exportResultOnnx(modelId) {
        const container = document.getElementById('result-onnx-status');
        if (container) {
            container.innerHTML = '<span class="loading-text">Exporting to ONNX... (this may take a minute)</span>';
        }
        
        try {
            const result = await api.exportToOnnx(modelId);
            showToast(result.message, 'success');
            this.loadResultOnnxStatus(modelId);
            
            // Refresh model list to update size display
            if (window.app) {
                window.app.loadModels();
            }
        } catch (error) {
            showToast('Failed to export to ONNX: ' + error.message, 'error');
            this.loadResultOnnxStatus(modelId);
        }
    }
    
    /**
     * Delete ONNX export from results page
     * @param {string} modelId - Model ID
     */
    async deleteResultOnnx(modelId) {
        if (!confirm('Delete ONNX export?')) return;
        
        try {
            await api.deleteOnnxExport(modelId);
            showToast('ONNX export deleted', 'success');
            this.loadResultOnnxStatus(modelId);
        } catch (error) {
            showToast('Failed to delete ONNX export: ' + error.message, 'error');
        }
    }

    /**
     * Download the trained model
     */
    async downloadModel() {
        const btn = this.elements.btnDownloadModel;
        const originalContent = btn ? btn.innerHTML : null;
        const modelId = this.wakeWord.toLowerCase().replace(/\s+/g, '_');
        
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
            
            const blob = await api.downloadTrainedModel(this.jobId);
            const filename = modelId + '_model.zip';
            downloadBlob(blob, filename);
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
     * Go to test page with current model
     */
    goToTest() {
        const modelId = this.wakeWord.toLowerCase().replace(/\s+/g, '_');
        // Navigate to test page and select this model
        window.app.navigateTo('test');
        // Set model in tester after ensuring models are loaded
        const trySelectModel = async (retries = 10) => {
            if (window.tester) {
                // Wait for models to be loaded if needed
                if (!window.tester.models || window.tester.models.length === 0) {
                    await window.tester.loadModels();
                }
                window.tester.selectModel(modelId);
            } else if (retries > 0) {
                setTimeout(() => trySelectModel(retries - 1), 100);
            }
        };
        setTimeout(() => trySelectModel(), 100);
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
