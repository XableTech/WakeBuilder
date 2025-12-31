/**
 * WakeBuilder - API Client
 */

const API_BASE = '';  // Same origin

/**
 * API client for WakeBuilder backend
 */
const api = {
    /**
     * Make an API request
     * @param {string} endpoint - API endpoint
     * @param {object} options - Fetch options
     * @returns {Promise<any>} Response data
     */
    async request(endpoint, options = {}) {
        const url = `${API_BASE}${endpoint}`;
        const response = await fetch(url, {
            ...options,
            headers: {
                ...options.headers,
            },
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ message: response.statusText }));
            throw new Error(error.detail || error.message || 'Request failed');
        }

        // Check if response is JSON
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            return response.json();
        }

        return response;
    },

    /**
     * GET request
     * @param {string} endpoint - API endpoint
     * @returns {Promise<any>} Response data
     */
    async get(endpoint) {
        return this.request(endpoint);
    },

    /**
     * POST request with JSON body
     * @param {string} endpoint - API endpoint
     * @param {object} data - Request body
     * @returns {Promise<any>} Response data
     */
    async post(endpoint, data) {
        return this.request(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });
    },

    /**
     * POST request with FormData
     * @param {string} endpoint - API endpoint
     * @param {FormData} formData - Form data
     * @returns {Promise<any>} Response data
     */
    async postForm(endpoint, formData) {
        return this.request(endpoint, {
            method: 'POST',
            body: formData,
        });
    },

    /**
     * DELETE request
     * @param {string} endpoint - API endpoint
     * @returns {Promise<any>} Response data
     */
    async delete(endpoint) {
        return this.request(endpoint, {
            method: 'DELETE',
        });
    },

    // ========================================================================
    // Health & Info
    // ========================================================================

    /**
     * Check API health
     * @returns {Promise<object>} Health status
     */
    async getHealth() {
        return this.get('/health');
    },

    /**
     * Get API info
     * @returns {Promise<object>} API information
     */
    async getInfo() {
        return this.get('/api/info');
    },

    // ========================================================================
    // Training
    // ========================================================================

    /**
     * Start a training job
     * @param {string} wakeWord - Wake word to train
     * @param {Blob[]} recordings - Audio recordings
     * @param {object} options - Training options
     * @returns {Promise<object>} Job info
     */
    async startTraining(wakeWord, recordings, options = {}) {
        const formData = new FormData();
        formData.append('wake_word', wakeWord);
        formData.append('model_type', options.modelType || 'ast');

        // Data generation settings
        if (options.targetPositiveSamples !== undefined) {
            formData.append('target_positive_samples', options.targetPositiveSamples);
        }
        if (options.maxRealNegatives !== undefined) {
            formData.append('max_real_negatives', options.maxRealNegatives);
        }
        if (options.useTtsPositives !== undefined) {
            formData.append('use_tts_positives', options.useTtsPositives);
        }
        if (options.useRealNegatives !== undefined) {
            formData.append('use_real_negatives', options.useRealNegatives);
        }
        if (options.negativeRatio !== undefined) {
            formData.append('negative_ratio', options.negativeRatio);
        }
        if (options.hardNegativeRatio !== undefined) {
            formData.append('hard_negative_ratio', options.hardNegativeRatio);
        }
        if (options.useHardNegatives !== undefined) {
            formData.append('use_hard_negatives', options.useHardNegatives);
        }

        // Training hyperparameters
        if (options.batchSize) {
            formData.append('batch_size', options.batchSize);
        }
        if (options.numEpochs) {
            formData.append('num_epochs', options.numEpochs);
        }
        if (options.learningRate) {
            formData.append('learning_rate', options.learningRate);
        }
        if (options.dropout !== undefined) {
            formData.append('dropout', options.dropout);
        }
        if (options.labelSmoothing !== undefined) {
            formData.append('label_smoothing', options.labelSmoothing);
        }
        if (options.mixupAlpha !== undefined) {
            formData.append('mixup_alpha', options.mixupAlpha);
        }

        // Model enhancements
        if (options.useFocalLoss !== undefined) {
            formData.append('use_focal_loss', options.useFocalLoss);
        }
        if (options.focalAlpha !== undefined) {
            formData.append('focal_alpha', options.focalAlpha);
        }
        if (options.focalGamma !== undefined) {
            formData.append('focal_gamma', options.focalGamma);
        }
        if (options.useAttention !== undefined) {
            formData.append('use_attention', options.useAttention);
        }
        if (options.useSeBlock !== undefined) {
            formData.append('use_se_block', options.useSeBlock);
        }
        if (options.useTcn !== undefined) {
            formData.append('use_tcn', options.useTcn);
        }
        if (options.classifierHiddenDims !== undefined) {
            formData.append('classifier_hidden_dims', JSON.stringify(options.classifierHiddenDims));
        }

        recordings.forEach((blob, index) => {
            formData.append('recordings', blob, `recording_${index + 1}.wav`);
        });

        return this.postForm('/api/train/start', formData);
    },

    /**
     * Get training job status
     * @param {string} jobId - Job ID
     * @returns {Promise<object>} Job status
     */
    async getTrainingStatus(jobId) {
        return this.get(`/api/train/status/${jobId}`);
    },

    /**
     * Download trained model
     * @param {string} jobId - Job ID
     * @returns {Promise<Blob>} Model ZIP file
     */
    async downloadTrainedModel(jobId) {
        const response = await this.request(`/api/train/download/${jobId}`);
        return response.blob();
    },

    /**
     * List all training jobs
     * @returns {Promise<object>} Jobs list
     */
    async listJobs() {
        return this.get('/api/train/jobs');
    },

    /**
     * Delete a training job
     * @param {string} jobId - Job ID
     * @returns {Promise<object>} Delete result
     */
    async deleteJob(jobId) {
        return this.delete(`/api/train/${jobId}`);
    },

    // ========================================================================
    // Models
    // ========================================================================

    /**
     * List all models
     * @param {string} category - Filter by category ('default', 'custom', or null for all)
     * @returns {Promise<object>} Models list
     */
    async listModels(category = null) {
        const params = category ? `?category=${category}` : '';
        return this.get(`/api/models/list${params}`);
    },

    /**
     * Get model metadata
     * @param {string} modelId - Model ID
     * @returns {Promise<object>} Model metadata
     */
    async getModelMetadata(modelId) {
        return this.get(`/api/models/${modelId}/metadata`);
    },

    /**
     * Get training history for a model
     * @param {string} modelId - Model ID
     * @returns {Promise<object>} Training history (loss, accuracy per epoch)
     */
    async getTrainingHistory(modelId) {
        return this.get(`/api/models/${modelId}/training-history`);
    },

    /**
     * Download model
     * @param {string} modelId - Model ID
     * @returns {Promise<Blob>} Model ZIP file
     */
    async downloadModel(modelId) {
        const response = await this.request(`/api/models/${modelId}/download`);
        return response.blob();
    },

    /**
     * Delete a model
     * @param {string} modelId - Model ID
     * @returns {Promise<object>} Delete result
     */
    async deleteModel(modelId) {
        return this.delete(`/api/models/${modelId}`);
    },

    /**
     * List orphaned recordings (recordings without associated models)
     * @returns {Promise<object>} List of orphaned recordings
     */
    async listOrphanedRecordings() {
        return this.get('/api/models/recordings/orphaned');
    },

    /**
     * Clean up orphaned recordings
     * @returns {Promise<object>} Cleanup result
     */
    async cleanOrphanedRecordings() {
        return this.delete('/api/models/recordings/orphaned');
    },

    // ========================================================================
    // Testing
    // ========================================================================

    /**
     * Test model with audio file
     * @param {string} modelId - Model ID
     * @param {Blob} audioBlob - Audio file
     * @param {number} threshold - Detection threshold (optional)
     * @returns {Promise<object>} Test result
     */
    async testWithFile(modelId, audioBlob, threshold = null) {
        const formData = new FormData();
        formData.append('model_id', modelId);
        formData.append('audio_file', audioBlob, 'test.wav');
        if (threshold !== null) {
            formData.append('threshold', threshold);
        }
        return this.postForm('/api/test/file', formData);
    },

    /**
     * List models available for testing
     * @returns {Promise<object>} Testable models
     */
    async listTestableModels() {
        return this.get('/api/test/models');
    },

    /**
     * Create WebSocket connection for real-time testing
     * @param {string} modelId - Model ID
     * @param {number} threshold - Detection threshold
     * @param {number} cooldownMs - Cooldown between detections
     * @param {boolean} noiseReduction - Enable noise reduction
     * @returns {WebSocket} WebSocket connection
     */
    createTestWebSocket(modelId, threshold = 0.5, cooldownMs = 1000, noiseReduction = false) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        const params = new URLSearchParams({
            model_id: modelId,
            threshold: threshold,
            cooldown_ms: cooldownMs,
            noise_reduction: noiseReduction,
        });
        return new WebSocket(`${protocol}//${host}/api/test/realtime?${params}`);
    },

    /**
     * Get device info for inference
     * @returns {Promise<object>} Device info
     */
    async getDeviceInfo() {
        return this.get('/api/test/device');
    },

    /**
     * Set device for inference
     * @param {string} device - Device to use ('cpu' or 'cuda')
     * @returns {Promise<object>} Updated device info
     */
    async setDevice(device) {
        const formData = new FormData();
        formData.append('device', device);
        return this.postForm('/api/test/device', formData);
    },

    // ========================================================================
    // Cache Management
    // ========================================================================

    /**
     * Get comprehensive cache status
     * @returns {Promise<object>} Cache status
     */
    async getCacheStatus() {
        return this.get('/api/cache/status');
    },

    /**
     * Get negative data cache info (legacy)
     * @returns {Promise<object>} Cache info with chunk count
     */
    async getNegativeCacheInfo() {
        return this.get('/api/train/negative-cache/info');
    },

    /**
     * Build audio cache with SSE progress
     * @param {function} onProgress - Progress callback
     * @returns {Promise<object>} Build result
     */
    async buildAudioCache(onProgress) {
        return this.streamSSE('/api/cache/build/audio', onProgress);
    },

    /**
     * Build spectrogram cache with SSE progress
     * @param {function} onProgress - Progress callback
     * @returns {Promise<object>} Build result
     */
    async buildSpectrogramCache(onProgress) {
        return this.streamSSE('/api/cache/build/spectrograms', onProgress);
    },

    /**
     * Build all caches with SSE progress
     * @param {function} onProgress - Progress callback
     * @returns {Promise<object>} Build result
     */
    async buildAllCaches(onProgress) {
        return this.streamSSE('/api/cache/build/all', onProgress);
    },

    /**
     * Clear all caches
     * @returns {Promise<object>} Clear status
     */
    async clearAllCaches() {
        return this.delete('/api/cache/all');
    },

    /**
     * Clear negative data cache (legacy)
     * @returns {Promise<object>} Clear status
     */
    async clearNegativeCache() {
        return this.delete('/api/train/negative-cache');
    },

    /**
     * Get negative data status (source files availability)
     * @returns {Promise<object>} Negative data status
     */
    async getNegativeDataStatus() {
        return this.get('/api/cache/negative-data/status');
    },

    /**
     * Download negative data (UNAC dataset) from Kaggle
     * @param {function} onProgress - Progress callback
     * @returns {Promise<object>} Download result
     */
    async downloadNegativeData(onProgress) {
        return this.streamSSE('/api/cache/negative-data/download', onProgress);
    },


    /**
     * Stream SSE events from a POST endpoint
     * @param {string} endpoint - API endpoint
     * @param {function} onProgress - Progress callback
     * @returns {Promise<object>} Final result
     */
    async streamSSE(endpoint, onProgress) {
        return new Promise((resolve, reject) => {
            fetch(`${API_BASE}${endpoint}`, {
                method: 'POST',
            }).then(response => {
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                let lastEvent = null;

                function read() {
                    reader.read().then(({ done, value }) => {
                        if (done) {
                            resolve(lastEvent);
                            return;
                        }

                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split('\n');
                        buffer = lines.pop() || '';

                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                try {
                                    const event = JSON.parse(line.slice(6));
                                    lastEvent = event;
                                    if (onProgress) {
                                        onProgress(event);
                                    }
                                    if (event.type === 'error') {
                                        reject(new Error(event.message));
                                        return;
                                    }
                                } catch (e) {
                                    // Ignore parse errors
                                }
                            }
                        }
                        read();
                    }).catch(reject);
                }
                read();
            }).catch(reject);
        });
    },

    // ========================================================================
    // Model Recordings
    // ========================================================================

    /**
     * Get recordings for a model
     * @param {string} modelId - Model ID
     * @returns {Promise<object>} Recordings list
     */
    async getModelRecordings(modelId) {
        return this.get(`/api/models/${modelId}/recordings`);
    },

    // ========================================================================
    // ONNX Export
    // ========================================================================

    /**
     * Check ONNX export status for a model
     * @param {string} modelId - Model ID
     * @returns {Promise<object>} ONNX status
     */
    async getOnnxStatus(modelId) {
        return this.get(`/api/models/${modelId}/onnx-status`);
    },

    /**
     * Export model to ONNX format
     * @param {string} modelId - Model ID
     * @returns {Promise<object>} Export result
     */
    async exportToOnnx(modelId) {
        return this.postForm(`/api/models/${modelId}/export-onnx`, new FormData());
    },

    /**
     * Delete ONNX export for a model
     * @param {string} modelId - Model ID
     * @returns {Promise<object>} Delete result
     */
    async deleteOnnxExport(modelId) {
        return this.delete(`/api/models/${modelId}/onnx`);
    },

    /**
     * Move a custom model to the default folder
     * @param {string} modelId - Model ID
     * @returns {Promise<object>} Move result
     */
    async moveModelToDefault(modelId) {
        return this.postForm(`/api/models/${modelId}/move-to-default`, new FormData());
    },

    /**
     * Test model with audio file using ONNX
     * @param {string} modelId - Model ID
     * @param {Blob} audioBlob - Audio file
     * @param {number} threshold - Detection threshold (optional)
     * @param {boolean} useOnnx - Use ONNX model
     * @returns {Promise<object>} Test result
     */
    async testWithFileOnnx(modelId, audioBlob, threshold = null, useOnnx = false) {
        const formData = new FormData();
        formData.append('model_id', modelId);
        formData.append('audio_file', audioBlob, 'test.wav');
        if (threshold !== null) {
            formData.append('threshold', threshold);
        }
        formData.append('use_onnx', useOnnx);
        return this.postForm('/api/test/file', formData);
    },

    /**
     * Create WebSocket connection for real-time testing with ONNX support
     * @param {string} modelId - Model ID
     * @param {number} threshold - Detection threshold
     * @param {number} cooldownMs - Cooldown between detections
     * @param {boolean} noiseReduction - Enable noise reduction
     * @param {boolean} useOnnx - Use ONNX model
     * @returns {WebSocket} WebSocket connection
     */
    createTestWebSocketOnnx(modelId, threshold = 0.5, cooldownMs = 1000, noiseReduction = false, useOnnx = false) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        const params = new URLSearchParams({
            model_id: modelId,
            threshold: threshold,
            cooldown_ms: cooldownMs,
            noise_reduction: noiseReduction,
            use_onnx: useOnnx,
        });
        return new WebSocket(`${protocol}//${host}/api/test/realtime?${params}`);
    },
};
