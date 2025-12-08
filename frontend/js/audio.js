/**
 * WakeBuilder - Audio Recording & Processing
 */

/**
 * Audio recorder for capturing wake word samples
 */
class AudioRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioContext = null;
        this.analyser = null;
        this.stream = null;
        this.chunks = [];
        this.isRecording = false;
        this.onDataAvailable = null;
        this.onVisualize = null;
    }

    /**
     * Initialize audio context and get microphone access
     * @returns {Promise<boolean>} Success status
     */
    async initialize() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true,
                },
            });

            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000,
            });

            const source = this.audioContext.createMediaStreamSource(this.stream);
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            source.connect(this.analyser);

            return true;
        } catch (error) {
            console.error('Failed to initialize audio:', error);
            throw new Error('Microphone access denied. Please allow microphone access and try again.');
        }
    }

    /**
     * Start recording
     */
    startRecording() {
        if (!this.stream) {
            throw new Error('Audio not initialized');
        }

        this.chunks = [];
        this.mediaRecorder = new MediaRecorder(this.stream, {
            mimeType: 'audio/webm;codecs=opus',
        });

        this.mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                this.chunks.push(event.data);
            }
        };

        this.mediaRecorder.onstop = async () => {
            const webmBlob = new Blob(this.chunks, { type: 'audio/webm' });
            // Convert to WAV for API compatibility
            const wavBlob = await this.convertToWav(webmBlob);
            if (this.onDataAvailable) {
                this.onDataAvailable(wavBlob);
            }
        };

        this.mediaRecorder.start();
        this.isRecording = true;
        this.startVisualization();
    }

    /**
     * Stop recording
     * @returns {Promise<Blob>} Recorded audio as WAV blob
     */
    stopRecording() {
        return new Promise((resolve) => {
            if (!this.mediaRecorder || !this.isRecording) {
                resolve(null);
                return;
            }

            this.mediaRecorder.onstop = async () => {
                const webmBlob = new Blob(this.chunks, { type: 'audio/webm' });
                const wavBlob = await this.convertToWav(webmBlob);
                this.isRecording = false;
                resolve(wavBlob);
            };

            this.mediaRecorder.stop();
        });
    }

    /**
     * Convert audio blob to WAV format
     * @param {Blob} blob - Audio blob
     * @returns {Promise<Blob>} WAV blob
     */
    async convertToWav(blob) {
        const arrayBuffer = await blob.arrayBuffer();
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        
        // Resample to 16kHz mono
        const offlineContext = new OfflineAudioContext(
            1,
            audioBuffer.duration * 16000,
            16000
        );
        
        const source = offlineContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(offlineContext.destination);
        source.start();
        
        const renderedBuffer = await offlineContext.startRendering();
        const wavBlob = this.audioBufferToWav(renderedBuffer);
        
        audioContext.close();
        
        return wavBlob;
    }

    /**
     * Convert AudioBuffer to WAV blob
     * @param {AudioBuffer} buffer - Audio buffer
     * @returns {Blob} WAV blob
     */
    audioBufferToWav(buffer) {
        const numChannels = 1;
        const sampleRate = buffer.sampleRate;
        const format = 1; // PCM
        const bitDepth = 16;
        
        const bytesPerSample = bitDepth / 8;
        const blockAlign = numChannels * bytesPerSample;
        
        const samples = buffer.getChannelData(0);
        const dataLength = samples.length * bytesPerSample;
        const bufferLength = 44 + dataLength;
        
        const arrayBuffer = new ArrayBuffer(bufferLength);
        const view = new DataView(arrayBuffer);
        
        // WAV header
        this.writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + dataLength, true);
        this.writeString(view, 8, 'WAVE');
        this.writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, format, true);
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * blockAlign, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitDepth, true);
        this.writeString(view, 36, 'data');
        view.setUint32(40, dataLength, true);
        
        // Audio data
        let offset = 44;
        for (let i = 0; i < samples.length; i++) {
            const sample = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
        
        return new Blob([arrayBuffer], { type: 'audio/wav' });
    }

    /**
     * Write string to DataView
     * @param {DataView} view - Data view
     * @param {number} offset - Offset
     * @param {string} string - String to write
     */
    writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    /**
     * Start waveform visualization
     */
    startVisualization() {
        if (!this.analyser || !this.onVisualize) return;

        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        const draw = () => {
            if (!this.isRecording) return;

            requestAnimationFrame(draw);
            this.analyser.getByteTimeDomainData(dataArray);
            this.onVisualize(dataArray);
        };

        draw();
    }

    /**
     * Get current audio level (0-1)
     * @returns {number} Audio level
     */
    getLevel() {
        if (!this.analyser) return 0;

        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        this.analyser.getByteTimeDomainData(dataArray);

        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {
            const value = (dataArray[i] - 128) / 128;
            sum += value * value;
        }

        return Math.sqrt(sum / bufferLength);
    }

    /**
     * Clean up resources
     */
    dispose() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }
        if (this.audioContext) {
            this.audioContext.close();
        }
    }
}

/**
 * Real-time audio streamer for WebSocket testing
 */
class AudioStreamer {
    constructor() {
        this.audioContext = null;
        this.processor = null;
        this.source = null;
        this.stream = null;
        this.onAudioData = null;
        this.isStreaming = false;
    }

    /**
     * Initialize audio streaming
     * @returns {Promise<boolean>} Success status
     */
    async initialize() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true,
                },
            });

            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000,
            });

            return true;
        } catch (error) {
            console.error('Failed to initialize audio streaming:', error);
            throw new Error('Microphone access denied');
        }
    }

    /**
     * Get the actual sample rate of the audio context
     * @returns {number} The actual sample rate
     */
    getActualSampleRate() {
        return this.audioContext ? this.audioContext.sampleRate : 16000;
    }

    /**
     * Start streaming audio
     */
    start() {
        if (!this.stream || !this.audioContext) {
            throw new Error('Audio not initialized');
        }

        // Log actual sample rate for debugging
        console.log(`AudioContext actual sample rate: ${this.audioContext.sampleRate}`);

        this.source = this.audioContext.createMediaStreamSource(this.stream);
        
        // Use ScriptProcessorNode for audio processing
        // Note: This is deprecated but still widely supported
        // AudioWorklet would be the modern alternative
        this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
        
        this.processor.onaudioprocess = (event) => {
            if (!this.isStreaming || !this.onAudioData) return;

            const inputData = event.inputBuffer.getChannelData(0);
            
            // Convert float32 to int16
            const int16Data = new Int16Array(inputData.length);
            for (let i = 0; i < inputData.length; i++) {
                const sample = Math.max(-1, Math.min(1, inputData[i]));
                int16Data[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
            }

            this.onAudioData(int16Data.buffer);
        };

        this.source.connect(this.processor);
        this.processor.connect(this.audioContext.destination);
        this.isStreaming = true;
    }

    /**
     * Stop streaming audio
     */
    stop() {
        this.isStreaming = false;
        
        if (this.processor) {
            this.processor.disconnect();
            this.processor = null;
        }
        
        if (this.source) {
            this.source.disconnect();
            this.source = null;
        }
    }

    /**
     * Clean up resources
     */
    dispose() {
        this.stop();
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }
        
        if (this.audioContext) {
            this.audioContext.close();
        }
    }
}

/**
 * Draw waveform on canvas
 * @param {HTMLCanvasElement} canvas - Canvas element
 * @param {Uint8Array} dataArray - Audio data
 */
function drawWaveform(canvas, dataArray) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.fillStyle = '#334155';
    ctx.fillRect(0, 0, width, height);

    ctx.lineWidth = 2;
    ctx.strokeStyle = '#6366f1';
    ctx.beginPath();

    const sliceWidth = width / dataArray.length;
    let x = 0;

    for (let i = 0; i < dataArray.length; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * height) / 2;

        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }

        x += sliceWidth;
    }

    ctx.lineTo(width, height / 2);
    ctx.stroke();
}

/**
 * Draw idle waveform (flat line)
 * @param {HTMLCanvasElement} canvas - Canvas element
 */
function drawIdleWaveform(canvas) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.fillStyle = '#334155';
    ctx.fillRect(0, 0, width, height);

    ctx.lineWidth = 2;
    ctx.strokeStyle = '#64748b';
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();
}
