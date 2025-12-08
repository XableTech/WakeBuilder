/**
 * WakeBuilder - Utility Functions
 */

/**
 * Format a number as percentage
 * @param {number} value - Value between 0 and 1
 * @param {number} decimals - Number of decimal places
 * @returns {string} Formatted percentage
 */
function formatPercent(value, decimals = 1) {
    return (value * 100).toFixed(decimals) + '%';
}

/**
 * Format a number with fixed decimals
 * @param {number} value - Number to format
 * @param {number} decimals - Number of decimal places
 * @returns {string} Formatted number
 */
function formatNumber(value, decimals = 4) {
    if (value === null || value === undefined) return '-';
    return value.toFixed(decimals);
}

/**
 * Format file size in KB or MB
 * @param {number} kb - Size in kilobytes
 * @returns {string} Formatted size
 */
function formatSize(kb) {
    if (kb < 1024) {
        return kb.toFixed(1) + ' KB';
    }
    return (kb / 1024).toFixed(2) + ' MB';
}

/**
 * Format duration in seconds to human readable string
 * @param {number} seconds - Duration in seconds
 * @returns {string} Formatted duration
 */
function formatDuration(seconds) {
    if (seconds < 60) {
        return seconds.toFixed(0) + 's';
    } else if (seconds < 3600) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}m ${secs}s`;
    } else {
        const hours = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${mins}m`;
    }
}

/**
 * Format date to locale string
 * @param {string} isoString - ISO date string
 * @returns {string} Formatted date
 */
function formatDate(isoString) {
    if (!isoString) return '-';
    const date = new Date(isoString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

/**
 * Format time as HH:MM:SS
 * @param {Date} date - Date object
 * @returns {string} Formatted time
 */
function formatTime(date) {
    return date.toLocaleTimeString();
}

/**
 * Debounce function calls
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in ms
 * @returns {Function} Debounced function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Show a toast notification
 * @param {string} message - Message to display
 * @param {string} type - Type: 'success', 'error', 'info'
 * @param {number} duration - Duration in ms
 */
function showToast(message, type = 'info', duration = 3000) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

/**
 * Generate a unique ID
 * @returns {string} UUID-like string
 */
function generateId() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

/**
 * Sleep for specified duration
 * @param {number} ms - Duration in milliseconds
 * @returns {Promise} Promise that resolves after duration
 */
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Validate wake word
 * @param {string} wakeWord - Wake word to validate
 * @returns {{valid: boolean, error: string|null}} Validation result
 */
function validateWakeWord(wakeWord) {
    const trimmed = wakeWord.trim().replace(/\s+/g, ' ');
    
    // Check minimum length (4 characters)
    if (trimmed.length < 4) {
        return { valid: false, error: 'Wake word must be at least 4 characters' };
    }
    
    // Check maximum length (12 characters)
    if (trimmed.length > 12) {
        return { valid: false, error: 'Wake word must be at most 12 characters' };
    }
    
    // Check for valid characters (letters and single space between words)
    if (!/^[A-Za-z]+( [A-Za-z]+)?$/.test(trimmed)) {
        return { valid: false, error: 'Wake word must be 1-2 words, letters only, single space between words' };
    }
    
    const words = trimmed.split(' ').filter(w => w.length > 0);
    
    // Check word count (1-2 words)
    if (words.length > 2) {
        return { valid: false, error: 'Wake word must be 1-2 words' };
    }
    
    // For 2-word wake words, ensure each word has at least 2 characters
    if (words.length === 2) {
        if (words[0].length < 2 || words[1].length < 2) {
            return { valid: false, error: 'Each word must be at least 2 characters' };
        }
    }
    
    return { valid: true, error: null };
}

/**
 * Convert ArrayBuffer to Base64
 * @param {ArrayBuffer} buffer - Array buffer
 * @returns {string} Base64 string
 */
function arrayBufferToBase64(buffer) {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
}

/**
 * Download a blob as a file
 * @param {Blob} blob - Blob to download
 * @param {string} filename - Filename
 */
function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}
