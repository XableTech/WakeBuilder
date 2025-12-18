/**
 * WakeBuilder - Chart Drawing Utilities
 * Simple canvas-based charts without external dependencies
 */

/**
 * Chart configuration
 */
const chartConfig = {
    colors: {
        primary: '#6366f1',
        secondary: '#818cf8',
        success: '#22c55e',
        warning: '#f59e0b',
        error: '#ef4444',
        grid: '#334155',
        text: '#94a3b8',
        background: '#1e293b',
    },
    padding: {
        top: 20,
        right: 20,
        bottom: 40,
        left: 50,
    },
};

/**
 * Training history chart
 */
class TrainingChart {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.data = {
            trainLoss: [],
            valLoss: [],
            valAccuracy: [],
        };
    }

    /**
     * Update chart with new epoch data
     * @param {object} epochData - Epoch metrics
     */
    update(epochData) {
        this.data.trainLoss.push(epochData.train_loss);
        this.data.valLoss.push(epochData.val_loss);
        this.data.valAccuracy.push(epochData.val_accuracy);
        this.draw();
    }

    /**
     * Set full history data
     * @param {Array} history - Array of epoch metrics
     */
    setHistory(history) {
        this.data.trainLoss = history.map(h => h.train_loss);
        this.data.valLoss = history.map(h => h.val_loss);
        this.data.valAccuracy = history.map(h => h.val_accuracy);
        this.draw();
    }

    /**
     * Set data directly from arrays
     * @param {Array} trainLoss - Array of train loss values
     * @param {Array} valLoss - Array of validation loss values
     */
    setData(trainLoss, valLoss) {
        this.data.trainLoss = trainLoss;
        this.data.valLoss = valLoss;
        this.draw();
    }

    /**
     * Clear chart data
     */
    clear() {
        this.data = {
            trainLoss: [],
            valLoss: [],
            valAccuracy: [],
        };
        this.draw();
    }

    /**
     * Draw the chart
     */
    draw() {
        const { ctx, canvas } = this;
        const { padding, colors } = chartConfig;
        
        const width = canvas.width;
        const height = canvas.height;
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;

        // Clear canvas
        ctx.fillStyle = colors.background;
        ctx.fillRect(0, 0, width, height);

        if (this.data.trainLoss.length === 0) {
            ctx.fillStyle = colors.text;
            ctx.font = '14px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Training data will appear here...', width / 2, height / 2);
            return;
        }

        // Calculate scales
        const epochs = this.data.trainLoss.length;
        const allLosses = [...this.data.trainLoss, ...this.data.valLoss];
        const maxLoss = Math.max(...allLosses) * 1.1;
        const minLoss = 0;

        const xScale = chartWidth / Math.max(epochs - 1, 1);
        const yScale = chartHeight / (maxLoss - minLoss);

        // Draw grid
        ctx.strokeStyle = colors.grid;
        ctx.lineWidth = 1;
        
        // Horizontal grid lines
        for (let i = 0; i <= 5; i++) {
            const y = padding.top + (chartHeight / 5) * i;
            ctx.beginPath();
            ctx.moveTo(padding.left, y);
            ctx.lineTo(width - padding.right, y);
            ctx.stroke();
            
            // Y-axis labels
            const value = maxLoss - (maxLoss / 5) * i;
            ctx.fillStyle = colors.text;
            ctx.font = '11px sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText(value.toFixed(2), padding.left - 5, y + 4);
        }

        // X-axis labels
        ctx.textAlign = 'center';
        const labelInterval = Math.ceil(epochs / 10);
        for (let i = 0; i < epochs; i += labelInterval) {
            const x = padding.left + i * xScale;
            ctx.fillText(String(i + 1), x, height - padding.bottom + 20);
        }

        // Draw train loss line
        this.drawLine(this.data.trainLoss, colors.primary, xScale, yScale, maxLoss);

        // Draw val loss line
        this.drawLine(this.data.valLoss, colors.error, xScale, yScale, maxLoss);

        // Draw legend
        this.drawLegend();
    }

    /**
     * Draw a line on the chart
     * @param {Array} data - Data points
     * @param {string} color - Line color
     * @param {number} xScale - X scale factor
     * @param {number} yScale - Y scale factor
     * @param {number} maxY - Maximum Y value
     */
    drawLine(data, color, xScale, yScale, maxY) {
        const { ctx } = this;
        const { padding } = chartConfig;

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();

        for (let i = 0; i < data.length; i++) {
            const x = padding.left + i * xScale;
            const y = padding.top + (maxY - data[i]) * yScale;

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }

        ctx.stroke();

        // Draw points
        ctx.fillStyle = color;
        for (let i = 0; i < data.length; i++) {
            const x = padding.left + i * xScale;
            const y = padding.top + (maxY - data[i]) * yScale;
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    /**
     * Draw chart legend
     */
    drawLegend() {
        const { ctx, canvas } = this;
        const { colors } = chartConfig;

        const legendX = canvas.width - 120;
        const legendY = 15;

        ctx.font = '11px sans-serif';
        ctx.textAlign = 'left';

        // Train loss
        ctx.fillStyle = colors.primary;
        ctx.fillRect(legendX, legendY, 12, 12);
        ctx.fillStyle = colors.text;
        ctx.fillText('Train Loss', legendX + 18, legendY + 10);

        // Val loss
        ctx.fillStyle = colors.error;
        ctx.fillRect(legendX, legendY + 18, 12, 12);
        ctx.fillStyle = colors.text;
        ctx.fillText('Val Loss', legendX + 18, legendY + 28);
    }
}

/**
 * Accuracy and F1 history chart
 */
class AccuracyChart {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.data = {
            valAccuracy: [],
            valF1: [],
        };
    }

    /**
     * Set full history data
     * @param {Array} history - Array of epoch metrics
     */
    setHistory(history) {
        this.data.valAccuracy = history.map(h => h.val_accuracy);
        this.data.valF1 = history.map(h => h.val_f1);
        this.draw();
    }

    /**
     * Set data directly from arrays
     * @param {Array} valAccuracy - Array of validation accuracy values
     * @param {Array} valF1 - Array of validation F1 values
     */
    setData(valAccuracy, valF1) {
        this.data.valAccuracy = valAccuracy;
        this.data.valF1 = valF1 || [];
        this.draw();
    }

    /**
     * Clear chart data
     */
    clear() {
        this.data = {
            valAccuracy: [],
            valF1: [],
        };
        this.draw();
    }

    /**
     * Draw the chart
     */
    draw() {
        const { ctx, canvas } = this;
        const { padding, colors } = chartConfig;
        
        const width = canvas.width;
        const height = canvas.height;
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;

        // Clear canvas
        ctx.fillStyle = colors.background;
        ctx.fillRect(0, 0, width, height);

        if (this.data.valAccuracy.length === 0) {
            ctx.fillStyle = colors.text;
            ctx.font = '14px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Accuracy data will appear here...', width / 2, height / 2);
            return;
        }

        // Calculate scales (0 to 1 for accuracy/F1)
        const epochs = this.data.valAccuracy.length;
        const maxY = 1.0;
        const minY = Math.min(
            Math.min(...this.data.valAccuracy),
            Math.min(...this.data.valF1)
        ) * 0.9;

        const xScale = chartWidth / Math.max(epochs - 1, 1);
        const yScale = chartHeight / (maxY - minY);

        // Draw grid
        ctx.strokeStyle = colors.grid;
        ctx.lineWidth = 1;
        
        // Horizontal grid lines
        for (let i = 0; i <= 5; i++) {
            const y = padding.top + (chartHeight / 5) * i;
            ctx.beginPath();
            ctx.moveTo(padding.left, y);
            ctx.lineTo(width - padding.right, y);
            ctx.stroke();
            
            // Y-axis labels (percentage)
            const value = maxY - ((maxY - minY) / 5) * i;
            ctx.fillStyle = colors.text;
            ctx.font = '11px sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText((value * 100).toFixed(0) + '%', padding.left - 5, y + 4);
        }

        // X-axis labels
        ctx.textAlign = 'center';
        const labelInterval = Math.ceil(epochs / 10);
        for (let i = 0; i < epochs; i += labelInterval) {
            const x = padding.left + i * xScale;
            ctx.fillText(String(i + 1), x, height - padding.bottom + 20);
        }

        // Draw val accuracy line
        this.drawLine(this.data.valAccuracy, colors.success, xScale, yScale, maxY, minY);

        // Draw val F1 line
        this.drawLine(this.data.valF1, colors.warning, xScale, yScale, maxY, minY);

        // Draw legend
        this.drawLegend();
    }

    /**
     * Draw a line on the chart
     */
    drawLine(data, color, xScale, yScale, maxY, minY) {
        const { ctx } = this;
        const { padding } = chartConfig;

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();

        for (let i = 0; i < data.length; i++) {
            const x = padding.left + i * xScale;
            const y = padding.top + (maxY - data[i]) * yScale;

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }

        ctx.stroke();

        // Draw points
        ctx.fillStyle = color;
        for (let i = 0; i < data.length; i++) {
            const x = padding.left + i * xScale;
            const y = padding.top + (maxY - data[i]) * yScale;
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    /**
     * Draw chart legend
     */
    drawLegend() {
        const { ctx, canvas } = this;
        const { colors } = chartConfig;

        const legendX = canvas.width - 120;
        const legendY = 15;

        ctx.font = '11px sans-serif';
        ctx.textAlign = 'left';

        // Val Accuracy
        ctx.fillStyle = colors.success;
        ctx.fillRect(legendX, legendY, 12, 12);
        ctx.fillStyle = colors.text;
        ctx.fillText('Val Accuracy', legendX + 18, legendY + 10);

        // Val F1
        ctx.fillStyle = colors.warning;
        ctx.fillRect(legendX, legendY + 18, 12, 12);
        ctx.fillStyle = colors.text;
        ctx.fillText('Val F1', legendX + 18, legendY + 28);
    }
}

/**
 * Threshold analysis chart (FAR/FRR curves)
 */
class ThresholdChart {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.data = null;
        this.optimalThreshold = null;
    }

    /**
     * Set threshold analysis data
     * @param {Array} data - Threshold metrics array
     * @param {number} optimalThreshold - Optimal threshold value
     */
    setData(data, optimalThreshold) {
        this.data = data;
        this.optimalThreshold = optimalThreshold;
        this.draw();
    }

    /**
     * Draw the chart
     */
    draw() {
        const { ctx, canvas, data } = this;
        const { padding, colors } = chartConfig;

        const width = canvas.width;
        const height = canvas.height;
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;

        // Clear canvas
        ctx.fillStyle = colors.background;
        ctx.fillRect(0, 0, width, height);

        if (!data || data.length === 0) {
            ctx.fillStyle = colors.text;
            ctx.font = '14px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Threshold analysis will appear here...', width / 2, height / 2);
            return;
        }

        // Scales
        const xScale = chartWidth / (data.length - 1);
        const yScale = chartHeight;

        // Draw grid
        ctx.strokeStyle = colors.grid;
        ctx.lineWidth = 1;

        for (let i = 0; i <= 5; i++) {
            const y = padding.top + (chartHeight / 5) * i;
            ctx.beginPath();
            ctx.moveTo(padding.left, y);
            ctx.lineTo(width - padding.right, y);
            ctx.stroke();

            // Y-axis labels (percentage)
            const value = 100 - (100 / 5) * i;
            ctx.fillStyle = colors.text;
            ctx.font = '11px sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText(value + '%', padding.left - 5, y + 4);
        }

        // X-axis labels
        ctx.textAlign = 'center';
        for (let i = 0; i <= 10; i++) {
            const x = padding.left + (chartWidth / 10) * i;
            const threshold = (i / 10).toFixed(1);
            ctx.fillText(threshold, x, height - padding.bottom + 20);
        }

        // Draw FAR line
        ctx.strokeStyle = colors.error;
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < data.length; i++) {
            const x = padding.left + i * xScale;
            const y = padding.top + (1 - data[i].far) * yScale;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Draw FRR line
        ctx.strokeStyle = colors.warning;
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < data.length; i++) {
            const x = padding.left + i * xScale;
            const y = padding.top + (1 - data[i].frr) * yScale;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Draw optimal threshold line
        if (this.optimalThreshold !== null) {
            const optimalX = padding.left + (this.optimalThreshold * chartWidth);
            ctx.strokeStyle = colors.success;
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            ctx.moveTo(optimalX, padding.top);
            ctx.lineTo(optimalX, height - padding.bottom);
            ctx.stroke();
            ctx.setLineDash([]);

            // Label
            ctx.fillStyle = colors.success;
            ctx.font = '11px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Optimal: ' + this.optimalThreshold.toFixed(2), optimalX, padding.top - 5);
        }
    }
}

/**
 * Simple bar chart for metrics comparison
 */
class MetricsChart {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.data = null;
    }

    /**
     * Set metrics data
     * @param {object} data - Metrics object {label: value}
     */
    setData(data) {
        this.data = data;
        this.draw();
    }

    /**
     * Draw the chart
     */
    draw() {
        const { ctx, canvas, data } = this;
        const { colors } = chartConfig;

        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.fillStyle = colors.background;
        ctx.fillRect(0, 0, width, height);

        if (!data) return;

        const labels = Object.keys(data);
        const values = Object.values(data);
        const barWidth = (width - 60) / labels.length - 10;
        const maxValue = 1;

        labels.forEach((label, i) => {
            const x = 40 + i * (barWidth + 10);
            const barHeight = (values[i] / maxValue) * (height - 60);
            const y = height - 30 - barHeight;

            // Bar
            ctx.fillStyle = colors.primary;
            ctx.fillRect(x, y, barWidth, barHeight);

            // Value
            ctx.fillStyle = colors.text;
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText((values[i] * 100).toFixed(1) + '%', x + barWidth / 2, y - 5);

            // Label
            ctx.fillText(label, x + barWidth / 2, height - 10);
        });
    }
}
