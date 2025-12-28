/**
 * ECG Renderer Module
 * 
 * Handles the main ECG waveform canvas rendering with grid, 
 * signal trace, R-peak markers, drag interaction, and export.
 */

class ECGRenderer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.setupCanvas();
        
        // Drag state
        this.isDragging = false;
        this.lastDragX = 0;
        this.onDragCallback = null;
        
        // Setup event listeners for dragging
        this.setupDragListeners();
        
        // Bind resize handler
        window.addEventListener('resize', () => this.setupCanvas());
    }
    
    /**
     * Set up canvas with proper DPI scaling
     */
    setupCanvas() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * window.devicePixelRatio;
        this.canvas.height = rect.height * window.devicePixelRatio;
        this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        this.width = rect.width;
        this.height = rect.height;
    }
    
    /**
     * Set up mouse/touch drag listeners for interactive scrolling
     */
    setupDragListeners() {
        // Mouse events
        this.canvas.addEventListener('mousedown', (e) => this.startDrag(e.clientX));
        this.canvas.addEventListener('mousemove', (e) => this.drag(e.clientX));
        this.canvas.addEventListener('mouseup', () => this.endDrag());
        this.canvas.addEventListener('mouseleave', () => this.endDrag());
        
        // Touch events for mobile
        this.canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.startDrag(e.touches[0].clientX);
        });
        this.canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            this.drag(e.touches[0].clientX);
        });
        this.canvas.addEventListener('touchend', () => this.endDrag());
        
        // Change cursor style
        this.canvas.style.cursor = 'grab';
    }
    
    /**
     * Start dragging
     */
    startDrag(x) {
        this.isDragging = true;
        this.lastDragX = x;
        this.canvas.style.cursor = 'grabbing';
    }
    
    /**
     * Handle drag motion
     */
    drag(x) {
        if (!this.isDragging) return;
        
        const deltaX = x - this.lastDragX;
        this.lastDragX = x;
        
        // Convert pixel delta to time delta (negative = go back in time)
        // Assuming 5 seconds displayed across the canvas width
        const secondsPerPixel = 5 / this.width;
        const deltaSeconds = -deltaX * secondsPerPixel;
        
        if (this.onDragCallback && Math.abs(deltaSeconds) > 0.01) {
            this.onDragCallback(deltaSeconds);
        }
    }
    
    /**
     * End dragging
     */
    endDrag() {
        this.isDragging = false;
        this.canvas.style.cursor = 'grab';
    }
    
    /**
     * Set callback for drag events
     * @param {Function} callback - Called with deltaSeconds when dragging
     */
    setDragCallback(callback) {
        this.onDragCallback = callback;
    }
    
    /**
     * Clear canvas and draw background
     */
    clear() {
        this.ctx.fillStyle = '#0a0a1a';
        this.ctx.fillRect(0, 0, this.width, this.height);
    }
    
    /**
     * Draw grid lines
     */
    drawGrid() {
        this.ctx.strokeStyle = 'rgba(0, 255, 136, 0.1)';
        this.ctx.lineWidth = 1;
        
        // Vertical lines
        for (let x = 0; x < this.width; x += 50) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.height);
            this.ctx.stroke();
        }
        
        // Horizontal lines
        for (let y = 0; y < this.height; y += 50) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.width, y);
            this.ctx.stroke();
        }
    }
    
    /**
     * Draw ECG signal trace
     * @param {Array} samples - ECG sample values
     */
    drawSignal(samples) {
        if (!samples || samples.length < 2) return;
        
        const minVal = Math.min(...samples);
        const maxVal = Math.max(...samples);
        const range = maxVal - minVal || 1;
        
        this.ctx.strokeStyle = '#00ff88';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        
        for (let i = 0; i < samples.length; i++) {
            const x = (i / samples.length) * this.width;
            const y = this.height - ((samples[i] - minVal) / range) * (this.height - 40) - 20;
            
            if (i === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        }
        
        this.ctx.stroke();
    }
    
    /**
     * Draw R-peak markers
     * @param {Array} annotations - Beat annotations with sample_index and beat_type
     * @param {Array} samples - ECG sample values
     * @param {number} startSample - Start sample index of the visible window
     * @param {Array} classifications - Classification results for false detection markers
     */
    drawRPeakMarkers(annotations, samples, startSample, classifications = []) {
        if (!annotations || !samples || samples.length < 2) return;
        
        const minVal = Math.min(...samples);
        const maxVal = Math.max(...samples);
        const range = maxVal - minVal || 1;
        const endSample = startSample + samples.length;
        
        annotations.forEach(ann => {
            if (ann.sample_index > startSample && ann.sample_index <= endSample) {
                const bufferIdx = ann.sample_index - startSample;
                
                if (bufferIdx >= 0 && bufferIdx < samples.length) {
                    const x = (bufferIdx / samples.length) * this.width;
                    const y = this.height - ((samples[bufferIdx] - minVal) / range) * (this.height - 40) - 20;
                    
                    // Check for false detection
                    // Check for false detection (prediction mismatch)
                    const classResult = classifications.find(c => c.r_peak === ann.sample_index);
                    if (classResult && !classResult.correct) {
                        this.ctx.strokeStyle = '#ffd700';
                        this.ctx.lineWidth = 3;
                        this.ctx.beginPath();
                        this.ctx.arc(x, y, 10, 0, Math.PI * 2);
                        this.ctx.stroke();
                    }
                    
                    // Draw R-peak marker
                    this.ctx.fillStyle = ann.beat_type === 'N' ? '#00ff88' : '#ff4757';
                    this.ctx.beginPath();
                    this.ctx.arc(x, y, 6, 0, Math.PI * 2);
                    this.ctx.fill();
                }
            }
        });
    }
    
    /**
     * Draw history indicator when not in live mode
     */
    drawHistoryIndicator() {
        this.ctx.fillStyle = 'rgba(255, 215, 0, 0.9)';
        this.ctx.font = 'bold 14px Arial';
        this.ctx.fillText('📜 VIEWING HISTORY', 10, 25);
    }
    
    /**
     * Export ECG graph as a clean medical format image
     * @param {Object} options - Export options
     * @returns {string} Data URL of the exported image
     */
    exportAsImage(options = {}) {
        const {
            samples = [],
            annotations = [],
            startSample = 0,
            classifications = [],
            patientInfo = '',
            timestamp = new Date().toISOString(),
            modelName = 'ECG Model',
            showGrid = true,
            format = 'png'  // 'png' or 'jpeg'
        } = options;
        
        // Create a new canvas for export (larger, white background for print)
        const exportCanvas = document.createElement('canvas');
        const exportWidth = 1200;
        const exportHeight = 600;
        exportCanvas.width = exportWidth;
        exportCanvas.height = exportHeight;
        const ctx = exportCanvas.getContext('2d');
        
        // White background for medical printing
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, exportWidth, exportHeight);
        
        // Header section
        ctx.fillStyle = '#333333';
        ctx.font = 'bold 18px Arial';
        ctx.fillText('ECG Analysis Report', 20, 30);
        
        ctx.font = '12px Arial';
        ctx.fillStyle = '#666666';
        ctx.fillText(`Model: ${modelName}`, 20, 50);
        ctx.fillText(`Timestamp: ${timestamp}`, 20, 68);
        if (patientInfo) {
            ctx.fillText(`Patient: ${patientInfo}`, 300, 50);
        }
        
        // Graph area
        const graphX = 50;
        const graphY = 90;
        const graphWidth = exportWidth - 100;
        const graphHeight = exportHeight - 180;
        
        // Graph border
        ctx.strokeStyle = '#cccccc';
        ctx.lineWidth = 1;
        ctx.strokeRect(graphX, graphY, graphWidth, graphHeight);
        
        // Grid (medical ECG paper style - 1mm = 0.04s at 25mm/s)
        if (showGrid) {
            // Small grid (light)
            ctx.strokeStyle = '#ffcccc';
            ctx.lineWidth = 0.5;
            for (let x = graphX; x <= graphX + graphWidth; x += 20) {
                ctx.beginPath();
                ctx.moveTo(x, graphY);
                ctx.lineTo(x, graphY + graphHeight);
                ctx.stroke();
            }
            for (let y = graphY; y <= graphY + graphHeight; y += 20) {
                ctx.beginPath();
                ctx.moveTo(graphX, y);
                ctx.lineTo(graphX + graphWidth, y);
                ctx.stroke();
            }
            
            // Large grid (bold) - every 5 small squares
            ctx.strokeStyle = '#ff9999';
            ctx.lineWidth = 1;
            for (let x = graphX; x <= graphX + graphWidth; x += 100) {
                ctx.beginPath();
                ctx.moveTo(x, graphY);
                ctx.lineTo(x, graphY + graphHeight);
                ctx.stroke();
            }
            for (let y = graphY; y <= graphY + graphHeight; y += 100) {
                ctx.beginPath();
                ctx.moveTo(graphX, y);
                ctx.lineTo(graphX + graphWidth, y);
                ctx.stroke();
            }
        }
        
        // Draw ECG signal (black for medical print)
        if (samples && samples.length > 1) {
            const minVal = Math.min(...samples);
            const maxVal = Math.max(...samples);
            const range = maxVal - minVal || 1;
            
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            
            for (let i = 0; i < samples.length; i++) {
                const x = graphX + (i / samples.length) * graphWidth;
                const y = graphY + graphHeight - ((samples[i] - minVal) / range) * (graphHeight - 20) - 10;
                
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            
            ctx.stroke();
            
            // Draw R-peak markers
            const endSample = startSample + samples.length;
            annotations.forEach(ann => {
                if (ann.sample_index > startSample && ann.sample_index <= endSample) {
                    const bufferIdx = ann.sample_index - startSample;
                    
                    if (bufferIdx >= 0 && bufferIdx < samples.length) {
                        const x = graphX + (bufferIdx / samples.length) * graphWidth;
                        const y = graphY + graphHeight - ((samples[bufferIdx] - minVal) / range) * (graphHeight - 20) - 10;
                        
                        // Check for false detection
                        const classResult = classifications.find(c => c.r_peak === ann.sample_index);
                        const isFalse = classResult && !classResult.correct;
                        
                        // Marker color based on beat type
                        ctx.fillStyle = ann.beat_type === 'N' ? '#00aa00' : '#cc0000';
                        ctx.beginPath();
                        ctx.arc(x, y, 5, 0, Math.PI * 2);
                        ctx.fill();
                        
                        // False detection highlight
                        if (isFalse) {
                            ctx.strokeStyle = '#ff8800';
                            ctx.lineWidth = 2;
                            ctx.beginPath();
                            ctx.arc(x, y, 8, 0, Math.PI * 2);
                            ctx.stroke();
                        }
                        
                        // Label
                        ctx.fillStyle = '#333333';
                        ctx.font = '10px Arial';
                        ctx.fillText(ann.beat_type, x - 3, y - 12);
                    }
                }
            });
        }
        
        // Legend
        const legendY = exportHeight - 70;
        ctx.font = '11px Arial';
        ctx.fillStyle = '#333333';
        ctx.fillText('Legend:', 50, legendY);
        
        // Normal marker
        ctx.fillStyle = '#00aa00';
        ctx.beginPath();
        ctx.arc(110, legendY - 4, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#333333';
        ctx.fillText('Normal (N)', 120, legendY);
        
        // Abnormal marker
        ctx.fillStyle = '#cc0000';
        ctx.beginPath();
        ctx.arc(200, legendY - 4, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#333333';
        ctx.fillText('Abnormal', 210, legendY);
        
        // False detection marker
        ctx.strokeStyle = '#ff8800';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(290, legendY - 4, 6, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = '#333333';
        ctx.fillText('False Detection', 300, legendY);
        
        // Time scale
        const duration = samples.length / 360;  // Assuming 360 Hz sampling rate
        ctx.fillStyle = '#666666';
        ctx.font = '10px Arial';
        ctx.fillText(`Duration: ${duration.toFixed(2)}s | Sampling Rate: 360 Hz`, graphX, graphY + graphHeight + 20);
        
        // Footer
        ctx.fillStyle = '#999999';
        ctx.font = '9px Arial';
        ctx.fillText('Generated by ECG Real-Time Classification System', 50, exportHeight - 20);
        ctx.fillText(timestamp, exportWidth - 180, exportHeight - 20);
        
        // Return data URL
        const mimeType = format === 'jpeg' ? 'image/jpeg' : 'image/png';
        return exportCanvas.toDataURL(mimeType, 0.95);
    }
    
    /**
     * Download ECG as image
     * @param {Object} options - Export options
     * @param {string} filename - Filename without extension
     */
    downloadAsImage(options = {}, filename = 'ecg_report') {
        const dataUrl = this.exportAsImage(options);
        const format = options.format || 'png';
        
        const link = document.createElement('a');
        link.download = `${filename}.${format}`;
        link.href = dataUrl;
        link.click();
    }
    
    /**
     * Main render method
     * @param {Object} options - Rendering options
     */
    render(options) {
        const {
            samples,
            annotations,
            startSample,
            classifications,
            isLive
        } = options;
        
        // Store for export
        this.lastRenderOptions = options;
        
        this.clear();
        this.drawGrid();
        this.drawSignal(samples);
        this.drawRPeakMarkers(annotations, samples, startSample, classifications);
        
        if (!isLive) {
            this.drawHistoryIndicator();
        }
    }
}

// Export for use in other modules
window.ECGRenderer = ECGRenderer;
