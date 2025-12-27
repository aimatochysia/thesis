/**
 * ECG Renderer Module
 * 
 * Handles the main ECG waveform canvas rendering with grid, 
 * signal trace, R-peak markers, and history navigation.
 */

class ECGRenderer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.setupCanvas();
        
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
