/**
 * Beat Renderer Module
 * 
 * Handles the individual beat waveform canvas rendering
 * showing the extracted beat that is fed to the ONNX model.
 */

class BeatRenderer {
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
     * Clear canvas
     */
    clear() {
        this.ctx.fillStyle = '#0a0a1a';
        this.ctx.fillRect(0, 0, this.width, this.height);
    }
    
    /**
     * Draw grid lines for the beat canvas
     */
    drawGrid() {
        this.ctx.strokeStyle = 'rgba(0, 255, 136, 0.1)';
        this.ctx.lineWidth = 1;
        
        // Vertical lines
        for (let x = 0; x < this.width; x += 30) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.height);
            this.ctx.stroke();
        }
        
        // Horizontal lines
        for (let y = 0; y < this.height; y += 30) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.width, y);
            this.ctx.stroke();
        }
    }
    
    /**
     * Draw beat waveform
     * @param {Array} waveform - Beat sample values
     * @param {boolean} isAbnormal - Whether the beat is classified as abnormal
     */
    drawWaveform(waveform, isAbnormal = false) {
        if (!waveform || waveform.length < 2) return;
        
        const minVal = Math.min(...waveform);
        const maxVal = Math.max(...waveform);
        const range = maxVal - minVal || 1;
        
        this.ctx.strokeStyle = isAbnormal ? '#ff4757' : '#00ff88';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        
        for (let i = 0; i < waveform.length; i++) {
            const x = (i / waveform.length) * this.width;
            const y = this.height - ((waveform[i] - minVal) / range) * (this.height - 20) - 10;
            
            if (i === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        }
        
        this.ctx.stroke();
    }
    
    /**
     * Draw R-peak marker on the beat
     * @param {number} rPeakPos - Position of R-peak in the waveform
     * @param {Array} waveform - Beat sample values
     */
    drawRPeakMarker(rPeakPos, waveform) {
        if (!waveform || waveform.length < 2 || rPeakPos < 0) return;
        
        const minVal = Math.min(...waveform);
        const maxVal = Math.max(...waveform);
        const range = maxVal - minVal || 1;
        
        // Clamp position to valid array bounds to prevent rendering errors
        const safePos = Math.min(rPeakPos, waveform.length - 1);
        const x = (safePos / waveform.length) * this.width;
        const y = this.height - ((waveform[safePos] - minVal) / range) * (this.height - 20) - 10;
        
        // Draw marker
        this.ctx.fillStyle = '#ffcc00';
        this.ctx.beginPath();
        this.ctx.arc(x, y, 6, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Draw label
        this.ctx.font = '11px Arial';
        this.ctx.fillText('R-peak', x - 18, y - 10);
    }
    
    /**
     * Main render method
     * @param {Object} options - Rendering options
     */
    render(options) {
        const {
            waveform,
            rPeakPos,
            isAbnormal
        } = options;
        
        this.clear();
        this.drawGrid();
        this.drawWaveform(waveform, isAbnormal);
        this.drawRPeakMarker(rPeakPos, waveform);
    }
}

// Export for use in other modules
window.BeatRenderer = BeatRenderer;
