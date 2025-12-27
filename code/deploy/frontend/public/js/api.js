/**
 * API Module
 * 
 * Handles all communication with the backend API.
 * Provides a clean interface for ECG data fetching and classification.
 */

class API {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
    }
    
    /**
     * Generic fetch wrapper with error handling
     * @param {string} endpoint - API endpoint
     * @param {Object} options - Fetch options
     * @returns {Promise<Object>} Response data
     */
    async request(endpoint, options = {}) {
        try {
            const response = await fetch(`${this.baseUrl}${endpoint}`, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`API error (${endpoint}):`, error);
            throw error;
        }
    }
    
    /**
     * Load ECG signal and annotations
     * @returns {Promise<Object>} Object with signal array and annotations
     */
    async loadData() {
        return this.request('/ecg/data');
    }
    
    /**
     * Get system status including model info
     * @returns {Promise<Object>} Status object
     */
    async getStatus() {
        return this.request('/ecg/status');
    }
    
    /**
     * Classify a beat at the given R-peak position
     * @param {number} rPeak - R-peak sample index
     * @param {string} beatType - Ground truth beat type
     * @returns {Promise<Object>} Classification result
     */
    async classify(rPeak, beatType) {
        return this.request('/ecg/infer', {
            method: 'POST',
            body: JSON.stringify({
                r_peak: rPeak,
                beat_type: beatType
            })
        });
    }
    
    /**
     * Control playback (start, stop, reset, set_speed)
     * @param {string} action - Control action
     * @param {Object} params - Additional parameters
     * @returns {Promise<Object>} Response
     */
    async control(action, params = {}) {
        return this.request('/ecg/control', {
            method: 'POST',
            body: JSON.stringify({
                action,
                ...params
            })
        });
    }
    
    /**
     * Get ECG window for streaming display
     * @param {number} windowSeconds - Window duration in seconds
     * @param {number} endSample - End sample index (optional)
     * @returns {Promise<Object>} Window data
     */
    async getStream(windowSeconds = 5.0, endSample = null) {
        let url = `/ecg/stream?window_seconds=${windowSeconds}`;
        if (endSample !== null) {
            url += `&end_sample=${endSample}`;
        }
        return this.request(url);
    }
    
    /**
     * Get annotations in a sample range
     * @param {number} start - Start sample index
     * @param {number} end - End sample index
     * @returns {Promise<Array>} Annotations
     */
    async getAnnotations(start, end) {
        return this.request(`/ecg/annotations?start=${start}&end=${end}`);
    }
    
    /**
     * Get classification results and false detections
     * @param {number} count - Number of results to return
     * @returns {Promise<Object>} Results object
     */
    async getResults(count = 50) {
        return this.request(`/ecg/results?count=${count}`);
    }
    
    /**
     * Check health of frontend and backend
     * @returns {Promise<Object>} Health status
     */
    async healthCheck() {
        return this.request('/health');
    }
}

// Create global API instance
window.api = new API();
