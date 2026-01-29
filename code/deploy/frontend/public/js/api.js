class API {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
    }
    
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
    
    async loadData() {
        return this.request('/ecg/data');
    }
    
    async getStatus() {
        return this.request('/ecg/status');
    }
    
    async classify(rPeak, beatType) {
        return this.request('/ecg/infer', {
            method: 'POST',
            body: JSON.stringify({
                r_peak: rPeak,
                beat_type: beatType
            })
        });
    }
    
    async control(action, params = {}) {
        return this.request('/ecg/control', {
            method: 'POST',
            body: JSON.stringify({
                action,
                ...params
            })
        });
    }
    
    async getStream(windowSeconds = 5.0, endSample = null) {
        let url = `/ecg/stream?window_seconds=${windowSeconds}`;
        if (endSample !== null) {
            url += `&end_sample=${endSample}`;
        }
        return this.request(url);
    }
    
    async getAnnotations(start, end) {
        return this.request(`/ecg/annotations?start=${start}&end=${end}`);
    }
    
    async getResults(count = 50) {
        return this.request(`/ecg/results?count=${count}`);
    }
    
    async healthCheck() {
        return this.request('/health');
    }
}

window.api = new API();
