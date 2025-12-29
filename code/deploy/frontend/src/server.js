/**
 * ECG Real-Time Classification Frontend Server
 * 
 * A Node.js/Express server that serves the frontend static files
 * and proxies API requests to the Python Flask backend.
 * 
 * Architecture:
 * - Frontend (Node.js): Serves HTML/CSS/JS static files
 * - Backend (Python/Flask): Handles ECG data processing and ONNX inference
 * 
 * Usage:
 *   npm start                  # Production mode
 *   npm run dev               # Development mode with auto-reload
 * 
 * Environment Variables:
 *   PORT: Frontend server port (default: 3000)
 *   BACKEND_URL: Python backend URL (default: http://localhost:5000)
 */

const express = require('express');
const path = require('path');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();
const PORT = process.env.PORT || 3000;
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:5000';

// Middleware for parsing JSON
app.use(express.json());

// Serve static files from public directory
app.use(express.static(path.join(__dirname, '../public')));

// Proxy API requests to Python backend
// All /ecg/* and /api/* routes are forwarded to the Flask backend
app.use('/ecg', createProxyMiddleware({
    target: BACKEND_URL,
    changeOrigin: true,
    logLevel: 'warn',
    onError: (err, req, res) => {
        console.error(`Proxy error: ${err.message}`);
        res.status(503).json({
            error: 'Backend service unavailable',
            message: `Make sure the Python backend is running on ${BACKEND_URL}`,
            details: err.message
        });
    }
}));

// Legacy API endpoints (backward compatibility)
app.use('/api', createProxyMiddleware({
    target: BACKEND_URL,
    changeOrigin: true,
    logLevel: 'warn',
    onError: (err, req, res) => {
        console.error(`Proxy error: ${err.message}`);
        res.status(503).json({
            error: 'Backend service unavailable',
            message: 'Make sure the Python backend is running on ' + BACKEND_URL
        });
    }
}));

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        service: 'frontend',
        timestamp: new Date().toISOString(),
        backend_url: BACKEND_URL
    });
});

// Catch-all route - serve index.html for SPA-style navigation
// Note: Rate limiting not implemented for thesis demo; production deployments
// should use nginx rate limiting or express-rate-limit package
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, '../public/index.html'));
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Server error:', err.stack);
    res.status(500).json({
        error: 'Internal server error',
        message: err.message
    });
});

// Start server
app.listen(PORT, () => {
    console.log('='.repeat(60));
    console.log('ECG Real-Time Classification Frontend');
    console.log('Node.js Express Server');
    console.log('='.repeat(60));
    console.log(`\nFrontend running at: http://localhost:${PORT}`);
    console.log(`Backend proxy to: ${BACKEND_URL}`);
    console.log('\nMake sure the Python backend is running:');
    console.log('  cd ../backend && python app.py --model v6');
    console.log('='.repeat(60));
});

module.exports = app;
