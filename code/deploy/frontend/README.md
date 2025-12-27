# ECG Real-Time Classification Frontend

A Node.js/Express frontend for real-time ECG arrhythmia classification.

## Architecture

This project uses a separated frontend/backend architecture:

```
code/deploy/
├── frontend/                    # Node.js frontend (this directory)
│   ├── package.json            # Node.js dependencies
│   ├── src/
│   │   └── server.js           # Express server with API proxy
│   └── public/                 # Static files
│       ├── index.html          # Main HTML page
│       ├── css/
│       │   └── styles.css      # All CSS styles
│       └── js/
│           ├── api.js          # Backend API client
│           ├── ecgRenderer.js  # ECG waveform canvas renderer
│           ├── beatRenderer.js # Beat snapshot canvas renderer
│           └── app.js          # Main application logic
│
├── backend/                     # Python Flask backend
│   ├── __init__.py
│   ├── ecg_streamer.py         # ECG signal loading and streaming
│   ├── inference_engine.py     # ONNX model inference
│   └── evaluation_layer.py     # Performance metrics
│
├── app.py                       # Flask API server
└── sample/                      # Model files and test data
    ├── *.onnx                   # ONNX models
    ├── *.pkl                    # Scalers
    ├── 119.csv                  # Test ECG signal
    └── 119annotations.txt       # Test annotations
```

## Quick Start

### Option 1: Standalone (Backend serves frontend)

```bash
cd code/deploy
python app.py --model v6 --port 5000
# Open http://localhost:5000
```

### Option 2: Separated (Node.js + Python)

```bash
# Terminal 1: Start Python backend
cd code/deploy
python app.py --model v6 --port 5000

# Terminal 2: Start Node.js frontend
cd code/deploy/frontend
npm install
npm start
# Open http://localhost:3000
```

## Development

```bash
cd frontend
npm install
npm run dev   # Auto-reload on file changes
```

## API Endpoints

The frontend proxies these endpoints to the Python backend:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ecg/stream` | GET | Returns ECG window data |
| `/ecg/infer` | POST | Classify beat at R-peak |
| `/ecg/status` | GET | System status and model info |
| `/ecg/control` | POST | Playback control |
| `/ecg/annotations` | GET | Get annotations in range |
| `/ecg/results` | GET | Classification history |
| `/ecg/data` | GET | Full signal and annotations |

## Frontend Features

- **Real-time ECG visualization** with grid overlay
- **Beat classification** with ONNX model inference
- **Speed control**: 0.1x, 0.5x, 1x, 5x, 10x presets
- **History navigation**: ±1s, ±5s buttons, Live mode
- **Beat snapshot**: Shows individual beat fed to model
- **Classification history**: Scrollable list with click navigation
- **False detection log**: Highlights incorrect predictions
- **Statistics**: Total beats, normal/abnormal counts, accuracy, BPM

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 3000 | Frontend server port |
| `BACKEND_URL` | http://localhost:5000 | Python backend URL |

## Production Deployment

### Using PM2

```bash
cd frontend
npm install
pm2 start src/server.js --name ecg-frontend
```

### Using Docker

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
EXPOSE 3000
CMD ["npm", "start"]
```

## Dependencies

- **express**: Web server framework
- **http-proxy-middleware**: API request proxying to Python backend

## License

MIT - Part of ECG Classification Thesis Project
