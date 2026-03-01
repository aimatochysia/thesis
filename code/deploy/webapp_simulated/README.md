# ECG Real-Time Classification Web Application (Simulated)

A Flask-based web application for real-time ECG heartbeat classification using
the Context-Aware CNN1D (v6) PyTorch ONNX model. This app simulates real-time
ECG monitoring by playing back recorded MIT-BIH data through a browser-based
interface with live classification.

## Features

- Real-time ECG signal visualization with interactive canvas
- Beat-by-beat classification using Context-Aware CNN1D (v6) ONNX inference
- 7-beat rolling context window for improved accuracy
- Adjustable playback speed (0.1x to 10x)
- History navigation with drag-to-scroll
- Auto-batch export system with ZIP download
- Beat snapshot panel showing model input
- False detection tracking and statistics
- BPM calculation from R-peak intervals

## Quick Start with Docker (macOS / Linux / Windows)

Docker is the recommended way to run this app on any platform.

**Prerequisites:** Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) for your OS.

### Option 1 -- docker compose (simplest)

```bash
cd code/deploy/webapp_simulated
docker compose up --build
```

Open `http://localhost:5000` in your browser.

### Option 2 -- platform helper scripts

| Platform | Command |
|----------|---------|
| macOS    | `./run_macos.sh`   |
| Linux    | `./run_linux.sh`   |
| Windows  | `run_windows.bat`  |

Each script builds the Docker image and starts the container on port 5000.

### Option 3 -- manual docker commands

```bash
docker build -t ecg-flask-app .
docker run -p 5000:5000 ecg-flask-app
```

### Stopping

- `docker compose up`: press Ctrl+C or run `docker compose down`
- `docker run`: press Ctrl+C

## Running without Docker

```bash
cd code/deploy/webapp_simulated
pip install -r requirements.txt
python app.py              # development server on port 5000
python app.py --port 8080  # custom port
```

## Model

The application uses the Context-Aware CNN1D (v6) model:

- Beat length: 200 samples (90 before + 110 after R-peak)
- Context window: 7 beats (3 previous + center + 3 next)
- Input shape: (1, 7, 200) after normalization
- First 3 beats show WAITING status until the context buffer is full

## Data

The application loads ECG data from MIT-BIH record 119, which was excluded from
model training to serve as true validation data. Data files are loaded from the
`model/` directory.

## Project Structure

```
webapp_simulated/
  app.py              - Flask backend with model loading and classification API
  Dockerfile          - Container definition (Python 3.12 + gunicorn)
  docker-compose.yml  - One-command container startup
  requirements.txt    - Python dependencies
  .dockerignore       - Files excluded from the Docker image
  run_linux.sh        - Docker helper script for Linux
  run_macos.sh        - Docker helper script for macOS
  run_windows.bat     - Docker helper script for Windows
  README.md           - This file
  model/              - ONNX model, scaler, and ECG data
  templates/
    index.html        - Main HTML template
  static/
    css/
      styles.css      - Application styles (dark theme)
    js/
      app.js          - Frontend JavaScript (ECG rendering, controls, export)
```

## API Endpoints

- `GET /` - Serves the main HTML page
- `GET /api/data` - Returns the full ECG signal and annotations
- `POST /api/classify` - Classifies a beat at a given R-peak index
- `GET /api/model_info` - Returns current model metadata
- `POST /api/reset` - Resets the backend beat buffer
