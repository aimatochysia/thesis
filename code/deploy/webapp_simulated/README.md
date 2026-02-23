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

## Installation

```bash
cd code/deploy/webapp_simulated
pip install -r requirements.txt
```

## Usage

```bash
# Run the application
python app.py

# Run on a custom port
python app.py --port 8080
```

Then open your browser to `http://localhost:5000` (or the specified port).

## Model

The application uses the Context-Aware CNN1D (v6) model:

- Beat length: 200 samples (90 before + 110 after R-peak)
- Context window: 7 beats (3 previous + center + 3 next)
- Input shape: (1, 7, 200) after normalization
- First 3 beats show WAITING status until the context buffer is full

## Data

The application loads ECG data from MIT-BIH record 119, which was excluded from
model training to serve as true validation data. Data files are loaded from the
`../sample/` directory relative to this application.

## Project Structure

```
webapp_simulated/
  app.py              - Flask backend with model loading and classification API
  requirements.txt    - Python dependencies
  README.md           - This file
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
