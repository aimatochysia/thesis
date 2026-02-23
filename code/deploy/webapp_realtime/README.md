# ECG Real-Input Classification Webapp

A web application for classifying ECG heartbeats from real CSV input data.
Unlike the simulated webapp, this version:

- Accepts **real ECG input** via CSV file upload (no pre-loaded demo data)
- Has **built-in R-peak detection** using the Pan-Tompkins algorithm (no pre-annotated R-peaks needed)
- Uses the same PyTorch ONNX models for beat classification


## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`


## Installation and Usage

```bash
cd code/deploy/webapp_realtime
pip install -r requirements.txt
python app.py
```

Then open your browser at `http://localhost:5001`.

### Command-line options

```
python app.py --model v3 --port 5001
```

- `--model` / `-m`: Model version to use. Options: `v2` (CNN), `v3` (LSTM, default), `v5` (Transformer), `v6` (Context-Aware CNN1D)
- `--port` / `-p`: Port number (default: 5001)


## How It Works

1. Upload a CSV file containing raw ECG data
2. Optionally specify the sampling rate (default: 360 Hz) and ECG column name
3. Select a model version from the dropdown
4. Click "Process ECG" to run the full pipeline:
   - Bandpass filtering (0.5-40 Hz, 2nd order Butterworth)
   - Pan-Tompkins R-peak detection
   - Adaptive beat segmentation around each detected R-peak
   - Beat resampling and normalization
   - ONNX model inference for each beat
5. View results: ECG strip with R-peak markers, beat classification list, beat snapshot panel
6. Click on individual beats to inspect their waveform and classification
7. Export the ECG strip as PNG or JPEG


## Input Data Format

The input must be a CSV file with at least one numeric column containing ECG amplitude values.

Example CSV structure:
```
'Elapsed time','MLII','V5'
0:00.000,995,1011
0:00.003,995,1011
...
```

The application auto-detects the first numeric column if no column name is specified.
Sample ECG files (MIT-BIH format) are available in the `../sample/` directory:
- `119.csv` - MIT-BIH record 119
- `100.csv` - MIT-BIH record 100

These files use the `MLII` column for the ECG signal at 360 Hz.


## Available Models

| Version | Architecture | Input Shape | Notes |
|---------|-------------|-------------|-------|
| v2 | CNN | (1, 1, 188) | Convolutional neural network |
| v3 | LSTM | (1, 188, 1) | Long short-term memory (default) |
| v5 | Transformer | (1, 188, 1) | Transformer-based model |
| v6 | Context-Aware CNN1D | (1, 7, 200) | Uses 7-beat rolling context window |

All models are stored as ONNX files in the `../sample/` directory alongside their corresponding scaler files.


## Project Structure

```
webapp_realtime/
  app.py              - Flask backend with Pan-Tompkins R-peak detection + ONNX inference
  requirements.txt    - Python dependencies
  README.md           - This file
  templates/
    index.html        - HTML template
  static/
    css/
      styles.css      - CSS styling (dark theme)
    js/
      app.js          - Frontend JavaScript (canvas rendering, upload, interaction)
```


## Differences from Simulated Webapp

| Feature | Simulated | Real-Input |
|---------|-----------|------------|
| Data source | Pre-loaded MIT-BIH record | User-uploaded CSV file |
| R-peak detection | Uses pre-annotated R-peak positions | Built-in Pan-Tompkins algorithm |
| Ground truth | Available from annotation files | Not available (no annotations) |
| Accuracy tracking | Yes (compares to ground truth) | No (real-world use case) |
| Real-time simulation | Yes (streaming playback) | No (batch processing) |
