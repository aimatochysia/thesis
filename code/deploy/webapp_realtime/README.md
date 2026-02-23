# ECG Real-Input Classification Webapp

A web application for classifying ECG heartbeats from real CSV input data.
Unlike the simulated webapp, this version:

- Accepts **real ECG input** via CSV file upload (no pre-loaded demo data)
- Has **built-in R-peak detection** using the Pan-Tompkins algorithm (no pre-annotated R-peaks needed)
- Uses the Context-Aware CNN1D (v6) PyTorch ONNX model for beat classification


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
python app.py --port 5001
```

- `--port` / `-p`: Port number (default: 5001)


## How It Works

1. Upload a CSV file containing raw ECG data
2. Optionally specify the sampling rate (default: 360 Hz) and ECG column name
3. Click "Process ECG" to run the full pipeline:
   - Bandpass filtering (0.5-40 Hz, 2nd order Butterworth)
   - Pan-Tompkins R-peak detection
   - Beat extraction: 200 samples (90 before + 110 after R-peak)
   - Context-aware classification using 7-beat rolling window
   - ONNX model inference for each beat
4. View results: ECG strip with R-peak markers, beat classification list, beat snapshot panel
5. Click on individual beats to inspect their waveform and classification
6. Export the ECG strip as PNG or JPEG


## Input Data Format

The input must be a CSV file with at least one numeric column containing ECG amplitude values.

Example CSV structure:
```
'Elapsed time','MLII','V5'
0:00.000,995,1011
0:00.003,995,1011
...
```

The application auto-detects known ECG column names (MLII, ECG, V1) or uses the
first numeric column if no column name is specified.
Sample ECG files (MIT-BIH format) are available in the `../sample/` directory:
- `119.csv` - MIT-BIH record 119
- `100.csv` - MIT-BIH record 100

These files use the `MLII` column for the ECG signal at 360 Hz.


## Model

The application uses the Context-Aware CNN1D (v6) model:

| Architecture | Input Shape | Beat Length | Context Window |
|-------------|-------------|-------------|----------------|
| Context-Aware CNN1D | (1, 7, 200) | 200 samples | 7 beats (3 prev + center + 3 next) |

The model and scaler files (`context_ecg_model.onnx`, `context_ecg_scaler.pkl`) are
stored in the `../sample/` directory.


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
