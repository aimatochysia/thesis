# CardioMonitor -- User Guide

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Starting the Application](#starting-the-application)
4. [Using CardioMonitor](#using-cardiomonitor)
5. [Understanding the Interface](#understanding-the-interface)
6. [Exporting Data](#exporting-data)
7. [Stopping the Application](#stopping-the-application)

---

## System Requirements

- **Docker Desktop** (recommended) -- available for macOS, Linux, and Windows
- Alternatively: Python 3.10 or higher with pip

---

## Installation

### Using Docker (recommended)

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) for your operating system.
2. Open a terminal and navigate to the application directory:
   ```
   cd code/deploy/webapp_simulated
   ```
3. Build and start the application with a single command:
   ```
   docker compose up --build
   ```

### Using Platform Helper Scripts

| Platform | Command            |
|----------|--------------------|
| macOS    | `./run_macos.sh`   |
| Linux    | `./run_linux.sh`   |
| Windows  | `run_windows.bat`  |

### Without Docker

1. Navigate to the application directory:
   ```
   cd code/deploy/webapp_simulated
   ```
2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python app.py
   ```
4. To use a different port:
   ```
   python app.py --port 8080
   ```

---

## Starting the Application

1. After installation, open a web browser.
2. Go to `http://localhost:5000` (or the custom port you specified).
3. The CardioMonitor interface will load with the ECG canvas and control panel.

---

## Using CardioMonitor

### Starting a Monitoring Session

1. Click the **Start** button to begin the ECG monitoring session.
2. The ECG signal will begin streaming across the canvas from left to right.
3. As each heartbeat is detected, the model will classify it as Normal or Abnormal.
4. The first few beats will show a "WAITING" status while the 7-beat context window fills up.

### Controlling Playback Speed

Use the speed buttons to adjust how fast the ECG signal plays back:

- **0.1x** -- Very slow, useful for detailed inspection of individual beats
- **0.5x** -- Half speed
- **1x** -- Real-time speed (default)
- **5x** -- Fast forward
- **10x** -- Maximum speed for quick review

### Pausing and Resetting

- Click **Stop** to pause the monitoring session at the current position.
- Click **Reset** to return to the beginning and clear all classification data.

### Navigating History

- **Drag** the ECG graph left or right to scroll through past data.
- Use the **<< -5s** and **< -1s** buttons to jump backward.
- Use the **> +1s** and **>> +5s** buttons to jump forward (available when viewing history).
- Click the **LIVE** button to return to the current position.
- A "VIEWING HISTORY" indicator appears when you are not at the live position.

### Reviewing a Specific Beat

- Click on any beat entry in the **Classification History** panel to navigate the ECG view to that beat's position.
- The **Beat Snapshot** panel shows the waveform that was fed to the model for the most recent classification, along with the annotation type, ground truth label, and model prediction.

---

## Understanding the Interface

### Statistics Bar

The statistics bar at the top shows real-time metrics:

| Metric            | Description                                                      |
|-------------------|------------------------------------------------------------------|
| Total Beats       | Total number of heartbeats classified so far                     |
| Normal            | Count of beats classified as normal                              |
| Abnormal          | Count of beats classified as abnormal                            |
| Accuracy          | Model accuracy compared to ground truth annotations              |
| BPM               | Current heart rate calculated from recent R-peak intervals       |
| False Predictions | Count of classifications that did not match the ground truth     |

### ECG Canvas

The main canvas displays the ECG signal waveform with color-coded R-peak markers:

- **Green dots** -- Normal beats
- **Red dots** -- Abnormal beats
- **Yellow circles** -- False detections (model prediction differs from ground truth)

### Beat Snapshot Panel

This panel shows exactly what the model sees for the latest classified beat:

- The waveform segment extracted around the R-peak
- A yellow marker indicating the R-peak position
- The annotation beat type, ground truth label, and model prediction

### Current Classification Panel

Shows the latest classification result with:

- The predicted label (Normal or Abnormal)
- A probability bar indicating the model's confidence toward abnormal

### Classification History Panel

A scrollable list of all past classification results, ordered from most recent to oldest. Each entry shows:

- The annotation beat type and model prediction
- The timestamp and abnormal probability
- Click any entry to navigate the ECG view to that beat

### False Detections Panel

Lists all beats where the model prediction did not match the ground truth annotation. Click any entry to navigate to it.

---

## Exporting Data

### Auto-Batch Export

CardioMonitor automatically saves ECG strip segments every 2 minutes during a session. Each batch contains a PNG image of the ECG waveform with annotated R-peak markers.

### Downloading Batches

1. Click the **Download Batches (ZIP)** button at any time.
2. A ZIP file will be generated containing all saved batches as PNG images.
3. The file will automatically download to your default download location.

The batch status indicator below the ECG canvas shows how many batches have been saved and their status.

---

## Stopping the Application

- **Docker Compose:** Press `Ctrl+C` in the terminal, or run `docker compose down` from the application directory.
- **Docker Run:** Press `Ctrl+C` in the terminal.
- **Python (direct):** Press `Ctrl+C` in the terminal.
