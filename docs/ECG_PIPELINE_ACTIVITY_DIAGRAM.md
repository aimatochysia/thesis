# ECG Classification Pipeline Activity Diagram

This document contains the activity diagram for the ECG Classification Pipeline, showing the complete flow from MIT-BIH data gathering to deployment with vertical swimlanes.

## Overview

The pipeline consists of 4 main partitions:
1. **Data Gathering** - Loading MIT-BIH Arrhythmia Database
2. **Data Modification** - Processing and preparing data (Note: Record 119 is skipped)
3. **AI Modeling** - Training the Context-Aware CNN1D model (v6)
4. **Result / Deployment** - Real-time classification and export

## Activity Diagram (Mermaid)

```mermaid
flowchart TB
    subgraph DG["📊 Data Gathering (MIT-BIH)"]
        A1[Load MIT-BIH Arrhythmia Database<br/>48 records, 360 Hz sampling]
        A2[Parse ECG signals - MLII lead]
        A3[Load annotation files<br/>R-peak locations & beat types]
        A1 --> A2 --> A3
    end

    subgraph DM["🔧 Data Modification"]
        B1[Apply Pan-Tompkins R-peak detection]
        B2[Extract beats: 200 samples<br/>90 pre-R + 110 post-R]
        B3["⚠️ SKIP Record 119<br/>(Reserved for live testing)<br/>47 records used for training"]
        B4[Create 7-beat context windows<br/>3 prev + 1 center + 3 next]
        B5[Binary labeling<br/>N=Normal, Others=Abnormal]
        B6[Record-wise split<br/>70% train / 15% val / 15% test<br/>NO patient leakage]
        B7[Normalize with StandardScaler<br/>Fitted on training data ONLY]
        B1 --> B2 --> B3 --> B4 --> B5 --> B6 --> B7
    end

    subgraph AI["🤖 AI Modeling"]
        C1[Build Context-Aware CNN1D v6<br/>Input: batch × 7 × 200]
        C2[Train with class weights<br/>Handle imbalanced classes]
        C3[Validate on held-out records]
        C4[Evaluate on test records]
        C5[Export to ONNX format<br/>context_ecg_model.onnx<br/>context_ecg_scaler.pkl]
        C1 --> C2 --> C3 --> C4 --> C5
    end

    subgraph RD["🚀 Result / Deployment"]
        D1[Deploy realtime_frontend.py<br/>Flask web application]
        D2["✅ Load Record 119 for testing<br/>(True unseen validation)"]
        D3[Real-time beat classification<br/>7-beat rolling buffer + ONNX]
        D4[Display results with accuracy]
        D5[Export complete ECG to PNG/JPEG<br/>From 0s to current position]
        D1 --> D2 --> D3 --> D4 --> D5
    end

    DG --> DM --> AI --> RD

    style B3 fill:#ffcccc,stroke:#cc0000,stroke-width:2px
    style D2 fill:#ccffcc,stroke:#00cc00,stroke-width:2px
```

## Key Points

### Record 119 Exclusion
- **Record 119 is completely skipped during data modification**
- It is reserved exclusively for live testing during deployment
- This ensures true unseen validation - the model has never seen this data

### Data Flow Summary
| Stage | Input | Output |
|-------|-------|--------|
| Data Gathering | MIT-BIH Database (48 records) | Raw ECG signals + annotations |
| Data Modification | Raw signals | Normalized context windows (47 records) |
| AI Modeling | Training/validation data | ONNX model + scaler |
| Deployment | ONNX model + Record 119 | Real-time classification results |

### Export Feature (Updated)
The export to PNG/JPEG now captures the **complete recording from 0 seconds to the current realtime position**, not just the visible graph window.

## PlantUML Version

For a more detailed swimlane diagram, see `ecg_pipeline_activity_diagram.puml` which can be rendered using PlantUML.

## Files Created
- `ecg_pipeline_activity_diagram.puml` - PlantUML source file
- `ECG_PIPELINE_ACTIVITY_DIAGRAM.md` - This documentation with Mermaid diagram
