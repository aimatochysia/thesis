# Evaluasi Sistem Klasifikasi ECG Real-Time

## BAB 4: Evaluasi Sistem Frontend Deployment

### 4.1 Ringkasan Sistem

Sistem klasifikasi ECG real-time ini terdiri dari tiga komponen utama yang bekerja secara terintegrasi:

1. **Dataset Modifier** - Pemrosesan dan modifikasi dataset MIT-BIH
2. **Model Training (v6)** - Pelatihan model CNN1D Context-Aware
3. **Frontend Deployment** - Antarmuka web untuk simulasi dan klasifikasi real-time

Dokumen ini fokus pada **evaluasi sistem deployment frontend** yang mensimulasikan lingkungan klinis nyata.

---

### 4.2 Arsitektur Frontend Deployment

#### 4.2.1 Aliran Data Sistem

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FRONTEND DEPLOYMENT (v6)                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  119.csv                119annotations.txt                              │
│     │                         │                                         │
│     ▼                         ▼                                         │
│  ┌──────────┐          ┌──────────────┐                                │
│  │ Sinyal   │          │ Lokasi R-peak│                                │
│  │ ECG MLII │          │ + Jenis beat │                                │
│  └────┬─────┘          └──────┬───────┘                                │
│       │                       │                                         │
│       ▼                       ▼                                         │
│  ┌─────────────────────────────────────┐                               │
│  │     Ekstraksi Beat (200 sampel)     │                               │
│  │     90 pre-R + 110 post-R           │                               │
│  └─────────────────┬───────────────────┘                               │
│                    ▼                                                    │
│  ┌─────────────────────────────────────┐                               │
│  │   Rolling Buffer (7 beat)           │                               │
│  │   [beat-3, beat-2, beat-1, CENTER,  │                               │
│  │    beat+1, beat+2, beat+3]          │                               │
│  └─────────────────┬───────────────────┘                               │
│                    ▼                                                    │
│  ┌─────────────────────────────────────┐                               │
│  │   Flatten (7×200 = 1400) →          │                               │
│  │   Normalize (StandardScaler) →      │                               │
│  │   Reshape (1, 7, 200)               │                               │
│  └─────────────────┬───────────────────┘                               │
│                    ▼                                                    │
│  ┌─────────────────────────────────────┐                               │
│  │   ONNX Inference                    │                               │
│  │   context_ecg_model.onnx            │                               │
│  └─────────────────┬───────────────────┘                               │
│                    ▼                                                    │
│  ┌─────────────────────────────────────┐                               │
│  │   Hasil: NORMAL / ABNORMAL          │                               │
│  │   + Perbandingan Ground Truth       │                               │
│  └─────────────────────────────────────┘                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 4.2.2 Komponen Teknis

| Komponen | Deskripsi | Detail |
|----------|-----------|--------|
| **Web Framework** | Flask (Python) | Server HTTP ringan untuk antarmuka web |
| **Model Runtime** | ONNX Runtime | Inferensi model cross-platform |
| **Visualisasi** | HTML5 Canvas + JavaScript | Rendering sinyal ECG real-time |
| **Data Source** | MIT-BIH Record 119 | Data validasi yang tidak pernah dilihat model saat training |

---

### 4.3 Mengapa Menggunakan Record 119?

Record 119 dipilih sebagai data pengujian karena beberapa alasan penting:

1. **Tidak Termasuk dalam Training** - Record 119 dikecualikan dari proses pelatihan model v6
2. **Validasi Sejati** - Mewakili data pasien yang benar-benar baru
3. **Tidak Ada Kebocoran Data** - Menjamin estimasi performa yang realistis
4. **Distribusi Berbeda** - Menguji kemampuan generalisasi model

```python
# Konfigurasi: Record 119 selalu digunakan untuk semua model
use_record_119 = True  # Default untuk v2, v3, v5, dan v6
```

---

### 4.4 Pipeline Preprocessing (Sama Persis dengan Training)

#### 4.4.1 Ekstraksi Beat

Setiap beat diekstraksi dengan 200 sampel yang berpusat pada R-peak:

```python
def extract_beat_v6(signal, r_peak_idx):
    """Ekstraksi beat 200 sampel berpusat pada R-peak.
    
    Sesuai training: PRE_R_SAMPLES=90, POST_R_SAMPLES=110
    """
    start_idx = r_peak_idx - 90   # 90 sampel sebelum R-peak
    end_idx = r_peak_idx + 110    # 110 sampel setelah R-peak
    
    # Penanganan kasus tepi dengan zero padding
    if start_idx < 0:
        beat = np.zeros(200)
        available = signal[:end_idx]
        beat[-len(available):] = available
    elif end_idx > len(signal):
        beat = np.zeros(200)
        available = signal[start_idx:]
        beat[:len(available)] = available
    else:
        beat = signal[start_idx:end_idx]
    
    return beat  # Shape: (200,)
```

**Mengapa 200 sampel (90 + 110)?**

| Komponen | Jumlah Sampel | Durasi (360Hz) | Fungsi |
|----------|---------------|----------------|--------|
| Pre-R | 90 | ~250ms | Menangkap gelombang P |
| Post-R | 110 | ~306ms | Menangkap ST-segment dan gelombang T |
| **Total** | **200** | **~556ms** | Cukup untuk kompleks PQRST lengkap |

#### 4.4.2 Rolling Buffer 7 Beat

```python
# Buffer global untuk model v6 context-aware
beat_buffer = []  # List dari (beat_waveform, beat_type) tuples

def process_beat(beat, beat_type):
    global beat_buffer
    
    # Tambahkan beat baru ke buffer
    beat_buffer.append((beat, beat_type))
    
    # Simpan hanya 7 beat terakhir
    if len(beat_buffer) > 7:
        beat_buffer = beat_buffer[-7:]
    
    # Butuh 7 beat untuk inferensi
    if len(beat_buffer) < 7:
        return {"status": "MENUNGGU", "buffer_size": len(beat_buffer)}
    
    # Siap untuk inferensi
    return run_inference()
```

**Mengapa 7 beat (3+1+3)?**

Window konteks 7 beat dipilih untuk menangkap pola aritmia multi-beat:

- **3 beat sebelumnya**: Menangkap pola "pendekatan"
- **1 beat tengah**: Target klasifikasi
- **3 beat selanjutnya**: Memberikan konteks konfirmasi

Pola aritmia yang dapat dideteksi:
- **Bigeminy/Trigeminy**: PVC bergantian dengan beat normal
- **Variabilitas R-R**: Pola irregular pada atrial fibrillation
- **Pause Kompensatori**: Pause setelah PVC
- **AV Block**: Perubahan sistematis pada interval PR

#### 4.4.3 Normalisasi (Kritis - Harus Sama dengan Training)

```python
def prepare_input():
    # Stack 7 beat: (7, 200)
    context_beats = np.stack([b for b, _ in beat_buffer], axis=0)
    
    # Flatten untuk scaler: (1, 1400)
    flat_size = 7 * 200  # = 1400
    context_flat = context_beats.reshape(1, flat_size)
    
    # Normalisasi menggunakan scaler TRAINING
    # Scaler ini di-fit hanya pada X_train (tidak ada kebocoran data)
    normalized = scaler.transform(context_flat)
    
    # Reshape untuk input model: (1, 7, 200)
    model_input = normalized.reshape(1, 7, 200).astype(np.float32)
    
    return model_input
```

**Prinsip Normalisasi:**

1. **Flatten order sama**: Row-major (C-order) reshaping sesuai training
2. **Scaler sama**: Dimuat dari `context_ecg_scaler.pkl`
3. **Reshape sama**: Shape akhir (1, 7, 200) sesuai input model

---

### 4.5 Proses Inferensi ONNX

```python
def run_inference():
    model_input = prepare_input()
    
    # Inferensi sesi ONNX
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    output = model.run([output_name], {input_name: model_input})[0]
    
    # Output adalah logits: [logit_normal, logit_abnormal]
    # Aplikasikan softmax untuk mendapatkan probabilitas
    exp_output = np.exp(output - np.max(output))
    proba = exp_output / np.sum(exp_output)
    
    prob_abnormal = proba[0, 1]
    predicted_class = 1 if prob_abnormal >= 0.5 else 0
    
    return {
        "predicted": "ABNORMAL" if predicted_class == 1 else "NORMAL",
        "probability": prob_abnormal,
        "ground_truth": get_ground_truth(beat_buffer[3][1])  # Beat tengah
    }
```

---

### 4.6 Perbandingan Ground Truth

```python
def get_ground_truth(beat_type):
    """N = Normal, lainnya = Abnormal"""
    return "NORMAL" if beat_type == 'N' else "ABNORMAL"
```

Beat tengah (indeks 3 dalam window 7 beat) digunakan untuk ground truth karena:
- Model memprediksi klasifikasi beat tengah
- Beat sekitarnya hanya memberikan konteks

---

### 4.7 Fitur Antarmuka Pengguna

#### 4.7.1 Kontrol Kecepatan

```python
# Preset kecepatan (pengali dari real-time)
speeds = [0.1, 0.5, 1, 5, 10]
# 1x = 360 sampel/detik (real-time MIT-BIH)
# 10x = 3600 sampel/detik (playback 10x lebih cepat)
```

#### 4.7.2 Perhitungan BPM

```python
def calculateBPM(currentBeatSample):
    beatTimes.append(currentBeatSample)
    
    # Simpan 10 beat terakhir untuk smoothing
    if len(beatTimes) > 10:
        beatTimes.pop(0)
    
    # Rata-rata interval (filter outlier: rentang 30-200 BPM)
    intervals = []
    for i in range(1, len(beatTimes)):
        interval = (beatTimes[i] - beatTimes[i-1]) / 360  # detik
        if 0.3 < interval < 2.0:  # 30-200 BPM
            intervals.append(interval)
    
    if intervals:
        avg_interval = sum(intervals) / len(intervals)
        return round(60 / avg_interval)
    return None
```

**Mengapa filter ke 30-200 BPM?**
- Rentang yang masuk akal secara fisiologis
- Memfilter beat yang terlewat atau deteksi ganda
- Menghaluskan tampilan agar tidak berfluktuasi

#### 4.7.3 Navigasi Riwayat

```python
# Kontrol navigasi
scrollHistory(-5)  # Mundur 5 detik
scrollHistory(-1)  # Mundur 1 detik
goToLive()         # Kembali ke tampilan live
scrollHistory(+1)  # Maju 1 detik (jika melihat riwayat)
scrollHistory(+5)  # Maju 5 detik
```

#### 4.7.4 Log Deteksi Salah

```python
if result.ground_truth != result.predicted:
    falseDetections.append({
        "time": result.r_peak / 360,  # Konversi ke detik
        "expected": result.ground_truth,
        "got": result.predicted,
        "r_peak": result.r_peak
    })
    updateFalseDetectionList()  # Update UI
```

Deteksi salah yang dapat diklik memungkinkan navigasi ke waktu spesifik dalam sinyal.

---

### 4.8 Fitur Stabilitas Grafik

#### 4.8.1 Tinggi Grafik Stabil

Tinggi grafik ECG diimplementasikan dengan prinsip "hanya bisa membesar, tidak bisa mengecil":

```javascript
let maxGraphHeight = MIN_GRAPH_HEIGHT;  // Tinggi minimum awal

function updateGraphHeight(newHeight) {
    // Hanya update jika tinggi baru lebih besar
    if (newHeight > maxGraphHeight) {
        maxGraphHeight = newHeight;
        canvas.style.height = maxGraphHeight + 'px';
    }
}
```

#### 4.8.2 Skala Y-Axis Stabil

Skala vertikal juga diimplementasikan dengan prinsip yang sama:

```javascript
let globalMinVal = Infinity;
let globalMaxVal = -Infinity;

function updateYAxisScale(bufferMin, bufferMax) {
    // Hanya perluas rentang, tidak pernah mengecilkan
    if (bufferMin < globalMinVal) globalMinVal = bufferMin;
    if (bufferMax > globalMaxVal) globalMaxVal = bufferMax;
    
    // Gunakan nilai global untuk scaling
    return { min: globalMinVal, max: globalMaxVal };
}
```

**Manfaat:**
- Beat dengan amplitudo tinggi di masa lalu tetap menjadi referensi
- Beat dengan amplitudo rendah tidak menyebabkan "zoom in" yang membingungkan
- Konsistensi visual yang lebih baik untuk evaluasi klinis

---

### 4.9 Sistem Ekspor Otomatis

#### 4.9.1 Auto-Batch Export

Sistem ekspor dirancang untuk kemudahan pengguna (dokter/perawat):

```javascript
const BATCH_INTERVAL_MS = 2 * 60 * 1000;  // 2 menit
const savedBatches = [];  // Array untuk menyimpan batch PNG

// Auto-save setiap 2 menit
setInterval(function() {
    if (isRecording && pendingData.length > MIN_BATCH_SAMPLES) {
        saveBatchToMemory();
    }
}, BATCH_CHECK_INTERVAL_MS);
```

#### 4.9.2 Download sebagai ZIP

Semua batch dibundel dalam satu file ZIP untuk kemudahan:

```javascript
async function downloadAllBatches() {
    const zip = new JSZip();
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    
    // Tambahkan semua batch ke ZIP
    for (let i = 0; i < savedBatches.length; i++) {
        const batch = savedBatches[i];
        const filename = `ecg_batch_${i+1}_${batch.startTime}s-${batch.endTime}s.png`;
        
        // Konversi data URL ke blob
        const data = batch.dataUrl.split(',')[1];
        zip.file(filename, data, {base64: true});
    }
    
    // Generate dan download ZIP
    const content = await zip.generateAsync({type: "blob"});
    saveAs(content, `ecg_recording_${timestamp}.zip`);
}
```

**Alur Kerja untuk Dokter/Perawat:**
1. Tekan **Start** → rekaman dimulai
2. (Batch otomatis tersimpan setiap 2 menit di memori)
3. Tekan **Stop** → batch terakhir tersimpan
4. Klik **"Download Batches (ZIP)"** → satu file ZIP terdownload
5. Selesai!

---

### 4.10 Performa yang Diharapkan

Berdasarkan evaluasi pada data test set (yang termasuk pasien yang tidak pernah dilihat seperti record 119):

| Metrik | Nilai yang Diharapkan |
|--------|----------------------|
| Akurasi | ~69% (data validasi sejati) |
| Recall (Abnormal) | ~55% |
| AUC-ROC | ~0.80 |
| Akurasi (test set distribusi serupa) | ~94-98% |

**Mengapa performa pada record 119 lebih rendah dari test set?**
- Model v6 menggunakan split record-wise (tidak ada kebocoran pasien)
- Record 119 memiliki karakteristik yang mungkin berbeda dari data training
- Metrik v6 lebih realistis untuk deployment dunia nyata

---

### 4.11 Troubleshooting

#### 4.11.1 Status "MENUNGGU"

Jika tampilan menunjukkan "MENUNGGU" untuk 3 beat pertama:
- Ini adalah perilaku yang diharapkan
- Buffer 7 beat perlu terisi sebelum inferensi
- Tunggu sampai 7 R-peak terdeteksi

#### 4.11.2 Error Loading Model

```bash
# Periksa file model ada
ls sample/context_ecg_model.onnx
ls sample/context_ecg_scaler.pkl
```

#### 4.11.3 Ketidakcocokan Prediksi

Jika prediksi tidak cocok dengan ground truth:
1. **Diharapkan**: ~55% recall pada beat abnormal
2. **Distribusi Record 119**: Periksa rasio normal/abnormal
3. **Distribution Shift**: Record 119 mungkin memiliki karakteristik berbeda dari data training

---

### 4.12 Kesimpulan Evaluasi Sistem

Sistem deployment frontend berhasil mengimplementasikan:

1. **Simulasi Real-Time** - Mensimulasikan pemantauan ECG live
2. **Model Context-Aware** - Menggunakan pola temporal antar beat
3. **Tidak Ada Kebocoran Data** - Record 119 tidak pernah digunakan dalam training
4. **Preprocessing yang Tepat** - Sesuai persis dengan training
5. **Utilitas Klinis** - Perbandingan ground truth dan logging deteksi salah
6. **Kemudahan Penggunaan** - Ekspor otomatis, stabilitas grafik, navigasi intuitif

Sistem ini siap untuk demonstrasi dan validasi lebih lanjut dalam lingkungan klinis yang terkontrol.

---

## Referensi

[1] Moody, G. B., & Mark, R. G. (2001). *The impact of the MIT-BIH Arrhythmia Database.* IEEE Engineering in Medicine and Biology Magazine, 20(3), 45-50.

[2] Goldberger, A. L., et al. (2000). *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation, 101(23), e215-e220.

[3] Pan, J., & Tompkins, W. J. (1985). *A real-time QRS detection algorithm.* IEEE Transactions on Biomedical Engineering, BME-32(3), 230-236.

[4] Hannun, A. Y., et al. (2019). *Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network.* Nature Medicine, 25(1), 65-69.

[5] Acharya, U. R., et al. (2017). *A deep convolutional neural network model to classify heartbeats.* Computers in Biology and Medicine, 89, 389-396.

[6] de Chazal, P., et al. (2004). *Automatic classification of heartbeats using ECG morphology and heartbeat interval features.* IEEE Transactions on Biomedical Engineering, 51(7), 1196-1206.

[7] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press.

[8] Yildirim, O., et al. (2018). *Arrhythmia detection using deep convolutional neural network with long duration ECG signals.* Computers in Biology and Medicine, 102, 411-420.
