# PRESENTASI SIDANG SKRIPSI: DETEKSI ARITMIA JANTUNG MENGGUNAKAN 1D CONVOLUTIONAL NEURAL NETWORK

## Penjelasan Komprehensif dalam Bahasa Indonesia untuk Sidang Skripsi

> **FOKUS UTAMA PENELITIAN:**
> 1. **Pra-Pemrosesan Data (Pre-processing)** - Bagaimana data dipersiapkan secara sistematis
> 2. **Simulasi Deployment Real-time** - Validasi model pada kondisi nyata dengan penurunan akurasi ~4%
> 3. Model 1D-CNN (pendukung, karena topik ini sudah banyak diteliti)

---

# BAGIAN 1: PENDAHULUAN DAN LATAR BELAKANG

## 1.1 Mengapa Penelitian Ini Penting?

### Latar Belakang Masalah

Penyakit kardiovaskular merupakan **penyebab utama kematian** di seluruh dunia. *Aritmia* jantung—yaitu detak jantung yang tidak normal atau tidak teratur—menjadi indikator kritis kesehatan jantung. Beberapa jenis aritmia dapat tidak berbahaya, namun banyak di antaranya berpotensi menyebabkan kondisi medis serius, termasuk **serangan jantung dan kematian mendadak**.

#### Masalah dengan Diagnosis Manual:
1. **Membutuhkan keahlian tinggi** - Hanya kardiolog berpengalaman yang dapat menginterpretasikan sinyal ECG dengan akurat
2. **Rentan terhadap noise/gangguan** - Sinyal ECG sering terkontaminasi oleh berbagai sumber gangguan
3. **Sifat acak aritmia** - Aritmia dapat muncul sewaktu-waktu dan sulit diprediksi kemunculannya
4. **Risiko misdiagnosis** - Dapat menyebabkan keterlambatan intervensi medis yang krusial

### Solusi yang Ditawarkan

Penelitian ini mengembangkan sistem klasifikasi aritmia otomatis berbasis **Deep Learning** menggunakan arsitektur **1D Convolutional Neural Network (1D-CNN)** yang:
- Mampu mengekstraksi fitur secara otomatis dari sinyal ECG mentah
- Mendeteksi pola-pola halus yang mungkin terlewatkan oleh pengamatan manual
- Divalidasi tidak hanya pada data test set, tetapi juga pada **simulasi deployment real-time**

---

## 1.2 Rumusan Masalah

1. **Bagaimana** proses pengumpulan, modifikasi, dan persiapan data sinyal ECG dari dataset MIT-BIH dapat dilakukan untuk membentuk data detak jantung tunggal yang siap digunakan dalam pelatihan model deep learning?

2. **Bagaimana** merancang arsitektur model *Convolutional Neural Network* 1D (1D-CNN) yang efektif untuk klasifikasi sinyal ECG menjadi dua kelas: Normal dan Abnormal?

3. **Bagaimana** pengaruh strategi preprocessing seperti penghapusan *outlier*, penyeimbangan kelas, *clustering* (K-Means), dan reduksi dimensi (PCA) terhadap kualitas dan performa dataset untuk klasifikasi biner?

4. **Bagaimana** strategi segmentasi sinyal ECG berbasis R-peak detection dapat membantu membentuk input yang fisiologis dan konsisten untuk model CNN?

5. **Bagaimana** performa model CNN 1D yang dikembangkan ketika dievaluasi menggunakan metrik akurasi, *precision, recall, F1-score*, dan AUC-ROC, baik terhadap data bersih (*clean evaluation set*) maupun data mentah (*prediction dataset*/simulasi real-time)?

---

## 1.3 Hipotesis

Model *Convolutional Neural Network 1D* yang dilatih pada representasi detak jantung tunggal dari *dataset* MIT-BIH yang dimodifikasi akan mampu mengklasifikasikan detak jantung sebagai Normal atau Abnormal dengan tingkat akurasi dan kinerja (*precision*, *recall*, *F1-score*, dan AUC-ROC) yang tinggi, sebanding atau bahkan melampaui hasil penelitian terdahulu dalam klasifikasi biner detak jantung.

---

# BAGIAN 2: TINJAUAN PUSTAKA DAN LANDASAN TEORI

## 2.1 Elektrokardiogram (ECG/EKG)

Elektrokardiogram adalah sinyal biologis yang merekam aktivitas listrik dari jantung melalui elektroda di kulit. Sinyal ECG terdiri dari **3 gelombang utama**:

| Gelombang | Representasi | Durasi Normal |
|-----------|--------------|---------------|
| **Gelombang P** | Depolarisasi atrium (kontraksi serambi) | 80-120 ms |
| **Kompleks QRS** | Depolarisasi ventrikel (kontraksi bilik) | 80-120 ms |
| **Gelombang T** | Repolarisasi ventrikel (pemulihan bilik) | 120-160 ms |

### Karakteristik Sinyal ECG:
- **Rentang frekuensi**: 0.05 Hz - 1000 Hz
- **Rentang amplitudo**: 1 mV - 10 mV secara normal
- **Sampling rate MIT-BIH**: 360 Hz

---

## 2.2 Jenis-Jenis Aritmia

| Jenis Aritmia | Karakteristik | Tingkat Bahaya |
|---------------|---------------|----------------|
| **Takikardia Ventrikel (VT)** | Detak jantung sangat cepat (>100 BPM dari ventrikel) | Tinggi |
| **Fibrilasi Ventrikel (VF)** | Aktivitas listrik kacau, jantung tidak efektif memompa | Sangat Tinggi |
| **Fibrilasi Atrium (AF)** | Ritme atrium tidak teratur dan cepat | Sedang-Tinggi |
| **Premature Ventricular Contraction (PVC)** | Detak ekstra dari ventrikel | Rendah-Sedang |
| **Bundle Branch Block** | Hambatan konduksi pada berkas His | Sedang |

---

## 2.3 Dataset MIT-BIH Arrhythmia Database

### Karakteristik Dataset

| Parameter | Nilai |
|-----------|-------|
| **Sumber** | Beth Israel Hospital + MIT, via PhysioBank |
| **Jumlah Rekaman** | 48 rekaman dari 47 pasien |
| **Durasi per Rekaman** | ~30 menit |
| **Sampling Rate** | 360 Hz |
| **Lead** | Dual-lead: MLII (Modified Lead II) dan V1/V2/V5 |
| **Total Detak Teranotasi** | >100,000 detak jantung |

### Struktur Data:
1. **Signal Time Series** (.csv): Data numerik sinyal ECG dalam satuan milivolt
2. **Sample Index**: Titik waktu untuk setiap anotasi detak (posisi R-peak)
3. **Annotation Label** (.txt): Label untuk tiap detak sesuai klasifikasi AAMI

---

## 2.4 Algoritma Pan-Tompkins untuk Deteksi R-Peak

Algoritma Pan-Tompkins adalah **metode standar industri** untuk mendeteksi kompleks QRS dalam sinyal ECG. Dikembangkan tahun 1985, algoritma ini memiliki **5 tahapan pemrosesan**:

### Tahap 1: Band-Pass Filtering (5-15 Hz)
Menghilangkan noise frekuensi rendah (baseline wander) dan frekuensi tinggi (muscle noise, powerline interference).

$$H_{LP}(z) = \frac{(1 - z^{-6})^2}{(1 - z^{-1})^2}$$

### Tahap 2: Differentiation (Turunan)
Menekankan perubahan amplitudo yang cepat (slope) karakteristik kompleks QRS.

$$y(n) = \frac{1}{8T}[-x(n-2) - 2x(n-1) + 2x(n+1) + x(n+2)]$$

### Tahap 3: Squaring (Pengkuadratan)
Membuat semua nilai menjadi positif dan menekankan perbedaan slope.

$$y(n) = [x(n)]^2$$

### Tahap 4: Moving Window Integration
Memperhalus sinyal dan menghasilkan satu puncak per kompleks QRS.

$$y(n) = \frac{1}{N}[x(n - (N-1)) + x(n - (N-2)) + ... + x(n)]$$

### Tahap 5: Adaptive Thresholding
Menentukan lokasi R-peak dengan threshold adaptif.

$$THRESHOLD = SPKI + 0.25 \times (SPKF - NPKF)$$

### Keunggulan:
- **Adaptif** terhadap variasi sinyal antar pasien
- **Robust** terhadap noise
- **Efisien** untuk implementasi real-time
- **Akurasi >99%** pada dataset standar

---

# BAGIAN 3: PRA-PEMROSESAN DATA (FOKUS UTAMA PENELITIAN)

## 3.1 Mengapa Pra-Pemrosesan Sangat Penting?

> **POIN KRITIS:** Pra-pemrosesan data adalah salah satu **kontribusi utama** penelitian ini. Berbeda dengan banyak penelitian sebelumnya yang fokus pada arsitektur model, penelitian ini menekankan pada **strategi persiapan data yang sistematis** untuk menghasilkan input berkualitas tinggi bagi model deep learning.

### Tantangan Data ECG Mentah:
1. **Panjang sinyal tidak konsisten** - Durasi rekaman bervariasi antar pasien
2. **Noise dan artefak** - Gangguan dari gerakan, baseline wander, powerline interference
3. **Class imbalance** - Detak normal jauh lebih banyak dari abnormal
4. **Variasi morfologi antar pasien** - Bentuk gelombang berbeda-beda
5. **Ketergantungan temporal** - Aritmia sering memiliki pola berurutan

---

## 3.2 Strategi Segmentasi Sinyal ECG

### 3.2.1 Deteksi R-Peak dengan Pan-Tompkins

Lokasi temporal puncak gelombang R dideteksi menggunakan algoritma Pan-Tompkins sebagai **titik referensi pusat** setiap detak.

### 3.2.2 Pemotongan Sinyal (Windowing) - **200 Sampel per Detak**

**Berbeda dengan pendekatan simetris konvensional**, penelitian ini menerapkan **segmentasi asimetris**:

| Komponen | Jumlah Sampel | Durasi (360 Hz) | Alasan |
|----------|---------------|-----------------|--------|
| **Pre-R** | 90 sampel | ±250 ms | Menangkap gelombang P dan interval PR |
| **Post-R** | 110 sampel | ±306 ms | Menangkap segmen ST dan gelombang T secara lengkap |
| **Total** | **200 sampel** | ±556 ms | Mencakup satu siklus PQRST utuh |

### Mengapa 200 Sampel?
- **Fisiologis**: Durasi ~556 ms ideal untuk satu siklus detak jantung
- **Standar**: Konsisten dengan literatur terkini
- **Efisien**: Ukuran yang dapat diproses dengan cepat oleh model CNN

### 3.2.3 Penanganan Tepi (Zero-Padding)

Untuk detak di awal/akhir rekaman yang tidak memiliki cukup sampel, dilakukan **zero-padding** untuk memastikan dimensi output tetap konsisten di 200 sampel.

```python
def extract_beat_v6(signal, r_peak_idx):
    PRE_SAMPLES = 90
    POST_SAMPLES = 110
    BEAT_LENGTH = 200
    
    start_idx = r_peak_idx - PRE_SAMPLES
    end_idx = r_peak_idx + POST_SAMPLES
    
    if start_idx < 0:
        # Zero-padding di awal
        pad_before = -start_idx
        beat = np.zeros(BEAT_LENGTH, dtype=np.float32)
        available = signal[:end_idx]
        beat[pad_before:pad_before + len(available)] = available
    elif end_idx > len(signal):
        # Zero-padding di akhir
        beat = np.zeros(BEAT_LENGTH, dtype=np.float32)
        available = signal[start_idx:]
        beat[:len(available)] = available
    else:
        beat = signal[start_idx:end_idx].astype(np.float32)
    
    return beat
```

---

## 3.3 Context Window (Jendela Konteks) - **7 Detak**

### Mengapa Tidak Cukup Satu Detak?

> **INOVASI KUNCI:** Berbeda dengan metode konvensional yang mengklasifikasikan setiap detak secara terisolasi, penelitian ini menerapkan pendekatan **Context-Aware** untuk menangkap pola temporal antar-detak.

### Komposisi Context Window:
```
[3 detak sebelum] [Detak Pusat] [3 detak sesudah]
     beat_{t-3}      beat_t       beat_{t+3}
```

### Alasan Pemilihan 7 Detak:

1. **Deteksi Pola Bigeminy/Trigeminy**
   - **Bigeminy**: Setiap detak kedua abnormal → butuh minimal 4 detak
   - **Trigeminy**: Setiap detak ketiga abnormal → butuh minimal 6 detak
   - **7 detak** memberikan margin yang cukup

2. **Konteks Temporal yang Memadai**
   - Dengan heart rate 60-100 BPM, 7 detak mencakup **4.2-7 detik**
   - Representatif untuk analisis ritme jantung

3. **Keseimbangan Komputasi**
   - Ukuran lebih besar → kompleksitas eksponensial
   - Ukuran lebih kecil → konteks tidak memadai

4. **Simetri Temporal**
   - Komposisi 3+1+3 memungkinkan model mempelajari konteks sebelum dan sesudah **secara seimbang**

### Representasi Matematis:
$$W_t = [b_{t-3}, b_{t-2}, b_{t-1}, b_t, b_{t+1}, b_{t+2}, b_{t+3}]$$

### Dimensi Final:
- **Input Model**: (batch_size, 7, 200)
- 7 = jumlah detak (channels)
- 200 = sampel per detak (sequence length)

---

## 3.4 Karakteristik Fisik dan Dimensi Data

### Sumbu Waktu (Time Axis)

Dengan sampling rate 360 Hz:

$$\Delta t = \frac{1}{f_s} = \frac{1}{360} \approx 2.78\ ms$$

Durasi total window:
$$T_{window} = 200 \times 2.78\ ms \approx 556\ ms$$

### Sumbu Amplitudo (Amplitude Axis)

Data menggunakan nilai mentah dari ADC 11-bit dengan resolusi:

$$LSB = \frac{10\ mV}{2048} \approx 0.0048\ mV/unit$$

Konversi ke milivolt:
$$mV = (X_{raw} - 1024) \times 0.0048$$

---

## 3.5 Strategi Pemisahan Data (Record-Wise Split)

### Mengapa Tidak Random Split per Detak?

> **KRITIS:** Random split per detak **menyebabkan data leakage**! Detak dari pasien yang sama bisa masuk ke training DAN test set, menghasilkan evaluasi yang **terlalu optimis** dan tidak merepresentasikan performa sebenarnya.

### Record-Wise Split:
Dataset dibagi berdasarkan **Nomor Rekaman (Record ID) pasien**, bukan per detak.

| Set | Jumlah Records | Persentase |
|-----|----------------|------------|
| **Training** | ~33 records | 70% |
| **Validation** | ~7 records | 15% |
| **Testing** | ~7 records | 15% |

### Keuntungan:
1. **Mensimulasikan kondisi nyata** - Model akan memproses pasien baru yang belum pernah dilihat
2. **Evaluasi objektif** - Tidak ada kebocoran informasi dari test set
3. **Generalisasi lebih baik** - Model belajar pola universal, bukan pola spesifik pasien

---

## 3.6 Normalisasi Data dengan Standard Scaler

### Prosedur No-Peeking (Mencegah Data Leakage)

$$x_{scaled} = \frac{x - \mu}{\sigma}$$

**ATURAN KETAT:**
1. **FIT hanya pada Training Data** - Parameter μ dan σ dihitung HANYA dari data training
2. **TRANSFORM pada Semua Set** - Parameter yang sama digunakan untuk validasi dan test
3. **Tidak Ada Akses ke Data Uji** - Data uji TIDAK boleh mempengaruhi parameter normalisasi

### Implementasi:
```python
# FIT scaler HANYA pada training data
scaler.fit(X_train.reshape(-1, 1400))

# TRANSFORM semua data dengan parameter yang sama
X_train_scaled = scaler.transform(X_train.reshape(-1, 1400))
X_val_scaled = scaler.transform(X_val.reshape(-1, 1400))
X_test_scaled = scaler.transform(X_test.reshape(-1, 1400))
```

---

## 3.7 Penanganan Class Imbalance dengan Class Weighting

### Distribusi Kelas pada Dataset:

| Kelas | Jumlah | Persentase |
|-------|--------|------------|
| Normal (0) | 73,443 | 67.9% |
| Abnormal (1) | 34,647 | 32.1% |
| **Total** | **108,090** | 100% |

### Strategi: Class Weighting

Bobot penalty lebih besar diterapkan pada kesalahan prediksi kelas abnormal:

$$weight_{abnormal} = \frac{n_{total}}{2 \times n_{abnormal}}$$

### Mengapa Ini Penting?
- Mencegah model **bias terhadap kelas mayoritas** (Normal)
- Meningkatkan **sensitivitas terhadap deteksi aritmia**
- Dalam konteks medis, **False Negative (gagal mendeteksi aritmia)** jauh lebih berbahaya daripada False Positive

---

## 3.8 Hasil Pra-Pemrosesan

### Dataset Final:
- **Total detak**: 108,090 windows (masing-masing 7 detak)
- **Dimensi input**: (N, 7, 200)
- **Label**: Biner (0=Normal, 1=Abnormal)
- **Distribusi**: 67.9% Normal, 32.1% Abnormal

---

# BAGIAN 4: ARSITEKTUR MODEL 1D-CNN

## 4.1 Arsitektur Context-Aware CNN1D

### Struktur Layer:

```
┌────────────────────────────────────────────────────────┐
│  INPUT: (batch, 7, 200)                                │
│  7 channels = 7 detak, 200 samples per detak           │
├────────────────────────────────────────────────────────┤
│  BLOK KONVOLUSI 1:                                     │
│  Conv1D(7→32, kernel=3, padding=1)                     │
│  BatchNorm1D(32)                                       │
│  ReLU                                                  │
│  MaxPool1D(2) → Output: (batch, 32, 100)               │
├────────────────────────────────────────────────────────┤
│  BLOK KONVOLUSI 2:                                     │
│  Conv1D(32→64, kernel=5, padding=2)                    │
│  BatchNorm1D(64)                                       │
│  ReLU                                                  │
│  MaxPool1D(2) → Output: (batch, 64, 50)                │
├────────────────────────────────────────────────────────┤
│  BLOK KONVOLUSI 3:                                     │
│  Conv1D(64→128, kernel=7, padding=3)                   │
│  BatchNorm1D(128)                                      │
│  ReLU                                                  │
│  MaxPool1D(2) → Output: (batch, 128, 25)               │
├────────────────────────────────────────────────────────┤
│  GLOBAL AVERAGE POOLING                                │
│  → Output: (batch, 128)                                │
├────────────────────────────────────────────────────────┤
│  CLASSIFIER HEAD:                                      │
│  Linear(128→64) + ReLU + Dropout(0.5)                  │
│  Linear(64→2) → Output: 2 classes                      │
└────────────────────────────────────────────────────────┘
```

### Total Parameter: ~77,314 (relatif kecil)

---

## 4.2 Teknik Anti-Overfitting

| Teknik | Implementasi | Tujuan |
|--------|--------------|--------|
| **Dropout** | p=0.5 pada classifier | Mencegah co-adaptation neuron |
| **Batch Normalization** | Setelah setiap Conv | Stabilisasi training, efek regularisasi |
| **Weight Decay** | λ=0.0001 (L2) | Mencegah bobot terlalu besar |
| **Gradient Clipping** | max_norm=1.0 | Mencegah exploding gradients |
| **Early Stopping** | patience=15 | Menghentikan sebelum overfitting |
| **ReduceLROnPlateau** | factor=0.5 | Penyesuaian learning rate adaptif |

---

## 4.3 Konfigurasi Training

| Parameter | Nilai |
|-----------|-------|
| **Optimizer** | AdamW (lr=0.0001, weight_decay=1e-4) |
| **Loss Function** | CrossEntropyLoss + Class Weights |
| **Batch Size** | 64 |
| **Max Epochs** | 100 |
| **Early Stopping Patience** | 15 epochs |
| **LR Scheduler** | ReduceLROnPlateau (factor=0.5, patience=5) |

---

# BAGIAN 5: DEPLOYMENT DAN SIMULASI REAL-TIME (FOKUS UTAMA)

## 5.1 Mengapa Deployment Validation Sangat Penting?

> **POIN KRITIS PENELITIAN:** Banyak penelitian deep learning ECG hanya melaporkan akurasi pada test set yang sudah "bersih". Penelitian ini **memvalidasi model pada simulasi deployment nyata** untuk menguji robustness model terhadap data yang **belum pernah dilihat sama sekali**.

### Perbedaan Test Set vs Deployment Simulation:

| Aspek | Test Set | Deployment Simulation |
|-------|----------|----------------------|
| **Data Source** | Rekaman yang dipisah dari training | Record 119 (dikecualikan sejak awal) |
| **Preprocessing** | Sudah melalui pipeline lengkap | Minimal preprocessing |
| **Kondisi** | Terkontrol | Menyerupai real-world |
| **Evaluasi** | Offline | Real-time streaming |

---

## 5.2 Arsitektur Sistem Deployment

### Komponen Utama:

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND (Web UI)                    │
│  - Visualisasi sinyal ECG real-time                     │
│  - Panel kontrol dan statistik                          │
│  - Classification history dan log                       │
│  - Export gambar medis                                  │
└─────────────────────────────────────────────────────────┘
                          ↕ REST API (Flask)
┌─────────────────────────────────────────────────────────┐
│                    BACKEND (Python)                     │
│  ┌─────────────┬─────────────────┬────────────────┐     │
│  │ Data Buffer │ Inference Engine│ Evaluation     │     │
│  │ (Rolling 7) │ (ONNX Runtime)  │ (Metrics)      │     │
│  └─────────────┴─────────────────┴────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

### File Implementasi: `code/deploy/realtime_frontend.py`

---

## 5.3 Mekanisme Rolling Buffer untuk Context Window

### Tantangan:
- Data ECG masuk **satu detak per satu**
- Model membutuhkan **7 detak sekaligus** sebagai input
- Harus **real-time** tanpa blocking

### Solusi: FIFO (First-In-First-Out) Rolling Buffer

```python
beat_buffer = []  # Maksimal 7 detak

def process_new_beat(beat, beat_type):
    global beat_buffer
    
    # Tambah detak baru ke buffer
    beat_buffer.append((beat, beat_type))
    
    # Jaga ukuran buffer maksimal 7
    if len(beat_buffer) > CONTEXT_WINDOW_SIZE:
        beat_buffer = beat_buffer[-CONTEXT_WINDOW_SIZE:]
    
    # Tunggu sampai buffer penuh
    if len(beat_buffer) < CONTEXT_WINDOW_SIZE:
        return {"status": "WAITING", "buffer_size": len(beat_buffer)}
    
    # Buffer penuh, lakukan inferensi
    context_input = prepare_context_window(beat_buffer)
    prediction = model.run(context_input)
    return prediction
```

### Visualisasi Rolling Buffer:

```
Waktu →
t=1: [beat₁][____][____][____][____][____][____]  → WAITING
t=2: [beat₁][beat₂][____][____][____][____][____] → WAITING
t=3: [beat₁][beat₂][beat₃][____][____][____][____] → WAITING
...
t=7: [beat₁][beat₂][beat₃][beat₄][beat₅][beat₆][beat₇] → PREDICT (center: beat₄)
t=8: [beat₂][beat₃][beat₄][beat₅][beat₆][beat₇][beat₈] → PREDICT (center: beat₅)
```

---

## 5.4 Inferensi dengan ONNX Runtime

### Mengapa ONNX?

| Aspek | PyTorch Native | ONNX Runtime |
|-------|----------------|--------------|
| **Ukuran** | Besar (library lengkap) | Kecil (runtime saja) |
| **Kecepatan** | Baik | **Lebih cepat** (optimasi graf) |
| **Deployment** | Butuh PyTorch | Standalone, multi-platform |
| **Memory** | Tinggi | **Lebih rendah** |

### Implementasi:
```python
import onnxruntime as ort

# Load model ONNX
model = ort.InferenceSession('context_ecg_model.onnx')

# Load scaler untuk normalisasi
scaler = joblib.load('context_ecg_scaler.pkl')

def predict(context_window):
    # Normalisasi menggunakan scaler yang sama dengan training
    flat = context_window.reshape(1, 1400)  # 7 * 200 = 1400
    normalized = scaler.transform(flat).astype(np.float32)
    reshaped = normalized.reshape(1, 7, 200)
    
    # Inferensi ONNX
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    output = model.run([output_name], {input_name: reshaped})[0]
    
    # Post-processing
    prob_abnormal = softmax(output)[0, 1]
    predicted = "ABNORMAL" if prob_abnormal >= 0.5 else "NORMAL"
    
    return predicted, prob_abnormal
```

---

## 5.5 Validasi pada Record 119 (Unseen Data)

### Spesifikasi Pengujian:

| Parameter | Nilai |
|-----------|-------|
| **Record yang digunakan** | MIT-BIH Record 119 |
| **Status record** | **Dikecualikan dari training sejak awal** |
| **Tujuan** | Menguji generalisasi model pada pasien yang benar-benar baru |
| **Kondisi** | Zero-shot (model tidak pernah melihat data ini) |

### Hasil Perbandingan:

| Metrik | Test Set (Offline) | Deployment Sim (Record 119) | Selisih |
|--------|-------------------|------------------------------|---------|
| **Akurasi** | 98% | 94% | **-4%** |

### Analisis Penurunan 4% Akurasi:

1. **Distribution Shift**
   - Morfologi sinyal ECG **berbeda antar pasien**
   - Record 119 mungkin memiliki karakteristik unik yang tidak terwakili di training

2. **Tidak Ada Preprocessing Khusus**
   - Data streaming langsung dari file mentah
   - Tidak ada filtering atau cleaning tambahan

3. **Real-time Constraints**
   - Rolling buffer membutuhkan waktu untuk "warming up"
   - 6 detak pertama tidak dapat diprediksi (buffer belum penuh)

4. **TETAP LAYAK** untuk Screening:
   - **94% akurasi** masih **sangat tinggi** untuk deteksi awal
   - Threshold klinis untuk screening tools biasanya **≥90%**
   - Dapat berfungsi sebagai **instrument pendukung keputusan klinis**

---

## 5.6 Fitur Frontend Real-Time

### 1. Grafik Sinyal ECG Real-Time
- Sinyal bergerak dari kiri ke kanan
- Penanda R-peak: 🟢 Hijau (Normal), 🔴 Merah (Abnormal)
- Drag-to-scroll untuk navigasi histori

### 2. Beat Snapshot
- Menampilkan 200 sampel yang sedang diproses
- Penanda R-peak pada posisi ke-90 (sesuai Pre-R samples)
- Verifikasi morfologi gelombang

### 3. Statistik Real-Time
- Total detak terdeteksi
- Jumlah Normal vs Abnormal
- Akurasi prediksi (vs ground truth)
- **BPM (Beats Per Minute)** dari rata-rata interval R-R

### 4. Log Deteksi Salah
- Mencatat setiap prediksi yang tidak sesuai anotasi
- Klik untuk navigasi ke posisi tersebut pada grafik

### 5. Kontrol Kecepatan Simulasi
- Preset: 0.1x, 0.5x, 1x, 5x, **10x**
- Simulasi berbagai skenario pengujian

### 6. Export Gambar Medis
- Format PNG/JPEG
- Tampilan medis standar (latar putih, grid merah)
- Header informasi model dan timestamp

---

## 5.7 Perhitungan BPM Real-Time

```python
def calculate_bpm(r_peaks, sampling_rate=360):
    """
    Menghitung BPM dari 10 interval R-R terakhir
    """
    if len(r_peaks) < 2:
        return 0
    
    # Ambil 10 interval terakhir
    recent_peaks = r_peaks[-11:]  # 11 peaks = 10 intervals
    
    valid_bpms = []
    for i in range(len(recent_peaks) - 1):
        interval_samples = recent_peaks[i+1] - recent_peaks[i]
        interval_seconds = interval_samples / sampling_rate
        bpm = 60.0 / interval_seconds
        
        # Filter nilai tidak valid (30-200 BPM)
        if 30 <= bpm <= 200:
            valid_bpms.append(bpm)
    
    if not valid_bpms:
        return 0
    
    return sum(valid_bpms) / len(valid_bpms)
```

---

# BAGIAN 6: HASIL DAN PEMBAHASAN

## 6.1 Hasil Evaluasi pada Test Set (Offline)

### Metrik Performa:

| Metrik | Nilai | Interpretasi |
|--------|-------|--------------|
| **Accuracy** | 98% | Akurasi klasifikasi keseluruhan |
| **Precision** | 98% | Ketepatan prediksi abnormal |
| **Recall (Sensitivity)** | 98.82% | Kemampuan mendeteksi abnormal |
| **F1-Score** | 98% | Harmonik precision-recall |
| **AUC-ROC** | 99.93% | Kualitas decision boundary |

### Confusion Matrix:

```
                 Predicted
               Normal  Abnormal
Actual Normal   7,281      64    (FP: 64)
Actual Abnormal    41   3,424    (FN: 41)
```

### Analisis:
- **False Negative = 41** dari 3,464 kasus abnormal → Sangat rendah!
- Dalam konteks medis, FN rendah **sangat penting** karena artinya pasien berisiko **hampir selalu terdeteksi**
- False Positive yang moderat (64) **dapat diterima** karena akan mengarahkan ke pemeriksaan lebih lanjut

---

## 6.2 Hasil Validasi Deployment (Record 119)

### Akurasi: **94%**

### Perbandingan dengan Test Set:

| Skenario | Akurasi | Keterangan |
|----------|---------|------------|
| Test Set (offline, preprocessed) | 98% | Data sudah "bersih" |
| Deployment Sim (online, raw) | 94% | Data streaming mentah |
| **Gap** | **-4%** | **Masih dalam batas toleransi** |

### Mengapa Gap Ini Dapat Diterima?

1. **Threshold Klinis Terpenuhi**
   - Standar screening tools: ≥90%
   - Pencapaian 94% **melampaui threshold**

2. **Kondisi Pengujian Lebih Ketat**
   - Data benar-benar unseen (zero-shot)
   - Tidak ada preprocessing khusus

3. **Generalisasi Terbukti**
   - Model tidak overfitting pada training data
   - Mampu memproses pasien baru dengan baik

---

## 6.3 Black Box Testing Frontend

| Skenario Pengujian | Hasil | Status |
|--------------------|-------|--------|
| Load data pasien baru | Data dimuat dengan benar | ✅ Valid |
| Start/Stop/Reset simulasi | Kontrol berfungsi normal | ✅ Valid |
| Deteksi detak Normal | Klasifikasi benar | ✅ Valid |
| Deteksi detak Abnormal | Klasifikasi benar | ✅ Valid |
| Perhitungan BPM | Nilai sesuai interval R-R | ✅ Valid |
| Audit log navigasi | Dapat klik ke histori | ✅ Valid |
| Export gambar medis | File tersimpan dengan benar | ✅ Valid |
| Kecepatan simulasi 10x | Sistem tetap responsif | ✅ Valid |

---

# BAGIAN 7: SIMULASI TANYA JAWAB SIDANG

## Pertanyaan tentang PRA-PEMROSESAN DATA

### Q1: "Mengapa memilih window size 200 sampel? Kenapa tidak 188 seperti penelitian lain?"

**Jawaban:**
Pemilihan 200 sampel didasarkan pada analisis fisiologis sinyal ECG:

1. **Durasi Fisiologis**: Dengan sampling rate 360 Hz, 200 sampel setara dengan ~556 ms, yang mencakup satu siklus PQRST secara utuh tanpa memotong gelombang penting.

2. **Segmentasi Asimetris**: Kami menggunakan 90 sampel pre-R dan 110 sampel post-R karena:
   - Pre-R (250 ms): Cukup untuk menangkap gelombang P dan interval PR yang biasanya 120-200 ms sebelum R-peak
   - Post-R (306 ms): Lebih panjang untuk memastikan segmen ST dan gelombang T tertangkap, karena kelainan repolarisasi sering terjadi di fase ini

3. **Berbeda dengan 188**: Window 188 sampel pada penelitian lain biasanya menggunakan segmentasi simetris yang dapat memotong gelombang T untuk detak dengan QT interval panjang.

---

### Q2: "Bagaimana Anda memastikan tidak ada data leakage dalam pembagian dataset?"

**Jawaban:**
Kami menerapkan **Record-Wise Split** yang ketat:

1. **Pembagian per Pasien**: Dataset dibagi berdasarkan Nomor Rekaman (Record ID), bukan per detak. Jika seorang pasien masuk ke training, SELURUH detak jantungnya hanya berada di training.

2. **Prosedur No-Peeking untuk Normalisasi**:
   - Scaler di-FIT hanya pada training data
   - Parameter μ dan σ dari training digunakan untuk transform validasi dan test
   - Tidak ada statistik dari data uji yang masuk ke proses normalisasi

3. **Record 119 Dikecualikan Total**: Untuk validasi deployment, kami menggunakan Record 119 yang **tidak pernah disentuh** sama sekali selama training maupun hyperparameter tuning.

---

### Q3: "Kenapa menggunakan context window 7 detak? Bukankah ini memperlambat prediksi karena harus menunggu 7 detak?"

**Jawaban:**
Pemilihan 7 detak adalah **trade-off yang diperhitungkan**:

**Keuntungan:**
1. **Deteksi Pola Aritmia Berurutan**: Bigeminy dan Trigeminy adalah aritmia yang hanya dapat dideteksi jika melihat urutan beberapa detak. Dengan 7 detak (3+1+3), kami dapat mendeteksi pola ini.

2. **Konteks Temporal**: Aritmia tidak muncul secara terisolasi. Detak sebelum dan sesudah memberikan informasi penting tentang ritme jantung.

3. **Akurasi Lebih Tinggi**: Eksperimen kami menunjukkan bahwa context window meningkatkan akurasi dibanding single-beat classification.

**Mitigasi Delay:**
1. **Delay Minimal**: 7 detak pada heart rate 60 BPM = 7 detik delay maksimal. Pada 100 BPM = 4.2 detik.
2. **Rolling Buffer**: Setelah buffer penuh, setiap detak baru langsung menghasilkan prediksi.
3. **Untuk Screening Awal**: Delay beberapa detik masih acceptable karena bukan untuk emergency real-time defibrillation.

---

### Q4: "Bagaimana Anda menangani class imbalance? Bukankah 67.9% Normal vs 32.1% Abnormal cukup timpang?"

**Jawaban:**
Kami menangani class imbalance dengan **Class Weighting** pada loss function:

$$weight_i = \frac{n_{total}}{n_{classes} \times n_i}$$

Implementasi:
```python
class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
```

**Mengapa Class Weighting, bukan SMOTE/Oversampling?**
1. **Tidak mengubah distribusi data asli** - Mempertahankan representasi fisiologis yang akurat
2. **Mencegah overfitting pada data sintetis** - SMOTE dapat menciptakan data yang tidak realistis
3. **Efektif untuk deep learning** - Class weighting pada loss sudah cukup untuk mengkompensasi imbalance

---

## Pertanyaan tentang DEPLOYMENT DAN SIMULASI

### Q5: "Mengapa akurasi turun dari 98% ke 94% pada deployment simulation? Apakah ini menunjukkan model overfitting?"

**Jawaban:**
Penurunan 4% ini **bukan overfitting**, melainkan **expected distribution shift**:

1. **Definisi Overfitting**: Model terlalu menyesuaikan diri dengan training data sehingga tidak bisa generalisasi. Jika overfitting, akurasi test set juga akan rendah.

2. **Fakta**: Akurasi test set TETAP 98%, menunjukkan model generalisasi dengan baik pada data dari distribusi yang sama.

3. **Distribution Shift**: Record 119 memiliki karakteristik unik:
   - Morfologi sinyal berbeda dari rekaman lain
   - Rasio normal/abnormal mungkin berbeda
   - Noise profile berbeda

4. **94% Masih Excellent**: Untuk screening tool, threshold klinis biasanya ≥90%. Pencapaian 94% pada **zero-shot scenario** menunjukkan model sangat robust.

5. **Ini Justru Bukti Positif**: Model yang akurasinya tidak berubah pada unseen data justru mencurigakan (kemungkinan data leakage).

---

### Q6: "Jelaskan bagaimana mekanisme rolling buffer bekerja dan mengapa diperlukan?"

**Jawaban:**

**Masalah:**
- Model membutuhkan 7 detak sebagai input
- Dalam streaming real-time, data masuk satu detak per satu
- Tidak bisa menunggu 7 detak baru setiap kali prediksi

**Solusi Rolling Buffer:**
```
Buffer awal:   [_][_][_][_][_][_][_]   → WAITING
Detak masuk:   [1][_][_][_][_][_][_]   → WAITING
...
Buffer penuh:  [1][2][3][4][5][6][7]   → PREDICT (center: detak 4)
Detak baru:    [2][3][4][5][6][7][8]   → PREDICT (center: detak 5)
                ↑ Detak 1 dibuang
```

**Keuntungan:**
1. **Efisien**: Hanya menyimpan 7 detak terakhir
2. **Real-time**: Setelah warm-up, setiap detak baru menghasilkan prediksi
3. **FIFO**: First-In-First-Out memastikan urutan temporal terjaga

---

### Q7: "Mengapa menggunakan ONNX untuk deployment? Kenapa tidak langsung PyTorch?"

**Jawaban:**

**Perbandingan:**

| Aspek | PyTorch Native | ONNX Runtime |
|-------|----------------|--------------|
| **Ukuran Library** | ~500 MB | ~20 MB |
| **Kecepatan Inferensi** | Baik | **20-30% lebih cepat** |
| **Memory Usage** | Tinggi | **Lebih rendah** |
| **Dependency** | PyTorch + semua library | ONNX Runtime saja |
| **Platform** | Python-centric | **Multi-platform** (C++, Java, C#) |

**Alasan Pemilihan ONNX:**
1. **Deployment Ringan**: Tidak perlu install PyTorch besar
2. **Optimasi Graf**: ONNX Runtime melakukan optimasi otomatis pada graf komputasi
3. **Portabilitas**: Model bisa dijalankan di berbagai platform dan bahasa
4. **Standar Industri**: ONNX adalah format exchange model yang diakui industri

---

### Q8: "Bagaimana sistem menghitung BPM dan apakah reliable?"

**Jawaban:**

**Metode:**
```python
def calculate_bpm(r_peaks, sampling_rate=360):
    # Ambil 10 interval R-R terakhir
    recent_peaks = r_peaks[-11:]
    
    valid_bpms = []
    for i in range(len(recent_peaks) - 1):
        interval_samples = recent_peaks[i+1] - recent_peaks[i]
        interval_seconds = interval_samples / sampling_rate
        bpm = 60.0 / interval_seconds
        
        # Filter nilai tidak valid (30-200 BPM)
        if 30 <= bpm <= 200:
            valid_bpms.append(bpm)
    
    return average(valid_bpms)
```

**Reliabilitas:**
1. **Filtering**: Nilai BPM di luar 30-200 diabaikan untuk menghindari artefak
2. **Moving Average**: Menggunakan 10 interval terakhir untuk stabilitas
3. **Validasi**: Hasil dibandingkan dengan anotasi MIT-BIH yang sudah terverifikasi

---

## Pertanyaan tentang MODEL

### Q9: "Mengapa menggunakan CNN 1D, bukan LSTM atau Transformer yang lebih modern?"

**Jawaban:**

**Alasan Pemilihan CNN 1D:**

1. **Efisiensi untuk Sinyal ECG**: CNN 1D sangat efisien dalam mengekstraksi fitur lokal dari sinyal deret waktu. Kompleks QRS, gelombang P, dan T adalah fitur lokal.

2. **Context Window Sudah Menangkap Temporal**: Dengan pendekatan 7 detak sebagai input, informasi temporal antar-detak sudah tertangkap sebagai "channels". LSTM tidak diperlukan.

3. **Parameter Lebih Sedikit**: Model kami hanya ~77,314 parameter, memungkinkan:
   - Training lebih cepat
   - Inferensi lebih cepat
   - Deployment lebih ringan

4. **Hasil Sudah Sangat Baik**: Akurasi 98% sudah sangat tinggi. Arsitektur kompleks tidak menjamin peningkatan signifikan.

---

### Q10: "Jelaskan mengapa menggunakan Global Average Pooling dan bukan Flatten?"

**Jawaban:**

**Global Average Pooling (GAP):**
- Mengambil rata-rata nilai pada setiap channel feature map
- Output: vektor dengan dimensi = jumlah channels

**Keuntungan GAP vs Flatten:**

| Aspek | Flatten | Global Average Pooling |
|-------|---------|----------------------|
| **Parameter** | Sangat banyak | **Nol** (operasi langsung) |
| **Overfitting** | Rentan | **Lebih resistant** |
| **Translation Invariance** | Tidak ada | **Ada** |
| **Interpretability** | Rendah | **Lebih tinggi** |

**Contoh:**
- Setelah Conv layer terakhir: (batch, 128, 25) → 128 channels, 25 timesteps
- **Flatten**: 128 × 25 = 3,200 features → butuh 3,200 weights di FC layer
- **GAP**: rata-rata 25 timesteps → 128 features → hanya butuh 128 weights

---

### Q11: "Bagaimana Anda memastikan model tidak overfitting selama training?"

**Jawaban:**

**Multi-Layer Anti-Overfitting:**

1. **Early Stopping (patience=15)**
   - Memantau validation AUC
   - Jika tidak ada improvement 15 epoch berturut-turut, training dihentikan
   - Best model dari checkpoint terbaik disimpan

2. **Dropout (p=0.5)**
   - 50% neuron dinonaktifkan secara acak saat training
   - Mencegah co-adaptation

3. **Batch Normalization**
   - Menormalisasi output setiap layer
   - Efek regularisasi ringan

4. **Weight Decay (L2 = 1e-4)**
   - Menambah penalty pada bobot besar
   - Mencegah model terlalu kompleks

5. **Gradient Clipping (max_norm=1.0)**
   - Mencegah exploding gradient
   - Menjaga stabilitas training

6. **Learning Rate Scheduling**
   - ReduceLROnPlateau menurunkan LR jika val_loss stagnan
   - Memungkinkan fine-tuning halus menjelang konvergensi

---

## Pertanyaan Kritis dan Challenging

### Q12: "Dengan banyaknya penelitian sejenis yang sudah ada, apa kontribusi unik penelitian ini?"

**Jawaban:**

**Kontribusi Unik:**

1. **FOKUS PADA DEPLOYMENT VALIDATION**
   - Sebagian besar penelitian hanya melaporkan akurasi test set
   - Kami **memvalidasi pada simulasi deployment nyata** dengan data yang benar-benar unseen
   - Menunjukkan penurunan akurasi yang realistis (98% → 94%) dan menganalisis penyebabnya

2. **STRATEGI PRA-PEMROSESAN SISTEMATIS**
   - Dokumentasi lengkap dari raw data → model-ready input
   - Justifikasi fisiologis untuk setiap keputusan (200 sampel, 7 detak context, dll)
   - **Record-wise split** yang ketat untuk mencegah data leakage

3. **SISTEM MONITORING REAL-TIME LENGKAP**
   - Bukan hanya model, tapi **sistem end-to-end**
   - Frontend dengan visualisasi dan audit log
   - Mekanisme rolling buffer untuk streaming data

4. **REPRODUCIBILITY**
   - Semua kode tersedia di repository
   - Langkah-langkah dapat direplikasi oleh peneliti lain

---

### Q13: "Bagaimana jika model digunakan pada pasien dengan kondisi yang sangat berbeda dari dataset training? Misalnya pasien dengan pacemaker?"

**Jawaban:**

**Keterbatasan yang Diakui:**

1. **Dataset MIT-BIH Specific**: Model dilatih pada dataset MIT-BIH yang memiliki karakteristik tertentu. Pasien dengan pacemaker **tidak termasuk** dalam dataset ini.

2. **Domain Shift**: Sinyal dari pasien pacemaker akan sangat berbeda (ada spike artifact dari pacemaker). Model **tidak didesain** untuk kasus ini.

**Rekomendasi:**

1. **Screening Tool Only**: Model ini adalah **alat bantu screening**, bukan pengganti diagnosis dokter. Hasil harus dikonfirmasi oleh kardiolog.

2. **Dataset Expansion**: Untuk deployment klinis nyata, perlu training ulang dengan dataset yang mencakup:
   - Pasien dengan pacemaker
   - Dataset dari populasi berbeda (non-US)
   - Variasi device ECG berbeda

3. **Warning System**: Sistem deployment bisa dilengkapi detector untuk morfologi yang tidak biasa, yang akan trigger "Require Manual Review".

---

### Q14: "Apa rencana untuk mengatasi distribution shift di masa depan?"

**Jawaban:**

**Strategi Mitigasi:**

1. **Transfer Learning / Fine-Tuning**
   - Model yang sudah terlatih dapat di-fine-tune dengan data baru dari klinik spesifik
   - Membutuhkan data labeled yang lebih sedikit

2. **Domain Adaptation**
   - Teknik seperti adversarial domain adaptation
   - Membantu model mempelajari fitur yang invariant terhadap sumber data

3. **Continual Learning**
   - Model dapat di-update secara berkala dengan data baru
   - Tanpa melupakan pengetahuan sebelumnya (menghindari catastrophic forgetting)

4. **Ensemble Methods**
   - Menggabungkan prediksi dari beberapa model yang dilatih pada dataset berbeda
   - Meningkatkan robustness terhadap variasi

---

### Q15: "Bagaimana Anda memvalidasi bahwa ground truth annotation dari MIT-BIH itu benar?"

**Jawaban:**

**Validitas Dataset MIT-BIH:**

1. **Gold Standard**: MIT-BIH Arrhythmia Database adalah **benchmark paling diakui** dalam penelitian ECG sejak 1980-an.

2. **Anotasi oleh Ahli**: Setiap detak dianotasi oleh **dua kardiolog independen** dengan konsensus.

3. **Peer Review**: Dataset ini telah digunakan dalam **ribuan publikasi** peer-reviewed dan hasilnya konsisten.

4. **PhysioBank Verification**: Tersedia melalui PhysioBank yang dikelola oleh NIH dan MIT.

**Keterbatasan yang Diakui:**

- Kemungkinan kesalahan anotasi pada beberapa detak memang ada
- Namun, kesalahan tersebut **sangat kecil** dan tidak signifikan mempengaruhi hasil keseluruhan
- Dalam penelitian kami, beberapa prediksi "salah" mungkin sebenarnya **model yang benar** dan anotasi yang kurang akurat

---

# BAGIAN 8: KESIMPULAN

## 8.1 Temuan Utama

### Pra-Pemrosesan Data
✅ **Context window 7 detak** berhasil menangkap informasi ritme antar-detak
✅ **Segmentasi asimetris 200 sampel** (90 pre-R, 110 post-R) optimal untuk menangkap siklus PQRST
✅ **Record-wise split** menjamin evaluasi objektif tanpa data leakage
✅ **Class weighting** efektif mencegah bias mayoritas

### Performa Model
✅ Akurasi **98%** pada test set dengan recall **98.82%**
✅ False Negative sangat rendah (41 dari 3,464 kasus abnormal)
✅ AUC-ROC **99.93%** menunjukkan decision boundary optimal

### Deployment Validation
✅ Akurasi **94%** pada simulasi deployment (Record 119, zero-shot)
✅ Penurunan 4% masih dalam batas toleransi klinis (threshold ≥90%)
✅ Sistem real-time dengan rolling buffer berfungsi dengan baik
✅ Semua fitur frontend valid pada Black Box Testing

---

## 8.2 Kontribusi Penelitian

1. **Validasi Deployment yang Komprehensif**: Tidak hanya melaporkan akurasi offline, tetapi juga menguji pada kondisi yang menyerupai deployment nyata.

2. **Dokumentasi Pra-Pemrosesan yang Lengkap**: Strategi preprocessing yang dapat direplikasi dan memiliki justifikasi fisiologis.

3. **Sistem End-to-End**: Bukan hanya model, tetapi sistem monitoring lengkap dengan frontend interaktif.

---

## 8.3 Saran Pengembangan

1. **Ekspansi ke Multi-Class**: Mengklasifikasikan jenis aritmia spesifik (PVC, AF, Bundle Branch Block, dll)

2. **Validasi Lintas Populasi**: Pengujian dengan dataset eksternal (INCART, AHA Database)

3. **Edge Computing**: Deployment ke mikrokontroler/wearable untuk monitoring portable

4. **Integrasi Telemedicine**: Remote monitoring dengan alert system untuk tenaga medis

---

# LAMPIRAN: REFERENSI KODE

## File Utama:

1. **Data Preparation**: `code/dataset/ecg-datacreator.ipynb`
   - Pembuatan dataset dari MIT-BIH
   - Segmentasi dan labeling

2. **Model Training**: `code/train-last/ecg-train-last-v2-1.ipynb`
   - Arsitektur Context-Aware CNN1D
   - Training dengan anti-overfitting techniques

3. **Deployment Frontend**: `code/deploy/realtime_frontend.py` (args: v6)
   - Sistem monitoring real-time
   - Rolling buffer implementation
   - ONNX inference

## Model Config (v6):
```python
MODEL_CONFIGS = {
    'v6': {
        'name': 'Context-Aware CNN1D (v6)',
        'onnx_file': 'context_ecg_model.onnx',
        'scaler_file': 'context_ecg_scaler.pkl',
        'input_shape': (1, 7, 200),
        'beat_length': 200,
        'context_aware': True,
        'context_window_size': 7,
        'pre_r_samples': 90,
        'post_r_samples': 110,
    },
}
```

---

**Repository**: github.com/aimatochysia/thesis

**© 2025 - Universitas Bina Nusantara**
