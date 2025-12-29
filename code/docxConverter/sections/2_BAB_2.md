**BAB 2****

**TINJAUAN REFERENSI**

**2.1 Elektrokardiogram**

Elektrokardiogram (EKG) adalah sinyal biologis yang merekam aktivitas
listrik dari jantung melalui elektroda di kulit (Ashley & Niebauer,
2004). Sinyal EKG terdiri dari 3 gelombang utama, seperti yang
ditunjukkan pada **Gambar 2.1**, yaitu gelombang P, yang
merepresentasikan depolarisasi atrium, gelombang QRS, yang
merepresentasikan depolarisasi ventrikel, dan gelombang T, yang
merepresentasikan repolarisasi ventrikel. Sinyal EKG ini memiliki
rentang frekuensi sekitar 0,05 Hz - 1000 Hz dan rentang listrik sekitar
1 mV - 10 mV secara normal (Velayudhan & Peter, 2016).

<figure>
<img src="media/image5.png" style="width:5.95756in;height:2.37506in" />
<figcaption><blockquote>
<p><strong>Gambar 2.1</strong> Sinyal Elektrokardiogram</p>
</blockquote></figcaption>
</figure>

Mungkin terdapat sinyal yang tidak diinginkan atau suara bising yang
dapat digabungkan dengan sinyal EKG yang sebenarnya selama perekaman.
Derau ini harus dihilangkan, jika tidak, maka akan menyebabkan masalah
analisis yang dapat menyebabkan diagnosis yang salah. Derau itu sendiri
memiliki karakteristik frekuensi, sehingga dengan menghilangkan atau
melewatkan rentang frekuensi tertentu akan menghasilkan sinyal EKG yang
sebenarnya.

**2.2 Denoising dalam pemrosesan sinyal ECG**

Dalam pengolahan sinyal biomedis, khususnya sinyal Electrocardiogram
(ECG), keberadaan dari noises merupakan salah satu tantangan utama yang
dapat mempengaruhi akurasi dari analisis. Noise ini dapat berasal dari
berbagai sumber seperti baseline wander, powerline interface, Gerakan
pasien maupun aktivitas dari otot. Oleh karena itu, tahapan denoising
ini menjadi Langkah esesial pada pra-pemrosesan sinyal ECG sebelum
dilakukan pelatihan model CNN. Dalam Konteks penelitian ini, proses
denoising tidak dilakukan melalui metode eksplisit namun dilakukan
melalui pendekatan yang structural dan segmentatif. Data ECG yang
digunakan bersumber dari MIT-BIH Arrhytmia Database yang telah dianotas
oleh ahli dan hanya mencakup sinyal yang relevan dengan peristiwa detak
jantung. Proses seperti trimming Panjang sinyal normal, pemindahan label
ke posisi akhir dan shifting secara tidak langsung berkontribusi dalam
menghilangkan bagian sinyal yang mengandung noise, seperti area sebelum
dan sesudah terdeteksi seperti pada **Gambar 2.2**

<figure>
<img src="media/image6.png" style="width:4.49792in;height:2.42456in" />
<figcaption><p><strong>Gambar 2.2</strong> Denoising
ECG</p></figcaption>
</figure>

Strategi ini termasuk dalam pendekatan *implicit noise reduction*, yaitu
penghapusan *noise* melalui seleksi segmen dan normalisasi struktur data
alih-alih melalui filter matematis. Hal ini terbukti efektif karena
model deep learning, khususnya *hybrid CNN-LSTM*, sangat sensitif
terhadap variasi struktural maupun numerik pada input data (Faust et
al., 2018). Dengan demikian, meskipun tidak menggunakan teknik denoising
eksplisit, penelitian ini telah menerapkan prinsip denoising secara
adaptif melalui strategi segmentasi dan pemangkasan berbasis anotasi
yang memperkuat robustness model terhadap variasi sinyal yang berasal
dari noise.

**2.3 Deteksi *Arrhythmia*, Sinyal Abnormal dan Variabilitas Sinyal**

*Arrhythmia* adalah kondisi abnormal dari ritme jantung yang dapat
mengancam jiwa, seperti takikardia ventrikular (VT) pada **Gambar 3**,
fibrilasi ventrikular (VF) pada **Gambar 4**, dan fibrilasi atrium (AF)
pada **Gambar 5**. Kondisi ini menyebabkan denyut jantung menjadi
terlalu cepat, terlalu lambat, atau tidak teratur. *Arrhythmia* terjadi
ketika impuls listrik yang mengatur detak jantung tidak bekerja dengan
baik, sehingga menyebabkan aktivitas elektrik jantung menjadi kacau
(Zipes & Jalife, 2013). Menurut *American Heart Association* (AHA),
*arrhythmia* merupakan penyebab utama dari berbagai komplikasi
kardiovaskular, dan deteksi dini melalui ECG menjadi komponen penting
dalam pencegahan gagal jantung dan stroke (AHA, 2021).

<figure>
<img src="media/image7.jpeg" style="width:5.71875in;height:1.75962in" />
<figcaption><p><strong>Gambar 2.3</strong> Takikardia Ventrikel
(VT)</p></figcaption>
</figure>

<figure>
<img src="media/image8.jpeg" style="width:5.71875in;height:2.06302in" />
<figcaption><p><strong>Gambar 2.1</strong> fibrilasi ventrikular
(VF)</p></figcaption>
</figure>

<figure>
<img src="media/image9.jpeg" style="width:5.71875in;height:1.43885in" />
<figcaption><p><strong>Gambar 2.2</strong> fibrilasi atrium
(AF)</p></figcaption>
</figure>

Dalam konteks klasifikasi sinyal ECG, seluruh bentuk detak jantung
abnormal dikategorikan sebagai kelas \"1\" atau *arrhythmia*, sedangkan
detak jantung normal diklasifikasikan sebagai \"0\". Klasifikasi biner
ini disesuaikan dengan pendekatan machine learning yang digunakan dalam
penelitian ini agar lebih efisien dalam pemodelan dan analisis.
Pengkategorian seluruh jenis kelainan detak jantung ke dalam satu label
umum (*arrhythmia*) dianggap cukup representatif pada tahap awal
klasifikasi, terutama saat model difokuskan pada deteksi keberadaan
kelainan, bukan spesifikasinya (Faust et al., 2018). Dalam konteks
klasifikasi sinyal ECG, seluruh bentuk detak jantung abnormal
dikategorikan sebagai kelas \"1\" atau *arrhythmia*, sedangkan detak
jantung normal diklasifikasikan sebagai \"0\". Klasifikasi biner ini
disesuaikan dengan pendekatan machine learning yang digunakan dalam
penelitian ini agar lebih efisien dalam pemodelan dan analisis.
Pengkategorian seluruh jenis kelainan detak jantung ke dalam satu label
umum (*arrhythmia*) dianggap cukup representatif pada tahap awal
klasifikasi, terutama saat model difokuskan pada deteksi keberadaan
kelainan, bukan spesifikasinya (Faust et al., 2018).

**2.4 *Heart Rate Variability***

*Heart Rate Variability* (HRV) adalah ukuran fluktuasi waktu antara dua
detak jantung berturut-turut, yang biasanya dihitung dari interval RR
pada sinyal ECG. HRV mencerminkan keseimbangan antara sistem saraf
simpatis dan parasimpatis, dan sering digunakan sebagai indikator
kesehatan kardiovaskular serta stres fisiologis (Shaffer & Ginsberg,
2017). Secara umum, HRV dapat dianalisis melalui tiga pendekatan utama:
domain waktu (misalnya SDNN, RMSSD), domain frekuensi (misalnya LF, HF,
LF/HF ratio), dan domain non-linear (misalnya *Poincaré plot, entropy*).
Dalam penelitian berbasis *machine learning*, fitur-fitur HRV ini sering
digunakan untuk membedakan kondisi detak jantung normal dan abnormal
(Acharya et al., 2006). Fitur HRV yang rendah biasanya menunjukkan
kondisi abnormal seperti *arrhythmia* atau stres jantung, sementara
nilai HRV yang tinggi menandakan sistem otonom yang sehat dan adaptif.
Dalam konteks penelitian ini, HRV memiliki relevansi sebagai sinyal
pembeda yang dapat digunakan sebagai input untuk model CNN atau LSTM,
atau sebagai penanda penurunan kualitas ritme jantung.

**2.5 Dataset MIT-BH *Arrythmia***

MIT-BIH *Arrhythmia* Database merupakan salah satu dataset benchmark
paling komprehensif dan banyak digunakan dalam penelitian analisis
sinyal elektrokardiogram (ECG). Dataset ini dikembangkan oleh Beth
Israel Hospital bekerja sama dengan Massachusetts Institute of
Technology (MIT) dan tersedia secara publik melalui platform PhysioNet
dan Kaggle (Goldberger et al., 2000; Yoon, 2023). Dataset ini terdiri
dari 48 rekaman ECG dua channel berdurasi sekitar 30 menit, yang direkam
dari 47 pasien rawat jalan dan rumah sakit. Setiap rekaman memiliki dua
lead: MLII (Modified Lead II) dan V1, serta disampling pada frekuensi
360 Hz. Rekaman ini telah dianotasi secara manual oleh ahli jantung,
mencakup lebih dari 100.000 detak jantung yang diklasifikasikan ke dalam
lebih dari 15 tipe detak, seperti detak normal (N), *premature
ventricular contraction* (V), *atrial premature beat* (A), dan lainnya.

Struktur data terdiri dari:

1.  *Signal Time Series*: data numerik sinyal ECG dalam satuan milivolt.

2.  *Sample Index*: titik waktu untuk setiap anotasi detak.

3.  *Annotation Label*: label untuk tiap detak, sesuai klasifikasi AAMI
    (Association for the Advancement of Medical Instrumentation).

Dalam penelitian ini, dataset digunakan dalam format yang telah
disederhanakan dan diproses ulang oleh Taejoong Yoon melalui Kaggle,
yang menggabungkan sinyal berdurasi pendek (segmentasi) dan
mengelompokkan tipe detak jantung menjadi dua kelas utama, yaitu:

1.  Label 0: Detak jantung normal

2.  Label 1: Detak jantung abnormal (*arrhythmia*)

Penggunaan dataset ini sangat sesuai untuk pengembangan dan evaluasi
sistem diagnosis otomatis berbasis machine learning karena:

1.  Anotasi dilakukan secara manual oleh ahli jantung

2.  Sinyal tersedia dalam format digital dan dapat diakses

3.  Mencakup variasi besar dalam kondisi fisiologis dan patologis

Dataset ini juga dilengkapi dengan metadata tambahan seperti jenis
kelamin pasien, lead yang digunakan, dan waktu perekaman. Dengan
demikian, MIT-BIH *Arrhythmia* Database tidak hanya memberikan sinyal
ECG mentah, tetapi juga konteks klinis yang dapat membantu dalam
interpretasi hasil klasifikasi dan generalisasi model terhadap populasi
yang lebih luas. Dataset versi Kaggle oleh Taejoong Yoon (2023)
menyediakan data ini dalam format yang lebih mudah diakses, dengan
pembagian detak jantung menjadi normal dan abnormal berdasarkan tipe
anotasi. Dataset ini sangat berguna untuk melatih dan mengevaluasi model
machine learning karena mengandung variasi besar dalam kondisi kardiak.

**2.6 Modifikasi Data**

Setelah data terkumpul, dilakukan proses modifikasi guna membentuk
struktur data yang siap digunakan untuk pelatihan model machine
learning. Salah satu tahapan utama adalah pembentukan sub-dataset dengan
mengambil sampel rekaman secara acak dari database besar untuk menjaga
keberagaman representasi sinyal. Kemudian, dilakukan proses segmentasi
sinyal menjadi fixed-length windows yang merepresentasikan satu siklus
detak jantung. Proses ini merujuk pada studi Murali et al. (2025), yang
menyarankan pemotongan sinyal ECG berdasarkan jarak antar R-peak atau
menggunakan pendekatan sliding window dengan jumlah sampel konstan
seperti 256 titik. Tujuan dari segmentasi ini adalah untuk menciptakan
representasi spasial-temporal yang konsisten untuk setiap heartbeat
cycle, mengingat variasi panjang sinyal dapat mengganggu pelatihan model
konvolusional.

Tahap selanjutnya adalah *reclassification* label dari bentuk
multikategori menjadi dua kelas biner yaitu Normal dan Abnormal. Panwar
et al. (2025) menyebutkan bahwa pendekatan klasifikasi biner tidak hanya
menyederhanakan kompleksitas model, tetapi juga lebih efisien pada
aplikasi real-time dan perangkat portabel. Semua kategori *arrhythmia*
seperti *Premature Ventricular Contraction* (PVC), *Atrial Fibrillation*
(AF), dan *Ventricular Tachycardia* (VT) diklasifikasikan ke dalam kelas
Abnormal. Hal ini juga bertujuan mempermudah penilaian performa awal
model sebelum dilakukan klasifikasi lanjutan secara multi-kelas. Sebagai
langkah validasi kemampuan generalisasi model, sebagian kecil data yang
belum dibersihkan secara ekstensif juga disisihkan sebagai prediction
dataset, yang akan digunakan untuk menguji performa model terhadap data
noisy yang lebih menyerupai kondisi di lapangan.

**2.7 Persiapan data**

Persiapan data dimulai dengan eksplorasi data menggunakan metode
exploratory data analysis (EDA) untuk memahami distribusi data,
mendeteksi nilai kosong, dan mengidentifikasi outlier. (Zahid et al.,
2020) menunjukkan bahwa *outlier* pada data ECG dapat berdampak besar
terhadap performa model deteksi, terutama dalam hal sinyal ekstrem
akibat gangguan perangkat keras atau noise. Oleh karena itu, dilakukan
pembersihan dengan teknik Z-score *normalization.* Sedangkan untukm
menghitung Z-score itu sendiri menggunakan **persamaan (1)**:

  -----------------------------------------------------------------------
  **(1)**
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

$$z = \frac{x - \mu}{\sigma}$$

di mana x merupakan nilai amplitudo sinyal, μ adalah rata-rata populasi,
dan σ adalah standar deviasi. Nilai Z\>3 atau Z\<−3 dikategorikan
sebagai *outlier* dan dieliminasi. Untuk mengatasi ketidakseimbangan
kelas (*class imbalance*) antara Normal dan Abnormal, digunakan teknik
*resampling* atau seleksi data terstratifikasi. (Alinsaif, 2024)
menyarankan bahwa distribusi data yang tidak seimbang dapat menyebabkan
model cenderung bias terhadap kelas mayoritas, sehingga proses
*balancing* harus dilakukan sebelum pelatihan. Selain itu, dilakukan
validasi terhadap keberagaman sinyal *arrhythmia* menggunakan analisis
*K-Means Clustering*, guna memastikan bahwa data dalam kelas abnormal
mencakup spektrum variasi yang luas. Fungsi objektif dari *K-Means*
dapat ditulisakan dengan **persamaan (2)**:

  -----------------------------------------------------------------------
  **(2)**
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

$$\arg{\min_{\text{\{}C_{1},C_{2},\ldots,C_{k}\text{\}}}\left\{ \sum_{i = 1}^{k}{\sum_{x \in C_{i}}^{}{\left. \ |\text{|}x - \mu_{i} \right.\ {\text{|}|}^{2}}} \right\}}$$

sdi mana Ci adalah cluster ke-i dan μ adalah centroid-nya pada rumus.
Rumus *K-Means* tersebut merepresentasikan fungsi objektif yang
bertujuan meminimalkan total kuadrat jarak antara tiap data dengan pusat
klaster (*centroid*) tempat data tersebut berada. Dalam hal ini, x
adalah titik data dan ∥x−μi​∥menunjukkan jarak Euclidean kuadrat. Semakin
kecil nilai total fungsi tersebut, semakin kompak data dalam
masing-masing klaster, menandakan bahwa proses pengelompokan telah
berhasil mengidentifikasi struktur alami dalam data.. Terakhir,
dilakukan *dimensionality reduction* menggunakan *Principal Component
Analysis (PCA)* untuk mengekstraksi komponen utama dan mengurangi
redudansi dimensi dengan **persamaan(3)**:

  -----------------------------------------------------------------------
  **(3)**
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

$$T\  = \ X\  \cdot W$$

di mana T adalah hasil transformasi, X adalah matriks data input, dan W
adalah matriks *eigenvector* dari kovariansi X. PCA dinyatakan efektif
(Eleyan & Alboghbaish, 2024) dalam meningkatkan efisiensi pelatihan
model ECG dengan tetap mempertahankan informasi utama dari sinyal.

**2.7.1 Standard Scaler untuk Normalisasi Data**

Dalam konteks pembelajaran mesin untuk klasifikasi sinyal ECG, normalisasi data menggunakan *Standard Scaler* merupakan teknik krusial untuk memastikan konvergensi model yang optimal. *Standard Scaler* mentransformasi setiap fitur sehingga memiliki rata-rata 0 dan standar deviasi 1, mengikuti distribusi Gaussian standar. Transformasi ini diberikan oleh **persamaan (4)**:

$$x_{scaled} = \frac{x - \mu}{\sigma}$$

di mana:
- $x$ adalah nilai fitur asli
- $\mu$ adalah rata-rata fitur pada data pelatihan
- $\sigma$ adalah standar deviasi fitur pada data pelatihan
- $x_{scaled}$ adalah nilai fitur yang telah dinormalisasi

**Prosedur No-Peeking (Mencegah Kebocoran Data)**

Aspek kritis dalam penerapan *Standard Scaler* adalah prosedur *no-peeking* untuk mencegah kebocoran informasi (*data leakage*). Prosedur ini mengharuskan:

1. **Fit pada Data Training SAJA**: Parameter statistik ($\mu$ dan $\sigma$) hanya dihitung dari data pelatihan
2. **Transform pada Semua Set**: Parameter yang sama digunakan untuk mentransformasi data validasi dan pengujian
3. **Tidak Ada Akses ke Data Uji**: Data uji tidak boleh mempengaruhi parameter normalisasi

Pelanggaran prosedur ini akan menyebabkan model memiliki akses tidak langsung ke informasi dari data uji, menghasilkan metrik evaluasi yang terlalu optimis dan tidak merepresentasikan performa sebenarnya pada data baru.

**2.7.2 Context Window (Jendela Konteks)**

Pendekatan *context-aware* dalam klasifikasi ECG menggunakan konsep jendela konteks (*context window*) untuk menangkap informasi temporal antar-detak jantung. Berbeda dengan klasifikasi per-detak yang mengisolasi setiap detak, pendekatan ini mempertimbangkan pola ritme dalam urutan detak.

**Mengapa 7 Detak?**

Pemilihan ukuran jendela konteks sebesar 7 detak (3 sebelum + 1 pusat + 3 sesudah) didasarkan pada pertimbangan berikut:

1. **Deteksi Pola Bigeminy/Trigeminy**: Pola aritmia seperti *bigeminy* (setiap detak kedua abnormal) dan *trigeminy* (setiap detak ketiga abnormal) memerlukan minimal 4-6 detak untuk diidentifikasi. Jendela 7 detak memberikan margin yang cukup.

2. **Konteks Temporal yang Memadai**: Dengan frekuensi jantung normal 60-100 BPM, 7 detak mencakup sekitar 4.2-7 detik, memberikan konteks temporal yang representatif.

3. **Keseimbangan Komputasi**: Ukuran yang lebih besar akan meningkatkan kompleksitas model dan kebutuhan memori secara eksponensial, sementara ukuran yang lebih kecil mungkin tidak cukup untuk menangkap pola temporal.

4. **Simetri Temporal**: Komposisi simetris (3+1+3) memungkinkan model mempelajari konteks sebelum dan sesudah detak pusat secara seimbang.

**Representasi Matematis**

Untuk detak pusat pada posisi $t$, jendela konteks didefinisikan sebagai **persamaan (5)**:

$$W_t = [b_{t-3}, b_{t-2}, b_{t-1}, b_t, b_{t+1}, b_{t+2}, b_{t+3}]$$

di mana $b_i$ adalah representasi vektor dari detak ke-$i$ dengan dimensi 200 sampel. Label klasifikasi ditentukan oleh detak pusat $b_t$ saja, sementara detak sekitarnya berfungsi sebagai fitur kontekstual.

**2.8 Algoritma Pan-Tompkins untuk Deteksi R-Peak**

Algoritma Pan-Tompkins adalah metode standar industri untuk mendeteksi kompleks QRS dalam sinyal elektrokardiogram (ECG). Dikembangkan oleh Jiapu Pan dan Willis J. Tompkins pada tahun 1985, algoritma ini telah menjadi referensi utama dalam pemrosesan sinyal ECG digital karena kemampuannya mendeteksi puncak gelombang R (R-peak) secara akurat bahkan dalam kondisi noise yang tinggi (Pan & Tompkins, 1985).

**2.8.1 Tahapan Pemrosesan Algoritma Pan-Tompkins**

Algoritma Pan-Tompkins terdiri dari lima tahapan pemrosesan berurutan yang dirancang untuk menekan noise sekaligus memperkuat karakteristik kompleks QRS:

**1. Band-Pass Filtering**

Tahap pertama adalah filtering untuk menghilangkan noise frekuensi rendah (baseline wander) dan frekuensi tinggi (muscle noise, powerline interference). Filter ini merupakan kombinasi dari low-pass filter dan high-pass filter dengan rentang frekuensi 5-15 Hz. Persamaan transfer function untuk low-pass filter adalah:

$$H_{LP}(z) = \frac{(1 - z^{-6})^2}{(1 - z^{-1})^2}$$

Sedangkan untuk high-pass filter:

$$H_{HP}(z) = \frac{(-1 + 32z^{-16} + z^{-32})}{(1 + z^{-1})}$$

Hasil kombinasi kedua filter menghasilkan band-pass filter dengan passband 5-15 Hz, yang merupakan rentang frekuensi dominan dari kompleks QRS.

**2. Differentiation (Turunan)**

Setelah filtering, sinyal diturunkan untuk menekankan perubahan amplitudo yang cepat (slope) yang merupakan karakteristik kompleks QRS. Operasi diferensiasi diberikan oleh persamaan:

$$y(n) = \frac{1}{8T}[-x(n-2) - 2x(n-1) + 2x(n+1) + x(n+2)]$$

di mana T adalah periode sampling dan x(n) adalah sampel sinyal pada waktu n. Turunan lima titik ini memberikan estimasi slope yang lebih stabil dibandingkan turunan dua titik sederhana.

**3. Squaring (Pengkuadratan)**

Langkah pengkuadratan bertujuan untuk membuat semua nilai menjadi positif dan menekankan perbedaan yang dihasilkan dari turunan. Persamaan pengkuadratan adalah:

$$y(n) = [x(n)]^2$$

Operasi ini menghasilkan sinyal yang memiliki puncak tajam pada lokasi kompleks QRS karena slope yang tinggi pada gelombang R.

**4. Moving Window Integration**

Integrasi jendela bergerak digunakan untuk memperhalus sinyal dan menghasilkan satu puncak per kompleks QRS. Persamaan integrasi adalah:

$$y(n) = \frac{1}{N}[x(n - (N-1)) + x(n - (N-2)) + ... + x(n)]$$

di mana N adalah lebar jendela integrasi. Untuk frekuensi sampling 360 Hz, N biasanya dipilih sekitar 30 sampel (≈83 ms), yang sesuai dengan durasi kompleks QRS normal (80-120 ms).

**5. Adaptive Thresholding**

Tahap akhir adalah penerapan threshold adaptif untuk menentukan lokasi R-peak. Dua threshold digunakan secara bersamaan: threshold pada sinyal terintegrasi dan threshold pada sinyal terfilter. Threshold ini diadaptasi secara dinamis berdasarkan statistik sinyal:

$$THRESHOLD = SPKI + 0.25 \times (SPKF - NPKF)$$

di mana:
- SPKI = Signal Peak (rata-rata puncak sinyal yang terdeteksi sebagai QRS)
- NPKF = Noise Peak (rata-rata puncak noise)
- SPKF = Signal Peak pada sinyal terfilter

**2.8.2 Refractory Period dan Search-Back**

Algoritma juga menerapkan *refractory period* minimal 200 ms setelah deteksi QRS untuk mencegah deteksi ganda pada kompleks QRS yang sama. Selain itu, mekanisme *search-back* diterapkan jika tidak ada QRS terdeteksi dalam interval yang lebih panjang dari yang diharapkan (berdasarkan interval RR sebelumnya), algoritma akan mencari kembali dengan threshold yang lebih rendah.

**2.8.3 Keunggulan Algoritma Pan-Tompkins**

1. **Adaptif terhadap variasi sinyal**: Threshold yang menyesuaikan secara dinamis memungkinkan deteksi akurat pada berbagai kondisi pasien
2. **Robust terhadap noise**: Kombinasi filtering dan pengkuadratan efektif menekan berbagai jenis noise
3. **Efisiensi komputasional**: Operasi sederhana memungkinkan implementasi real-time bahkan pada perangkat dengan sumber daya terbatas
4. **Akurasi tinggi**: Studi menunjukkan sensitivitas >99% pada dataset standar seperti MIT-BIH

**2.9 CNN dan Conv-1D**

*Convolutional Neural Network* (CNN) adalah salah satu jenis deep neural
network yang menggunakan operasi linear matematika antar matriks yang
disebut dengan *convolution* (Bayat et al., 2017). CNN dirancang untuk
mendeteksi pola atau struktur spasial dari data dan telah menjadi metode
utama dalam berbagai bidang seperti pengenalan pola (pattern
recognition), klasifikasi sinyal, dan pengolahan citra medis. Aspek
penting dari CNN adalah kemampuannya untuk melakukan reduksi parameter
secara signifikan dibandingkan dengan *Artificial Neural Network* (ANN),
sehingga menjadikan CNN lebih efisien dan unggul dalam mendeteksi fitur
kompleks dari data masukan (Bayat et al., 2017). Arsitektur CNN terdiri
dari beberapa layer konvolusi, aktivasi, pooling, dan fully connected
yang saling terhubung, sehingga memungkinkan proses ekstraksi fitur
secara bertingkat dari input mentah hingga representasi tingkat tinggi.

Berdasarkan dimensi input yang ditangani, CNN terbagi menjadi:

1.  1-D CNN untuk data sinyal dan deret waktu (time-series), seperti
    sinyal ECG, pengenalan suara (*voice recognition*), dan analisis
    teks.

2.  2-D CNN untuk data gambar atau peta fitur dua dimensi.

3.  3-D CNN untuk data spasio-temporal seperti video atau citra medis 3D
    (MRI, CT-scan).

![](media/image10.png){width="5.377450787401575in"
height="1.9928204286964128in"}

**Gambar 2.6** Arsitektur 1-D CNN

Dalam konteks penelitian ini, kami menggunakan 1-D CNN untuk
menganalisis sinyal ECG satu dimensi dan contoh visualisasinya ada pada
**Gambar 2.6**. Layer konvolusi satu dimensi (Conv1D) secara spesifik
melakukan operasi konvolusi dengan cara menggeser kernel sepanjang
dimensi waktu. Setiap kernel akan mengekstrak fitur lokal seperti pola
QRS kompleks, gelombang P dan T. Dengan parameter seperti kernel size,
stride, dan padding, struktur Conv1D dapat disesuaikan dengan panjang
dan resolusi sinyal input. Penggunaan Conv1D pada model CNN terbukti
efektif dalam mengekstraksi fitur spasial dari sinyal ECG.

**2.9 *Auto Encoder***

*Autoencoder* merupakan jenis jaringan saraf tiruan yang digunakan untuk
melakukan kompresi data (*encoding*) dan kemudian merekonstruksi kembali
data tersebut (*decoding*) dengan tujuan mempertahankan informasi
penting dari input awal. Arsitektur *autoencoder* terdiri atas tiga
bagian utama: *encoder, bottleneck (latent space)*, dan *decoder*
(Hinton & Salakhutdinov, 2006). Dalam konteks sinyal ECG, *autoencoder*
berguna untuk mengurangi dimensi data dan menghilangkan komponen yang
tidak signifikan atau bersifat *noise*. Proses ini sangat membantu dalam
ekstraksi fitur, terutama ketika sinyal mengandung fluktuasi kompleks
akibat variasi fisiologis atau gangguan luar. Dengan menekan sinyal ke
dalam representasi berdimensi rendah, autoencoder dapat memfasilitasi
proses klasifikasi dan meningkatkan efisiensi komputasi. Dalam
penelitian ini, autoencoder berperan sebagai tahap awal pra-pemrosesan
untuk mendukung model CNN-LSTM. Representasi hasil encoder dapat
digunakan sebagai input ke CNN atau digabungkan kembali ke struktur
hybrid. *Autoencoder* juga dapat berfungsi untuk denoising sinyal ECG
secara tidak langsung, karena ia belajar merekonstruksi sinyal ideal
dari input yang mungkin tercampur noise. Penggunaan *autoencoder* pada
sinyal fisiologis telah terbukti mampu meningkatkan akurasi klasifikasi
dalam berbagai studi, termasuk klasifikasi *arrhythmia* dan deteksi
serangan jantung (Zhao et al., 2019; Xiong et al., 2021).

**2.10 Kinerja Model**

Evaluasi kinerja merupakan elemen krusial dalam validasi model
klasifikasi, khususnya dalam konteks medis seperti pendeteksian
*arrhythmia* jantung dari sinyal elektrokardiogram (ECG). Model yang
digunakan perlu dievaluasi secara menyeluruh agar tidak hanya akurat
secara numerik, tetapi juga dapat diandalkan dalam skenario nyata,
termasuk kasus dengan distribusi kelas yang tidak seimbang.

Matrik evaluasi yang digunakan untu menilai performa kinerja model ini
antara lain adalah:

1.  Akurasi : Mengukur proporsi total prediksi yang benar. Meskipun umum
    digunakan, metrik ini kurang representatif pada dataset dengan
    distribusi kelas yang timpang (Luz et al., 2016).

2.  Presisi : menunjukkan seberapa besar proporsi prediksi positif yang
    benar dari seluruh prediksi positif dan ditunjukkan dengan
    **persamaan (4)** :

  -----------------------------------------------------------------------
  **(4)**
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

$$Precision = \frac{TP}{TP + FP}$$

Dimana *True Positive (TP)* adalah jumlah data positif yang berhasil
diklasifikasikan dengan benar oleh model. *False Positive (FP)* adalah
jumlah data negatif yang secara keliru diklasifikasikan sebagai positif
oleh model.

3.  *Recall (Sensitivity)* : mengukur seberapa jauh dan seberapa baik
    model dalam mendeteksi kasus positif yang sebenarnya ditunjukkan
    dengan **persamaan(5)**:

  -----------------------------------------------------------------------
  **(5)**
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

$$\text{Recall} = \frac{TP}{TP + FN}$$

Dimana TP sudah dijelaskan pada poin bagian 2. Sedangkan untuk FN itu
sendiri adalah *False Negative (FN)* dimana jumlah data positif yang
salah itu diklasifikasikan sebagai negatif oleh model.

4.  F1-Score : F1-Score ini adalah rata-rata dari harmonisasi presisi
    dan recall dan digunakan untuk evaluasi pada dataset yang tidak
    seimbang ditunjukkan dengan **persamaan(6)**:

  -----------------------------------------------------------------------
  **(6)**
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

$$F_{1} = \frac{2 \times \left( \text{Precision} \times \text{Recall} \right)}{\text{Precision} + \text{Recall}}$$

5.  *Confusion matrix*

Matriks ini menampilkan perbandingan antara prediksi dan kenyataan dalam
empat komponen: true positives (TP), true negatives (TN), false
positives (FP), dan false negatives (FN) ditunjukkan pada **Gambar
2.7.**

![](media/image11.png){width="2.7136253280839897in"
height="1.8229166666666667in"}

**Gambar 2.7** Confusion Matrix

**2.10.1 Early Stopping dan Regularisasi**

*Early Stopping* adalah teknik regularisasi yang mencegah *overfitting* dengan menghentikan proses pelatihan sebelum model terlalu menyesuaikan diri dengan data pelatihan. Mekanisme ini bekerja dengan memantau metrik kinerja pada data validasi selama pelatihan:

1. **Prinsip Kerja**: Setiap epoch, kinerja model pada data validasi dievaluasi. Jika tidak ada peningkatan selama sejumlah epoch tertentu (disebut *patience*), pelatihan dihentikan.

2. **Pemilihan Model Terbaik**: Model dengan kinerja validasi terbaik disimpan sebagai *checkpoint* dan digunakan sebagai model akhir.

3. **Metrik Pemantauan**: Dalam klasifikasi ECG, AUC-ROC sering digunakan sebagai metrik pemantauan karena lebih robust terhadap ketidakseimbangan kelas dibandingkan akurasi.

**ReduceLROnPlateau**

Teknik *learning rate scheduling* yang menurunkan *learning rate* secara otomatis ketika metrik validasi stagnan:

$$lr_{new} = lr_{current} \times factor$$

di mana *factor* biasanya bernilai 0.1-0.5. Teknik ini memungkinkan model melakukan penyesuaian bobot yang lebih halus saat mendekati konvergensi.

**2.10.2 Distribution Shift dalam Pembagian Data Record-Wise**

Fenomena *distribution shift* terjadi ketika distribusi kelas pada data validasi berbeda signifikan dari data pelatihan. Dalam konteks klasifikasi ECG dengan pembagian *record-wise*:

1. **Penyebab**: Pembagian berdasarkan rekaman pasien (bukan per-detak) dapat menghasilkan distribusi kelas yang tidak seimbang antar subset. Beberapa rekaman pasien mungkin mengandung lebih banyak aritmia dibandingkan yang lain.

2. **Manifestasi**: Model menunjukkan:
   - Peningkatan loss pada training tetapi penurunan pada validasi
   - AUC training sangat tinggi (>0.99) tetapi AUC validasi lebih rendah
   - *Early stopping* terpicu di epoch awal

3. **Implikasi**: Ini bukan *overfitting* dalam pengertian tradisional, melainkan model kesulitan mengeneralisasi ke distribusi yang berbeda. Model yang dihentikan lebih awal seringkali memiliki generalisasi yang lebih baik karena belum terlalu menyesuaikan dengan distribusi spesifik data pelatihan.

4. **Mitigasi**: 
   - Stratified record-wise split (mengelompokkan rekaman berdasarkan rasio abnormal sebelum pembagian)
   - Penggunaan class weights yang dinamis
   - Validasi pada data yang benar-benar tidak terlihat (seperti record 119 yang dikecualikan dari training)

**2.11 Arsitektur Context-Aware CNN1D untuk Klasifikasi ECG**

Arsitektur Context-Aware CNN1D yang digunakan dalam penelitian ini dirancang khusus untuk mengekstraksi fitur dari jendela konteks 7 detak jantung. Berbeda dengan CNN1D konvensional yang memproses sinyal ECG secara keseluruhan, pendekatan ini memperlakukan setiap detak dalam jendela konteks sebagai kanal input terpisah.

**2.11.1 Struktur Layer Konvolusi**

Model terdiri dari tiga blok konvolusi berurutan dengan jumlah filter yang meningkat:

1. **Blok Konvolusi 1**: 
   - Input: 7 kanal × 200 sampel (7 detak)
   - Conv1D: 32 filter, kernel size 3, padding 1
   - BatchNorm1D(32)
   - ReLU activation
   - MaxPool1D(2) → Output: 32 × 100

2. **Blok Konvolusi 2**:
   - Conv1D: 64 filter, kernel size 5, padding 2
   - BatchNorm1D(64)
   - ReLU activation
   - MaxPool1D(2) → Output: 64 × 50

3. **Blok Konvolusi 3**:
   - Conv1D: 128 filter, kernel size 7, padding 3
   - BatchNorm1D(128)
   - ReLU activation
   - MaxPool1D(2) → Output: 128 × 25

**2.11.2 Global Average Pooling**

Setelah layer konvolusi, *Global Average Pooling* diterapkan untuk mereduksi dimensi spasial menjadi vektor fitur berdimensi 128. Teknik ini lebih robust dibandingkan *Flatten* tradisional karena:

1. Mengurangi jumlah parameter secara signifikan
2. Mencegah *overfitting*
3. Memberikan invariansi terhadap translasi kecil dalam sinyal

**2.11.3 Classifier Head**

Classifier terdiri dari:
- Linear: 128 → 64
- ReLU activation
- Dropout(0.5)
- Linear: 64 → 2 (output classes)

Total parameter model adalah sekitar 77,314, yang relatif kecil dan memungkinkan inferensi cepat pada perangkat dengan sumber daya terbatas.

**2.11.4 Alasan Pemilihan Arsitektur**

Arsitektur ini dipilih karena:
1. **Efisiensi**: Parameter yang relatif sedikit memungkinkan training dan inferensi cepat
2. **Konteks Temporal**: Memperlakukan 7 detak sebagai 7 kanal memungkinkan model mempelajari hubungan antar-detak
3. **Hierarki Fitur**: Kernel size yang berbeda (3, 5, 7) menangkap pola pada skala temporal yang berbeda

**2.12 Teknik Anti-Overfitting dalam Deep Learning**

Overfitting adalah masalah umum dalam deep learning di mana model terlalu menyesuaikan diri dengan data pelatihan sehingga tidak dapat mengeneralisasi ke data baru. Beberapa teknik anti-overfitting yang diterapkan dalam penelitian ini:

**2.12.1 Dropout**

Dropout adalah teknik regularisasi di mana sebagian neuron dinonaktifkan secara acak selama pelatihan dengan probabilitas $p$. Dalam penelitian ini, $p = 0.5$ digunakan pada classifier head. Secara matematis, output neuron dengan dropout diberikan oleh:

$$y_i = r_i \cdot a_i, \quad r_i \sim \text{Bernoulli}(1-p)$$

di mana $a_i$ adalah aktivasi asli dan $r_i$ adalah variabel acak Bernoulli (Srivastava et al., 2014).

**2.12.2 Batch Normalization**

Batch Normalization menormalisasi input setiap layer dengan statistik mini-batch selama pelatihan:

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

di mana $\mu_B$ dan $\sigma_B^2$ adalah rata-rata dan varians mini-batch. Teknik ini mempercepat konvergensi dan memiliki efek regularisasi ringan (Ioffe & Szegedy, 2015).

**2.12.3 Weight Decay (L2 Regularization)**

Weight Decay menambahkan penalti pada loss function berdasarkan magnitude bobot:

$$L_{total} = L_{CE} + \lambda \sum_{i} w_i^2$$

di mana $L_{CE}$ adalah Cross-Entropy Loss, $\lambda$ adalah koefisien regularisasi (0.0001 dalam penelitian ini), dan $w_i$ adalah bobot model.

**2.12.4 Gradient Clipping**

Gradient Clipping membatasi norm gradien untuk mencegah *exploding gradients*:

$$g' = \min\left(1, \frac{\theta}{||g||}\right) \cdot g$$

di mana $\theta$ adalah threshold (1.0 dalam penelitian ini) dan $g$ adalah vektor gradien.

**2.13 AdamW Optimizer**

AdamW adalah varian dari Adam optimizer yang memisahkan weight decay dari pembaruan gradien (Loshchilov & Hutter, 2019). Pembaruan parameter diberikan oleh:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$w_t = w_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda w_{t-1} \right)$$

AdamW lebih efektif daripada Adam standar karena weight decay yang terpisah menghasilkan generalisasi yang lebih baik pada model deep learning.

**2.14 Area Under the Curve - Receiver Operating Characteristic (AUC-ROC)**

AUC-ROC adalah metrik evaluasi yang mengukur kemampuan model dalam membedakan antara kelas positif dan negatif pada berbagai threshold. Kurva ROC memplot:
- *True Positive Rate* (TPR/Recall) pada sumbu y
- *False Positive Rate* (FPR) pada sumbu x

AUC-ROC memiliki nilai antara 0 dan 1, di mana:
- 0.5 = Random classifier
- 1.0 = Perfect classifier

AUC-ROC lebih informatif daripada akurasi untuk dataset tidak seimbang karena mempertimbangkan seluruh range threshold keputusan.

**2.15 *Open Neural Network Exchange* (ONNX)**

Dalam pengembangan model *machine learning,* seringkali terdapat
kebutuhan untuk memindahkan model dari data training ke *deployment*
yang memiliki karakteristik hardware yang berbeda. *Open Neural Network
Exchange* (ONNX) ini adalah format yang dirancang untuk mengatasi
masalah ini dengan menyediakan representasi model yang sergama pada
berbagai framework (Bai et al., 2019). Format ini memungkingkan model
yang telah dilatih menggunakan *library* seperti PyTorch, TensorFlow dan
Scikit-learn untuk diekspor menjadi format biner. Keuntungan utama dari
penggunaan ONNX ini adalah kemampuan optimasi graf komputasi secara
otomatis yang dapat mempercepat proses prediksi dan penggunaan memori
yang lebih sedikit dibandingkan menjalankan model pada framework
aslinya. Efisiensi ini menjadikan model ONNX adalah solusi yang tepat
untuk implementasi sistem pada perangkat yang ada.

**2.12 *Unified Modeling Language* (UML) dan *Use Case Diagram***

Untuk visualisasi arsitektur dan interaksi fungsional pada sistem yang
dikembangkan, penelitian ini menggunakan standar *Unified Modeling
Language* (UML). Menurut Pressman (2014), UML adalah bahasa pemodelan
standar yang digunakan untuk mendokumentasikan, menspesifikasikan, dan
membangun sistem *software* berorientasi objek. Dalam kontek perancangan
*interface* pada penelitian ini, fokus pemodelan dilakukan dengan
menggunakan *Use Case Diagram.*

Diagarm ini digunakan untuk menggambarkan hubungan antara sistem dan
lingkungan eskternalnya sertaa mendefinisikan apa yang dilakukan oleh
sistem tanpa harus menjelaskan secara terperinci bagaimana sistem
melakukannya. Diagram ini terdiri dari 4 komponen utama. Pertama, Aktor
yang dilambangkan dengan gambar orang yang merepresentasikan entitas
eksternal yang berinteraksi dengan sistem. Kedua, *Use Case* yang
dilambangkan dengan bentuk oval yang merepresentasikan fungsional secara
spesifik atau layanan yang disediakan oleh sistem. Ketigas, Asosiasi
yang berupa garis penghubung yang menampilkan hubungan aktif antara
aktor dan use case. Terakhir, batasan sistem yang dilambangkan dengan
kota persegi panjang yang membungkus seluruh *use case* untuk memisahkan
ruang lingkup internal sistem dari lingkungan eksternalnya. Secara umum,
*Use Case Diagram* dapat divisualisasikan pada gambar berikut.

![A screenshot of a computer AI-generated content may be
incorrect.](media/image12.png){width="3.9054593175853016in"
height="2.636363735783027in"}

**2.13 Related works**

Penelitian ini bukanlah yang pertama yang mencoba memprediksi
*arrhythmia* jantung. Sejumlah penelitian sebelumnya telah
mengeksplorasi domain ini menggunakan berbagai teknik pembelajaran
mesin, mulai dari jaringan saraf dasar hingga arsitektur yang lebih
canggih seperti CNN-LSTM. Beberapa karya terkait ini dirangkum dalam
**Tabel 2.1**.

Tabel 2.1 Related Works

+------------------+------------------+---------+----------+--------+
| *Author*         | *Description*    | *       | *        | *Acc   |
|                  |                  | Method* | Dataset* | uracy* |
+:=================+:=================+:========+:=========+:=======+
| (Ebrahimzadeh et | *Single          | *Neural | Holter   | 99.7%  |
| al., 2014)       | patient's HRV    | N       |          |        |
|                  | and  One*        | etwork* |          |        |
|                  |                  |         |          |        |
|                  | *minute prior    |         |          |        |
|                  | VF*              |         |          |        |
+------------------+------------------+---------+----------+--------+
| (Ebrahimzadeh et | *Single          | *Neural | Holter   | 90.3%  |
| al., 2014)       | patient's*       | N       |          |        |
|                  |                  | etwork* |          |        |
|                  | *HRV and three   |         |          |        |
|                  | minutes prior    |         |          |        |
|                  | VF*              |         |          |        |
+------------------+------------------+---------+----------+--------+
| (Ebrahimzadeh et | *Single          | *Neural | Holter   | 83.9%  |
| al., 2014)       | patient's*       | N       |          |        |
|                  |                  | etwork* |          |        |
|                  | *HRV and four    |         |          |        |
|                  | minutes prior    |         |          |        |
|                  | VF*              |         |          |        |
+------------------+------------------+---------+----------+--------+
| (Zheng et al.,   | *Classifying     | *CNN-   | MIT-BIH  | 99%    |
| 2020)            | non-SCD*         | LSTM*   |          |        |
+------------------+------------------+---------+----------+--------+
| (Warrick &       | *Classifying     | *CNN-   | P        | 83%    |
| Nabhan Homsi,    | non- SCD*        | LSTM*   | hysionet |        |
| 2017)            |                  |         |          |        |
|                  |                  |         | C        |        |
|                  |                  |         | hallenge |        |
|                  |                  |         | 2017     |        |
+------------------+------------------+---------+----------+--------+
| (Oh et al.,      | *Classifying     | *CNN-   | MIT-BIH  | 99%    |
| 2018)            | non- SCD*        | LSTM*   |          |        |
+------------------+------------------+---------+----------+--------+
| (Shi et al.,     | *Classifying     | *CNN-   | MIT-BIH  | 98%    |
| 2019)            | non- SCD*        | LSTM*   |          |        |
+------------------+------------------+---------+----------+--------+

Sebagai contoh, Ibtehaz *et al*. (2019) melaporkan akurasi klasifikasi
yang mengesankan, yaitu hampir 99% ketika membedakan antara kejadian
fibrilasi ventrikel (VF) dan non-VF. Namun, model mereka secara khusus
dirancang untuk mendeteksi kejadian hanya beberapa detik sebelum
terjadinya kematian jantung mendadak (SCD), sehingga membatasi
kesempatan untuk intervensi medis yang tepat waktu. Peneliti lain,
termasuk Ebrahimzadeh *et al*. (2014), Lee *et al*. (2016), dan Joo *et
al*. (2010), mengadopsi metodologi prapemrosesan dan pemilihan fitur
yang serupa. Mereka terutama berfokus pada indikator *Heart Rate
Variability* (HRV), khususnya interval RR, yang mewakili waktu antara
puncak R yang berurutan dalam sinyal EKG. Ukuran statistik seperti
rata-rata dan deviasi standar interval RR, bersama dengan variabilitas
periode pernapasan, diekstraksi dan dievaluasi menggunakan nilai-p untuk
menentukan signifikansi statistiknya. Hanya fitur yang paling
berpengaruh yang dipilih sebagai masukan untuk masing-masing model.
Perbedaan utama di antara penelitian-penelitian ini terletak pada
arsitektur jaringan saraf yang digunakan. Lee *et al*. (2016)
mengimplementasikan jaringan saraf tiruan dengan satu dan dua lapisan
tersembunyi, masing-masing terdiri dari lima neuron. Sebaliknya, Joo *et
al*. (2010) menggunakan arsitektur yang lebih dalam dengan dua lapisan
tersembunyi, masing-masing berisi 25 neuron, yang menghasilkan
peningkatan waktu pelatihan dan kompleksitas komputasi. Sementara itu,
Ebrahimzadeh *et al*. (2014) membandingkan kinerja Multi-Layer
Perceptron (MLP) dengan pengklasifikasi K-Nearest Neighbor (KNN).

Pendekatan alternatif diusulkan oleh Farhadi *et al*. (2017), yang
menggunakan Algoritma Genetika (GA) sebagai teknik optimasi. Daripada
mengonfigurasi arsitektur model secara manual, GA digunakan untuk secara
otomatis mengidentifikasi kombinasi fitur dan struktur jaringan yang
optimal, menggunakan set data yang sama dan fitur berbasis HRV.
Khususnya, metode ini memungkinkan pengecualian fitur-fitur domain
frekuensi, yang sering kali sulit untuk diekstraksi dan
diinterpretasikan. Lebih lanjut, Rajput *et al*. (2019) menekankan bahwa
sinyal EKG mentah dapat secara efektif dimasukkan ke dalam model
pembelajaran mendalam, sehingga memungkinkan model untuk mempelajari
fitur-fitur yang relevan secara mandiri tanpa memerlukan rekayasa fitur
secara manual. Hal ini mendukung gagasan tentang jalur pembelajaran
ujung ke ujung untuk deteksi *arrhythmia*. Dalam penelitian lain, Verma
dan Dong (2016) mencapai lebih dari 94% akurasi dalam membedakan ritme
VF dari non-VF menggunakan fitur yang diturunkan dari HRV dan
pengklasifikasi hutan acak. Berdasarkan upaya ini, penelitian saat ini
bertujuan untuk memasukkan takikardia ventrikel (VT) sebagai kelas
tambahan yang akan diprediksi. Secara kolektif, model CNN-LSTM telah
menunjukkan efektivitas yang substansial dalam mendeteksi dan
mengklasifikasikan irama jantung. Meskipun sebagian besar penelitian
sebelumnya memanfaatkan arsitektur ini untuk ekstraksi fitur dan
klasifikasi VT/VF, mereka umumnya tidak menyertakan komponen prediksi
atau peramalan-sehingga membatasi potensi intervensi dini. Sebagai
tanggapan, penelitian ini mengusulkan integrasi CNN dan LSTM untuk tidak
hanya mengklasifikasikan tetapi juga meramalkan kemungkinan terjadinya
VF atau VT dan memberi waktu untuk tindakah pencegahan.