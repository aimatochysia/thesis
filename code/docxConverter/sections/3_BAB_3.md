**BAB 3****

**METODE PENELITIAN**

1.  **Pendekatan penelitian**

Penelitian ini mengadopsi pendekatan kuantitatif dengan desain
eksperimental, mengikuti kerangka kerja CRISP-DM (*Cross-Industry
Standard Process for Data Mining*)pada Gambar 6.. CRISP-DM adalah model
standar industri yang menyediakan panduan terstruktur untuk proyek
berbasis data, memastikan pendekatan yang sistematis dan iteratif.
Kerangka ini dipilih karena kemampuannya untuk mengelola kompleksitas
proyek *deep learning* secara efektif, dari pemahaman masalah bisnis
hingga implementasi solusi. Enam fase utama dalam CRISP-DM, yang akan
diterapkan dalam penelitian ini pada **Gambar 3.1**, meliputi:

![](media/image13.png){width="4.040277777777778in" height="3.15625in"}

**Gambar 3.1** CRISP-DM Model

Pendekatan penelitian ini dimulai dengan tahap *Business Understanding*,
di mana tujuan utama dari penelitian dipahami secara mendalam dari
perspektif domain medis, khususnya dalam konteks kebutuhan akan
diagnosis *Arrhythmia* yang akurat. Diagnosis yang tepat dan cepat
terhadap *cardiac Arrhythmia* sangat penting, mengingat keterkaitannya
dengan risiko *sudden cardiac arrest* dan *ischemic stroke* (Al-Zaiti et
al., 2020). Pemahaman yang komprehensif terhadap latar belakang klinis
menjadi dasar dalam merancang sistem pendeteksian berbasis *machine
learning* yang dapat mendukung keputusan medis secara objektif dan
efisien.

Tahap selanjutnya adalah *Data Understanding*, yang mencakup proses awal
pengumpulan data *electrocardiogram (ECG)* dan eksplorasi terhadap
karakteristik sinyal yang dikumpulkan. Proses ini bertujuan untuk
mengidentifikasi struktur data, distribusi nilai, serta mendeteksi
potensi masalah seperti *missing values*, *signal noise*, atau gangguan
artefak yang mungkin mempengaruhi kualitas data (Zahid et al., 2020).
Selain itu, dilakukan juga pengamatan terhadap label anotasi dan
korelasinya dengan jenis-jenis *Arrhythmia*.

Setelah karakteristik data dipahami, proses dilanjutkan ke tahap *Data
Preparation*. Pada fase ini, dilakukan serangkaian *pre-processing*
seperti *signal segmentation* berdasarkan titik *R-peak*,
*normalization* untuk memastikan keseragaman skala amplitudo, serta
penanganan *class imbalance* dengan metode *oversampling* atau
*undersampling* untuk menjaga proporsi representatif antar kelas
(Alinsaif, 2024). Tujuan utama dari tahap ini adalah untuk menghasilkan
data bersih dan siap pakai dalam pelatihan model *deep learning*.

Tahap krusial berikutnya adalah *Modeling*, di mana arsitektur *deep
neural network* yang dipilih adalah *1D-Convolutional Neural Network
(1D-CNN)*. Arsitektur ini telah terbukti efektif dalam menangkap pola
lokal dari sinyal sekuensial seperti ECG karena kemampuannya dalam
mengenali fitur spasial-temporal dari sinyal berdimensi satu (Yildirim,
2018). Pada fase ini juga dilakukan penentuan *hyperparameters* seperti
jumlah *filters*, ukuran *kernel*, *learning rate*, dan jumlah *epochs*
untuk memastikan model memiliki *generalization performance* yang
optimal.

Setelah model selesai dilatih, dilakukan proses *Evaluation* terhadap
performa model menggunakan metrik evaluasi seperti *Precision*,
*Recall*, *F1-Score*, *Accuracy*, dan *Area Under the Curve - Receiver
Operating Characteristic (AUC-ROC)*. Evaluasi ini bertujuan untuk
menilai seberapa baik model dalam membedakan antara sinyal *normal* dan
*abnormal*, serta memastikan bahwa model tidak overfitting terhadap data
latih (Eleyan & Alboghbaish, 2024). Meskipun implementasi penuh sistem
*real-time* tidak menjadi fokus utama dalam penelitian ini, tahap akhir
yaitu *Deployment* tetap dibahas sebagai bagian dari proyeksi aplikasi
nyata dari model yang dikembangkan. Diskusi pada tahap ini mencakup
kemungkinan integrasi sistem ke dalam perangkat *wearable* atau sistem
monitoring rumah sakit, serta tantangan dan peluang dalam penerapan
model di lingkungan klinis maupun non-klinis (Panwar et al., 2025).

**3.2 Tahapan penelitian**

Proses penelitian akan dibagi menjadi beberapa tahapan utama yang
terstruktur, dimulai dari persiapan data hingga evaluasi model, seperti
yang digambarkan pada **Gambar 3.2**.

![](media/image14.png){width="5.71875in" height="2.7708333333333335in"}

**Gambar 3.2** Tahapan penelitian

Penelitian ini dirancang dengan pendekatan sistematis yang terdiri dari
lima fase utama yang saling mendukung, yaitu: perolehan data, modifikasi
data, persiapan data, pelatihan model, serta evaluasi hasil. Seluruh
alur kerja ini bertujuan untuk membangun sistem klasifikasi *arrhythmia*
jantung berbasis sinyal ECG menggunakan metode pembelajaran mendalam
dengan klasifikasi dua kelas (normal vs abnormal). Pada tahap awal,
dilakukan eksplorasi literatur untuk menelusuri metode dan praktik
terbaik dalam deteksi *arrhythmia* berbasis sinyal fisiologis. Sumber
rujukan mencakup jurnal akademik, dokumen konferensi, hingga repositori
publik seperti PhysioNet dan Kaggle (Goldberger et al., 2000; Yoon,
2023). Berdasarkan hasil kajian tersebut, MIT-BIH *Arrhythmia* Database
dipilih sebagai dataset utama karena telah diakui secara luas dalam
studi klinis dan memiliki dokumentasi yang baik (Moody & Mark, 2001).

Tahapan berikutnya berfokus pada proses penyesuaian struktur data agar
kompatibel dengan kebutuhan pemodelan deep learning. Beberapa rekaman
ECG dipilih secara acak namun mewakili variasi sinyal dan distribusi
label yang seimbang. Sinyal berdurasi panjang kemudian dipotong menjadi
segmen-segmen pendek yang mencerminkan satu siklus jantung, dengan pusat
pada titik R sebagai referensi fisiologis utama. Proses ini dilakukan
agar masukan model memiliki struktur temporal yang konsisten (Acharya et
al., 2017). Selain itu, label awal yang terdiri dari banyak kelas
dikonsolidasikan menjadi dua kategori utama: normal dan abnormal, untuk
menyederhanakan proses klasifikasi dan meningkatkan efisiensi
pembelajaran model (Panwar et al., 2025). Sebagai tambahan, disiapkan
juga satu subset khusus yang berisi data prediksi acak yang tidak
melalui tahap pra-pemrosesan penuh. Tujuan dari subset ini adalah untuk
mengevaluasi kemampuan model dalam menghadapi data nyata yang tidak
terstruktur atau mengandung noise (Faust et al., 2018).

Pada tahap persiapan data, dilakukan berbagai teknik eksplorasi dan
pembersihan data. Analisis awal dilakukan melalui *exploratory data
analysis* (EDA) guna meninjau distribusi fitur, mendeteksi nilai kosong,
serta mengidentifikasi pola-pola data yang menyimpang (Tukey, 1977; Das
& Behera, 2021). Data yang menunjukkan nilai ekstrem berdasarkan
statistik Z-score (di luar ±3 standar deviasi) dihapus dari dataset guna
menghindari distorsi dalam proses pembelajaran (Iglewicz & Hoaglin,
1993). Komposisi kelas juga dianalisis, dan bila ditemukan ketimpangan
jumlah antara kelas normal dan abnormal, diterapkan strategi
penyeimbangan seperti *resampling* untuk menjaga netralitas pelatihan
(He & Garcia, 2009). Guna memastikan bahwa sinyal dalam kelas abnormal
tetap mencerminkan keragaman fisiologis, digunakan metode *K-Means
Clustering* untuk memverifikasi keberagaman bentuk gelombang yang masih
tersisa dalam dataset (Jain, 2010). Langkah lanjutan berupa reduksi
dimensi dilakukan menggunakan PCA (Principal Component Analysis), yang
digunakan untuk memahami struktur spasial data dan menyaring korelasi
antar fitur dengan cara yang efisien (Jolliffe & Cadima, 2016; Abdi &
Williams, 2010).

Pada fase pelatihan model, data dibagi ke dalam dua bagian utama:
*training set* dan *evaluation set*, dengan pembagian stratified guna
mempertahankan rasio kelas yang proporsional. Model utama yang
diimplementasikan adalah 1D Convolutional Neural Network (1D-CNN), yang
dirancang untuk mengekstraksi karakteristik lokal dari sinyal deret
waktu seperti ECG. Struktur jaringan mencakup beberapa lapisan konvolusi
dan pooling, diakhiri dengan lapisan fully-connected untuk klasifikasi
akhir. Proses pelatihan dilakukan dalam beberapa epoch dengan optimasi
menggunakan algoritma Adam dan fungsi aktivasi ReLU di lapisan
tersembunyi serta sigmoid di output layer (Zhao et al., 2019; Hannun et
al., 2019). Selain itu, checkpoint diterapkan selama pelatihan untuk
menyimpan model dengan performa terbaik berdasarkan metrik validasi.

Fase akhir dalam metodologi adalah evaluasi kinerja model. Evaluasi
dilakukan dalam dua skenario: pertama, menggunakan *evaluation set* yang
telah melalui pra-pemrosesan lengkap, dan kedua, dengan *prediction
dataset* yang tidak dibersihkan. Performa model dinilai menggunakan
metrik klasifikasi seperti akurasi, presisi, recall, F1-score, serta
AUC-ROC untuk memberikan penilaian yang menyeluruh (Chicco & Jurman,
2020). Evaluasi ganda ini tidak hanya mengukur efektivitas model dalam
kondisi ideal, tetapi juga menguji ketahanannya dalam menghadapi sinyal
yang menyerupai kondisi nyata di lapangan.

**3.3 Pemilihan dan Pemilahan Dataset**

Pada tahapan ini dijelaskan bagaimana prosedur sistematis didalam
pemilihan serta pemilahan dataset yang diterapkan untuk dapat mengubah
data mentah dari sinyal menjadi dataset yang terstruktur yang nantinya
akan digunakan sebagai model training dan model test.

**3.3.1 Sumber dan Karakteristik Data Mentah**

> Dataset utama pada penilitian ini berasal dari MIT-BIH Arrhythmia
> Databse yang diakses melalu repositori Kaggle. Data ini adalah data
> yang digunakan dalam penilitian aritmia yang terdiri dari rekaman
> elektrokardiogram(EKG) dalam dua saluran dari 48 pasien yang berbeda.
> Setiap rekaman terdiri dari dua file utama yang berpasangan:

a\. File Sinyal(.csv) File ini berisi data waktu dari amplitude sinyal
EKG yang direkamn secara terus menerus dengan frekuensi sampling 360Hz.
Fokus pada penilitian ini memakai saluran MLII (Modified Limb Lead II)
karena memiliki gelombang QRS yang cukup jelas.

b\. File Anotasi(.txt) File ini berisi data klinis hasil diagnosis akhli
kardiologi. File ini memiliki informasi berupa Sample sebagai indeks
Lokasi waktu terjadinya puncak gelombang (R-peak) serta *Type* yang
bertugas sebagai label diagnosis untuk setiap detail dari detak jantung.

Contoh visualisasi dari dataset mentah yang ada terdapat pada Gambar.

![](media/image15.png){width="5.191475284339457in"
height="3.0915594925634298in"}

**3.3.2 Segmentasi dan Ekstraksi Sinyal**

> Data sinyal mentah yang bersifat kontinu tidak dapat langsung
> digunakan sebagai input model klasifikasi karena model membutuhkan
> input yang terstandarisasi. Oleh karena itu, dilakukan proses
> segmentasi untuk memecah sinyal panjang menjadi segmen detak jantung
> individual yang berpusat pada gelombang R (*R-peak*). Proses dan
> tahapan transformasi sinyal tersebut antara lain:

1\. Deteksi R-Peak:Algoritma Pan-Tompkins digunakan untuk mendeteksi
lokasi temporal dari puncak gelombang R sebagai titik referensi pusat
setiap detak. Algoritma ini mencakup proses *bandpass filtering* (5-15
Hz) untuk meminimalisir *noise* otot dan *baseline wander* sebelum
melakukan diferensiasi dan integrasi sinyal untuk menemukan puncak QRS
yang valid.

2\. Pemotongan Sinyal (*Windowing*): Berbeda dengan pendekatan simetris
konvensional, penelitian ini menerapkan segmentasi asimetris dengan
total panjang jendela (window size) sebesar 200 sampel (setara dengan
±556 ms pada frekuensi 360 Hz). Pemilihan angka 200 sampel ini bertujuan
untuk menangkap kompleks PQRST secara utuh. Pembagian area potong adalah
sebagai berikut:

- Pre-R (90 Sampel): Mengambil 90 titik data sebelum puncak R. Durasi
  ini (±250 ms) cukup untuk menangkap gelombang P dan interval PR secara
  lengkap, yang sering bermulai 120-200 ms sebelum puncak R.

- Post-R (110 Sampel): Mengambil 110 titik data setelah puncak R. Durasi
  ini (±306 ms) dialokasikan lebih panjang untuk memastikan segmen ST
  dan gelombang T tertangkap sepenuhnya, karena kelainan pada
  repolarisasi ventrikel sering terjadi pada fase ini.

1.  Penanganan Tepi (*Padding*): Untuk detak jantung yang berada di awal
    atau akhir rekaman di mana jumlah sampel tidak mencukupi (kurang
    dari 90 sebelum atau 110 sesudah R-peak), dilakukan teknik
    *zero-padding* untuk memastikan dimensi output tetap konsisten di
    angka 200 sampel.

> Hasil dari segmentasi dan ekstrasi dari sinyal dapat terlihat pada
> gambar..
>
> ![A graph of a graph AI-generated content may be
> incorrect.](media/image16.png){width="5.71875in"
> height="2.4034722222222222in"}

**3.3.3. Pembentukan *Window Construction***

> Berbeda dengan metode konvensional yang mengklasifikasikan setiap
> detak secara terisolasi, penelitian ini menerapkan pendekatan
> *Context-Aware* untuk menangkap pola temporal antar-detak. Data input
> dibentuk menjadi jendela konteks (*context window*) dengan spesifikasi
> sebagai berikut:

1.  Ukuran Jendela (Window Size): Setiap input model terdiri dari 7
    detak berurutan.

    a.  Komposisi: 3 detak sebelumnya (*previous beats*), 1 detak pusat
        (*center beat*), dan 3 detak setelahnya (*subsequent beats*).

    b.  Tujuan: Konfigurasi ini memungkinkan model untuk mempelajari
        pola ritme jangka panjang, seperti pada kasus *Bigeminy* (setiap
        detak kedua abnormal) atau *Trigeminy*, yang sulit dideteksi
        jika hanya melihat satu detak saja.

2.  Penentuan Label: Label klasifikasi (Normal/Abnormal) untuk satu
    jendela konteks ditentukan berdasarkan label dari detak pusat
    (urutan ke-4 dalam jendela). Detak-detak di sekitarnya hanya
    berfungsi sebagai fitur pendukung (konteks) dan tidak mempengaruhi
    label target secara langsung.

> Hasil dari pembentukan *windows construction* ini dapat dilihat pada
> gambar
>
> ![A graph of a heart rate AI-generated content may be
> incorrect.](media/image17.png){width="5.71875in" height="2.2875in"}

**3.3.4 Dataset Final dan Labeling**

> Hasil akhir dari proses segmentasi dan pembentukan jendela adalah
> himpunan data tensor dengan dimensi (Jumlah Sampel, 7, 200).

a.  Dimensi 7: Merepresentasikan *channels* yang berisi 7 detak dalam
    satu jendela.

b.  Dimensi 200: Merepresentasikan panjang sampel (*sequence length*)
    per detak.

> Target klasifikasi disederhanakan menjadi dua kelas (biner) sesuai
> kebutuhan klinis untuk *screening* awal:

a.  Kelas 0 (Normal): Mencakup anotasi \'N\' (*Normal Sinus Rhythm*).

b.  Kelas 1 (Abnormal): Menggabungkan seluruh jenis aritmia lainnya
    (termasuk *Premature Ventricular Contraction*, *Atrial Premature
    Beat*, *Bundle Branch Block*, dll).

> Setelah melalui proses yang ada, terbentuklah berapa jumlah dataset
> yang ada. Dari semua proses yang ada, terdapat total 108090 detak
> jantung dengan rincian 73443 rekaman detak jantung secara normal dan
> 34647 rekaman jantung abnormal dengan presentasi sebanyak 67.9% normal
> dan 32.1% rekaman detak jantung tidak normal.

**3.4 Pra-pemrosesan dan Analisis Kualitas Data**

Setelah dataset telah terkumpul dan terbentuk pada satu file, tahapan
berikutnya adalah analisis mengenai karakteristik sinyal dan menentukan
penanganan kualitas data. Pada sub-bab ini akan dijabarkan bagaimana
representasi dari data serta evaluasi terhadap outlier untuk memastikan
informasi yang masuk ke dalam model yang digunakan.

**3.4.1 Karakteristik fisik dan dimensi data**

> Data yang tersimpan pada matriks dataset dijabarkan mengenai makna
> dari setiap sumbu data berdasarkan spesifikasi teknis MIT-BH. Data
> yang terekam pada dataset terbagi menjadi 2 yaitu:
>
> a\. Sumbu Waktu ( Time Axis ) Sumbu horizontal merepresentasikan
> urutan pengambilan data. Data rekaman digital ini memiliki frekuensi
> sampling sebesar 360Hz. Oleh karena itu, interval waktu antar titik
> data dapat dihitung dengan persamaan 7 :

$$\Delta t = \ \frac{1}{fs} = \frac{1}{360}\  \approx 2.78ms$$

> Ini menandakan setiap kolom memiliki jarak waktu sekitar 2.78
> milidetik. Total durasi window time untuk satu baris data ditunjukkan
> oleh persamaan 8

$$T_{window} = 188\  \times 2.78ms\  \approx 522.6ms$$

> Durasi sekitar 0,52 detik ini dipilih karena ideal dalam menangkap
> satu detak jantung secara utuh tanpa ada banyak gangguan dari detak
> lainnya
>
> b\. Sumbu Amplitudo ( Amplitudo Axis ) Sumbu vertikal pada
> merepresentasikan amplitudo pada setiap detak jantung. Data yang
> digunakan pada penelitian ini adalah nilai mentah (raw values) yang
> diperoleh dari konversi nilai ADC dengan resolusi 11-bit. Resolusi ini
> membagi tegangan yang masuk dengan total tegangan sebesar 10mV menjadi
> 2048 level diskrit. Oleh karena itu, resolusi tegangan per satuan
> (LSB) dapat dihitung pada persamaan 9.
>
> $$LSB = \frac{Rentang\ Tegangan\ Total}{Level\ Diskrit} = \ \frac{10mV}{2048} \approx 0,0048mV/unit$$
>
> Dikarenakan karakteristik sinyal pada dataset ini mempunyai titik
> tengah pada nilai 1024, maka konversi dari nilai raw digital
> ($X_{raw})$ menjadi satuan milivolt (mV) pada diformulasikan pada
> persamaan 10/
>
> $$mV = \left( X_{raw} - 1024\  \right)\  \times 0,0048$$
>
> Berdasarkan persamaan diatas, rentang nilai mentah yang diamati pada
> dataset yaitu pada rentang 910 -- 1200 yang merepresentasikan variasi
> amplitudo. Maka, nilai 910 setara dengan -0,56mV sedangkan 1200 setara
> dengan +0,86mV dimana rentang ini mencakup naik turun sinyal detak
> jantung dari baseline hingga R-Peak pada satu siklus detak jantung.

**3.4.2 Pemisahan Data**

> Penelitian ini tidak menggunakan pemisahan acak per detak (*random
> beat-wise split*) karena metode tersebut berisiko mencampurkan detak
> dari pasien yang sama ke dalam data latih dan data uji, yang
> menyebabkan bias evaluasi yang terlalu optimis. Oleh karena itu,
> diterapkan strategi lain yaitu Record-wise Split yang berupa :

1.  Metode Pembagian: Dataset dibagi berdasarkan Nomor Rekaman (Record
    ID) pasien. Artinya, jika seorang pasien masuk ke dalam data latih,
    maka seluruh detak jantungnya hanya berada di data latih dan tidak
    akan muncul di data validasi maupun uji. Hal ini mensimulasikan
    kondisi dunia nyata di mana model akan memproses pasien baru yang
    belum pernah dilihat sebelumnya.

2.  Proporsi Pembagian: Dataset total (47 rekaman) dibagi dengan
    proporsi mendekati 70% Pelatihan, 15% Validasi, dan 15% Pengujian.

> **3.4.3 Normalisasi Data**
>
> Normalisasi dilakukan untuk menyeragamkan rentang amplitudo sinyal
> agar mempercepat konvergensi model. Proses ini dilakukan dengan aturan
> ketat untuk menjaga integritas data uji:

a.  Metode: Menggunakan *Standard Scaler* yang mengubah distribusi data
    memiliki rata-rata 0 dan standar deviasi 1

b.  Prosedur *No-Peeking*: *Scaler* hanya dilatih (*fit*) menggunakan
    data Training. Parameter rata-rata (\$\\mu\$) dan standar deviasi
    (\$\\sigma\$) yang diperoleh dari data Training kemudian digunakan
    untuk mentransformasi data Validasi dan Test. Hal ini mencegah
    kebocoran statistik dari data uji ke dalam model.

c.  Implementasi Teknis: Data input 3D (Jumlah Sampel, 7, 200) diratakan
    (*flatten*) menjadi (Jumlah Sampel, 1400) terlebih dahulu sebelum
    normalisasi untuk memastikan konsistensi statistik di seluruh
    jendela konteks, kemudian dibentuk kembali (*reshape*) ke dimensi
    asli sebelum masuk ke model.

**3.5 Analisis dan perancangan model 1D-CNN**

**3.5.1. Analisis kebutuhan model**

> Berdasarkan tinjauan pustaka, 1D-CNN sangat sesuai untuk data deret
> waktu seperti sinyal ECG karena kemampuannya dalam mengekstraksi fitur
> spasial dan temporal secara otomatis dari data mentah (Ahmed et al.,
> 2023). Model ini diharapkan dapat menangani variasi dalam bentuk
> gelombang detak jantung dan mengidentifikasi pola-pola yang membedakan
> detak jantung Normal dan Abnormal.

**3.5.2. Perancangan arsitektur 1D-CNN**

> Arsitektur model akan dirancang dengan mempertimbangkan efektivitas
> dan efisiensi. Rancangan umum akan mencakup urutan lapisan konvolusi
> 1D, *pooling*, dan *dense layer* (lapisan *fully connected*).\
> Lapisan Konvolusi 1D**:** Beberapa lapisan konvolusi 1D akan digunakan
> untuk mengekstraksi fitur hierarkis dari sinyal detak jantung. Setiap
> lapisan konvolusi akan memiliki jumlah *filter* tertentu (misalnya,
> 64, 128, 256) dan ukuran *kernel* yang bervariasi (misalnya, 3, 5, 10)
> untuk menangkap pola lokal dengan skala berbeda. Fungsi aktivasi ReLU
> (Rectified Linear Unit) akan diterapkan setelah setiap lapisan
> konvolusi.

1.  Lapisan *Pooling*: Setelah kelompok lapisan konvolusi, lapisan
    *Max-Pooling 1D* akan digunakan untuk mengurangi dimensi spasial,
    menyorot fitur paling menonjol, dan membuat model lebih robust
    terhadap variasi posisi fitur (Ahmed et al., 2023).

2.  Lapisan *Batch Normalization*: Lapisan ini akan disisipkan untuk
    menstabilkan proses pelatihan dan mempercepat konvergensi dengan
    menormalisasi *input* ke setiap lapisan pada setiap *mini-batch*
    (Ahmed et al., 2023).

3.  Lapisan *Dropout*: Untuk mencegah *overfitting*, lapisan *Dropout*
    akan ditambahkan pada beberapa titik dalam arsitektur, secara acak
    \"mematikan\" sebagian neuron selama pelatihan (Ahmed et al., 2023).

> Lapisan *Dense (Fully Connected)*: Setelah fitur diekstraksi oleh
> lapisan konvolusi dan dipipihkan (*flatten*), beberapa lapisan *Dense*
> akan digunakan untuk melakukan klasifikasi akhir. Lapisan *output*
> akan menggunakan fungsi aktivasi Sigmoid untuk klasifikasi biner
> (Normal/Abnormal).

**3.6 Pelatihan dan scenario pengujian model**

Pada tahapan ini akan merinci konfigurasi secara teknis yang diterapkan
selama proses pembelajaran model serta strategi validasi yang akan
digunakan dalam menguji model sebelum nanti akan diimplementasikan pada
system.

**3.6.1 Konfigurasi Hiperparameter (*Hyperparameters*)**

Berdasarkan karakteristik dataset yang memiliki ketimpangan kelas dan
kompleksitas fitur, konfigurasi peltihan ditetapkan sebagai berikut:

1.  *Loss Function :* Penelitan ini menerapkan Cross Entropy Loss dengan
    mekanisme Class Weights. Mengingat jumlah data kelas normal jauh
    lebih dominan dibandingkan kelas abnormal, bobot penalty yang lebih
    besar akan diterapkan pada kesalahan prediksi pada kelas abnormal.
    Pendekatan ini bertujuan untuk mencegah model menjadi bias terhadap
    kelas mayoritas dan meningkatkan sensitivitas terhadap deteksi
    aritmia.

2.  *Optimizer* " Algoritma Adam W dipilih sebagai pengoptimal dengan
    learning rate awal sebesar 0.0001 dan weight decay sebesar 1e-4.
    Pemilihan AdamW didasarkan pada kemampuannya yang mampu memisahkan
    bobot dari pembaruan gradien, yang terbukti menghasilkan
    generalisasi yang lebih baik dibandingan dengan Adam standar pada
    model Deep Learning.

3.  Mekanisme Kendali Pelatihan:

<!-- -->

a.  Penjadwalan (*Scheduler)* : Menggunakan *ReduceLRONPlateau* yang
    secara otomatis menurunkan laju pembelajaran jika metrik AUC pada
    data validasi stagnan selama 5 epoch berturut-turut. Hal ini
    memungkinkan model untuk melakukan penyesuaian bobot yang lebih
    halus saat mendekati titik konvergensi.

b.  *Early Stopping :* Proses training akan dihentikan secara otomatis
    jika tidak terjadi peningkatan kinerja pada data validasi selama 15
    epoch. Mekanisme ini krusial untuk mencegah terjadinya overfitting.

**3.6.2 Skenario Pengujian Sistem**

> Selain evaluasi menggunakan data uji, validasi model ini diperkuat
> dengan simulasi system pada bagian *frontend.* Pengujian ini akan
> menggunakan 1 dataset secara acak yang tidak dimodifikasi sama sekali
> dari awal. Tujuan penggunaan dataset khusus ini adalah untuk menguji
> ketahanan model terhadap data pasien baru yang belum pernah terlihat
> sama sekali sebelumnya serta memverifikasi kinerja model saat
> memproses data menggunakan mekanismes *Rolling Buffer.* Skenario ini
> dirancang untuk merepresentasikan kondisi tantangan nyata pada alat
> pemantuann.

**3.7 Arsitektur dan Implementasi Sistem**

> Untuk merubah model klasifikasi menjadi alat bantu yang aplikatif,
> penerlitian ini merancang arsitektur software yang memisahkan
> *backend* dan *frontend.*
>
> **3.7.1 Logika Backend**
>
> Komponen pada *backend* befungsi sebagai mesin pemrosesan data yang
> beroperasi dilatar belakang system. Alur kerja teknis pada komponen
> ini nantinya meliputi:

1.  Data Buffering : Sistem dirancang untuk menerima aliran *raw data*
    secara terus menerus. Mengingat model ini membutuhkan konteks
    temporal maka data tidak akan diproses per titik detak jantung
    melainkan ditampung pada *Rolling Buffer* berkapasitas 7 detak.
    Prediksi baru akan dilakukan hanyaketika window lengkap terlah
    terbentuk untuk menjadi validitas input dari model.

2.  Pra-pemrosesan terstandarisasi : Data Buffering tadi
    dinormalisasikan secara real-time menggunakan parameter statistic
    yang identic dengan data training. Konsistensi Teknik pra-pemrosesan
    ini vital untuk menjaga akurasi inferensi.

3.  Inferensi model ONNX : Eksekusi prediksi dilukan menggunakan format
    model ONNX. Penggunaan model ini dipilih karena efisiensi memori dan
    kecepatan eksekusi yang lebih tinggi dibandingkan memuat *training
    library* secara utuh, sehingga lebih optimal untuk *deployment.*

> **3.7.2 Visualisasi Frontend**
>
> Komponen frontend bertugas untuk menampilkan hasil analisis komputasi
> menjadi informasi visual yang interaktif bagi pengguna medis. Fitur
> utama dari antarmuka mekiputi:

**3.8 Perancangan Fungsional Sistem (*Use Case*)**

Interaksi fungsional antara *user* dan system dimodelkan dengan
menggunakan *Use Case Diagram.* Diagram ini bertujuan untuk
memvisualisasikan bagaimana user nantinya akan berinteraksi dengan
interface aplikasi untuk melakukan pemantauan dan deteksi aritmia.
Pendekatan ini nantinya akan memastikan bahawa ada Batasan pada system
dan hak akses user dengan jelas sebelum impelementasi kode dilakukan.

**3.8.1 Definisi Aktor**

Aktor pada system ini definisikan sebagai *User* atau pengguna. User ini
nantinya akan merepresentasikan operator ataupun tenaga medis yang akan
bertanggung jawab untuk mengoperasikan aplikasi pemantauan. Dalam
konteks simulasi ini, user akan memiliki akses penuh terhadapa seluruh
fitur dari system tanpa memerlukan otentifikasi, guna memfasilitasi
proses demo dan penguji secara cepat.

**3.8.2 *Use Case Diagram***

> Berdasarkan analisis dari kebutuhan fungsional pada system, terdapat
> tiga aktivitas utama yang dapat dilakukan oleh actor. Hubungan antara
> actor dengan fungsionalitas system dapat terlihat pada gambar.

![A diagram of a person with text AI-generated content may be
incorrect.](media/image18.png){width="3.5201224846894137in"
height="4.147222222222222in"}

**3.8.3 Deskripsi Skenario Use Case**

> Pada bagian ini, akan dijelaskan secara ringi mengenai interaksi
> antara user dengan system dan spesifikasi untuk setiap kasus
> penggunaannya. Penjelasan setiap kasusnya adalah sebagai berikut:

1.  Memuat Data Pasien : Fungsi ini nantinya memungkinkan user untuk
    memuat system dan memilih sumber data rekaman ECG yang akan
    dianalisis. Alur kerja dari system ini adalah system memuat dataset
    pengujian dan membaca anotasi secara otomatis. Sistem kemudia
    menyiapkan mekanime Rolling Buffer untuk memulai aliran data sinyal

2.  Memantau Sinyal ECG : Fungsi ini nanti memberikan feedback visual
    kepada user berupa grafik detak jantung yang bisa diperbarui secara
    *real-time.* Alur kerja dai system ini alah mengambil data dari
    buffer dan rendering garfik gelombang yang bergerak dari kiri ke
    kanan pada layer interface. User dapat mengamati perubahan sinyal
    ini secara visual untuk memverifikasi pembacaan sensor.

3.  Melihat hasil diagnosis : Fungsi ini nantinya akan menyajikan hasil
    analisis model kepada user untuk mendukung pengambilan Keputusan.
    Alur kerjanya adalah setelah model melakukan prediksi pada detak
    jantung yang terdeteksi, system akan menampilkan status teks baik
    normal maupun abnormal dan mengubah warna indicator grafik yaitu
    biru untuk normal dan merah untuk abnormal. User akan menerima
    informasi ini secara langsung tanpa perlu melakukan apapun dan bisa
    mengambil Keputusan berdasrkan hasil pengetesan pada uji model
    tersebut.