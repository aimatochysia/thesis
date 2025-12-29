**BAB 1****

**PENDAHULUAN**

1.  **Latar Belakang**

Penyakit kardiovaskular masih menjadi penyebab utama morbiditas dan
mortalitas di seluruh dunia, dengan *Arrhythmia* jantung---detak jantung
yang tidak normal atau tidak teratur---menjadi indikator kritis
kesehatan jantung (Ahmed et al., 2025; Acharya et al., 2017; Sannino &
De Pietro, 2021). Beberapa jenis *Arrhythmia* dapat tidak berbahaya,
namun banyak di antaranya berpotensi menyebabkan kondisi medis serius,
termasuk serangan jantung dan kematian mendadak (Ahmed et al., 2025).
Oleh karena itu, deteksi dini *Arrhythmia* melalui analisis sinyal
Elektrokardiogram (ECG) memiliki peran yang sangat penting dalam upaya
mengurangi risiko kematian akibat henti jantung mendadak di kemudian
hari (Sannino & De Pietro, 2021). ECG sendiri merupakan alat diagnostik
yang cepat, murah, dan non-invasif, menjadikannya metode yang banyak
digunakan untuk mendiagnosis berbagai jenis *Arrhythmia* (Acharya et
al., 2017). Meskipun demikian, diagnosis *Arrhythmia* secara manual oleh
kardiolog memiliki keterbatasan yang signifikan. Proses ini membutuhkan
keahlian tinggi dan seringkali rentan terhadap pengaruh *noise* atau
gangguan pada sinyal ECG, serta sifat acak dari kemunculan *Arrhythmia*
(Acharya et al., 2017). Faktor-faktor ini dapat menyebabkan misdiagnosis
atau kesalahan interpretasi, yang pada gilirannya dapat menunda
intervensi medis yang krusial.

Mengingat tingginya angka mortalitas yang terkait dengan penyakit
kardiovaskular, tantangan dalam mengklasifikasikan sinyal ECG ke dalam
kondisi jantung spesifik mendorong pengembangan metode otomatis yang
lebih akurat dan efisien (Nab, 2023). Kontras yang jelas antara metode
diagnosis manual yang rentan terhadap kesalahan dan kebutuhan mendesak
akan sistem otomatis yang akurat ini membentuk justifikasi kuat untuk
pendekatan berbasis Artificial Intelligence (AI). Pendekatan ini secara
efektif merumuskan masalah inti yang dapat diatasi oleh penelitian di
bidang ini. Dalam beberapa tahun terakhir, teknologi Artificial
Intelligence (AI), khususnya *Deep Learning* (DL), telah menunjukkan
perkembangan pesat dan aplikasi yang luas dalam analisis sinyal biomedis
(Acharya et al., 2017; Ahmed et al., 2023; Guerra et al., 2025). DL
menawarkan alternatif yang jauh lebih baik untuk klasifikasi yang cepat
dan otomatis dibandingkan metode tradisional (Ahmed et al., 2023; Nab,
2023). Kemampuan DL untuk secara otomatis mengekstraksi fitur-fitur
penting dan belajar mandiri dari data mentah ECG memungkinkan deteksi
pola halus yang mungkin terlewatkan oleh pengamatan manusia (Ahmed et
al., 2023; Nab, 2023). Selain itu, model DL mampu menangani *dataset*
besar dan bising secara efektif, menjadikannya solusi yang menjanjikan
untuk tantangan diagnosis *Arrhythmia* (Ahmed et al., 2023). Penggunaan
AI dan DL dalam konteks ini bukan hanya sekadar mengikuti tren
teknologi, melainkan merupakan solusi fundamental yang mengatasi
keterbatasan diagnosis manual, menandai pergeseran paradigma dari
rekayasa fitur manual ke ekstraksi fitur otomatis, yang secara
signifikan dapat meningkatkan efisiensi dan akurasi diagnostik.
Berdasarkan uraian di atas, penelitian ini mengkaji penerapan
*Convolutional Neural Network 1D* pada *dataset* MIT-BIH yang telah
dimodifikasi untuk hanya berisi satu detak per data (*detak tunggal*)
dan kemudian mengklasifikasikannya menjadi hanya Normal dan Abnormal.

Penelitian ini termasuk dalam kategori penelitian terapan di bidang
*Artificial Intelligence* dan *Biomedical Engineering*, khususnya dalam
pengembangan sistem *Computer-Aided Diagnosis* (CAD) untuk *Arrhythmia*
jantung. Mengingat tingginya akurasi yang telah dicapai oleh model *Deep
Learning* pada *dataset* ini untuk klasifikasi biner, ide penelitian ini
sangat relevan dan memiliki potensi besar untuk validasi ulang serta
pengembangan aplikasi praktis dalam diagnosis medis.

2.  **Rumusan masalah**

Berdasarkan latar belakang yang telah diuraikan, maka rumusan masalah
dalam penelitian ini adalah sebagai berikut:

1.  Bagaimana proses pengumpulan, modifikasi, dan persiapan data sinyal
    ECG dari dataset MIT-BIH dapat dilakukan untuk membentuk data detak
    jantung tunggal yang siap digunakan dalam pelatihan model deep
    learning?

2.  Bagaimana merancang arsitektur model *Convolutional Neural Networ*k
    1D (1D-CNN) yang efektif untuk klasifikasi sinyal ECG menjadi dua
    kelas: Normal dan Abnormal?

3.  Bagaimana pengaruh strategi preprocessing seperti penghapusan
    *outlier*, penyeimbangan kelas, *clustering* (K-Means), dan reduksi
    dimensi (PCA) terhadap kualitas dan performa dataset untuk
    klasifikasi biner?

4.  Bagaimana strategi segmentasi sinyal ECG berbasis R-peak detection
    dapat membantu membentuk input yang fisiologis dan konsisten untuk
    model CNN?

5.  Bagaimana performa model CNN 1D yang dikembangkan ketika dievaluasi
    menggunakan metrik akurasi, *precision, recall, F1-score*, dan
    AUC-ROC, baik terhadap data bersih (*clean evaluation set*) maupun
    data mentah (*prediction dataset*)?

    1.  **Hipotesis**

Model *Convolutional Neural Network 1D* yang dilatih pada representasi
detak jantung tunggal dari *dataset* MIT-BIH yang dimodifikasi akan
mampu mengklasifikasikan detak jantung sebagai Normal atau Abnormal
dengan tingkat akurasi dan kinerja (*precision*, *recall*, *F1-score*,
dan *Area Under the Receiver Operating Characteristic Curve* (AUC-ROC))
yang tinggi, sebanding atau bahkan melampaui hasil penelitian terdahulu
dalam klasifikasi biner detak jantung.\
**1.4 Ruang lingkup penelitian**

> Penelitian ini memiliki batasan-batasan yang jelas untuk menjaga fokus
> dan ketercapaian tujuan. Ruang lingkup penelitian ini meliputi:
>
> **1.4.1. Dataset** : Penelitian ini akan menggunakan MIT-BIH
> *Arrhythmia* Database(Ahmed et al., 2025; Nab, 2024; Moody & Mark,
> 2001). Data akan di proses untukmemastikan satu detak jantung,
> berpusat pada R-peak dan memiliki ukuran window yang konsisten(Ahmed
> etl al., 2023).
>
> **1.4.2. Fokus klasifikasi** : Klasifikasi akan difokuskan pada
> masalah biner, yaitu membedakan detak jantung Normal dari detak
> jantung Abnormal. Semua jenis *Arrhythmia* selain Normal akan
> dikelompokkan ke dalam kategori Abnormal (Nab, 2024; Panwar et al.,
> 2025).

**1.4.3. Metode** : Implementasi dan evaluasi akan dilakukan menggunakan
arsitektur *Convolutional Neural Network 1D*. Model akan dilatih dan
dievaluasi dengan mempertimbangkan praktik terbaik seperti *stratified
splitting* dan penanganan ketidakseimbangan kelas (*class-weighting*
atau *weighted sampling*) (Ahmed et al., 2023; Nab, 2023).

> **1.4.4. Aspek yang tidak dibahas** : Penelitian ini tidak akan
> membahas deteksi atau klasifikasi *Arrhythmia* multi-kelas yang lebih
> spesifik (misalnya, membedakan *Supraventricular Ectopic* dari
> *Ventricular Ectopic* secara terpisah), pengembangan model untuk
> perangkat keras *real-time* yang spesifik, atau analisis fitur
> *time-frequency* yang lebih kompleks (Sannino & De Pietro, 2021).
> Fokus utamanya adalah validasi kinerja model *1D-CNN* untuk
> klasifikasi biner detak jantung tunggal. Pembatasan ruang lingkup
> secara jelas (apa yang dilakukan dan apa yang tidak) sangat penting
> untuk menjaga fokus penelitian. Dengan menyebutkan apa yang *tidak*
> dibahas, peneliti menunjukkan pemahaman tentang kompleksitas bidang
> dan memilih untuk fokus pada aspek yang dapat dikelola dan relevan
> dengan tujuan utama.

5.  **Tujuan dan manfaat penelitian**

> **1.5.1 Tujuan penelitian**
>
> Tujuan yang ingin dicapai dalam penelitian ini adalah:

1.  Mengembangkan dan mengimplementasikan model *Convolutional Neural
    Network 1D* untuk klasifikasi detak jantung Normal dan Abnormal dari
    *dataset* MIT-BIH yang telah dimodifikasi.

2.  Melatih dan mengoptimalkan model *1D-CNN* menggunakan teknik
    pra-pemrosesan data yang sesuai, termasuk segmentasi detak jantung
    tunggal dan penanganan ketidakseimbangan kelas.

3.  Mengevaluasi kinerja model *1D-CNN* dalam klasifikasi biner detak
    jantung menggunakan metrik evaluasi standar seperti akurasi,
    *precision*, *recall* (*sensitivity*), *F1-score*, dan *Area Under
    the Receiver Operating Characteristic Curve* (AUC-ROC).

> Tujuan-tujuan ini dirumuskan secara spesifik dan terukur
> (\"mengembangkan,\" \"melatih,\" \"mengevaluasi\"), yang merupakan
> ciri khas tujuan penelitian yang baik. Hal ini secara langsung mengacu
> pada langkah-langkah metodologis yang akan diambil, memberikan
> kejelasan tentang apa yang ingin dicapai.

**1.5.2 Manfaat penelitian**

Penelitian ini diharapkan dapat memberikan manfaat sebagai berikut:

1\. Bagi ilmu pengetahuan : Penelitian ini diharapkan dapat memberikan
validasi empiris lebih lanjut mengenai efektivitas *Convolutional Neural
Network 1D* dalam klasifikasi biner detak jantung tunggal pada *dataset*
ECG yang kompleks seperti MIT-BIH. Hasilnya dapat memperkaya literatur
ilmiah terkait aplikasi *Deep Learning* dalam analisis sinyal biomedis
dan menjadi referensi bagi penelitian serupa di masa mendatang.

2\. Bagi teknologi : Kontribusi penelitian ini dapat menjadi dasar bagi
pengembangan sistem diagnosis *Arrhythmia* jantung otomatis yang lebih
efisien dan akurat. Model yang dikembangkan berpotensi diintegrasikan ke
dalam perangkat medis atau aplikasi *telemedicine* untuk skrining awal
atau pemantauan jarak jauh, yang krusial untuk memberikan peringatan dan
intervensi tepat waktu (Panwar et al., 2025; Nehru Institute of
Technology et al., 2025).

3\. Bagi praktisi medis : Sistem klasifikasi otomatis yang akurat dapat
mendukung tenaga medis dalam proses skrining awal dan diagnosis
*Arrhythmia*, mengurangi beban kerja, dan meminimalkan potensi kesalahan
diagnosis manual, sehingga memungkinkan intervensi dini dan peningkatan
kualitas perawatan pasien (Ahmed et al., 2023; Sannino & De Pietro,
2021; Panwar et al., 2025).

> Manfaat penelitian diuraikan secara multi-dimensi (ilmu, teknologi,
> praktisi). Ini menunjukkan pemahaman yang komprehensif tentang dampak
> penelitian, dari kontribusi teoretis hingga aplikasi praktis di dunia
> nyata, meningkatkan nilai dan relevansi studi.

5.  **Sistematika penulisan**

> Sistematika penulisan pada laporan penelitian ini terbagi menjadi 5
> bab, yang kemudian masing-masing dibahas lebih detail dalam beberapa
> sub-bab. Berikut merupakan sistematika dari masing-masing bab beserta
> keterangannya :
>
> Bab I Pendahuluan
>
> Bab ini menguraikan latar belakang penelitian, termasuk tinjauan
> pustaka (*state of the art*) mengenai klasifikasi *Arrhythmia* dengan
> *Deep Learning* pada *dataset* MIT-BIH. Selanjutnya, bab ini
> merumuskan masalah, hipotesis, ruang lingkup, tujuan dan manfaat,
> metode penelitian secara umum, serta sistematika penulisan laporan.
>
> Bab II Tinjauan Pustaka
>
> Bab ini akan menyajikan landasan teori yang relevan, membahas secara
> lebih mendalam konsep Elektrokardiogram (ECG), *Arrhythmia* jantung,
> dasar-dasar *Deep Learning, Convolutional Neural Network* (CNN) 1D,
> serta dataset MIT-BIH *Arrhythmia*.
>
> Bab III Metode Penelitian
>
> Bab ini menjelaskan secara rinci tahapan penelitian yang dilakukan,
> mulai dari pengumpulan dan pra-pemrosesan data, perancangan arsitektur
> model *1D-CNN*, proses pelatihan dan validasi, hingga prosedur
> evaluasi kinerja model.
>
> Bab IV Hasil dan Pembahasan
>
> Bab ini menyajikan hasil eksperimen yang diperoleh dari implementasi
> model, termasuk metrik kinerja, confusion matrix, dan kurva AUC-ROC.
> Hasil akan dianalisis dan dibahas secara komprehensif, membandingkan
> dengan hipotesis dan penelitian terdahulu.
>
> Bab V Kesimpulan dan Saran
>
> Bab terakhir ini merangkum temuan utama penelitian,menarik kesimpulan
> berdasarkan hasil yang diperoleh, serta memberikan saran untuk
> penelitian lanjutan.