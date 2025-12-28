**DETEKSI SINYAL ECG NORMAL DAN ABNORMAL MENGGUNAKAN JARINGAN SARAF
KONVOLUSIONAL 1D PADA DATASET MIT-BIH YANG TELAH DIMODIFIKASI**

**SKRIPSI**

**Oleh**

**Abdiel Ivan Rivandi 2602164282**

**Petra Michael 2602075812**

![logo binus -
Kontenesia](media/image1.png){width="2.8118055555555554in"
height="1.7201388888888889in"}

**Computer Science Study Program**

**School of Computer Science**

**Universitas Bina Nusantara**

**BANDUNG**

**2025**

**DETEKSI SINYAL ECG NORMAL DAN ABNORMAL MENGGUNAKAN JARINGAN SARAF
KONVOLUSIONAL 1D PADA DATASET MIT-BIH YANG TELAH DIMODIFIKASI**

**SKRIPSI**

**diajukan sebagai salah satu syarat**

**untuk gelar kesarjanaan pada**

**Program Studi Teknik Informatika**

**Jenjang Pendidikan Strata-1**

**Oleh**

**ABDIEL IVAN RIVANDI 2602164282**

**PETRA MICHAEL 2602075812**

![logo binus -
Kontenesia](media/image1.png){width="2.8118055555555554in"
height="1.7201388888888889in"}

**Computer Science Study Program**

**School of Computer Science**

**Universitas Bina Nusantara**

**BANDUNG**

**2025**

**Universitas Bina Nusantara**

**Pernyataan Kesiapan Skripsi untuk Ujian Pendadaran**

**Pernyataan Penyusunan Skripsi**

**Kami, Abdiel Ivan Rivandi,**

**Petra Michael**

**Dengan ini menyatakan bahwa Skripsi yang berjudul:**

**DETEKSI SINYAL ECG NORMAL DAN ABNORMAL MENGGUNAKAN JARINGAN SARAF
KONVOLUSIONAL 1D PADA DATASET MIT-BIH YANG TELAH DIMODIFIKASI**

***DETECTION OF NORMAL AND ABNORMAL ECG SIGNALS USING A 1D CONVOLUTIONAL
NEURAL NETWORK ON A MODIFIED MIT-BIH DATASET***

**Adalah benar hasil karya kami dan belum pernah diajukan sebagai karya
ilmiah, sebagian atau seluruhnya, atas nama kami atau pihak lain**

![](media/image2.png){width="0.9531255468066492in"
height="0.9531255468066492in"}
![](media/image3.png){width="0.8721609798775153in"
height="0.8630391513560804in"}

**Abdiel Ivan Rivandi PetraMichael**

**2602164282 2602075812**

**Disetujui oleh Pembimbing**

**Saya setuju Skripsi tersebut layak diajukan untuk Ujian Pendadaran**

**Dr. Dani Suandi, S.Si., M.Si.**

> **D6532**

**UNIVERSITAS BINA NUSANTARA**

Computer Science Study Program

School of Computer Science

Skripsi Sarjana Teknik Informatika

**DETECTION OF NORMAL AND ABNORMAL ECG SIGNALS USING A 1D CONVOLUTIONAL
NEURAL NETWORK ON A MODIFIED MIT-BIH DATASET**

**Abdiel Ivan Rivandi 2602164282**

**Petra Michael 2602164282**

***ABSTRAK***

*This study explores the application of a one-dimensional Convolutional
Neural Network (1D CNN) to classify electrocardiogram (ECG) signals into
two categories: normal and abnormal. The dataset used is a modified
version of the MIT-BIH Arrhythmia Dataset, which has undergone
preprocessing steps such as normalization and outlier removal to enhance
signal quality. The 1D CNN model was designed to identify temporal
patterns in ECG signals without manual feature engineering. Evaluation
using metrics such as accuracy, precision, recall, F1-score, and AUC-ROC
demonstrates reliable performance in binary classification. The results
highlight the effectiveness of CNN-based deep learning approaches in
Arrhythmia detection and indicate strong potential for integration into
wearable devices for real-time cardiac monitoring. **Keywords:** ECG,
Arrhythmia, 1D CNN, Signal Classification, Cardiac Detection*

**ABSTRAK**

Penelitian ini membahas penerapan model Convolutional Neural Network
(CNN) satu dimensi (1D) untuk klasifikasi sinyal elektrokardiogram (ECG)
menjadi dua kategori: normal dan abnormal. Data yang digunakan berasal
dari MIT-BIH *Arrhythmia* Dataset yang telah dimodifikasi melalui proses
normalisasi dan penghapusan outlier guna meningkatkan kualitas sinyal.
Model CNN 1D dirancang untuk mengenali pola temporal dalam sinyal
jantung tanpa memerlukan rekayasa fitur manual. Evaluasi dilakukan
dengan metrik akurasi, precision, recall, F1-score, dan AUC-ROC, yang
menunjukkan kinerja model yang andal dalam deteksi biner. Hasil
penelitian ini menunjukkan bahwa pendekatan deep learning berbasis CNN
efektif dalam mengidentifikasi *arrhythmia* dan memiliki potensi besar
untuk diterapkan pada perangkat wearable sebagai sistem pemantauan
jantung otomatis dan real-time. Studi ini berkontribusi dalam
pengembangan metode klasifikasi sinyal medis yang efisien, akurat, dan
dapat diandalkan di lingkungan klinis maupun non-klinis.

**Kata Kunci:** ECG, *Arrhythmia*, CNN 1D, Klasifikasi Sinyal, Deteksi
Jantung

**Kata Pengantar**

Puji dan syukur penulis panjatkan ke hadirat Tuhan Yang Maha Esa atas
berkat dan rahmat-Nya, sehingga penulis dapat menyelesaikan penyusunan
skripsi yang berjudul "Deteksi Sinyal ECG Normal dan Abnormal
Menggunakan Jaringan Saraf Konvolusional 1D pada Dataset MIT-BIH yang
Telah Dimodifikasi" dengan baik dan tepat waktu.

Skripsi ini disusun sebagai salah satu syarat untuk memperoleh gelar
Sarjana pada Program Studi Teknik Informatika, Universitas Bina
Nusantara. Penulis menyadari bahwa keberhasilan penyusunan skripsi ini
tidak terlepas dari dukungan, bimbingan, dan bantuan dari berbagai
pihak. Oleh karena itu, penulis dengan tulus mengucapkan terima kasih
kepada:

1.  Ibu Dr. Nelly, S.Kom., M.M., CSCA , selaku Rektor Binus University

2.  Bapak Dr. Johan Muliadi Kerta, S.Kom., M.M. selaku *Campus Director*
    Universitas Bina Nusantara \@Bandung.

3.  Bapak Prof. Dr. Ir. Derwin Suhartono, S.Kom., MTI , selaku *Dean of
    School of Computer Science.*

4.  Bapak Prof. Dr. Ir. Derwin Suhartono, S.Kom., MTI , selaku *Head of
    Computer Science Study Program.*

5.  Bapak Dr. Dani Suandi, S.Si., M.Si., selaku dosen pembimbing, atas
    bimbingan, waktu, dan pengetahuan yang sangat berarti dalam proses
    penelitian ini.

6.  Kedua orang tua dan keluarga kami, atas doa, dukungan moral dan
    material, serta semangat yang tak ternilai selama masa studi dan
    penyusunan skripsi ini.

7.  Teman-teman dan semua pihak yang tidak dapat disebutkan satu per
    satu, yang telah membantu secara langsung maupun tidak langsung
    dalam menyelesaikan skripsi ini.

Penulis menyadari bahwa skripsi ini masih jauh dari sempurna. Oleh
karena itu, penulis membuka diri terhadap kritik dan saran yang
membangun demi penyempurnaan karya ini. Semoga skripsi ini dapat
memberikan manfaat bagi pengembangan ilmu pengetahuan, khususnya di
bidang pengolahan sinyal medis dan kecerdasan buatan.

Jakarta, 13 Juni 2025

Tim Penulis

**DAFTAR ISI**

**HALAMAN
SAMPUL...........................................................i**

**HALAMAN
JUDUL.............................................................ii**

**HALAMAN PERNYATAAN ORISINALITAS...........................iii**

**ABSTRAK.........................................................................iv**

**KATA
PENGANTAR.........................................................\...v**

**DAFTAR
ISI......................................................................vi**

**DAFTAR
TABEL................................................................vii**

**DAFTAR
GAMBAR.........................................................\...viii**

**BAB 1
PENDAHULUAN......................................................\...1**

1.  Latar
    Belakang.....................................................................\...1

2.  Rumusan
    masalah....................................................................2

3.  Hipotesis...............................................................................2

4.  Ruang lingkup
    penelitian............................................................3

    1.  Dataset....................................................................3

    2.  Fokus
        klasifikasi.......................................................
        3

    3.  Metode....................................................................3

    4.  Aspek yang tidak
        dibahas..............................................3

5.  Tujuan dan manfaat
    penelitian......................................................3

    1.  Tujuan
        penelitian........................................................3

    2.  Manfaat
        penelitian...................................................\....4

6.  Sistematika
    penulisan................................................................5

**BAB 2 TINJAUAN
REFERENSI...............................................6**

1.  Elektrokardiogram...............................................................\.....6

2.  Denoising dalam pemrosesan sinyal
    *ECG*\...\...\...\...\...\...\...\...\...\...\...\...\...\...\...\...\.....6

3.  Deteksi *arrhythmia*, sinyal abnormal dan variabilitas
    sinyal\...\...\...\...\...\...\...\....7

4.  *Heart rate
    variability*.................................................................9

5.  DatasetMIT-BH
    *arrhythmia*......................................................\...9

6.  Modifikasi
    Data.....................................................................10

7.  Persiapan
    data.....................................................................\...11

8.  CNN dan
    CONV-1D................................................................12

9.  *Auto
    Encoder*........................................................................13

10. Kinerja
    Model...................................................................14

11. *Open Neural Network X* (ONXX)

12. *Unified Modeling Language* (UML) dan *Use Case Diagram*

13. Related
    works...................................................................15

**BAB 3 METODE
PENELITIAN..............................................19**

1.  Pendekatan
    penelitian............................................................\...19

2.  Tahapan
    penelitian...............................................................\...20

3.  Pemilihan dan Pemilahan dataset

> 3.3.1 Sumber dan Karakteristik Data Mentah
>
> 3.3.2 Segmentasi dan Ekstrasi Sinyal
>
> 3.3.3 Pembentukan Windows Construction
>
> 3.3.3 Dataset Final dan Labelling

4.  Pra-pemrosesan dan Analisis Kualitas Data

> 3.4.1 Karakteristik fisik dan dimensi data
>
> 3.4.2 Pemisahan Data
>
> 3.4.3 Normalisasi Data

5.  Analisis dan perancangan model
    1D-CNN......................................23

3.4.1 Analisis kebutuhan
model..............................................23

3.4.2 Perancangan arsitektur
1D-CNN....................................\...23

3.4 Pengembangan dan implementasi
model.........................................24

3.5 Pelatihan dan validasi
model......................................................24

3.6 Analisis hasil dan pembuktian
hipotesis.......................................\...24

**BAB 4 HASIL DAN PEMBAHASAN**

**BAB 5 KESIMPULAN DAN SARAN**

**REFERENSI......................................................................26**

**DAFTAR TABEL**

Tabel 2.1 Tabel Related
Works.........................................................16

**DAFTAR GAMBAR**

Gambar 2.1. Sinyal
Elektrokardiogram......................................................7

Gambar 2.2 Denoising
Elektrokardiogram..................................................8

Gambar 2.3. Takikardia Ventrikel
(VT)...................................................\...8

Gambar 2.4. Fibrilasi Ventrikular (VF)
...................................................\...8

Gambar 2.5. Fibrilasi Atrium (AF)
...........................................................8

Gambar 2.6. Arsitektur
1D-CNN............................................................14

Gambar 2.7. *Confusion
Matrix*...........................................................\.....16

Gambar 3.1. CRISP-DM
Model............\..................................................19

Gambar 3.2. Tahapan
Penelitian\......................................................\...\...\....21

**BAB 1**

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

**BAB 2**

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

**2.8 CNN dan Conv-1D**

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

**2.11 *Open Neural Network Exchange* (ONNX)**

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

**BAB 3**

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

**BAB 4**

**HASIL DAN PEMBAHASAN\
BAB 5**

**SIMPULAN DAN SARAN**

**5.1 Simpulan**

Penelitian ini telah berhasil mengembangkan dan menguji model
Convolutional Neural Network 1D untuk klasifikasi sinyal ECG, yang mampu
mencapai akurasi 98.73%. Hasil evaluasi yang menggembirakan ini
menunjukkan bahwa model ini tidak hanya efektif dalam membedakan antara
sinyal normal dan abnormal, tetapi juga sangat stabil dalam kondisi yang
memiliki noise atau fluktuasi, yang sering menjadi tantangan dalam
aplikasi dunia nyata. Metrik lain seperti precision, recall, F1-score,
dan ROC-AUC menunjukkan bahwa model ini dapat digunakan dengan
kepercayaan tinggi untuk aplikasi deteksi dini penyakit jantung,
terutama dalam sistem pemantauan berbasis perangkat wearable.

Secara keseluruhan, model ini menunjukkan potensi besar dalam aplikasi
medis, khususnya dalam memberikan solusi deteksi dini untuk masalah
jantung. Namun, meskipun hasilnya sudah sangat baik, penelitian ini juga
menemukan bahwa masih ada beberapa area untuk perbaikan lebih lanjut,
terutama dalam hal pengurangan kesalahan prediksi pada kelas abnormal.

**5.2 Saran**

Meskipun model ini telah menunjukkan hasil yang sangat positif, ada
beberapa aspek yang dapat diperbaiki untuk memperluas aplikabilitas dan
meningkatkan akurasi lebih lanjut. Salah satu area utama yang perlu
ditangani adalah peningkatan recall pada kelas abnormal. Penggunaan
teknik augmentasi data lebih lanjut atau peningkatan jumlah data pada
kelas abnormal dapat membantu dalam mengurangi kesalahan klasifikasi
pada sinyal abnormal.

Selain itu, pengembangan model untuk mengklasifikasikan jenis arrhythmia
yang lebih spesifik daripada hanya dua kategori (normal dan abnormal)
bisa menjadi langkah selanjutnya yang signifikan dalam memperluas
kemampuan diagnosis model. Penelitian ini juga dapat diperluas dengan
menerapkan model ini dalam sistem pemantauan jantung berbasis perangkat
wearable, yang memungkinkan deteksi penyakit jantung secara real-time.
Dengan meningkatkan kecepatan dan efisiensi model, serta meminimalkan
penggunaan sumber daya, model ini dapat dioptimalkan lebih lanjut untuk
diterapkan pada perangkat dengan kemampuan komputasi terbatas.

Secara keseluruhan, dengan beberapa pengembangan lebih lanjut, model ini
memiliki potensi besar untuk diterapkan dalam dunia medis, khususnya
dalam meningkatkan kemampuan diagnosis otomatis dan monitoring kesehatan
berbasis teknologi AI.

**\
**

**REFERENSI**

Acharya, U. R., Fujita, H., Lih, O. H., Adam, M., Tan, J. H., & Chua, C.
K. (2017). *Automated Arrhythmia detection using spectrogram and deep
convolutional neural network with long duration ECG signals. Information
Sciences, 405, 112--127*.

Acharya, U. R., Joseph, K. P., Kannathal, N., Lim, C. M., & Suri, J. S.
(2006). *Heart rate variability: A review. Medical & Biological
Engineering & Computing, 44(12), 1031--1051*.

Acharya, U. R., Oh, S. L., Hagiwara, Y., Tan, J. H., & Adam, M. (2017).
*A deep convolutional neural network model to classify heartbeats.
Computers in Biology and Medicine, 89, 389--396.*

Ahmed, M. S., Khan, A. S., & Ali, N. (2023). *ECG signal classification
using deep learning techniques: A systematic review. Computers in
Biology and Medicine, 161*, 107380.

Ahmed, M. S., Khan, A. S., & Ali, N. (2025). *Detection of normal and
abnormal ECG signals using a 1D convolutional neural network on a
modified MIT-BIH dataset (Unpublished master's thesis)*. Universitas
Bina Nusantara, Bandung.

American Heart Association. (2021). *What is Arrhythmia? Retrieved from*
https://www.heart.org/en/health-topics/*Arrhythmia*/about-*Arrhythmia*

Bayat, O., Aljawarneh, S., Carlak, H. F., International Association of
Researchers, Institute of Electrical and Electronics Engineers, &
Akdeniz Üniversitesi. (2017). *Understanding of a convolutional neural
network. In IEEE International Conference on Engineering & Technology
(ICET) (pp. 21--23)*. Antalya.

Chen, Y. (2015). *Convolutional neural network for sentence
classification*. Ontario.

Chicco, D., & Jurman, G. (2020). *The advantages of the Matthews
correlation coefficient over F1 score and accuracy in binary
classification evaluation. BMC Genomics*, 21(1), 6.

Clifford, G. D., Azuaje, F., & McSharry, P. (2006). *Advanced methods
and tools for ECG data analysis*. Artech House.

Faust, O., Hagiwara, Y., Hong, T. J., Lih, O. S., & Acharya, U. R.
(2018). *Deep learning for healthcare applications based on
physiological signals: A review. Computer Methods and Programs in
Biomedicine, 161, 1--13*.

Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M.,
Ivanov, P. C., Mark, R. G., \... & Stanley, H. E. (2000). *PhysioBank,
PhysioToolkit, and PhysioNet. Circulation, 101(23)*, e215--e220.

Guerra, I., Castro, J. M., Silva, R. A., & Fonseca, J. M. (2025). *Deep
learning for ECG signal analysis: A systematic review. Biomedical Signal
Processing and Control, 97*, 106367.

Hannun, A. Y., Rajpurkar, P., Haghpanahi, M., Tison, G. H., Bourn, C.,
Turakhia, M. P., & Ng, A. Y. (2019). *Cardiologist-level Arrhythmia
detection and classification in ambulatory electrocardiograms using a
deep neural network*. Nature Medicine, 25(1), 65--69.

Hinton, G. E., & Salakhutdinov, R. R. (2006). *Reducing the
dimensionality of data with neural networks*. Science, 313(5786),
504--507.

Kwon, S., Hong, J., & Park, Y. (2018). *Smart wearable systems for
personalized health monitoring. IEEE Reviews in Biomedical Engineering*,
11, 356--367.

Luz, E. J. d. S., Schwartz, W. R., Cámara-Chávez, G., & Menotti, D.
(2016). *ECG-based heartbeat classification for Arrhythmia detection: A
survey. Computer Methods and Programs in Biomedicine,* 127, 144--164.

Martis, R. J., Acharya, U. R., Adeli, H., & Prasad, H. (2014).
*Application of higher order statistics for atrial Arrhythmia
classification. Biomedical Signal Processing and Control*, 8(6),
888--900.

Moody, G. B., & Mark, R. G. (2001). *The impact of the MIT-BIH
Arrhythmia Database. IEEE Engineering in Medicine and Biology Magazine,
20(3), 45--50*.

Nab, A. (2023). *ECG Arrhythmia classification using deep learning
(Unpublished master's thesis)*. University of Twente.

Nehru Institute of Technology, P. J., & R. K. (2025). *Early detection
of cardiac Arrhythmia using deep learning. Journal of Medical
Engineering & Technology*, 49(2), 118--124.

Panwar, M., Singh, S., & Singh, R. (2025). *Real-time ECG Arrhythmia
detection using deep learning. Biomedical Signal Processing and
Control*, 95, 106300.

Pantelopoulos, A., & Bourbakis, N. G. (2010). *A survey on wearable
sensor-based systems for health monitoring and prognosis. IEEE
Transactions on Systems, Man, and Cybernetics,* 40(1), 1--12.

Rajpurkar, P., Hannun, A. Y., Haghpanahi, M., Bourn, C., & Ng, A. Y.
(2017). *Cardiologist-level Arrhythmia detection with convolutional
neural networks. Nature Medicine,* 25(1), 65--69.

Sannino, G., & De Pietro, G. (2021). *Deep learning for ECG signal
classification: A review. Artificial Intelligence in Medicine*, 118,
102142.

Shaffer, F., & Ginsberg, J. P. (2017). *An overview of heart rate
variability metrics and norms. Frontiers in Public Health*, 5, 258.
https://doi.org/10.3389/fpubh.2017.00258

Shi, H., Zhang, C., He, J., & Wang, Z. (2019). *Automatic detection of
Arrhythmia based on multi-scale CNN and attention-based RNN. Journal of
Biomedical Informatics, 100*, 103395.

Taejoong Yoon. (2023). *MIT-BIH Arrhythmia Dataset. Kaggle*.
https://www.kaggle.com/datasets/taejoongyoon/mitbit-*Arrhythmia*-database

Xiong, Z., Stiles, M. K., & Zhao, J. (2021). *A deep learning framework
for automatic diagnosis of Arrhythmias using a single-lead ECG. Journal
of Electrocardiology, 66*, 29--35.

Yildirim, O., Baloglu, U. B., Tan, R. S., & Acharya, U. R. (2018).
*Arrhythmia detection using deep convolutional neural network with long
duration ECG signals. Computers in Biology and Medicine,* 102, 411--420.

Zhao, Z., Zhang, Y., Deng, Y., & Zhou, X. (2019). *ECG signal denoising
and classification using deep feature learning. Medical & Biological
Engineering & Computing, 57*, 1987--1998.

Zheng, J., Zhang, Y., Wang, L., Chen, H., & Wu, X. (2020). *A deep
learning framework for ECG-based heartbeat classification using spatial
pyramid pooling. Neural Networks, 122*, 160--169.

Zipes, D. P., & Jalife, J. (2013). *Cardiac electrophysiology: From cell
to bedside (6th ed.).* Elsevier.

Jain, A. K. (2010). Data clustering: 50 years beyond K-means. *Pattern
Recognition Letters, 31*(8), 651--666.

Lloyd, S. (1982). Least squares quantization in PCM. *IEEE Transactions
on Information Theory, 28*(2), 129--137.

Xu, R., & Wunsch, D. (2005). Survey of clustering algorithms. *IEEE
Transactions on Neural Networks, 16*(3), 645--678.

Alinsaif, S. (2024). *Unraveling arrhythmias with graph-based analysis:
A survey of the MIT-BIH database. Computation*, 12(2), 21.
https://doi.org/10.3390/computation12020021

Eleyan, A., & Alboghbaish, E. (2024). *Electrocardiogram signals
classification using deep-learning-based incorporated convolutional
neural network and long short-term memory framework*. IEEE Access, 12,
14223--14232. https://doi.org/10.1109/ACCESS.2024.3334562

Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M.,
Ivanov, P. C., Mark, R. G., Mietus, J. E., Moody, G. B., Peng, C. K., &
Stanley, H. E. (2000). *PhysioBank, PhysioToolkit, and PhysioNet:
Components of a new research resource for complex physiologic signals.
Circulation,* 101(23), e215--e220.
https://doi.org/10.1161/01.CIR.101.23.e215
