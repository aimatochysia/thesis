## 4.x Evaluasi Sistem Frontend Deployment

Subbab ini menjelaskan evaluasi sistem deployment frontend yang mensimulasikan pemantauan ECG real-time dalam lingkungan klinis.

### 4.x.1 Arsitektur Frontend Deployment

Sistem frontend deployment dirancang untuk mensimulasikan pemantauan ECG real-time menggunakan data MIT-BIH Record 119 sebagai sumber sinyal. Arsitektur sistem terdiri dari beberapa komponen utama yang bekerja secara terintegrasi.

Komponen pertama adalah sumber data yang terdiri dari file sinyal ECG (119.csv) berisi nilai-nilai MLII dan file anotasi (119annotations.txt) yang menyimpan lokasi R-peak beserta jenis beat-nya. Kedua file ini dibaca oleh server Flask yang berperan sebagai backend aplikasi.

Komponen kedua adalah modul ekstraksi beat yang mengambil 200 sampel dari sinyal ECG untuk setiap R-peak. Ekstraksi ini mengambil 90 sampel sebelum R-peak dan 110 sampel setelah R-peak, sesuai dengan konfigurasi yang digunakan saat training model.

Komponen ketiga adalah rolling buffer yang menyimpan 7 beat terakhir. Buffer ini diperlukan karena model v6 menggunakan pendekatan context-aware yang membutuhkan informasi dari beat-beat sebelum dan sesudah beat yang akan diklasifikasi.

Komponen keempat adalah modul preprocessing yang melakukan flatten data dari bentuk 7 beat menjadi vektor 1400 elemen, lalu menormalisasi menggunakan StandardScaler yang sama dengan training, dan terakhir me-reshape kembali menjadi bentuk yang sesuai untuk input model.

Komponen kelima adalah ONNX Runtime yang menjalankan inferensi model context_ecg_model.onnx untuk menghasilkan prediksi NORMAL atau ABNORMAL.

Komponen keenam adalah antarmuka visualisasi berbasis HTML5 Canvas dan JavaScript yang menampilkan sinyal ECG secara real-time beserta hasil prediksi dan perbandingan dengan ground truth.

---

### 4.x.2 Penggunaan Record 119

Record 119 dipilih sebagai data pengujian karena beberapa alasan penting dalam konteks validasi model.

Pertama, record 119 tidak termasuk dalam proses training model v6. Record ini secara sengaja dikecualikan dari dataset training sehingga mewakili data yang benar-benar baru bagi model. Hal ini memastikan tidak ada kebocoran data antara training dan testing.

Kedua, penggunaan record 119 memberikan validasi sejati terhadap kemampuan generalisasi model. Karena record ini berasal dari pasien yang berbeda dengan pasien-pasien dalam data training, performa model pada record ini mencerminkan kemampuan model dalam menangani variasi karakteristik ECG antar pasien.

Ketiga, pendekatan ini mengikuti paradigma evaluasi inter-pasien yang direkomendasikan oleh standar AAMI EC57:2012 untuk evaluasi sistem klasifikasi beat ECG. Paradigma ini mensyaratkan bahwa data testing berasal dari pasien yang tidak digunakan dalam training.

Keempat, record 119 dipilih secara konsisten untuk semua versi model (v2, v3, v5, dan v6) sehingga memungkinkan perbandingan performa yang fair antar versi.

---

### 4.x.3 Pipeline Preprocessing

Pipeline preprocessing pada deployment harus persis sama dengan pipeline yang digunakan saat training untuk memastikan konsistensi hasil prediksi.

Tahap pertama adalah ekstraksi beat. Setiap beat diekstraksi dengan mengambil 200 sampel yang berpusat pada lokasi R-peak. Pengambilan dilakukan dengan mengambil 90 sampel sebelum R-peak yang berfungsi untuk menangkap gelombang P dengan durasi sekitar 250 milidetik, dan 110 sampel setelah R-peak yang berfungsi untuk menangkap segmen ST dan gelombang T dengan durasi sekitar 306 milidetik. Total 200 sampel ini memberikan window sekitar 556 milidetik pada frekuensi sampling 360 Hz, yang cukup untuk menangkap kompleks PQRST secara lengkap. Untuk menangani kasus tepi di awal atau akhir sinyal, sistem menggunakan zero padding.

Tahap kedua adalah pengisian rolling buffer dengan 7 beat. Sistem mempertahankan buffer yang selalu berisi 7 beat terakhir. Setiap kali beat baru terdeteksi, beat tersebut ditambahkan ke buffer dan beat tertua dikeluarkan jika buffer sudah penuh. Sistem membutuhkan buffer yang terisi penuh (7 beat) sebelum dapat melakukan inferensi.

Window konteks 7 beat dipilih dengan konfigurasi 3 beat sebelumnya, 1 beat tengah sebagai target klasifikasi, dan 3 beat sesudahnya. Konfigurasi ini memungkinkan model mendeteksi pola aritmia multi-beat seperti bigeminy dan trigeminy dimana PVC bergantian dengan beat normal, variabilitas R-R yang menunjukkan pola irregular pada atrial fibrillation, pause kompensatori yang muncul setelah PVC, dan AV block yang menunjukkan perubahan sistematis pada interval PR.

Tahap ketiga adalah normalisasi. Data 7 beat di-stack menjadi matriks dengan bentuk 7 kali 200, kemudian di-flatten menjadi vektor dengan 1400 elemen. Vektor ini dinormalisasi menggunakan StandardScaler yang sama dengan yang digunakan saat training. Scaler ini di-fit hanya pada data training sehingga tidak ada kebocoran informasi dari data testing. Hasil normalisasi kemudian di-reshape menjadi bentuk 1 kali 7 kali 200 untuk input ke model ONNX.

---

### 4.x.4 Proses Inferensi ONNX

Proses inferensi menggunakan ONNX Runtime untuk menjalankan model context_ecg_model.onnx. Setelah input disiapkan melalui pipeline preprocessing, data dikirim ke model untuk mendapatkan output berupa logits untuk kelas normal dan abnormal.

Output logits dari model dikonversi menjadi probabilitas menggunakan fungsi softmax. Softmax dihitung dengan mengambil eksponensial dari setiap logit setelah dikurangi nilai maksimum untuk stabilitas numerik, kemudian membagi dengan jumlah total eksponensial.

Klasifikasi akhir ditentukan berdasarkan threshold 0.5 pada probabilitas kelas abnormal. Jika probabilitas abnormal lebih besar atau sama dengan 0.5, beat diklasifikasikan sebagai ABNORMAL. Jika tidak, beat diklasifikasikan sebagai NORMAL.

Hasil inferensi mencakup kelas prediksi, probabilitas abnormal, dan ground truth dari beat tengah dalam window 7 beat.

---

### 4.x.5 Perbandingan Ground Truth

Ground truth untuk evaluasi diambil dari file anotasi MIT-BIH. Dalam konteks klasifikasi biner yang digunakan oleh sistem, beat dengan label N dianggap NORMAL, sedangkan semua label lainnya seperti V, A, L, R, dan lainnya dianggap ABNORMAL.

Beat yang digunakan untuk ground truth adalah beat tengah dalam window 7 beat, yaitu beat pada indeks ke-3 dari buffer (dengan penghitungan mulai dari 0). Hal ini karena model dilatih untuk memprediksi klasifikasi beat tengah, sementara beat-beat sekitarnya hanya memberikan konteks temporal.

Perbandingan antara prediksi model dan ground truth ditampilkan secara visual pada antarmuka dengan penanda berwarna. Prediksi yang benar ditandai dengan warna hijau, sedangkan prediksi yang salah ditandai dengan warna merah.

---

### 4.x.6 Fitur Antarmuka Pengguna

Antarmuka pengguna menyediakan beberapa fitur untuk memudahkan penggunaan dan evaluasi sistem.

Fitur pertama adalah kontrol kecepatan playback. Pengguna dapat memilih dari beberapa preset kecepatan mulai dari 0.1x (10 kali lebih lambat dari real-time) hingga 10x (10 kali lebih cepat dari real-time). Kecepatan 1x berarti playback berjalan pada kecepatan real-time yaitu 360 sampel per detik sesuai dengan frekuensi sampling MIT-BIH.

Fitur kedua adalah tampilan BPM (beats per minute). Sistem menghitung BPM berdasarkan rata-rata interval antara 10 beat terakhir. Perhitungan menggunakan filter untuk mengabaikan interval yang tidak masuk akal secara fisiologis, yaitu di bawah 0.3 detik (setara dengan 200 BPM) atau di atas 2 detik (setara dengan 30 BPM). Filtering ini membantu menghaluskan tampilan BPM dan menghindari fluktuasi yang disebabkan oleh beat yang terlewat atau deteksi ganda.

Fitur ketiga adalah navigasi riwayat. Pengguna dapat menelusuri sinyal ECG yang sudah direkam dengan menggunakan tombol navigasi. Terdapat tombol untuk mundur atau maju 1 detik, mundur atau maju 5 detik, dan tombol untuk kembali ke tampilan live.

Fitur keempat adalah log deteksi salah. Setiap kali prediksi model berbeda dengan ground truth, kejadian tersebut dicatat dalam log. Log menampilkan waktu kejadian, ground truth yang diharapkan, dan prediksi model. Setiap entri dalam log dapat diklik untuk navigasi langsung ke waktu tersebut dalam sinyal.

---

### 4.x.7 Fitur Stabilitas Grafik

Grafik ECG diimplementasikan dengan dua fitur stabilitas untuk meningkatkan keterbacaan.

Fitur pertama adalah stabilitas tinggi grafik. Tinggi area visualisasi ECG dapat bertambah sesuai kebutuhan tetapi tidak pernah mengecil. Saat beat dengan amplitudo tinggi terdeteksi, grafik akan melebar secara vertikal untuk mengakomodasi. Setelah itu, meskipun beat-beat selanjutnya memiliki amplitudo yang lebih rendah, tinggi grafik tetap dipertahankan. Pendekatan ini memastikan konsistensi visual dan menghindari efek "zoom" yang membingungkan.

Fitur kedua adalah stabilitas skala Y-axis. Sistem melacak nilai minimum dan maksimum global dari seluruh rekaman yang sudah ditampilkan. Skala vertikal grafik menggunakan nilai global ini sebagai referensi, bukan nilai dari buffer yang sedang ditampilkan. Dengan demikian, beat dengan amplitudo tinggi yang sudah lewat tetap menjadi referensi skala, dan beat dengan amplitudo rendah tidak menyebabkan efek "zoom in" yang dapat menyesatkan interpretasi klinis.

Kedua fitur ini direset saat simulasi di-restart, memungkinkan sesi baru dimulai dengan skala yang segar.

---

### 4.x.8 Sistem Ekspor Otomatis

Sistem ekspor dirancang untuk kemudahan penggunaan oleh dokter atau perawat tanpa memerlukan langkah manual yang rumit.

Sistem menggunakan pendekatan auto-batch dimana selama rekaman berlangsung, sistem secara otomatis menyimpan snapshot grafik setiap 2 menit ke dalam memori browser. Penyimpanan ini berjalan di background tanpa mengganggu tampilan atau performa sistem.

Saat pengguna menghentikan rekaman dengan tombol Stop, sistem secara otomatis menyimpan data terakhir yang belum ter-batch sebagai batch final. Pengguna kemudian dapat mengunduh semua batch dengan mengklik tombol Download Batches ZIP.

Semua batch dibundel dalam satu file ZIP untuk memudahkan pengelolaan. File ZIP berisi semua gambar batch dalam format PNG dengan penamaan yang mencakup nomor urut batch dan rentang waktu yang tercakup.

Alur kerja yang sederhana untuk dokter atau perawat adalah sebagai berikut. Pertama, tekan tombol Start untuk memulai rekaman. Kedua, biarkan rekaman berjalan dimana batch akan otomatis tersimpan setiap 2 menit. Ketiga, tekan tombol Stop saat selesai. Keempat, klik tombol Download Batches ZIP untuk mengunduh semua hasil dalam satu file ZIP.

---

### 4.x.9 Performa yang Diharapkan

Berdasarkan evaluasi pada data test set yang menggunakan paradigma split record-wise, performa yang diharapkan pada record 119 adalah sebagai berikut.

Akurasi pada data validasi sejati diperkirakan sekitar 69 persen. Recall untuk kelas abnormal diperkirakan sekitar 55 persen. Nilai AUC-ROC diperkirakan sekitar 0.80. Untuk perbandingan, akurasi pada test set dengan distribusi yang lebih serupa dengan training mencapai 94 sampai 98 persen.

Perbedaan performa antara record 119 dan test set disebabkan oleh beberapa faktor. Model v6 menggunakan split record-wise yang memastikan tidak ada kebocoran informasi antar pasien, sehingga performa lebih rendah tetapi lebih realistis. Record 119 mungkin memiliki karakteristik yang berbeda dari rekaman-rekaman yang digunakan dalam training. Angka-angka ini mencerminkan performa yang dapat diharapkan dalam deployment dunia nyata dimana model akan menemui pasien baru yang tidak pernah dilihat sebelumnya.

---

### 4.x.10 Kesimpulan Evaluasi Sistem

Sistem deployment frontend berhasil mengimplementasikan simulasi pemantauan ECG real-time yang komprehensif dengan beberapa keunggulan utama.

Pertama, sistem mensimulasikan kondisi pemantauan ECG live dengan menampilkan sinyal secara real-time dan memberikan prediksi untuk setiap beat yang terdeteksi.

Kedua, model context-aware yang digunakan memanfaatkan pola temporal antar beat untuk meningkatkan akurasi klasifikasi aritmia yang bergantung pada konteks.

Ketiga, penggunaan record 119 yang tidak termasuk dalam training memastikan tidak ada kebocoran data dan memberikan estimasi performa yang realistis.

Keempat, pipeline preprocessing pada deployment dirancang untuk persis sama dengan training, memastikan konsistensi antara bagaimana model dilatih dan bagaimana model digunakan.

Kelima, sistem menyediakan utilitas klinis berupa perbandingan dengan ground truth dan logging deteksi salah yang membantu evaluasi performa model.

Keenam, fitur kemudahan penggunaan seperti ekspor otomatis, stabilitas grafik, dan navigasi intuitif membuat sistem dapat digunakan oleh tenaga kesehatan tanpa keahlian teknis khusus.

Sistem ini siap untuk demonstrasi dan validasi lebih lanjut dalam lingkungan klinis yang terkontrol.
