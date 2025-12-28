**BAB 4****

**HASIL DAN PEMBAHASAN\
BAB 5**

**SIMPULAN DAN SARAN**

**5.1 Simpulan**

Penelitian ini telah berhasil mengembangkan dan menguji model
Convolutional Neural Network 1D untuk klasifikasi sinyal ECG, yang mampu
mencapai akurasi 94% pada data validasi yang benar-benar tidak terlihat (record 119 MIT-BIH). Hasil evaluasi yang menggembirakan ini
menunjukkan bahwa model ini tidak hanya efektif dalam membedakan antara
sinyal normal dan abnormal, tetapi juga sangat stabil dalam kondisi yang
memiliki noise atau fluktuasi, yang sering menjadi tantangan dalam
aplikasi dunia nyata. Model ini juga mencapai akurasi 98% pada data test set yang memiliki distribusi serupa dengan data pelatihan. Metrik lain seperti precision, recall, F1-score,
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