# Prediksi-Gagal-Bayar-Kartu-Kredit

# 💳 Prediksi Gagal Bayar Kartu Kredit

### Analisis Risiko Kredit | Exploratory Data Analysis | Machine Learning

> **Mengubah perilaku pembayaran pelanggan menjadi insight risiko kredit yang dapat mendukung pengambilan keputusan menggunakan Python dan Machine Learning.**

[Python](https://www.python.org/) ([image](https://img.shields.io/badge/Python-3.x-blue?logo=python))
[Pandas](https://pandas.pydata.org/) ([image](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas))
[Scikit-learn](https://scikit-learn.org/) ([image](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn))
[XGBoost](https://xgboost.readthedocs.io/) ([image](https://img.shields.io/badge/XGBoost-Classification-337AB7))

---

## 📌 Ringkasan Proyek

Gagal bayar kartu kredit merupakan salah satu tantangan penting dalam manajemen risiko di industri keuangan. Identifikasi lebih awal terhadap pelanggan yang memiliki risiko gagal bayar dapat membantu mendukung proses penilaian risiko kredit dan pengelolaan pelanggan secara lebih tepat.

Dalam proyek ini, dilakukan analisis terhadap **30.000 data pelanggan dengan 25 variabel** yang mencakup informasi demografis, batas kredit, riwayat pembayaran, jumlah tagihan, serta jumlah pembayaran.

Proyek ini menggabungkan **Exploratory Data Analysis (EDA)** dan **Machine Learning classification** untuk memahami perilaku pembayaran pelanggan serta memprediksi kemungkinan terjadinya gagal bayar kartu kredit.

### Hal yang Ditunjukkan dalam Proyek

* Eksplorasi dan pemahaman data
* Data preparation dan feature selection
* Analisis perilaku pembayaran pelanggan
* Pembangunan model klasifikasi
* Implementasi Random Forest dan XGBoost
* Hyperparameter tuning menggunakan `GridSearchCV`
* 5-fold cross-validation
* Evaluasi model menggunakan ROC-AUC
* Analisis feature importance
* Menerjemahkan hasil analisis menjadi insight yang relevan bagi bisnis

---

# 🎯 Permasalahan Bisnis

Dari perspektif perbankan, pertanyaan utama dalam proyek ini adalah:

> **Apakah riwayat perilaku pembayaran pelanggan dapat digunakan untuk mengidentifikasi pelanggan yang memiliki kemungkinan lebih tinggi untuk mengalami gagal bayar pada pembayaran kartu kredit berikutnya?**

Proyek ini menganalisis permasalahan tersebut dari dua perspektif:

### 1. Descriptive Analytics

Memahami **siapa pelanggan dan bagaimana perilaku finansial mereka**.

### 2. Predictive Analytics

Membangun model klasifikasi untuk membedakan antara:

* **0 — Tidak Gagal Bayar**
* **1 — Gagal Bayar**

Tujuan proyek bukan hanya memperoleh akurasi prediksi yang tinggi, tetapi juga memahami variabel yang berkontribusi terhadap prediksi serta bagaimana performa model dapat diinterpretasikan dalam konteks risiko kredit.

---

# 📊 Dataset

Dataset terdiri dari:

* **30.000 data pelanggan**
* **25 variabel**
* Informasi demografis
* Informasi batas kredit
* Riwayat status pembayaran
* Jumlah tagihan
* Jumlah pembayaran sebelumnya
* Status gagal bayar kartu kredit

### Target Variable

```text
default.payment.next.month
```

| Nilai | Keterangan                              |
| ----- | --------------------------------------- |
| `0`   | Tidak gagal bayar pada bulan berikutnya |
| `1`   | Gagal bayar pada bulan berikutnya       |

### Fitur Utama

| Fitur       | Makna Bisnis                            |
| ----------- | --------------------------------------- |
| `LIMIT_BAL` | Batas kredit yang diberikan             |
| `AGE`       | Usia pelanggan                          |
| `SEX`       | Jenis kelamin pelanggan                 |
| `EDUCATION` | Tingkat pendidikan                      |
| `MARRIAGE`  | Status pernikahan                       |
| `PAY_0`     | Status pembayaran sebelumnya            |
| `PAY_2`     | Status pembayaran dua bulan sebelumnya  |
| `PAY_3`     | Status pembayaran dari bulan sebelumnya |
| `BILL_AMT1` | Jumlah tagihan sebelumnya               |
| `PAY_AMT1`  | Jumlah pembayaran sebelumnya            |

Struktur dataset asli dan target variable digunakan dalam kedua notebook proyek.

---

# 🔎 Exploratory Data Analysis

Sebelum membangun model prediksi, proyek terlebih dahulu menganalisis karakteristik dan pola yang terdapat dalam data.

### Analisis yang Dilakukan

* Pemeriksaan dataset
* Analisis struktur dan tipe data
* Statistik deskriptif
* Analisis target variable
* Analisis perilaku pembayaran pelanggan
* Analisis batas kredit
* Analisis jumlah tagihan
* Analisis jumlah pembayaran
* Analisis korelasi
* Analisis hubungan status pembayaran dengan gagal bayar

Analisis dilakukan menggunakan **Pandas, Matplotlib, dan Seaborn** untuk eksplorasi dan visualisasi data.

### Mengapa EDA Penting?

EDA tidak hanya digunakan untuk membuat visualisasi, tetapi juga untuk memahami karakteristik pelanggan dan perilaku pembayaran yang berpotensi berhubungan dengan risiko gagal bayar sebelum model machine learning dibangun.

Alur analisis dalam proyek:

```text
Data Mentah
   ↓
Pemahaman Data
   ↓
Exploratory Data Analysis
   ↓
Feature Selection
   ↓
Pengembangan Model
   ↓
Evaluasi Model
   ↓
Interpretasi Bisnis
```

---

# 🤖 Pendekatan Machine Learning

Dalam proyek ini digunakan dua pendekatan machine learning:

## 1. Random Forest

Random Forest digunakan sebagai metode klasifikasi berbasis decision tree untuk memprediksi kemungkinan gagal bayar pelanggan.

Fitur yang digunakan antara lain:

```text
LIMIT_BAL
AGE
SEX
EDUCATION
MARRIAGE
PAY_0
PAY_2
PAY_3
BILL_AMT1
PAY_AMT1
```

Data dibagi menjadi:

* **80% data training**
* **20% data testing**
* `random_state = 42`
* Stratified splitting

Pembagian data 80:20 secara stratified tersebut diterapkan langsung dalam notebook XGBoost.

---

# 🚀 Pemodelan XGBoost

XGBoost digunakan sebagai pendekatan klasifikasi berbasis boosting.

Berbeda dari hanya menggunakan parameter default, proyek ini melakukan **hyperparameter tuning secara sistematis menggunakan GridSearchCV dengan 5-fold cross-validation**.

### Hyperparameter yang Dievaluasi

| Parameter           | Tujuan                                                             |
| ------------------- | ------------------------------------------------------------------ |
| `n_estimators`      | Jumlah boosting trees                                              |
| `max_depth`         | Kedalaman maksimum tree                                            |
| `min_child_weight`  | Bobot minimum yang diperlukan untuk child node                     |
| `learning_rate`     | Ukuran langkah dalam proses boosting                               |
| `gamma`             | Minimum pengurangan loss yang diperlukan untuk melakukan splitting |
| `colsample_bylevel` | Sampling fitur pada setiap level tree                              |
| `subsample`         | Sampling baris data untuk proses training                          |

Model dievaluasi menggunakan **ROC-AUC** sebagai scoring metric selama proses hyperparameter tuning.

---

# 📈 Hyperparameter Optimization

Proses tuning dilakukan secara bertahap.

### Hasil Cross-Validation Terbaik

| Hyperparameter      | Nilai Terbaik | CV ROC-AUC |
| ------------------- | ------------: | ---------: |
| `n_estimators`      |           100 |     0.7597 |
| `max_depth`         |             4 |     0.7716 |
| `min_child_weight`  |             7 |     0.7657 |
| `learning_rate`     |          0.05 | **0.7782** |
| `gamma`             |           2.0 |     0.7745 |
| `colsample_bylevel` |          0.25 |     0.7675 |
| `subsample`         |           1.0 |     0.7597 |

ROC-AUC cross-validation tertinggi yang diperoleh selama eksperimen tuning adalah sekitar **0.7782**, yaitu pada `learning_rate = 0.05`.

Konfigurasi XGBoost yang tercatat dalam notebook adalah:

```python
XGBClassifier(
    n_estimators=100,
    max_depth=4,
    min_child_weight=7,
    learning_rate=0.05,
    gamma=2.0,
    colsample_bylevel=0.25,
    subsample=1.0,
    eval_metric='logloss'
)
```

---

# 📊 Evaluasi Model

Karena proyek ini merupakan **permasalahan klasifikasi risiko kredit**, evaluasi model tidak cukup hanya menggunakan accuracy.

Metrik yang digunakan meliputi:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC Curve
* ROC-AUC

### Mengapa Menggunakan ROC-AUC?

ROC-AUC digunakan dalam proses hyperparameter tuning karena model perlu membedakan antara pelanggan yang mengalami gagal bayar dan pelanggan yang tidak mengalami gagal bayar.

Dalam konteks risiko kredit, tujuan model bukan hanya memaksimalkan jumlah prediksi yang benar, tetapi juga mengidentifikasi pelanggan dengan risiko lebih tinggi secara efektif.

---

# 🔍 Interpretasi Model

Model prediksi akan lebih bermanfaat apabila hasilnya dapat diinterpretasikan.

Oleh karena itu, proyek ini juga menganalisis **feature importance** untuk memahami karakteristik pelanggan yang berkontribusi terhadap prediksi model.

Perhatian khusus diberikan pada variabel status pembayaran sebelumnya, seperti:

```text
PAY_0
PAY_2
PAY_3
```

Variabel tersebut merepresentasikan riwayat perilaku pembayaran pelanggan sehingga relevan dalam analisis risiko kredit.

> **Interpretasi bisnis:** Riwayat perilaku pembayaran memberikan informasi penting dalam membedakan pelanggan dengan tingkat risiko gagal bayar yang berbeda.

---

# 💼 Insight Bisnis

Hasil analisis memberikan beberapa perspektif yang dapat relevan bagi institusi keuangan.

### 1. Perilaku pembayaran memiliki peran penting

Riwayat status pembayaran merupakan salah satu sinyal penting dalam memprediksi perilaku gagal bayar di masa mendatang.

### 2. Analisis risiko kredit perlu berbasis data

Variabel demografis dan finansial dapat dianalisis secara bersama-sama daripada hanya mengandalkan satu karakteristik pelanggan.

### 3. Pemilihan model harus mempertimbangkan tujuan bisnis

Model dengan accuracy lebih tinggi tidak selalu menjadi model terbaik untuk aplikasi risiko kredit.

Metrik seperti **recall, precision, F1-score, dan ROC-AUC** juga perlu dipertimbangkan.

### 4. Interpretasi model penting dalam pengambilan keputusan

Memahami variabel yang memengaruhi prediksi dapat membantu analis menjelaskan hasil machine learning kepada stakeholder non-teknis.

---

# 🧠 Kemampuan Analitis yang Ditunjukkan

Proyek ini menunjukkan workflow yang lebih luas daripada sekadar melakukan training model machine learning.

### Perspektif Data Analyst

```text
Memahami Data
        ↓
Mengidentifikasi Pola
        ↓
Menemukan Variabel Relevan
        ↓
Menghasilkan Insight
```

### Perspektif Data Science

```text
Persiapan Data
      ↓
Feature Selection
      ↓
Training Model
      ↓
Cross-Validation
      ↓
Hyperparameter Tuning
      ↓
Evaluasi Performa
```

### Perspektif Bisnis

```text
Data Pelanggan
      ↓
Pola Risiko
      ↓
Prediksi
      ↓
Dukungan Pengambilan Keputusan
```

Kombinasi tersebut membuat proyek ini relevan untuk menunjukkan kemampuan **Data Analyst** maupun **Junior Data Scientist**.

---

# 🛠️ Tools & Skills

### Programming & Data

* Python
* Pandas
* NumPy

### Data Analysis

* Exploratory Data Analysis
* Data Cleaning
* Data Preparation
* Feature Selection
* Statistical / Descriptive Analysis

### Machine Learning

* Random Forest
* XGBoost
* Classification
* Hyperparameter Tuning
* GridSearchCV
* Cross-Validation

### Model Evaluation

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC Curve
* ROC-AUC

### Visualization

* Matplotlib
* Seaborn

---

# 📁 Struktur Proyek

```text
Credit-Card-Default-Prediction/
│
├── README.md
│
├── Skripsi_perbankan_EDA+RM.ipynb
│   └── Exploratory Data Analysis & Random Forest
│
├── Skripsi_XGBoost.ipynb
│   └── XGBoost Modeling & Hyperparameter Tuning
│
└── Perbankan.csv
```

---

# 🎓 Konteks Proyek

Proyek ini dikembangkan sebagai bagian dari penelitian/skripsi akademik dalam bidang **Matematika, Analisis Data, dan Machine Learning**.

Proyek menunjukkan bagaimana analisis kuantitatif dapat diterapkan pada permasalahan finansial di dunia nyata, mulai dari data pelanggan hingga menghasilkan model prediksi dan interpretasi bisnis.

---

# 🚀 Potensi Penerapan Bisnis

Workflow analisis seperti ini dapat dikembangkan untuk berbagai kebutuhan, seperti:

* Analisis risiko kredit
* Segmentasi risiko pelanggan
* Early-warning system
* Monitoring portofolio pelanggan
* Analisis portofolio kartu kredit
* Risk-based customer management

> **Catatan:** Model dalam proyek ini merupakan model analitik/penelitian dan belum dapat dianggap sebagai sistem persetujuan kredit yang siap digunakan dalam lingkungan produksi tanpa validasi lebih lanjut, monitoring, evaluasi fairness, calibration, serta review dari pihak yang memiliki keahlian di bidang bisnis dan risiko kredit.

---

# 🔮 Pengembangan Selanjutnya

Untuk mengembangkan proyek ini menjadi solusi analitik yang lebih mendekati kebutuhan produksi, beberapa pengembangan yang dapat dilakukan antara lain:

* Menangani class imbalance.
* Meningkatkan recall untuk kelas gagal bayar.
* Membandingkan Random Forest dan XGBoost pada independent test set yang sama.
* Mengoptimalkan classification threshold berdasarkan biaya bisnis.
* Menggunakan SHAP untuk meningkatkan interpretasi model.
* Melakukan probability calibration.
* Mengembangkan dashboard risiko kredit.
* Melakukan monitoring performa model dari waktu ke waktu.
* Mengevaluasi fairness model pada berbagai segmen pelanggan.

---

# 👤 Tentang Penulis

**Faris Fatur Rohman**

🎓 Lulusan Matematika
📊 Aspiring Data Analyst / Junior Data Scientist
💡 Memiliki minat pada Data Analytics, Business Intelligence, Machine Learning, dan Financial Data Analysis.

### Keahlian Utama

`Python` · `SQL` · `Excel` · `Power BI` · `Pandas` · `NumPy` · `EDA` · `Data Visualization` · `Machine Learning`

---

## ⭐ Pendekatan Saya dalam Proyek Ini

> **Saya tidak hanya membangun model, tetapi memulai dari permasalahan bisnis, memahami data, mengevaluasi model secara tepat, dan menerjemahkan hasil analisis menjadi insight yang dapat mendukung pengambilan keputusan.**
