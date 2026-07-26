# Dashboard Analisis Performa Mahasiswa

Dashboard interaktif berbasis **Streamlit** untuk menganalisis pengaruh gaya hidup (jam belajar, kehadiran, tingkat stres, sesi tutoring, kecemasan ujian, dan faktor lain) terhadap nilai akhir mahasiswa.

**Live demo:** https://perkembangan-mahasiswa.streamlit.app/

Proyek ini dibuat sebagai bagian dari tugas UAS mata kuliah **Visualisasi Data**, Program Studi Informatika, Universitas Ahmad Dahlan (Kelas B, Semester VI, Tahun Ajaran 2025/2026).

## Anggota Kelompok

| No | Nama | NIM |
|----|------|-----|
| 1 | Wildan Mursalin Rizqia | 2300018372 |
| 2 | Rayhan Arkananta Adrian | 2300018384 |

## Isi Repositori

| File | Deskripsi |
|---|---|
| `app.py` | Kode utama dashboard Streamlit (load data, filter interaktif, visualisasi Plotly). |
| `requirements.txt` | Daftar dependensi Python yang dibutuhkan untuk menjalankan dashboard. |
| `student_performance.csv` | **Dataset mentah** (raw) sebelum proses cleaning, diunduh dari Kaggle. |
| `student_performance_preprocessed.csv` | **Dataset bersih** (hasil cleaning: penghapusan kolom `Student_ID`, penyaringan baris `Gender` di luar Male/Female — menyisakan **7.716 baris**, serta penambahan kolom kelompok/kategori seperti `Kelompok_Jam_Belajar` dan `Kelompok_Stres` untuk kebutuhan visualisasi). |
| `README.md` | Dokumen ini. |

## Sumber Dataset

- **Nama dataset:** Student Lifestyle & GPA Prediction Dataset
- **Sumber:** [Kaggle](https://www.kaggle.com/datasets/sarveshchhetri/student-lifestyle-vs-academic-performance-dataset)
- **Ukuran:** 8.000 baris × 18 kolom (mentah) → **7.716 baris × 17 kolom** (bersih, setelah penghapusan kolom `Student_ID` dan penyaringan baris `Gender` di luar kategori Male/Female)

## Fitur Dashboard

- **Filter interaktif** di sidebar: rentang usia, gender, status kerja part-time, metode belajar, tingkat pendapatan keluarga, dan keikutsertaan ekstrakurikuler.
- **Ringkasan KPI**: jumlah mahasiswa, rata-rata nilai akhir, nilai tertinggi, GPA tertinggi, rata-rata jam belajar, dan rata-rata kehadiran (menyesuaikan otomatis dengan filter yang dipilih).
- **Analisis faktor paling berpengaruh** terhadap nilai akhir (bar chart korelasi) dan **matriks korelasi** antar variabel numerik (heatmap).
- **Visualisasi hubungan gaya hidup dengan nilai akhir**, di antaranya:
  - Tren rata-rata nilai akhir berdasarkan kelompok jam belajar
  - Hubungan tingkat kehadiran dengan nilai akhir (scatter plot + garis tren)
  - Dampak tingkat stres terhadap nilai akhir (box plot)
  - Efektivitas jumlah sesi tutoring per minggu terhadap nilai akhir
  - Perbandingan nilai akhir berdasarkan metode belajar
  - Hubungan status kerja part-time dengan nilai akhir

## Teknologi yang Digunakan

- [Streamlit](https://streamlit.io/) — kerangka kerja dashboard interaktif
- [Pandas](https://pandas.pydata.org/) — pengolahan dan manipulasi data
- [Plotly Express & Graph Objects](https://plotly.com/python/) — visualisasi data interaktif
- [NumPy](https://numpy.org/) — komputasi numerik
- [statsmodels](https://www.statsmodels.org/) — garis tren (trendline OLS) pada scatter plot

## Cara Menjalankan Secara Lokal

1. Clone repositori ini:
   ```bash
   git clone https://github.com/weemmm/dashboarding-streamlit.git
   cd dashboarding-streamlit
   ```
2. (Opsional tapi disarankan) buat virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan dashboard:
   ```bash
   streamlit run app.py
   ```
5. Dashboard akan terbuka otomatis di browser pada `http://localhost:8501`.

## Proses Data Cleaning (Ringkas)

1. Pemeriksaan jumlah baris & kolom, serta tipe data setiap kolom.
2. Pemeriksaan missing value, hasil: **0 missing value** pada seluruh kolom.
3. Pemeriksaan data duplikat  hasil: **0 duplikat**.
4. Penghapusan kolom `Student_ID` karena bersifat identifier dan tidak relevan untuk analisis/visualisasi.
5. Penyaringan baris dengan nilai `Gender` di luar kategori `Male`/`Female`.
6. Penambahan kolom turunan (`Kelompok_Jam_Belajar`, `Kelompok_Stres`) untuk mempermudah visualisasi berbasis kategori pada dashboard.

Detail lengkap proses cleaning beserta tangkapan layar dapat dilihat pada laporan project (Bagian 2: Metodologi dan Jejak Data).

## Lisensi

Dataset digunakan mengikuti lisensi yang tercantum pada halaman dataset di Kaggle, untuk keperluan tugas akademik (edukasi/riset non-komersial).
