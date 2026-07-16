import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# scipy dipakai untuk uji ANOVA (signifikansi perbedaan antar kategori).
# Dibungkus try/except supaya app tetap jalan meski scipy belum terpasang;
# insight hanya akan menampilkan versi tanpa keterangan signifikansi.
try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Dashboard Performa Mahasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# LOAD DATA
# =========================================================
DATA_FILE = "student_performance_preprocessed.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    # Kategori terurut untuk kolom ordinal
    diet_order = ["Poor", "Average", "Good", "Excellent"]
    internet_order = ["Poor", "Average", "Good", "Excellent"]
    income_order = ["Low", "Middle", "High"]
    predikat_order = ["Kurang (<60)", "Cukup (60-75)", "Baik (75-90)", "Sangat Baik (>90)"]
    jam_belajar_order = ["0-2 jam", "2-4 jam", "4-6 jam", "6-8 jam", "> 8 jam"]
    kehadiran_order = ["<60%", "60-70%", "70-80%", "80-90%", ">90%"]
    stres_order = ["Rendah (1-3)", "Sedang (3-5)", "Tinggi (5-7)", "Sangat Tinggi (7-10)"]
    for col, order in [
        ("Diet_Quality", diet_order),
        ("Internet_Quality", internet_order),
        ("Family_Income_Level", income_order),
        ("Predikat", predikat_order),
        ("Kelompok_Jam_Belajar", jam_belajar_order),
        ("Kelompok_Kehadiran", kehadiran_order),
        ("Kelompok_Stres", stres_order),
    ]:
        cats = [c for c in order if c in df[col].unique()]
        extra = [c for c in df[col].unique() if c not in cats]
        df[col] = pd.Categorical(df[col], categories=cats + extra, ordered=True)
    return df

df = load_data()

NUMERIC_COLS = [
    "Age", "Hours_Studied", "Attendance", "Sleep_Hours", "Stress_Level",
    "Screen_Time", "Previous_GPA", "Tutoring_Sessions_Per_Week",
    "Exam_Anxiety_Score", "Final_Score",
]
# Kategori "asli" dari data mentah
CATEGORICAL_COLS = [
    "Gender", "Part_Time_Job", "Study_Method", "Diet_Quality",
    "Internet_Quality", "Extracurricular", "Family_Income_Level",
]
# Kategori hasil pengelompokan/binning yang sudah disiapkan di dataset preprocessed
GROUPED_COLS = [
    "Predikat", "Kelompok_Jam_Belajar", "Kelompok_Kehadiran", "Kelompok_Stres",
]
# Gabungan seluruh kolom kategorikal yang tersedia untuk dropdown perbandingan/eksplorasi
ALL_CATEGORICAL_COLS = CATEGORICAL_COLS + GROUPED_COLS
# Untuk perbandingan "rata-rata Skor Akhir per kategori", kolom Predikat dikecualikan
# karena Predikat adalah hasil pembagian rentang dari Final_Score itu sendiri —
# membandingkannya akan sirkular/tautologis (Predikat "Sangat Baik" pasti >90, dst.)
COMPARISON_COLS = [c for c in ALL_CATEGORICAL_COLS if c != "Predikat"]

# =========================================================
# LABEL BAHASA INDONESIA (konsisten di semua dropdown & chart)
# =========================================================
LABEL_MAP = {
    # Variabel numerik
    "Age": "Usia",
    "Hours_Studied": "Jam Belajar",
    "Attendance": "Kehadiran (%)",
    "Sleep_Hours": "Jam Tidur",
    "Stress_Level": "Tingkat Stres",
    "Screen_Time": "Waktu Layar",
    "Previous_GPA": "GPA Sebelumnya",
    "Tutoring_Sessions_Per_Week": "Sesi Les per Minggu",
    "Exam_Anxiety_Score": "Skor Kecemasan Ujian",
    "Final_Score": "Skor Akhir",
    # Variabel kategorikal asli
    "Gender": "Gender",
    "Part_Time_Job": "Kerja Part-Time",
    "Study_Method": "Metode Belajar",
    "Diet_Quality": "Kualitas Diet",
    "Internet_Quality": "Kualitas Internet",
    "Extracurricular": "Ekstrakurikuler",
    "Family_Income_Level": "Tingkat Pendapatan Keluarga",
    # Variabel kategorikal hasil pengelompokan
    "Predikat": "Predikat",
    "Kelompok_Jam_Belajar": "Kelompok Jam Belajar",
    "Kelompok_Kehadiran": "Kelompok Kehadiran",
    "Kelompok_Stres": "Kelompok Stres",
}


def to_label(col):
    """Terjemahkan nama kolom teknis menjadi label Bahasa Indonesia untuk
    ditampilkan di dropdown, judul sumbu, dan teks insight. Nilai yang tidak
    dikenal (mis. '(Tidak ada)') dikembalikan apa adanya."""
    return LABEL_MAP.get(col, col)


# =========================================================
# PEMETAAN WARNA & URUTAN KATEGORI GLOBAL
# (agar identitas & urutan kategori konsisten di semua chart,
#  tidak berubah-ubah antara satu chart dengan chart lainnya)
# =========================================================
ORDINAL_SCALES = {
    "Diet_Quality": px.colors.diverging.RdYlGn,
    "Internet_Quality": px.colors.diverging.RdYlGn,
    "Family_Income_Level": px.colors.sequential.Blues,
    "Predikat": px.colors.diverging.RdYlGn,          # Kurang=merah ... Sangat Baik=hijau
    "Kelompok_Jam_Belajar": px.colors.sequential.Purples,
    "Kelompok_Kehadiran": px.colors.sequential.Greens,
    "Kelompok_Stres": px.colors.diverging.RdYlGn,     # dibalik: Rendah=hijau (baik) ... Sangat Tinggi=merah (buruk)
}
# Kolom yang urutan "baik->buruk"-nya berlawanan arah dengan urutan kategorinya,
# sehingga skala warnanya perlu dibalik
REVERSED_ORDINAL_COLS = {"Kelompok_Stres"}
QUALITATIVE_PALETTE = px.colors.qualitative.Bold

# Batas bawah & atas rentang skala warna yang boleh dipakai (0.0 - 1.0).
# Untuk color scale jenis "sequential" (Blues, Purples, Greens, dst), titik 0.0
# hampir selalu SANGAT TERANG / nyaris putih -> tidak terlihat di atas
# background putih Streamlit. Dengan menggeser batas bawah ke 0.25, kategori
# "paling terang" pun tetap punya kontras yang cukup untuk dibaca.
COLOR_SCALE_RANGE = (0.25, 1.0)


def build_color_maps(base_df):
    """Bangun dict {kolom: {kategori: warna}} berdasarkan seluruh data (bukan
    data terfilter), sehingga warna tiap kategori selalu sama meski filter berubah."""
    color_maps = {}
    for col in ALL_CATEGORICAL_COLS:
        if col in ORDINAL_SCALES:
            cats = list(base_df[col].cat.categories) if hasattr(base_df[col], "cat") else sorted(base_df[col].dropna().unique())
            scale = ORDINAL_SCALES[col]
            if col in REVERSED_ORDINAL_COLS:
                scale = list(reversed(scale))
            n = max(len(cats), 1)

            # --- FIX: hindari warna nyaris-putih di ujung skala ---
            # Sebelumnya: idxs = np.linspace(0, len(scale) - 1, n) -> dimulai dari 0%
            # skala warna, yang untuk scale "sequential" berarti putih.
            # Sekarang: sampling dibatasi ke rentang COLOR_SCALE_RANGE (25%-100%).
            low_frac, high_frac = COLOR_SCALE_RANGE
            if n == 1:
                positions = np.array([high_frac])
            else:
                positions = np.linspace(low_frac, high_frac, n)
            idxs = (positions * (len(scale) - 1)).round().astype(int)
            colors = [scale[i] for i in idxs]
        else:
            cats = sorted(base_df[col].dropna().unique().tolist(), key=str)
            colors = [QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)] for i in range(len(cats))]
        color_maps[col] = dict(zip(cats, colors))
    return color_maps


COLOR_MAPS = build_color_maps(df)

# Urutan kategori tetap (dipakai di SEMUA chart: box, bar, histogram, radar)
# supaya chart yang menampilkan data sama tidak pernah berbeda urutan sumbunya.
CATEGORY_ORDERS = {col: list(COLOR_MAPS[col].keys()) for col in ALL_CATEGORICAL_COLS}


def to_rgba(color, alpha):
    """Ubah warna 'rgb(r,g,b)' atau '#rrggbb' menjadi 'rgba(r,g,b,alpha)'."""
    if isinstance(color, str) and color.startswith("rgb("):
        return color.replace("rgb(", "rgba(").replace(")", f", {alpha})")
    if isinstance(color, str) and color.startswith("#"):
        h = color.lstrip("#")
        r, g, b = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r}, {g}, {b}, {alpha})"
    return color


# =========================================================
# FUNGSI BANTUAN — INSIGHT DINAMIS
# =========================================================
def insight_distribusi(series, col):
    """Insight untuk distribusi variabel numerik."""
    s = series.dropna()
    if s.empty:
        return "📌 **Insight:** Tidak ada data untuk ditampilkan pada filter saat ini."
    mean, median, std = s.mean(), s.median(), s.std()
    skew = s.skew()
    if pd.isna(skew) or abs(skew) < 0.5:
        bentuk = "relatif simetris (mendekati distribusi normal)"
    elif skew > 0.5:
        bentuk = "condong ke kanan (skewed positif), ada sejumlah nilai tinggi yang menarik rata-rata ke atas"
    else:
        bentuk = "condong ke kiri (skewed negatif), ada sejumlah nilai rendah yang menarik rata-rata ke bawah"
    return (
        f"📌 **Insight:** Rata-rata **{to_label(col)}** adalah **{mean:.2f}**, dengan median **{median:.2f}** "
        f"dan standar deviasi **{std:.2f}**. Distribusi data {bentuk}."
    )


def insight_komposisi(series, col):
    """Insight untuk komposisi/proporsi kategori."""
    counts = series.value_counts()
    if counts.empty:
        return "📌 **Insight:** Tidak ada data untuk ditampilkan pada filter saat ini."
    top = counts.idxmax()
    pct = counts.max() / counts.sum() * 100
    if len(counts) > 1:
        second = counts.index[1]
        pct2 = counts.iloc[1] / counts.sum() * 100
        return (
            f"📌 **Insight:** Mayoritas mahasiswa (**{pct:.1f}%**) berada pada kategori **{top}** untuk "
            f"**{to_label(col)}**, diikuti oleh **{second}** (**{pct2:.1f}%**)."
        )
    return f"📌 **Insight:** Seluruh mahasiswa berada pada kategori **{top}** untuk **{to_label(col)}**."


def insight_korelasi(data, x, y):
    """Insight untuk hubungan dua variabel numerik (scatter plot)."""
    pair = data[[x, y]].dropna()
    if len(pair) < 3:
        return "📌 **Insight:** Data terlalu sedikit untuk menghitung korelasi yang bermakna."
    r = pair[x].corr(pair[y])
    if pd.isna(r):
        return "📌 **Insight:** Korelasi tidak dapat dihitung (variasi data terlalu kecil)."
    abs_r = abs(r)
    if abs_r < 0.1:
        kekuatan = "sangat lemah / nyaris tidak ada hubungan"
    elif abs_r < 0.3:
        kekuatan = "lemah"
    elif abs_r < 0.5:
        kekuatan = "sedang"
    elif abs_r < 0.7:
        kekuatan = "kuat"
    else:
        kekuatan = "sangat kuat"
    arah = "positif (searah)" if r >= 0 else "negatif (berlawanan arah)"
    return (
        f"📌 **Insight:** Korelasi antara **{to_label(x)}** dan **{to_label(y)}** adalah **{r:.2f}** — menunjukkan hubungan "
        f"**{kekuatan}** dan **{arah}**. *(Korelasi tidak selalu berarti hubungan sebab-akibat.)*"
    )


def insight_heatmap(corr):
    """Insight untuk matriks korelasi — mencari pasangan variabel paling berkorelasi."""
    corr_no_diag = corr.copy()
    for i in range(len(corr_no_diag)):
        corr_no_diag.iloc[i, i] = 0
    stacked = corr_no_diag.abs().stack()
    if stacked.empty:
        return "📌 **Insight:** Data tidak cukup untuk menghitung korelasi."
    max_pair = stacked.idxmax()
    val = corr.loc[max_pair]
    return (
        f"📌 **Insight:** Pasangan variabel dengan korelasi terkuat adalah **{to_label(max_pair[0])}** dan "
        f"**{to_label(max_pair[1])}** (r = **{val:.2f}**)."
    )


def insight_faktor(corr_final):
    """Insight untuk bar chart faktor paling berkorelasi dengan Skor Akhir."""
    if corr_final.empty:
        return "📌 **Insight:** Data tidak cukup untuk menghitung korelasi."
    top_pos, top_pos_val = corr_final.idxmax(), corr_final.max()
    top_neg, top_neg_val = corr_final.idxmin(), corr_final.min()
    return (
        f"📌 **Insight:** Faktor yang paling berkorelasi **positif** dengan Skor Akhir adalah "
        f"**{to_label(top_pos)}** (r = {top_pos_val:.2f}), sedangkan yang paling **negatif** adalah "
        f"**{to_label(top_neg)}** (r = {top_neg_val:.2f})."
    )


def hitung_signifikansi_kategori(data, cat_col, val_col):
    """Uji ANOVA satu arah (one-way ANOVA) untuk mengetahui apakah rata-rata
    val_col benar-benar berbeda antar kelompok pada cat_col, atau perbedaan
    yang tampak di chart hanyalah variasi acak (noise) pada sampel.

    Return: p-value (float), atau None jika tidak bisa dihitung
    (data terlalu sedikit, hanya 1 kelompok, atau scipy tidak tersedia).
    """
    if scipy_stats is None:
        return None
    groups = [
        g[val_col].dropna().values
        for _, g in data.groupby(cat_col, observed=True)
    ]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return None
    try:
        _, p_value = scipy_stats.f_oneway(*groups)
        return None if pd.isna(p_value) else p_value
    except Exception:
        return None


def insight_kategori(agg_df, cat_col, val_col, p_value=None):
    """Insight untuk perbandingan rata-rata antar kategori.

    Jika p_value diberikan, insight akan secara eksplisit menyatakan apakah
    perbedaan antar kelompok signifikan secara statistik (p < 0.05) atau
    hanya kebetulan/noise (p >= 0.05) — supaya tidak menyesatkan pembaca
    dengan menyimpulkan "kelompok A lebih baik" padahal bedanya tidak berarti.
    """
    if agg_df.empty:
        return "📌 **Insight:** Tidak ada data untuk ditampilkan pada filter saat ini."
    best = agg_df.loc[agg_df[val_col].idxmax()]
    worst = agg_df.loc[agg_df[val_col].idxmin()]
    gap = best[val_col] - worst[val_col]
    if best[cat_col] == worst[cat_col]:
        return f"📌 **Insight:** Hanya ada satu kelompok (**{best[cat_col]}**) pada filter saat ini."

    dasar = (
        f"Kelompok **{best[cat_col]}** memiliki rata-rata **{to_label(val_col)}** tertinggi "
        f"(**{best[val_col]:.1f}**), sedangkan **{worst[cat_col]}** terendah "
        f"(**{worst[val_col]:.1f}**) — selisih **{gap:.1f}** poin."
    )

    if p_value is None:
        return f"📌 **Insight:** {dasar}"

    if p_value < 0.05:
        return (
            f"📌 **Insight:** {dasar} Perbedaan ini **signifikan secara statistik** "
            f"(uji ANOVA, p = {p_value:.4f} < 0.05), sehingga kemungkinan besar "
            f"mencerminkan pola nyata, bukan kebetulan."
        )
    return (
        f"📌 **Insight:** Secara angka, kelompok **{best[cat_col]}** tampak sedikit lebih tinggi "
        f"({best[val_col]:.1f}) dibanding **{worst[cat_col]}** ({worst[val_col]:.1f}). Namun, "
        f"uji ANOVA menunjukkan perbedaan ini **tidak signifikan secara statistik** "
        f"(p = {p_value:.4f} ≥ 0.05) — variasi ini kemungkinan besar hanya kebetulan/noise, "
        f"bukan pengaruh nyata dari **{to_label(cat_col)}**."
    )


def insight_radar(group_avg_norm):
    """Insight untuk radar chart perbandingan multi-variabel antar kelompok."""
    if group_avg_norm.empty or len(group_avg_norm.index) < 2:
        return "📌 **Insight:** Perlu minimal dua kelompok untuk perbandingan pada radar chart."
    top_counts = group_avg_norm.idxmax(axis=0).value_counts()
    leader = top_counts.idxmax()
    n = top_counts.max()
    total = len(group_avg_norm.columns)
    return (
        f"📌 **Insight:** Kelompok **{leader}** unggul pada **{n} dari {total}** variabel yang diukur, "
        f"menjadikannya kelompok dengan profil rata-rata paling menonjol dibandingkan kelompok lain."
    )


# =========================================================
# SIDEBAR — FILTER
# =========================================================
st.sidebar.title("🎓 Filter Data")
st.sidebar.markdown("Gunakan filter berikut untuk menyaring data mahasiswa.")

f_df = df.copy()

age_min, age_max = int(df.Age.min()), int(df.Age.max())
age_range = st.sidebar.slider("Rentang Usia", age_min, age_max, (age_min, age_max))
f_df = f_df[f_df.Age.between(*age_range)]

gender_sel = st.sidebar.multiselect("Gender", sorted(df.Gender.unique()), default=list(sorted(df.Gender.unique())))
f_df = f_df[f_df.Gender.isin(gender_sel)]

job_sel = st.sidebar.multiselect("Kerja Part-Time", sorted(df.Part_Time_Job.unique()), default=list(sorted(df.Part_Time_Job.unique())))
f_df = f_df[f_df.Part_Time_Job.isin(job_sel)]

method_sel = st.sidebar.multiselect("Metode Belajar", sorted(df.Study_Method.unique()), default=list(sorted(df.Study_Method.unique())))
f_df = f_df[f_df.Study_Method.isin(method_sel)]

income_sel = st.sidebar.multiselect("Tingkat Pendapatan Keluarga", CATEGORY_ORDERS["Family_Income_Level"], default=CATEGORY_ORDERS["Family_Income_Level"])
f_df = f_df[f_df.Family_Income_Level.isin(income_sel)]

extra_sel = st.sidebar.multiselect("Ekstrakurikuler", sorted(df.Extracurricular.unique()), default=list(sorted(df.Extracurricular.unique())))
f_df = f_df[f_df.Extracurricular.isin(extra_sel)]

study_hours_range = st.sidebar.slider(
    "Jam Belajar per Hari",
    float(df.Hours_Studied.min()), float(df.Hours_Studied.max()),
    (float(df.Hours_Studied.min()), float(df.Hours_Studied.max())),
)
f_df = f_df[f_df.Hours_Studied.between(*study_hours_range)]

st.sidebar.markdown("---")
st.sidebar.caption(f"Menampilkan **{len(f_df):,}** dari **{len(df):,}** mahasiswa")

if f_df.empty:
    st.warning("Tidak ada data yang cocok dengan filter yang dipilih. Silakan ubah filter di sidebar.")
    st.stop()

# =========================================================
# HEADER & KPI
# =========================================================
st.title("🎓 Dashboard Analisis Performa Mahasiswa")
st.markdown("Eksplorasi interaktif terhadap faktor-faktor yang memengaruhi performa akademik mahasiswa.")

# 5 KPI utama: cukup untuk dipindai sekilas tanpa berdesakan, terutama di layar sempit.
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Mahasiswa", f"{len(f_df):,}")
k2.metric("Rata-rata Skor Akhir", f"{f_df.Final_Score.mean():.1f}")
k3.metric("Rata-rata Kehadiran", f"{f_df.Attendance.mean():.1f}%")
k4.metric("Rata-rata Jam Belajar", f"{f_df.Hours_Studied.mean():.1f} jam")
k5.metric("Predikat Terbanyak", f_df.Predikat.value_counts().idxmax())

# GPA Sebelumnya & Skor Stres dipindah ke sini: tetap bisa diakses,
# tapi tidak mendesak 5 KPI utama di atas.
with st.expander("📎 Statistik tambahan"):
    kx1, kx2 = st.columns(2)
    kx1.metric("Rata-rata GPA Sebelumnya", f"{f_df.Previous_GPA.mean():.2f}")
    kx2.metric("Rata-rata Skor Stres", f"{f_df.Stress_Level.mean():.1f}/10")

st.markdown("---")

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Ringkasan & Distribusi",
    "🔗 Korelasi & Hubungan",
    "🧑‍🤝‍🧑 Perbandingan Kategori",
    "🔍 Eksplorasi Bebas",
    "🗂️ Data Mentah",
])

# ---------------------------------------------------------
# TAB 1 — RINGKASAN & DISTRIBUSI
# ---------------------------------------------------------
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Distribusi Skor Akhir")
        fig = px.histogram(f_df, x="Final_Score", nbins=40, color_discrete_sequence=["#6C5CE7"],
                            labels=LABEL_MAP)
        fig.add_vline(x=f_df.Final_Score.mean(), line_dash="dash", line_color="red",
                      annotation_text=f"Rata-rata: {f_df.Final_Score.mean():.1f}")
        st.plotly_chart(fig, use_container_width=True, key="tab1_hist_final_score")
        st.info(insight_distribusi(f_df["Final_Score"], "Final_Score"))

    with c2:
        st.subheader("Distribusi Usia & Gender")
        fig = px.histogram(f_df, x="Age", color="Gender", barmode="group",
                            color_discrete_map=COLOR_MAPS["Gender"],
                            category_orders=CATEGORY_ORDERS, labels=LABEL_MAP)
        st.plotly_chart(fig, use_container_width=True, key="tab1_hist_age_gender")
        st.info(insight_distribusi(f_df["Age"], "Age"))

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Komposisi Gender")
        fig = px.pie(f_df, names="Gender", hole=0.45, color="Gender",
                     color_discrete_map=COLOR_MAPS["Gender"], category_orders=CATEGORY_ORDERS)
        st.plotly_chart(fig, use_container_width=True, key="tab1_pie_gender")
        st.info(insight_komposisi(f_df["Gender"], "Gender"))

    with c4:
        st.subheader("Tingkat Pendapatan Keluarga")
        counts = f_df.Family_Income_Level.value_counts().reindex(CATEGORY_ORDERS["Family_Income_Level"]).reset_index()
        counts.columns = ["Family_Income_Level", "Jumlah"]
        fig = px.bar(counts, x="Family_Income_Level", y="Jumlah", color="Family_Income_Level",
                     color_discrete_map=COLOR_MAPS["Family_Income_Level"],
                     category_orders=CATEGORY_ORDERS, labels=LABEL_MAP)
        st.plotly_chart(fig, use_container_width=True, key="tab1_bar_income")
        st.info(insight_komposisi(f_df["Family_Income_Level"], "Family_Income_Level"))

    c3b, c4b = st.columns(2)
    with c3b:
        st.subheader("Distribusi Predikat Kelulusan")
        counts_p = f_df["Predikat"].value_counts().reindex(CATEGORY_ORDERS["Predikat"]).reset_index()
        counts_p.columns = ["Predikat", "Jumlah"]
        fig = px.bar(counts_p, x="Predikat", y="Jumlah", color="Predikat",
                     color_discrete_map=COLOR_MAPS["Predikat"], category_orders=CATEGORY_ORDERS,
                     text_auto=True, labels=LABEL_MAP)
        st.plotly_chart(fig, use_container_width=True, key="tab1_bar_predikat")
        st.info(insight_komposisi(f_df["Predikat"], "Predikat"))

    with c4b:
        st.subheader("Sebaran Tingkat Stres")
        counts_s = f_df["Kelompok_Stres"].value_counts().reindex(CATEGORY_ORDERS["Kelompok_Stres"]).reset_index()
        counts_s.columns = ["Kelompok_Stres", "Jumlah"]
        fig = px.bar(counts_s, x="Kelompok_Stres", y="Jumlah", color="Kelompok_Stres",
                     color_discrete_map=COLOR_MAPS["Kelompok_Stres"], category_orders=CATEGORY_ORDERS,
                     text_auto=True, labels=LABEL_MAP)
        st.plotly_chart(fig, use_container_width=True, key="tab1_bar_stres")
        st.info(insight_komposisi(f_df["Kelompok_Stres"], "Kelompok_Stres"))

    st.subheader("Distribusi Variabel Numerik")
    num_choice = st.selectbox("Pilih variabel:", NUMERIC_COLS, index=NUMERIC_COLS.index("Hours_Studied"),
                               format_func=to_label)
    c5, c6 = st.columns([2, 1])
    with c5:
        fig = px.histogram(f_df, x=num_choice, nbins=40, marginal="box",
                            color_discrete_sequence=["#00B894"], labels=LABEL_MAP)
        st.plotly_chart(fig, use_container_width=True, key="tab1_hist_numeric_var")
        st.info(insight_distribusi(f_df[num_choice], num_choice))
    with c6:
        st.write("**Statistik Deskriptif**")
        desc = f_df[num_choice].rename(to_label(num_choice)).describe().round(2)
        st.dataframe(desc, use_container_width=True)

# ---------------------------------------------------------
# TAB 2 — KORELASI & HUBUNGAN
# ---------------------------------------------------------
with tab2:
    st.subheader("Matriks Korelasi Variabel Numerik")
    corr = f_df[NUMERIC_COLS].corr().round(2)
    corr_display = corr.rename(index=to_label, columns=to_label)
    fig = px.imshow(corr_display, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    st.plotly_chart(fig, use_container_width=True, key="tab2_heatmap_corr")
    st.info(insight_heatmap(corr))

    st.markdown("---")
    st.subheader("Scatter Plot Interaktif")
    c1, c2, c3 = st.columns(3)
    with c1:
        x_var = st.selectbox("Sumbu X", NUMERIC_COLS, index=NUMERIC_COLS.index("Hours_Studied"), format_func=to_label)
    with c2:
        y_var = st.selectbox("Sumbu Y", NUMERIC_COLS, index=NUMERIC_COLS.index("Final_Score"), format_func=to_label)
    with c3:
        color_var = st.selectbox("Warna berdasarkan", ["(Tidak ada)"] + ALL_CATEGORICAL_COLS, index=0, format_func=to_label)

    color_arg = None if color_var == "(Tidak ada)" else color_var
    fig = px.scatter(f_df, x=x_var, y=y_var, color=color_arg, trendline="ols",
                      opacity=0.6,
                      color_discrete_map=COLOR_MAPS.get(color_arg),
                      category_orders=CATEGORY_ORDERS, labels=LABEL_MAP)
    st.plotly_chart(fig, use_container_width=True, key="tab2_scatter_interaktif")
    st.info(insight_korelasi(f_df, x_var, y_var))

    st.subheader("Faktor Paling Berkorelasi dengan Skor Akhir")
    corr_final = f_df[NUMERIC_COLS].corr()["Final_Score"].drop("Final_Score").sort_values()
    corr_final_display = corr_final.rename(index=to_label)
    fig = px.bar(corr_final_display, orientation="h", color=corr_final_display.values,
                 color_continuous_scale="RdBu_r", labels={"value": "Korelasi", "index": "Variabel"})
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True, key="tab2_bar_faktor_korelasi")
    st.info(insight_faktor(corr_final))

# ---------------------------------------------------------
# TAB 3 — PERBANDINGAN KATEGORI
# ---------------------------------------------------------
with tab3:
    st.subheader("Bandingkan Skor Akhir Berdasarkan Kategori")
    cat_choice = st.selectbox("Pilih kategori pembanding:", COMPARISON_COLS, format_func=to_label)

    agg = f_df.groupby(cat_choice, observed=True)["Final_Score"].mean().reindex(CATEGORY_ORDERS[cat_choice]).reset_index()

    # Uji ANOVA: apakah perbedaan rata-rata Skor Akhir antar kelompok ini nyata
    # atau cuma kebetulan? Ditampilkan sebagai badge di atas chart supaya
    # user tahu sebelum membaca chart, bukan cuma di teks insight di bawahnya.
    p_val_final = hitung_signifikansi_kategori(f_df, cat_choice, "Final_Score")
    if p_val_final is not None:
        if p_val_final < 0.05:
            st.caption(
                f"✅ Perbedaan rata-rata Skor Akhir antar kelompok **{to_label(cat_choice)}** "
                f"signifikan secara statistik (ANOVA, p = {p_val_final:.4f})."
            )
        else:
            st.caption(
                f"⚠️ Perbedaan rata-rata Skor Akhir antar kelompok **{to_label(cat_choice)}** "
                f"**tidak signifikan** secara statistik (ANOVA, p = {p_val_final:.4f}) — "
                f"perbedaan yang tampak di chart kemungkinan besar hanya kebetulan."
            )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(f_df, x=cat_choice, y="Final_Score", color=cat_choice,
                     color_discrete_map=COLOR_MAPS[cat_choice],
                     category_orders=CATEGORY_ORDERS, labels=LABEL_MAP)
        st.plotly_chart(fig, use_container_width=True, key="tab3_box_kategori")
        st.info(insight_kategori(agg, cat_choice, "Final_Score", p_val_final))
    with c2:
        fig = px.bar(agg, x=cat_choice, y="Final_Score", color=cat_choice, text_auto=".1f",
                     color_discrete_map=COLOR_MAPS[cat_choice],
                     category_orders=CATEGORY_ORDERS, labels=LABEL_MAP)
        st.plotly_chart(fig, use_container_width=True, key="tab3_bar_kategori")
        st.info(insight_kategori(agg, cat_choice, "Final_Score", p_val_final))

    st.markdown("---")
    st.subheader("Perbandingan Multi-Variabel (Radar)")
    group_avg = f_df.groupby(cat_choice, observed=True)[NUMERIC_COLS].mean().reindex(CATEGORY_ORDERS[cat_choice])
    group_avg_norm = (group_avg - df[NUMERIC_COLS].min()) / (df[NUMERIC_COLS].max() - df[NUMERIC_COLS].min())
    theta_labels = [to_label(c) for c in NUMERIC_COLS]

    fig = go.Figure()
    for idx in group_avg_norm.index:
        warna = COLOR_MAPS[cat_choice].get(idx, "#636EFA")
        fig.add_trace(go.Scatterpolar(
            r=group_avg_norm.loc[idx].values,
            theta=theta_labels,
            fill="toself",
            name=str(idx),
            line=dict(color=warna, width=2.5),
            fillcolor=to_rgba(warna, 0.25),
            marker=dict(size=5, color=warna),
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
    st.plotly_chart(fig, use_container_width=True, key="tab3_radar")
    st.info(insight_radar(group_avg_norm))

    st.subheader("Tabel Ringkasan per Kategori")
    st.dataframe(group_avg.rename(columns=to_label).round(2), use_container_width=True)

# ---------------------------------------------------------
# TAB 4 — EKSPLORASI BEBAS
# ---------------------------------------------------------
with tab4:
    st.subheader("Bangun Visualisasi Anda Sendiri")
    st.caption("Pilih satu jenis analisis, lalu atur variabelnya. Sengaja dibuat ringkas agar mudah dipresentasikan.")

    chart_type = st.radio(
        "Jenis Analisis",
        ["🔗 Hubungan Dua Variabel", "📈 Distribusi Satu Variabel", "🧑‍🤝‍🧑 Perbandingan Antar Kategori"],
        horizontal=True,
    )
    st.markdown("")

    if chart_type == "🔗 Hubungan Dua Variabel":
        c1, c2, c3 = st.columns(3)
        # Default diselaraskan dengan Tab 2 (Hours_Studied vs Final_Score) —
        # sebelumnya default X=Age, Y=Hours_Studied yang kombinasinya acak
        # dan kurang informatif untuk tampilan awal.
        x = c1.selectbox("Variabel X", NUMERIC_COLS, index=NUMERIC_COLS.index("Hours_Studied"), key="s_x", format_func=to_label)
        y = c2.selectbox("Variabel Y", NUMERIC_COLS, index=NUMERIC_COLS.index("Final_Score"), key="s_y", format_func=to_label)
        color = c3.selectbox("Kelompokkan berdasarkan (opsional)", ["(Tidak ada)"] + ALL_CATEGORICAL_COLS, key="s_c", format_func=to_label)
        color_arg = None if color == "(Tidak ada)" else color
        fig = px.scatter(f_df, x=x, y=y, color=color_arg, trendline="ols",
                          opacity=0.6, color_discrete_map=COLOR_MAPS.get(color_arg),
                          category_orders=CATEGORY_ORDERS, labels=LABEL_MAP)
        st.plotly_chart(fig, use_container_width=True, key="tab4_scatter_bebas")
        st.info(insight_korelasi(f_df, x, y))

    elif chart_type == "📈 Distribusi Satu Variabel":
        c1, c2 = st.columns(2)
        x = c1.selectbox("Variabel", NUMERIC_COLS, key="h_x", format_func=to_label)
        color = c2.selectbox("Kelompokkan berdasarkan (opsional)", ["(Tidak ada)"] + ALL_CATEGORICAL_COLS, key="h_c", format_func=to_label)
        color_arg = None if color == "(Tidak ada)" else color
        # FIX: barmode="overlay" tanpa opacity < 1 membuat bar kelompok yang
        # dirender belakangan tertutup TOTAL oleh bar di depannya (opacity
        # default Plotly = 1.0), sehingga sebagian kelompok jadi tidak
        # terlihat sama sekali. opacity=0.6 membuat semua kelompok tetap
        # terlihat & saling tumpang tindih secara transparan.
        fig = px.histogram(f_df, x=x, color=color_arg, barmode="overlay", nbins=40,
                            opacity=0.6 if color_arg else 1.0,
                            color_discrete_map=COLOR_MAPS.get(color_arg),
                            category_orders=CATEGORY_ORDERS, labels=LABEL_MAP)
        st.plotly_chart(fig, use_container_width=True, key="tab4_hist_bebas")
        st.info(insight_distribusi(f_df[x], x))

    else:  # Perbandingan Antar Kategori
        c1, c2 = st.columns(2)
        y = c1.selectbox("Variabel Numerik", NUMERIC_COLS, key="b_y", format_func=to_label)
        # FIX: sebelumnya pakai ALL_CATEGORICAL_COLS (termasuk Predikat).
        # Predikat adalah hasil bin dari Final_Score itu sendiri, sehingga
        # membandingkannya (apalagi saat y = Final_Score) akan selalu
        # menghasilkan pemisahan "sempurna" yang tautologis, bukan temuan
        # nyata. COMPARISON_COLS sudah mengecualikan Predikat untuk alasan
        # ini (lihat definisinya di bagian atas file).
        x = c2.selectbox("Kategori", COMPARISON_COLS, key="b_x", format_func=to_label)

        p_val_b = hitung_signifikansi_kategori(f_df, x, y)
        if p_val_b is not None:
            if p_val_b < 0.05:
                st.caption(
                    f"✅ Perbedaan rata-rata **{to_label(y)}** antar kelompok **{to_label(x)}** "
                    f"signifikan secara statistik (ANOVA, p = {p_val_b:.4f})."
                )
            else:
                st.caption(
                    f"⚠️ Perbedaan rata-rata **{to_label(y)}** antar kelompok **{to_label(x)}** "
                    f"**tidak signifikan** secara statistik (ANOVA, p = {p_val_b:.4f}) — "
                    f"kemungkinan besar hanya kebetulan/noise."
                )

        # Box plot dengan garis rata-rata sekaligus -> menggantikan box + bar terpisah
        fig = px.box(f_df, x=x, y=y, color=x, color_discrete_map=COLOR_MAPS[x], points=False,
                     category_orders=CATEGORY_ORDERS, labels=LABEL_MAP)
        fig.update_traces(boxmean=True)  # tampilkan garis putus-putus rata-rata di tiap box
        st.plotly_chart(fig, use_container_width=True, key="tab4_box_bebas")
        agg_b = f_df.groupby(x, observed=True)[y].mean().reindex(CATEGORY_ORDERS[x]).reset_index()
        st.info(insight_kategori(agg_b, x, y, p_val_b))

# ---------------------------------------------------------
# TAB 5 — DATA MENTAH
# ---------------------------------------------------------
with tab5:
    st.subheader("Data Mentah (setelah difilter)")
    st.caption("Nama kolom pada tabel ini mengikuti nama kolom asli pada berkas CSV sumber.")
    st.dataframe(f_df, use_container_width=True, height=500)
    csv = f_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Unduh data terfilter (CSV)", csv, "data_terfilter.csv", "text/csv")