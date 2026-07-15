import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_mahasiswa.csv")
    # Kategori terurut untuk kolom ordinal
    diet_order = ["Poor", "Average", "Good", "Excellent"]
    internet_order = ["Poor", "Average", "Good", "Excellent"]
    income_order = ["Low", "Middle", "High"]
    for col, order in [
        ("Diet_Quality", diet_order),
        ("Internet_Quality", internet_order),
        ("Family_Income_Level", income_order),
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
CATEGORICAL_COLS = [
    "Gender", "Part_Time_Job", "Study_Method", "Diet_Quality",
    "Internet_Quality", "Extracurricular", "Family_Income_Level",
]

# =========================================================
# FUNGSI BANTUAN — INSIGHT DINAMIS
# =========================================================
def insight_distribusi(series, label):
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
        f"📌 **Insight:** Rata-rata **{label}** adalah **{mean:.2f}**, dengan median **{median:.2f}** "
        f"dan standar deviasi **{std:.2f}**. Distribusi data {bentuk}."
    )


def insight_komposisi(series, label):
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
            f"**{label}**, diikuti oleh **{second}** (**{pct2:.1f}%**)."
        )
    return f"📌 **Insight:** Seluruh mahasiswa berada pada kategori **{top}** untuk **{label}**."


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
        f"📌 **Insight:** Korelasi antara **{x}** dan **{y}** adalah **{r:.2f}** — menunjukkan hubungan "
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
        f"📌 **Insight:** Pasangan variabel dengan korelasi terkuat adalah **{max_pair[0]}** dan "
        f"**{max_pair[1]}** (r = **{val:.2f}**)."
    )


def insight_faktor(corr_final):
    """Insight untuk bar chart faktor paling berkorelasi dengan Skor Akhir."""
    if corr_final.empty:
        return "📌 **Insight:** Data tidak cukup untuk menghitung korelasi."
    top_pos, top_pos_val = corr_final.idxmax(), corr_final.max()
    top_neg, top_neg_val = corr_final.idxmin(), corr_final.min()
    return (
        f"📌 **Insight:** Faktor yang paling berkorelasi **positif** dengan Skor Akhir adalah "
        f"**{top_pos}** (r = {top_pos_val:.2f}), sedangkan yang paling **negatif** adalah "
        f"**{top_neg}** (r = {top_neg_val:.2f})."
    )


def insight_kategori(agg_df, cat_col, val_col):
    """Insight untuk perbandingan rata-rata antar kategori."""
    if agg_df.empty:
        return "📌 **Insight:** Tidak ada data untuk ditampilkan pada filter saat ini."
    best = agg_df.loc[agg_df[val_col].idxmax()]
    worst = agg_df.loc[agg_df[val_col].idxmin()]
    gap = best[val_col] - worst[val_col]
    if best[cat_col] == worst[cat_col]:
        return f"📌 **Insight:** Hanya ada satu kelompok (**{best[cat_col]}**) pada filter saat ini."
    return (
        f"📌 **Insight:** Kelompok **{best[cat_col]}** memiliki rata-rata **{val_col}** tertinggi "
        f"(**{best[val_col]:.1f}**), sedangkan **{worst[cat_col]}** terendah "
        f"(**{worst[val_col]:.1f}**) — selisih **{gap:.1f}** poin."
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

income_sel = st.sidebar.multiselect("Tingkat Pendapatan Keluarga", sorted(df.Family_Income_Level.unique().tolist()), default=list(sorted(df.Family_Income_Level.unique().tolist())))
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

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Mahasiswa", f"{len(f_df):,}")
k2.metric("Rata-rata Skor Akhir", f"{f_df.Final_Score.mean():.1f}")
k3.metric("Rata-rata GPA Sebelumnya", f"{f_df.Previous_GPA.mean():.2f}")
k4.metric("Rata-rata Kehadiran", f"{f_df.Attendance.mean():.1f}%")
k5.metric("Rata-rata Jam Belajar", f"{f_df.Hours_Studied.mean():.1f} jam")
k6.metric("Rata-rata Skor Stres", f"{f_df.Stress_Level.mean():.1f}/10")

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
        fig = px.histogram(f_df, x="Final_Score", nbins=40, color_discrete_sequence=["#6C5CE7"])
        fig.add_vline(x=f_df.Final_Score.mean(), line_dash="dash", line_color="red",
                      annotation_text=f"Rata-rata: {f_df.Final_Score.mean():.1f}")
        st.plotly_chart(fig, use_container_width=True)
        st.info(insight_distribusi(f_df["Final_Score"], "Skor Akhir"))

    with c2:
        st.subheader("Distribusi Usia & Gender")
        fig = px.histogram(f_df, x="Age", color="Gender", barmode="group",
                            color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
        st.info(insight_distribusi(f_df["Age"], "Usia"))

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Komposisi Gender")
        fig = px.pie(f_df, names="Gender", hole=0.45, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)
        st.info(insight_komposisi(f_df["Gender"], "Gender"))

    with c4:
        st.subheader("Tingkat Pendapatan Keluarga")
        counts = f_df.Family_Income_Level.value_counts().reset_index()
        counts.columns = ["Level", "Jumlah"]
        fig = px.bar(counts, x="Level", y="Jumlah", color="Level",
                     color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(fig, use_container_width=True)
        st.info(insight_komposisi(f_df["Family_Income_Level"], "Tingkat Pendapatan Keluarga"))

    st.subheader("Distribusi Variabel Numerik")
    num_choice = st.selectbox("Pilih variabel:", NUMERIC_COLS, index=NUMERIC_COLS.index("Hours_Studied"))
    c5, c6 = st.columns([2, 1])
    with c5:
        fig = px.histogram(f_df, x=num_choice, nbins=40, marginal="box",
                            color_discrete_sequence=["#00B894"])
        st.plotly_chart(fig, use_container_width=True)
        st.info(insight_distribusi(f_df[num_choice], num_choice))
    with c6:
        st.write("**Statistik Deskriptif**")
        st.dataframe(f_df[num_choice].describe().round(2), use_container_width=True)

# ---------------------------------------------------------
# TAB 2 — KORELASI & HUBUNGAN
# ---------------------------------------------------------
with tab2:
    st.subheader("Matriks Korelasi Variabel Numerik")
    corr = f_df[NUMERIC_COLS].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    st.plotly_chart(fig, use_container_width=True)
    st.info(insight_heatmap(corr))

    st.markdown("---")
    st.subheader("Scatter Plot Interaktif")
    c1, c2, c3 = st.columns(3)
    with c1:
        x_var = st.selectbox("Sumbu X", NUMERIC_COLS, index=NUMERIC_COLS.index("Hours_Studied"))
    with c2:
        y_var = st.selectbox("Sumbu Y", NUMERIC_COLS, index=NUMERIC_COLS.index("Final_Score"))
    with c3:
        color_var = st.selectbox("Warna berdasarkan", ["(Tidak ada)"] + CATEGORICAL_COLS, index=0)

    color_arg = None if color_var == "(Tidak ada)" else color_var
    fig = px.scatter(f_df, x=x_var, y=y_var, color=color_arg, trendline="ols",
                      opacity=0.6, color_discrete_sequence=px.colors.qualitative.Bold)
    st.plotly_chart(fig, use_container_width=True)
    st.info(insight_korelasi(f_df, x_var, y_var))

    st.subheader("Faktor Paling Berkorelasi dengan Skor Akhir")
    corr_final = f_df[NUMERIC_COLS].corr()["Final_Score"].drop("Final_Score").sort_values()
    fig = px.bar(corr_final, orientation="h", color=corr_final.values,
                 color_continuous_scale="RdBu_r", labels={"value": "Korelasi", "index": "Variabel"})
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.info(insight_faktor(corr_final))

# ---------------------------------------------------------
# TAB 3 — PERBANDINGAN KATEGORI
# ---------------------------------------------------------
with tab3:
    st.subheader("Bandingkan Skor Akhir Berdasarkan Kategori")
    cat_choice = st.selectbox("Pilih kategori pembanding:", CATEGORICAL_COLS)

    agg = f_df.groupby(cat_choice, observed=True)["Final_Score"].mean().reset_index()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(f_df, x=cat_choice, y="Final_Score", color=cat_choice,
                     color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(fig, use_container_width=True)
        st.info(insight_kategori(agg, cat_choice, "Final_Score"))
    with c2:
        fig = px.bar(agg, x=cat_choice, y="Final_Score", color=cat_choice, text_auto=".1f",
                     color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(fig, use_container_width=True)
        st.info(insight_kategori(agg, cat_choice, "Final_Score"))

    st.markdown("---")
    st.subheader("Perbandingan Multi-Variabel (Radar)")
    group_avg = f_df.groupby(cat_choice, observed=True)[NUMERIC_COLS].mean()
    group_avg_norm = (group_avg - df[NUMERIC_COLS].min()) / (df[NUMERIC_COLS].max() - df[NUMERIC_COLS].min())

    fig = go.Figure()
    for idx in group_avg_norm.index:
        fig.add_trace(go.Scatterpolar(
            r=group_avg_norm.loc[idx].values,
            theta=NUMERIC_COLS,
            fill="toself",
            name=str(idx),
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    st.info(insight_radar(group_avg_norm))

    st.subheader("Tabel Ringkasan per Kategori")
    st.dataframe(group_avg.round(2), use_container_width=True)

# ---------------------------------------------------------
# TAB 4 — EKSPLORASI BEBAS
# ---------------------------------------------------------
with tab4:
    st.subheader("Bangun Visualisasi Anda Sendiri")
    chart_type = st.radio("Jenis Grafik", ["Scatter", "Histogram", "Box Plot", "Bar (rata-rata)", "Violin"], horizontal=True)

    if chart_type == "Scatter":
        c1, c2, c3, c4 = st.columns(4)
        x = c1.selectbox("X", NUMERIC_COLS, key="s_x")
        y = c2.selectbox("Y", NUMERIC_COLS, index=1, key="s_y")
        color = c3.selectbox("Warna", ["(Tidak ada)"] + CATEGORICAL_COLS, key="s_c")
        size = c4.selectbox("Ukuran", ["(Tidak ada)"] + NUMERIC_COLS, key="s_s")
        fig = px.scatter(f_df, x=x, y=y,
                          color=None if color == "(Tidak ada)" else color,
                          size=None if size == "(Tidak ada)" else size,
                          opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)
        st.info(insight_korelasi(f_df, x, y))

    elif chart_type == "Histogram":
        c1, c2 = st.columns(2)
        x = c1.selectbox("Variabel", NUMERIC_COLS, key="h_x")
        color = c2.selectbox("Warna", ["(Tidak ada)"] + CATEGORICAL_COLS, key="h_c")
        fig = px.histogram(f_df, x=x, color=None if color == "(Tidak ada)" else color,
                            barmode="overlay", nbins=40)
        st.plotly_chart(fig, use_container_width=True)
        st.info(insight_distribusi(f_df[x], x))

    elif chart_type == "Box Plot":
        c1, c2 = st.columns(2)
        y = c1.selectbox("Variabel Numerik", NUMERIC_COLS, key="b_y")
        x = c2.selectbox("Kategori", CATEGORICAL_COLS, key="b_x")
        fig = px.box(f_df, x=x, y=y, color=x)
        st.plotly_chart(fig, use_container_width=True)
        agg_b = f_df.groupby(x, observed=True)[y].mean().reset_index()
        st.info(insight_kategori(agg_b, x, y))

    elif chart_type == "Bar (rata-rata)":
        c1, c2 = st.columns(2)
        y = c1.selectbox("Variabel Numerik", NUMERIC_COLS, key="ba_y")
        x = c2.selectbox("Kategori", CATEGORICAL_COLS, key="ba_x")
        agg = f_df.groupby(x, observed=True)[y].mean().reset_index()
        fig = px.bar(agg, x=x, y=y, color=x, text_auto=".2f")
        st.plotly_chart(fig, use_container_width=True)
        st.info(insight_kategori(agg, x, y))

    elif chart_type == "Violin":
        c1, c2 = st.columns(2)
        y = c1.selectbox("Variabel Numerik", NUMERIC_COLS, key="v_y")
        x = c2.selectbox("Kategori", CATEGORICAL_COLS, key="v_x")
        fig = px.violin(f_df, x=x, y=y, color=x, box=True, points="outliers")
        st.plotly_chart(fig, use_container_width=True)
        agg_v = f_df.groupby(x, observed=True)[y].mean().reset_index()
        st.info(insight_kategori(agg_v, x, y))

# ---------------------------------------------------------
# TAB 5 — DATA MENTAH
# ---------------------------------------------------------
with tab5:
    st.subheader("Data Mentah (setelah difilter)")
    st.dataframe(f_df, use_container_width=True, height=500)
    csv = f_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Unduh data terfilter (CSV)", csv, "data_terfilter.csv", "text/csv")