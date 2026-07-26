import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Dashboard Performa Mahasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #E3E6EC;
        border-radius: 12px;
        padding: 18px 20px 8px 20px;
        background-color: #FFFFFF;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] h3 {
        margin-top: 0;
        padding-top: 0;
        font-size: 1.05rem;
    }
    /* Beri jarak antar kartu supaya tidak menempel */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        margin-bottom: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CHART_HEIGHT = 420

DATA_FILE = "student_performance_preprocessed.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    # Kategori terurut untuk kolom ordinal yang dipakai di dashboard ini
    ordered = {
        "Family_Income_Level": ["Low", "Middle", "High"],
        "Kelompok_Jam_Belajar": ["0-2 jam", "2-4 jam", "4-6 jam", "6-8 jam", "> 8 jam"],
        "Kelompok_Stres": ["Rendah (1-3)", "Sedang (3-5)", "Tinggi (5-7)", "Sangat Tinggi (7-10)"],
    }
    for col, order in ordered.items():
        if col not in df.columns:
            continue
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
# Kolom kategorikal yang benar-benar dipakai (filter + pewarnaan chart)
ALL_CATEGORICAL_COLS = [
    "Study_Method", "Part_Time_Job", "Family_Income_Level",
    "Kelompok_Jam_Belajar", "Kelompok_Stres",
]

LABEL_MAP = {
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
    "Gender": "Gender",
    "Part_Time_Job": "Kerja Part-Time",
    "Study_Method": "Metode Belajar",
    "Extracurricular": "Ekstrakurikuler",
    "Family_Income_Level": "Tingkat Pendapatan Keluarga",
    "Kelompok_Jam_Belajar": "Kelompok Jam Belajar",
    "Kelompok_Stres": "Kelompok Stres",
}


def to_label(col):
    """Terjemahkan nama kolom teknis menjadi label Bahasa Indonesia."""
    return LABEL_MAP.get(col, col)

ORDINAL_SCALES = {
    "Family_Income_Level": px.colors.sequential.Blues,
    "Kelompok_Jam_Belajar": px.colors.sequential.Purples,
    "Kelompok_Stres": px.colors.diverging.RdYlGn,  # dibalik: Rendah=hijau ... Sangat Tinggi=merah
}
REVERSED_ORDINAL_COLS = {"Kelompok_Stres"}
QUALITATIVE_PALETTE = px.colors.qualitative.Bold

# Hindari ujung skala yang nyaris putih agar tetap kontras di background terang.
COLOR_SCALE_RANGE = (0.25, 1.0)


def build_color_maps(base_df):
    """Bangun dict {kolom: {kategori: warna}} dari seluruh data (bukan data
    terfilter), sehingga warna tiap kategori tetap sama meski filter berubah."""
    color_maps = {}
    for col in ALL_CATEGORICAL_COLS:
        if col not in base_df.columns:
            continue
        if col in ORDINAL_SCALES:
            cats = (list(base_df[col].cat.categories) if hasattr(base_df[col], "cat")
                    else sorted(base_df[col].dropna().unique()))
            scale = ORDINAL_SCALES[col]
            if col in REVERSED_ORDINAL_COLS:
                scale = list(reversed(scale))
            n = max(len(cats), 1)
            low_frac, high_frac = COLOR_SCALE_RANGE
            positions = np.array([high_frac]) if n == 1 else np.linspace(low_frac, high_frac, n)
            idxs = (positions * (len(scale) - 1)).round().astype(int)
            colors = [scale[i] for i in idxs]
        else:
            cats = sorted(base_df[col].dropna().unique().tolist(), key=str)
            colors = [QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)] for i in range(len(cats))]
        color_maps[col] = dict(zip(cats, colors))
    return color_maps


COLOR_MAPS = build_color_maps(df)
CATEGORY_ORDERS = {col: list(m.keys()) for col, m in COLOR_MAPS.items()}

st.sidebar.title("🎓 Filter Data")
st.sidebar.markdown("Gunakan filter berikut untuk menyaring data mahasiswa.")

f_df = df.copy()

age_min, age_max = int(df.Age.min()), int(df.Age.max())
age_range = st.sidebar.slider("Rentang Usia", age_min, age_max, (age_min, age_max))
f_df = f_df[f_df.Age.between(*age_range)]

gender_sel = st.sidebar.multiselect(
    "Gender", sorted(df.Gender.unique()), default=list(sorted(df.Gender.unique()))
)
f_df = f_df[f_df.Gender.isin(gender_sel)]

job_sel = st.sidebar.multiselect(
    "Kerja Part-Time", sorted(df.Part_Time_Job.unique()), default=list(sorted(df.Part_Time_Job.unique()))
)
f_df = f_df[f_df.Part_Time_Job.isin(job_sel)]

method_sel = st.sidebar.multiselect(
    "Metode Belajar", sorted(df.Study_Method.unique()), default=list(sorted(df.Study_Method.unique()))
)
f_df = f_df[f_df.Study_Method.isin(method_sel)]

income_sel = st.sidebar.multiselect(
    "Tingkat Pendapatan Keluarga",
    CATEGORY_ORDERS["Family_Income_Level"],
    default=CATEGORY_ORDERS["Family_Income_Level"],
)
f_df = f_df[f_df.Family_Income_Level.isin(income_sel)]

extra_sel = st.sidebar.multiselect(
    "Ekstrakurikuler", sorted(df.Extracurricular.unique()), default=list(sorted(df.Extracurricular.unique()))
)
f_df = f_df[f_df.Extracurricular.isin(extra_sel)]

st.sidebar.markdown("---")
st.sidebar.caption(f"Menampilkan **{len(f_df):,}** dari **{len(df):,}** mahasiswa")

if f_df.empty:
    st.warning("Tidak ada data yang cocok dengan filter yang dipilih. Silakan ubah filter di sidebar.")
    st.stop()

st.title("🎓 Dashboard Analisis Performa Mahasiswa")
st.markdown("Eksplorasi interaktif terhadap faktor-faktor yang memengaruhi performa akademik mahasiswa.")

KPI_ITEMS = [
    ("Jumlah Mahasiswa", f"{len(f_df):,}"),
    ("Rata-rata Nilai", f"{f_df.Final_Score.mean():.1f}"),
    ("Nilai Tertinggi", f"{f_df.Final_Score.max():.1f}"),
    ("GPA Tertinggi", f"{f_df.Previous_GPA.max():.2f}"),
    ("Rata-rata Jam Belajar", f"{f_df.Hours_Studied.mean():.1f} jam"),
    ("Rata-rata Kehadiran", f"{f_df.Attendance.mean():.1f}%"),
]
for col, (judul, nilai) in zip(st.columns(6), KPI_ITEMS):
    with col:
        with st.container(border=True):
            st.metric(judul, nilai)

st.write("")

with st.container(border=True):
    st.subheader("Faktor Paling Berpengaruh terhadap Nilai Akhir")
    corr_final = f_df[NUMERIC_COLS].corr()["Final_Score"].drop("Final_Score").sort_values()
    corr_final_display = corr_final.rename(index=to_label)
    fig = px.bar(
        corr_final_display, orientation="h", color=corr_final_display.values,
        color_continuous_scale="RdBu_r", labels={"value": "Korelasi", "index": "Variabel"},
    )
    fig.update_layout(coloraxis_showscale=False, showlegend=False, height=CHART_HEIGHT)
    st.plotly_chart(fig, use_container_width=True, key="bar_faktor_korelasi")


with st.container(border=True):
    st.subheader("Matriks Korelasi Variabel Numerik")
    corr = f_df[NUMERIC_COLS].corr().round(2)
    corr_display = corr.rename(index=to_label, columns=to_label)
    fig = px.imshow(
        corr_display, text_auto=True, color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto",
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True, key="heatmap_corr")

c1, c2 = st.columns(2)

with c1:
    with st.container(border=True):
        st.subheader("Tren Rata-rata Nilai Akhir Berdasarkan Jam Belajar")
        tren_jam = (
            f_df.groupby("Kelompok_Jam_Belajar", observed=True)["Final_Score"]
            .mean()
            .reindex(CATEGORY_ORDERS["Kelompok_Jam_Belajar"])
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tren_jam.index.astype(str), y=tren_jam.values, mode="lines+markers",
            line=dict(color="#6C5CE7", width=3), marker=dict(size=9),
        ))
        fig.update_layout(
            xaxis_title="Kelompok Jam Belajar", yaxis_title="Rata-rata Skor Akhir",
            height=CHART_HEIGHT,
        )
        st.plotly_chart(fig, use_container_width=True, key="tren_jam_belajar")

with c2:
    with st.container(border=True):
        st.subheader("Hubungan Tingkat Kehadiran dengan Nilai Akhir")
        fig = px.scatter(
            f_df, x="Attendance", y="Final_Score", trendline="ols", opacity=0.6,
            color_discrete_sequence=["#00B894"], labels=LABEL_MAP,
        )
        fig.update_layout(height=CHART_HEIGHT)
        st.plotly_chart(fig, use_container_width=True, key="scatter_kehadiran")

c3, c4 = st.columns(2)

with c3:
    with st.container(border=True):
        st.subheader("Dampak Tingkat Stres terhadap Nilai Akhir")
        fig = px.box(
            f_df, x="Kelompok_Stres", y="Final_Score", color="Kelompok_Stres", points=False,
            color_discrete_map=COLOR_MAPS["Kelompok_Stres"],
            category_orders=CATEGORY_ORDERS, labels=LABEL_MAP,
        )
        fig.update_traces(boxmean=True)
        fig.update_layout(showlegend=False, height=CHART_HEIGHT)
        st.plotly_chart(fig, use_container_width=True, key="box_stres")

with c4:
    with st.container(border=True):
        st.subheader("Sesi Les (Tutoring) per Minggu terhadap Nilai Akhir")
        tren_tutoring = (
            f_df.groupby("Tutoring_Sessions_Per_Week", observed=True)["Final_Score"].mean().sort_index()
        )
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=tren_tutoring.index.astype(str), y=tren_tutoring.values,
            marker_color="#0984E3", text=tren_tutoring.round(1), textposition="outside",
        ))
        fig.update_layout(
            xaxis_title="Sesi Les per Minggu", yaxis_title="Rata-rata Skor Akhir",
            height=CHART_HEIGHT,
        )
        st.plotly_chart(fig, use_container_width=True, key="bar_tutoring")

c5, c6 = st.columns(2)

with c5:
    with st.container(border=True):
        st.subheader("Rata-rata Nilai Akhir Berdasarkan Metode Belajar")
        agg_metode = (
            f_df.groupby("Study_Method", observed=True)["Final_Score"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        fig = px.bar(
            agg_metode, x="Study_Method", y="Final_Score", color="Study_Method",
            color_discrete_map=COLOR_MAPS.get("Study_Method"), text_auto=".1f", labels=LABEL_MAP,
        )
        fig.update_layout(showlegend=False, height=CHART_HEIGHT)
        st.plotly_chart(fig, use_container_width=True, key="bar_metode")

with c6:
    with st.container(border=True):
        st.subheader("Hubungan Kerja Part-Time dengan Nilai Akhir")
        fig = px.box(
            f_df, x="Part_Time_Job", y="Final_Score", color="Part_Time_Job", points=False,
            color_discrete_map=COLOR_MAPS.get("Part_Time_Job"), labels=LABEL_MAP,
        )
        fig.update_traces(boxmean=True)
        fig.update_layout(showlegend=False, height=CHART_HEIGHT)
        st.plotly_chart(fig, use_container_width=True, key="box_partime")
