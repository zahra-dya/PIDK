import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from pathlib import Path
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance · SNBP",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- Google Font ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ---- Global ---- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---- Force light mode everywhere ---- */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stBottomBlockContainer"], .main, .block-container {
    background-color: #f8fafc !important;
    color: #1e293b !important;
}
[data-testid="stHeader"] {
    background-color: rgba(248, 250, 252, 0.95) !important;
}

/* ---- Widget text colors ---- */
.stMarkdown, .stMarkdown p, label, span, div, p,
[data-testid="stWidgetLabel"], [data-testid="stText"] {
    color: #1e293b !important;
}

/* ---- Expander ---- */
[data-testid="stExpander"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
}
[data-testid="stExpander"] summary span {
    color: #1e293b !important;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #475569 !important;
}

/* ---- Sidebar slider fix ---- */
[data-testid="stSidebar"] [data-testid="stSlider"] div,
[data-testid="stSidebar"] [data-testid="stSlider"] span,
[data-testid="stSidebar"] [data-testid="stSlider"] p,
[data-testid="stSidebar"] [data-testid="stThumbValue"] {
    color: #475569 !important;
}
[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
    background-color: #3b82f6 !important;
}
[data-testid="stSidebar"] [data-baseweb="slider"] div[data-testid="stTickBar"] ~ div {
    background-color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div {
    background: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div > div {
    background: #3b82f6 !important;
}

/* ---- Sidebar header text ---- */
.sidebar-brand {
    padding: 1rem 0 1.5rem;
    text-align: center;
}
.sidebar-brand h2 {
    font-size: 1.25rem;
    font-weight: 700;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: 0.02em;
}
.sidebar-brand p {
    font-size: 0.72rem;
    color: #64748b;
    margin: 0.2rem 0 0;
}


/* ---- Navigation Radio Override ---- */
[data-testid="stSidebar"] div[role="radiogroup"] > label {
    padding: 0.6rem 0.8rem;
    border-radius: 8px;
    margin-bottom: 0.3rem;
    background-color: transparent;
    transition: all 0.2s ease;
    cursor: pointer;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background-color: #f1f5f9;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label[data-baseweb="radio"] div[data-testid="stMarkdownContainer"] p {
    font-size: 0.95rem;
    font-weight: 500;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    border-radius: 8px;
    box-shadow: 0 4px 10px -2px rgba(59, 130, 246, 0.4);
}
[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] p,
[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] span {
    color: #ffffff !important;
    font-weight: 600;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover:not([data-checked="true"]) {
    background-color: #f1f5f9;
}
/* ---- Section divider label ---- */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #94a3b8;
    margin: 1.5rem 0 0.5rem;
}

/* ---- KPI Cards ---- */
.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    border-radius: 12px 12px 0 0;
}
.kpi-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1e293b;
    line-height: 1;
}
.kpi-sub {
    font-size: 0.72rem;
    color: #2563eb;
    margin-top: 0.35rem;
}

/* ---- Chart card ---- */
.chart-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05);
    border-radius: 12px;
    padding: 1.4rem 1.6rem 1rem;
}
.chart-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 0.1rem;
    letter-spacing: 0.02em;
}
.chart-subtitle {
    font-size: 0.72rem;
    color: #64748b;
    margin-bottom: 0.8rem;
}

/* ---- Page heading ---- */
.page-heading {
    padding: 0.5rem 0 1.5rem;
}
.page-heading h1 {
    font-size: 1.55rem;
    font-weight: 800;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.page-heading p {
    font-size: 0.8rem;
    color: #64748b;
    margin: 0.25rem 0 0;
}

/* ---- Insight pills ---- */
.insight-card {
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
    margin-bottom: 0.75rem;
}
.insight-card.ok   { background: #f0fdf4; border: 1px solid #bbf7d0; }
.insight-card.warn { background: #fffbeb; border: 1px solid #fef3c7; }
.insight-icon { font-size: 1.1rem; line-height: 1.4; }
.insight-text { font-size: 0.8rem; color: #334155; line-height: 1.5; }

/* ---- Multiselect / widget overrides ---- */
.stMultiSelect [data-baseweb="tag"] {
    background-color: #eff6ff !important;
    border: 1px solid #bfdbfe !important;
    color: #1e40af !important;
}
[data-testid="stMetricValue"] {
    color: #1e293b !important;
}

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib global theme ───────────────────────────────────────────────────
LIGHT_BG  = "#ffffff"
GRID_CLR  = "#f1f5f9"
TEXT_CLR  = "#475569"
ACCENT    = "#3b82f6"
ACCENT2   = "#8b5cf6"
ACCENT3   = "#10b981"
PALETTE   = [ACCENT, ACCENT2, ACCENT3, "#f59e0b", "#ef4444"]

plt.rcParams.update({
    "figure.facecolor":  LIGHT_BG,
    "axes.facecolor":    LIGHT_BG,
    "axes.edgecolor":    "#e2e8f0",
    "axes.labelcolor":   TEXT_CLR,
    "axes.titlecolor":   TEXT_CLR,
    "axes.grid":         True,
    "grid.color":        GRID_CLR,
    "grid.linewidth":    0.6,
    "xtick.color":       TEXT_CLR,
    "ytick.color":       TEXT_CLR,
    "text.color":        TEXT_CLR,
    "legend.facecolor":  LIGHT_BG,
    "legend.edgecolor":  "#e2e8f0",
    "legend.labelcolor": TEXT_CLR,
    "font.family":       "sans-serif",
    "font.size":         9,
})

def make_fig(h=3.6):
    """Return a pre-styled figure."""
    fig, ax = plt.subplots(figsize=(6, h))
    fig.patch.set_facecolor(LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    return fig, ax

BASE_DIR = Path(__file__).resolve().parent.parent

# ── Load data ─────────────────────────────────────────────────────────────────
REMOTE_URL = "https://raw.githubusercontent.com/devyti/CAPSTONE-PROJECT/refs/heads/main/DATA/cleaned_data.csv"
LOCAL_CSV  = Path(__file__).resolve().parent / "cleaned_data.csv"

@st.cache_data
def load_data():
    # Coba file lokal terlebih dahulu (lebih cepat & tidak butuh internet)
    if LOCAL_CSV.exists():
        return pd.read_csv(LOCAL_CSV)
    # Fallback ke URL remote
    try:
        df = pd.read_csv(REMOTE_URL)
        # Simpan salinan lokal agar offline tetap bisa berjalan
        df.to_csv(LOCAL_CSV, index=False)
        return df
    except Exception as e:
        st.error(
            f"❌ Gagal memuat data dari internet dan file lokal tidak ditemukan.\n\n"
            f"**Solusi:** Unduh file `cleaned_data.csv` secara manual dari "
            f"[GitHub]({REMOTE_URL}) lalu taruh di folder yang sama dengan `dashboard.py`.\n\n"
            f"Detail error: `{e}`"
        )
        st.stop()

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h2>📈 SNBP Analytics</h2>
        <p>Student Performance Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Menu Utama</div>', unsafe_allow_html=True)
    page = st.radio("Navigasi", ["📊 Overview Analisis", "🤖 Prediksi Nilai"], label_visibility="collapsed")
    
    st.markdown("<div style='margin-bottom:1rem'></div>", unsafe_allow_html=True)

    if page == "📊 Overview Analisis":
        st.markdown('<div class="section-label">Filter Data</div>', unsafe_allow_html=True)

        gender_opts = sorted(df["gender"].unique())
        gender = st.multiselect(
            "Gender",
            options=gender_opts,
            default=gender_opts,
        )

        level_opts = sorted(df["academic_level"].unique()) if "academic_level" in df.columns else []
        if level_opts:
            levels = st.multiselect(
                "Jenjang Akademik",
                options=level_opts,
                default=level_opts,
            )
        else:
            levels = []

        st.markdown('<div class="section-label">Rentang Nilai</div>', unsafe_allow_html=True)
        score_min = float(df["exam_score"].min())
        score_max = float(df["exam_score"].max())
        score_range = st.slider(
            "Exam Score",
            min_value=score_min,
            max_value=score_max,
            value=(score_min, score_max),
            step=1.0,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown(
            '<p style="font-size:0.68rem;color:#334155;text-align:center;">Data: CAPSTONE PROJECT · 2025</p>',
            unsafe_allow_html=True,
        )

if page == "📊 Overview Analisis":
    # ── Filter data ───────────────────────────────────────────────────────────────
    df_filtered = df[df["gender"].isin(gender)]
    if level_opts and levels:
        df_filtered = df_filtered[df_filtered["academic_level"].isin(levels)]
    df_filtered = df_filtered[
        (df_filtered["exam_score"] >= score_range[0]) &
        (df_filtered["exam_score"] <= score_range[1])
    ]

    # ── Risk classification ───────────────────────────────────────────────────────
    def classify_student(row):
        if row["study_hours"] > 8 and row["sleep_hours"] < 5:
            return "Burnout Risk"
        elif row["study_hours"] < 2:
            return "Understudy"
        else:
            return "Normal"

    df_filtered = df_filtered.copy()
    df_filtered["risk_category"] = df_filtered.apply(classify_student, axis=1)

    # ── Page heading ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="page-heading">
        <h1>Analisis Performa Siswa untuk SNBP</h1>
        <p>Ringkasan data akademik, pola belajar, dan deteksi risiko siswa</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────────────
    total     = len(df_filtered)
    avg_score = round(df_filtered["exam_score"].mean(), 1)   if total else 0
    avg_study = round(df_filtered["study_hours"].mean(), 1)  if total else 0
    avg_sleep = round(df_filtered["sleep_hours"].mean(), 1)  if total else 0
    burnout_n = int((df_filtered["risk_category"] == "Burnout Risk").sum())
    normal_pct = round((df_filtered["risk_category"] == "Normal").sum() / total * 100, 1) if total else 0

    k1, k2, k3, k4, k5 = st.columns(5)

    def kpi(col, label, value, sub=""):
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {"<div class='kpi-sub'>" + sub + "</div>" if sub else ""}
        </div>
        """, unsafe_allow_html=True)

    kpi(k1, "Total Siswa",     total,      f"{gender} dipilih")
    kpi(k2, "Rata-rata Nilai", avg_score,  "dari 100")
    kpi(k3, "Jam Belajar / Hari", avg_study, "jam rata-rata")
    kpi(k4, "Jam Tidur / Hari",   avg_sleep, "jam rata-rata")
    kpi(k5, "Burnout Risk",    burnout_n,  f"{normal_pct}% normal")

    st.markdown("<div style='margin-bottom:1.5rem'></div>", unsafe_allow_html=True)

    # ── Row 1: Distribusi Nilai  |  Jam Belajar vs Nilai ─────────────────────────
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        st.markdown("""
        <div class="chart-card">
        <div class="chart-title">Distribusi Nilai Akademik</div>
        <div class="chart-subtitle">Histogram exam score seluruh siswa</div>
        """, unsafe_allow_html=True)

        fig, ax = make_fig(3.5)
        sns.histplot(
            df_filtered["exam_score"], bins=24, kde=False, ax=ax,
            color=ACCENT, edgecolor=LIGHT_BG, linewidth=0.4,
        )
        ax2 = ax.twinx()
        sns.kdeplot(
            df_filtered["exam_score"], ax=ax2,
            color=ACCENT2, linewidth=1.5,
        )
        ax2.set_ylabel("")
        ax2.set_yticks([])
        for spine in ax2.spines.values():
            spine.set_edgecolor(GRID_CLR)
        ax2.set_facecolor(LIGHT_BG)
        ax.set_xlabel("Exam Score", fontsize=8)
        ax.set_ylabel("Jumlah Siswa", fontsize=8)
        ax.set_title("")
        fig.tight_layout(pad=0.4)
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="chart-card">
        <div class="chart-title">Jam Belajar vs Nilai</div>
        <div class="chart-subtitle">Korelasi study hours terhadap exam score per gender</div>
        """, unsafe_allow_html=True)

        fig, ax = make_fig(3.5)
        gender_palette = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(gender_opts)}
        sns.scatterplot(
            data=df_filtered, x="study_hours", y="exam_score",
            hue="gender", palette=gender_palette,
            alpha=0.65, s=22, ax=ax, linewidth=0,
        )
        ax.set_xlabel("Study Hours", fontsize=8)
        ax.set_ylabel("Exam Score", fontsize=8)
        ax.get_legend().set_title("")
        fig.tight_layout(pad=0.4)
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:1rem'></div>", unsafe_allow_html=True)

    # ── Row 2: Jam Tidur vs Nilai  |  Social Media vs Nilai ──────────────────────
    c3, c4 = st.columns(2, gap="medium")

    with c3:
        st.markdown("""
        <div class="chart-card">
        <div class="chart-title">Jam Tidur vs Nilai</div>
        <div class="chart-subtitle">Pengaruh sleep hours terhadap performa akademik</div>
        """, unsafe_allow_html=True)

        fig, ax = make_fig(3.5)
        sns.regplot(
            data=df_filtered, x="sleep_hours", y="exam_score", ax=ax,
            scatter_kws={"color": ACCENT3, "alpha": 0.5, "s": 18, "linewidths": 0},
            line_kws={"color": "#f59e0b", "linewidth": 1.4},
        )
        ax.set_xlabel("Sleep Hours", fontsize=8)
        ax.set_ylabel("Exam Score", fontsize=8)
        fig.tight_layout(pad=0.4)
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="chart-card">
        <div class="chart-title">Social Media vs Nilai</div>
        <div class="chart-subtitle">Dampak screen time social media terhadap exam score</div>
        """, unsafe_allow_html=True)

        fig, ax = make_fig(3.5)
        sns.regplot(
            data=df_filtered, x="social_media_hours", y="exam_score", ax=ax,
            scatter_kws={"color": "#f59e0b", "alpha": 0.5, "s": 18, "linewidths": 0},
            line_kws={"color": "#ef4444", "linewidth": 1.4},
        )
        ax.set_xlabel("Social Media Hours", fontsize=8)
        ax.set_ylabel("Exam Score", fontsize=8)
        fig.tight_layout(pad=0.4)
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:1rem'></div>", unsafe_allow_html=True)

    # ── Row 3: Deteksi Risiko  |  Distribusi Mental Health ───────────────────────
    c5, c6 = st.columns([1, 1], gap="medium")

    with c5:
        st.markdown("""
        <div class="chart-card">
        <div class="chart-title">Deteksi Risiko Siswa</div>
        <div class="chart-subtitle">Klasifikasi berdasarkan pola belajar & tidur</div>
        """, unsafe_allow_html=True)

        risk_count = df_filtered["risk_category"].value_counts()
        risk_colors = {"Normal": ACCENT3, "Burnout Risk": "#ef4444", "Understudy": "#f59e0b"}
        colors = [risk_colors.get(r, ACCENT) for r in risk_count.index]

        fig, ax = make_fig(3.2)
        bars = ax.bar(risk_count.index, risk_count.values, color=colors,
                      width=0.5, zorder=3, edgecolor=LIGHT_BG, linewidth=0.5)
        for bar, val in zip(bars, risk_count.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(val), ha="center", va="bottom", fontsize=8, color=TEXT_CLR)
        ax.set_ylabel("Jumlah Siswa", fontsize=8)
        ax.set_xlabel("")
        fig.tight_layout(pad=0.4)
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c6:
        mental_col = "mental_health_score" if "mental_health_score" in df.columns else None
        if mental_col:
            st.markdown("""
            <div class="chart-card">
            <div class="chart-title">Mental Health Score</div>
            <div class="chart-subtitle">Distribusi skor kesehatan mental siswa</div>
            """, unsafe_allow_html=True)

            fig, ax = make_fig(3.2)
            sns.kdeplot(
                df_filtered[mental_col], fill=True, ax=ax,
                color=ACCENT2, alpha=0.35, linewidth=1.5,
            )
            ax.set_xlabel("Mental Health Score", fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
            fig.tight_layout(pad=0.4)
            st.pyplot(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="chart-card" style="height:100%;display:flex;align-items:center;justify-content:center;color:#334155;">
                Kolom mental_health_score tidak tersedia
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:1.5rem'></div>", unsafe_allow_html=True)

    # ── Insight ───────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Insight Otomatis</div>', unsafe_allow_html=True)

    avg_study_all = df_filtered["study_hours"].mean() if total else 0
    avg_sleep_all = df_filtered["sleep_hours"].mean() if total else 0
    avg_social    = df_filtered["social_media_hours"].mean() if total else 0
    burnout_pct   = round(burnout_n / total * 100, 1) if total else 0

    insights = [
        (
            avg_study_all > 6,
            "✅" if avg_study_all > 6 else "⚠️",
            f"Rata-rata jam belajar <b>{avg_study_all:.1f} jam/hari</b> — "
            + ("sudah cukup tinggi untuk memaksimalkan performa SNBP." if avg_study_all > 6
               else "masih di bawah target optimal 6 jam/hari.")
        ),
        (
            avg_sleep_all >= 6,
            "✅" if avg_sleep_all >= 6 else "⚠️",
            f"Rata-rata jam tidur <b>{avg_sleep_all:.1f} jam/hari</b> — "
            + ("kualitas istirahat baik." if avg_sleep_all >= 6
               else "banyak siswa kurang tidur, berpotensi burnout.")
        ),
        (
            avg_social <= 3,
            "✅" if avg_social <= 3 else "⚠️",
            f"Rata-rata social media <b>{avg_social:.1f} jam/hari</b> — "
            + ("dalam batas wajar." if avg_social <= 3
               else "konsumsi social media cukup tinggi, perlu perhatian.")
        ),
        (
            burnout_pct < 10,
            "✅" if burnout_pct < 10 else "⚠️",
            f"<b>{burnout_pct}%</b> siswa terdeteksi berisiko burnout "
            + ("— tingkat risiko rendah, pertahankan." if burnout_pct < 10
               else "— perlu intervensi lebih lanjut.")
        ),
    ]

    ins_cols = st.columns(2, gap="medium")
    for i, (is_ok, icon, text) in enumerate(insights):
        card_cls = "ok" if is_ok else "warn"
        ins_cols[i % 2].markdown(f"""
        <div class="insight-card {card_cls}">
            <span class="insight-icon">{icon}</span>
            <span class="insight-text">{text}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:1.5rem'></div>", unsafe_allow_html=True)

    # ── Data Table ────────────────────────────────────────────────────────────────
    with st.expander("📋  Lihat Data Siswa", expanded=False):
        display_cols = [
            "student_id", "gender", "age", "study_hours", "sleep_hours",
            "social_media_hours", "exam_score", "mental_health_score",
            "burnout_level", "risk_category",
        ]
        display_cols = [c for c in display_cols if c in df_filtered.columns]
        st.dataframe(
            df_filtered[display_cols].reset_index(drop=True),
            use_container_width=True,
            height=340,
        )
        st.caption(f"Menampilkan {len(df_filtered):,} baris dari {len(df):,} total data.")

    st.markdown("<div style='margin-bottom:1.5rem'></div>", unsafe_allow_html=True)

elif page == "🤖 Prediksi Nilai":
    # ══════════════════════════════════════════════════════════════════════════════
    # ── Regresi Linear ────────────────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="page-heading" style="padding-bottom:1rem;">
        <h1>🤖 Regresi Linear — Prediksi Exam Score</h1>
        <p>Model Machine Learning untuk memprediksi nilai ujian berdasarkan pola belajar & kebiasaan siswa</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Prepare features ──────────────────────────────────────────────────────────
    FEATURES = ["study_hours", "sleep_hours", "social_media_hours"]
    if "mental_health_score" in df.columns:
        FEATURES.append("mental_health_score")
    if "age" in df.columns:
        FEATURES.append("age")

    FEATURE_LABELS = {
        "study_hours":         "Jam Belajar",
        "sleep_hours":         "Jam Tidur",
        "social_media_hours":  "Social Media",
        "mental_health_score": "Mental Health",
        "age":                 "Usia",
    }

    @st.cache_data
    def train_model(data_hash):
        """Train linear regression model on full dataset (cached)."""
        df_model = df.copy()
        # Encode gender if present
        if "gender" in df_model.columns:
            le = LabelEncoder()
            df_model["gender_enc"] = le.fit_transform(df_model["gender"])
            feat_cols = FEATURES + ["gender_enc"]
        else:
            feat_cols = FEATURES

        df_model = df_model.dropna(subset=feat_cols + ["exam_score"])
        X = df_model[feat_cols].values
        y = df_model["exam_score"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return model, X_test, y_test, y_pred, feat_cols

    # Use a simple hash to bust cache if data changes
    _data_hash = len(df)
    model, X_test, y_test, y_pred, feat_cols = train_model(_data_hash)

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # ── Model Metric KPIs ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Metrik Model</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)

    def metric_kpi(col, label, value, sub="", color="#2563eb"):
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{color};font-size:1.7rem;">{value}</div>
            {"<div class='kpi-sub'>" + sub + "</div>" if sub else ""}
        </div>
        """, unsafe_allow_html=True)

    metric_kpi(m1, "R² Score",   f"{r2:.3f}",  "Seberapa baik model fit",         "#2563eb" if r2 >= 0.5 else "#ef4444")
    metric_kpi(m2, "MAE",        f"{mae:.2f}",  "Mean Absolute Error (poin)",      "#10b981")
    metric_kpi(m3, "RMSE",       f"{rmse:.2f}", "Root Mean Squared Error (poin)",  "#8b5cf6")
    metric_kpi(m4, "Fitur",      len(feat_cols), "variabel prediktor digunakan",   "#f59e0b")

    st.markdown("<div style='margin-bottom:1.5rem'></div>", unsafe_allow_html=True)

    # ── Row: Coefficient chart  |  Predicted vs Actual ───────────────────────────
    rl1, rl2 = st.columns(2, gap="medium")

    with rl1:
        st.markdown("""
        <div class="chart-card">
        <div class="chart-title">Koefisien Regresi Linear</div>
        <div class="chart-subtitle">Bobot setiap fitur terhadap prediksi exam score</div>
        """, unsafe_allow_html=True)

        coef_vals  = model.coef_
        coef_names = [FEATURE_LABELS.get(f, f) for f in feat_cols]
        coef_colors = [ACCENT3 if v >= 0 else "#ef4444" for v in coef_vals]

        fig, ax = make_fig(3.5)
        bars = ax.barh(coef_names, coef_vals, color=coef_colors,
                       height=0.55, zorder=3, edgecolor=LIGHT_BG, linewidth=0.4)
        ax.axvline(0, color="#cbd5e1", linewidth=1, linestyle="--")
        for bar, val in zip(bars, coef_vals):
            xpos = val + (0.05 if val >= 0 else -0.05)
            ha   = "left" if val >= 0 else "right"
            ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center", ha=ha, fontsize=7.5, color=TEXT_CLR)
        ax.set_xlabel("Koefisien", fontsize=8)
        ax.set_ylabel("")
        pos_patch = mpatches.Patch(color=ACCENT3,   label="Pengaruh positif")
        neg_patch = mpatches.Patch(color="#ef4444", label="Pengaruh negatif")
        ax.legend(handles=[pos_patch, neg_patch], fontsize=7, loc="lower right")
        fig.tight_layout(pad=0.4)
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with rl2:
        st.markdown("""
        <div class="chart-card">
        <div class="chart-title">Prediksi vs Aktual</div>
        <div class="chart-subtitle">Perbandingan nilai prediksi model terhadap nilai nyata (test set)</div>
        """, unsafe_allow_html=True)

        fig, ax = make_fig(3.5)
        ax.scatter(y_test, y_pred, color=ACCENT, alpha=0.55, s=18, linewidths=0, zorder=3)
        # Perfect prediction line
        mn = min(y_test.min(), y_pred.min())
        mx = max(y_test.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], color="#ef4444", linewidth=1.4,
                linestyle="--", label="Prediksi Sempurna")
        ax.set_xlabel("Nilai Aktual", fontsize=8)
        ax.set_ylabel("Nilai Prediksi", fontsize=8)
        ax.legend(fontsize=7)
        # Annotate R²
        ax.text(0.05, 0.93, f"R² = {r2:.3f}", transform=ax.transAxes,
                fontsize=8, color=TEXT_CLR,
                bbox=dict(facecolor=LIGHT_BG, edgecolor="#e2e8f0", boxstyle="round,pad=0.3"))
        fig.tight_layout(pad=0.4)
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:1.5rem'></div>", unsafe_allow_html=True)

    # ── Interactive Score Predictor ───────────────────────────────────────────────
    st.markdown('<div class="section-label">Kalkulator Prediksi Nilai</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="chart-card" style="margin-bottom:1rem;">
    <div class="chart-title">🎯 Prediksi Exam Score Siswa</div>
    <div class="chart-subtitle">Masukkan data siswa untuk mendapatkan prediksi nilai ujian</div>
    """, unsafe_allow_html=True)

    with st.form("predictor_form"):
        p1, p2, p3 = st.columns(3)
        inp_study  = p1.number_input("Jam Belajar / Hari",  min_value=0.0, max_value=16.0, value=6.0, step=0.5)
        inp_sleep  = p2.number_input("Jam Tidur / Hari",    min_value=0.0, max_value=12.0, value=7.0, step=0.5)
        inp_social = p3.number_input("Social Media (jam)",  min_value=0.0, max_value=12.0, value=2.0, step=0.5)

        p4, p5, _ = st.columns(3)
        inp_mental = None
        inp_age    = None
        if "mental_health_score" in feat_cols:
            inp_mental = p4.number_input("Mental Health Score", min_value=1, max_value=10, value=7, step=1)
        if "age" in feat_cols:
            inp_age = p5.number_input("Usia", min_value=10, max_value=25, value=17, step=1)

        if "gender_enc" in feat_cols:
            inp_gender = st.selectbox("Gender", options=sorted(df["gender"].unique()))
            le_pred = LabelEncoder().fit(df["gender"])
            gender_enc_val = int(le_pred.transform([inp_gender])[0])
        else:
            gender_enc_val = None

        submitted = st.form_submit_button("🔮 Prediksi Nilai", use_container_width=True)

    if submitted:
        row = [inp_study, inp_sleep, inp_social]
        if "mental_health_score" in feat_cols:
            row.append(inp_mental)
        if "age" in feat_cols:
            row.append(inp_age)
        if "gender_enc" in feat_cols:
            row.append(gender_enc_val)

        prediction = model.predict([row])[0]
        prediction = float(np.clip(prediction, 0, 100))

        # Color badge based on score
        if prediction >= 80:
            badge_color, badge_emoji, badge_text = "#10b981", "🏆", "Sangat Baik"
        elif prediction >= 65:
            badge_color, badge_emoji, badge_text = "#3b82f6", "👍", "Baik"
        elif prediction >= 50:  
            badge_color, badge_emoji, badge_text = "#f59e0b", "⚠️", "Cukup"
        else:
            badge_color, badge_emoji, badge_text = "#ef4444", "🚨", "Perlu Perhatian"

        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {badge_color}18, {badge_color}08);
            border: 2px solid {badge_color}60;
            border-radius: 14px;
            padding: 1.4rem 2rem;
            display: flex;
            align-items: center;
            gap: 1.2rem;
            margin-top: 0.5rem;
        ">
            <div style="font-size:2.5rem;">{badge_emoji}</div>
            <div>
                <div style="font-size:0.72rem;font-weight:600;text-transform:uppercase;
                            letter-spacing:0.08em;color:#64748b;">Prediksi Exam Score</div>
                <div style="font-size:2.6rem;font-weight:800;color:{badge_color};line-height:1.1;">
                    {prediction:.1f} <span style="font-size:1rem;font-weight:500;color:#64748b;">/ 100</span>
                </div>
                <div style="font-size:0.82rem;color:{badge_color};font-weight:600;margin-top:0.2rem;">
                    {badge_text}
                </div>
            </div>
            <div style="margin-left:auto;text-align:right;font-size:0.75rem;color:#64748b;line-height:1.9;">
                📚 Belajar: <b>{inp_study} jam</b><br>
                😴 Tidur: <b>{inp_sleep} jam</b><br>
                📱 Social Media: <b>{inp_social} jam</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Regression Insight ────────────────────────────────────────────────────────
    st.markdown("<div style='margin-bottom:1rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Interpretasi Model</div>', unsafe_allow_html=True)

    intercept_val = model.intercept_
    coef_dict = dict(zip(feat_cols, model.coef_))

    study_coef  = coef_dict.get("study_hours", 0)
    social_coef = coef_dict.get("social_media_hours", 0)
    sleep_coef  = coef_dict.get("sleep_hours", 0)

    reg_insights = [
        (
            study_coef > 0,
            "✅" if study_coef > 0 else "⚠️",
            f"Setiap tambahan <b>1 jam belajar</b> memprediksi perubahan nilai sebesar "
            f"<b>{study_coef:+.3f} poin</b> — {'positif, belajar lebih lama meningkatkan nilai.' if study_coef > 0 else 'negatif, perlu dicermati lebih lanjut.'}"
        ),
        (
            social_coef < 0,
            "✅" if social_coef < 0 else "⚠️",
            f"Setiap tambahan <b>1 jam social media</b> memprediksi perubahan nilai sebesar "
            f"<b>{social_coef:+.3f} poin</b> — {'negatif, konsisten dengan hipotesis gangguan belajar.' if social_coef < 0 else 'positif — model mungkin memerlukan lebih banyak data.'}"
        ),
        (
            r2 >= 0.5,
            "✅" if r2 >= 0.5 else "⚠️",
            f"Model menjelaskan <b>{r2*100:.1f}%</b> variansi exam score (R² = {r2:.3f}). "
            + ("Fit model cukup baik untuk insight eksploratif." if r2 >= 0.5
               else "Fit model masih rendah — faktor lain di luar dataset mungkin lebih berpengaruh.")
        ),
        (
            mae < 10,
            "✅" if mae < 10 else "⚠️",
            f"Rata-rata kesalahan prediksi model: <b>{mae:.2f} poin</b> (MAE). "
            + ("Akurasi prediksi relatif baik." if mae < 10 else "Masih ada ruang untuk perbaikan model.")
        ),
    ]

    ri_cols = st.columns(2, gap="medium")
    for i, (is_ok, icon, text) in enumerate(reg_insights):
        card_cls = "ok" if is_ok else "warn"
        ri_cols[i % 2].markdown(f"""
        <div class="insight-card {card_cls}">
            <span class="insight-icon">{icon}</span>
            <span class="insight-text">{text}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:2rem'></div>", unsafe_allow_html=True)