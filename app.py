import streamlit as st
import pandas as pd
from app.utils import show_logo, load_data
from app.ml import prepare_data
from app.visual import (
    show_summary_tab,
    show_explore_tab,
    show_predict_tab,
    show_compare_tab,
)
from app.config import LOGO_PATH, APP_TITLE

st.set_page_config(page_title=APP_TITLE, layout="wide")

# --- Logo & Caption ---
show_logo(LOGO_PATH)
st.divider()

# --- Load and Prepare Data ---
with st.spinner("Memuat dan menyiapkan data..."):
    try:
        df = load_data()
        df_prep = prepare_data(df)
        st.sidebar.header("Tugas Data Mining - Random Forest Regression")
        st.sidebar.markdown(
            """
            **Tujuan:**
            - Memprediksi rating produk Amazon berdasarkan fitur produk dan ulasan menggunakan Random Forest Regression.
            - Menjelaskan algoritma, proses penambangan data, dan insight bisnis.
            """
        )
        st.sidebar.success(
            f"✅ Data dimuat: {df.shape[0]} baris, {df['category'].nunique()} kategori."
        )
    except Exception as e:
        st.sidebar.error(f"❌ Gagal memuat data: {e}")
        st.stop()

# --- Navigation Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "💡 Ringkasan & Insight",
        "📊 Eksplorasi Data",
        "🤖 Prediksi Rating",
        "⚖️ Perbandingan Metode",
    ]
)

with tab1:
    show_summary_tab(df)
with tab2:
    show_explore_tab(df)
with tab3:
    show_predict_tab(df, df_prep)
with tab4:
    show_compare_tab(df, df_prep)
