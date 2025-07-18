import streamlit as st
import pandas as pd
import base64

# Tidak ada import antar modul app lain di sini


def clean_price_column(series: pd.Series) -> pd.Series:
    """Bersihkan kolom harga dari karakter non-digit dan konversi ke numerik."""
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce"
    ).fillna(0)


def format_rupiah(x) -> str:
    """Format angka ke dalam format Rupiah."""
    try:
        return f"Rp {int(x):,}".replace(",", ".")
    except Exception:
        return "-"


def get_base64_of_bin_file(bin_file: str) -> str:
    """Konversi file gambar ke base64 string."""
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def show_logo(path: str):
    """Tampilkan logo dan caption di tengah halaman."""
    img_base64 = get_base64_of_bin_file(path)
    st.markdown(
        f"""
        <div style='display: flex; flex-direction: column; align-items: center;'>
            <img src='data:image/png;base64,{img_base64}' width='140'/>
            <div style='font-size: 16px; margin-top: 4px; text-align: center;'>Universitas Indraprasta PGRI</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_data() -> pd.DataFrame:
    """Load dan bersihkan data dari CSV, konversi harga ke float."""
    columns = [
        "product_id",
        "product_name",
        "category",
        "discounted_price",
        "actual_price",
        "discount_percentage",
        "rating",
        "rating_count",
        "about_product",
        "user_id",
        "user_name",
        "review_id",
        "review_title",
        "review_content",
        "img_link",
        "product_link",
    ]
    df = pd.read_csv("amazon.csv", usecols=columns, dtype=str, low_memory=False)
    kurs_inr_to_idr = 190
    for col in ["discounted_price", "actual_price"]:
        if col in df.columns:
            df[col] = clean_price_column(df[col]) * kurs_inr_to_idr
            df[col] = (
                pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float32")
            )
    if "discount_percentage" in df.columns:
        df["discount_percentage"] = (
            pd.to_numeric(
                df["discount_percentage"]
                .astype(str)
                .str.replace("%", "")
                .str.replace(r"[^\d.]", "", regex=True),
                errors="coerce",
            )
            .fillna(0)
            .astype("float32")
        )
    for col in ["rating", "rating_count"]:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float32")
            )
    for col in ["category", "user_id", "user_name", "review_id", "product_id"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df
