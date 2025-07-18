import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import streamlit as st
# Tidak ada import antar modul app lain di sini


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Siapkan data untuk analisis dan prediksi."""
    df_prep = df.copy(deep=False)
    for col in df_prep.select_dtypes(include=["float", "int", "float32", "int32"]):
        df_prep[col] = df_prep[col].fillna(df_prep[col].median())
    for col in df_prep.select_dtypes(include=["object"]):
        df_prep[col] = df_prep[col].fillna("")
    if "review_content" in df_prep.columns:
        df_prep["panjang_review"] = (
            df_prep["review_content"].astype(str).str.len().astype("int32")
        )
        df_prep["jumlah_kata_review"] = (
            df_prep["review_content"].astype(str).str.split().apply(len).astype("int32")
        )
    if "discounted_price" in df_prep.columns and "actual_price" in df_prep.columns:
        df_prep["diskon_rupiah"] = (
            df_prep["actual_price"] - df_prep["discounted_price"]
        ).astype("float32")
    return df_prep


@st.cache_resource(show_spinner=False)
def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series
) -> RandomForestRegressor:
    """Latih model Random Forest."""
    return RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1).fit(
        X_train, y_train
    )


@st.cache_resource(show_spinner=False)
def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
    """Latih model XGBoost."""
    return xgb.XGBRegressor(
        n_estimators=50, random_state=42, n_jobs=-1, verbosity=0
    ).fit(X_train, y_train)
