import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
from app.utils import format_rupiah
from app.ml import prepare_data, train_random_forest, train_xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import plotly.graph_objects as go
import numpy as np


def show_summary_tab(df: pd.DataFrame):
    """Tampilkan tab ringkasan dan insight."""
    st.header("💡 Ringkasan & Insight")
    st.markdown(
        """
        Dashboard ini memberikan gambaran utama penjualan produk Amazon, seperti kategori terpopuler, pengaruh diskon dan review terhadap rating, serta insight bahwa produk dengan diskon besar dan review panjang cenderung memiliki rating lebih tinggi. Informasi ini dapat membantu Anda dalam mengambil keputusan promosi dan pengelolaan produk.
        """
    )
    st.divider()


def show_explore_tab(df: pd.DataFrame):
    """Tampilkan tab eksplorasi data."""
    st.header("Eksplorasi Data")
    st.write(f"Jumlah baris data: {df.shape[0]}")
    df_display = df.copy()
    for col in df_display.select_dtypes(include=["category", "object"]):
        df_display[col] = df_display[col].astype(str).fillna("")
    plot_sample = df_display
    if "category" in plot_sample.columns:
        st.markdown(
            "**Grafik 10 Kategori Produk Terpopuler**\nGrafik ini menunjukkan 10 kategori produk yang paling banyak dijual di Amazon. Semakin tinggi batang, semakin banyak produk di kategori tersebut."
        )
        top_cat = plot_sample["category"].value_counts().nlargest(10).reset_index()
        top_cat.columns = ["Kategori", "Jumlah"]
        fig_cat = px.bar(
            top_cat, x="Kategori", y="Jumlah", title="10 Kategori Produk Terpopuler"
        )
        st.plotly_chart(fig_cat, use_container_width=True, key="fig_cat")
        st.caption(
            f"Kategori terpopuler: {top_cat['Kategori'][0]} ({top_cat['Jumlah'][0]} produk)"
        )
    if "discounted_price" in plot_sample.columns:
        st.markdown(
            "**Histogram Distribusi Harga Diskon**\nGrafik ini memperlihatkan sebaran harga diskon produk. Anda bisa melihat rentang harga yang paling sering muncul."
        )
        fig_harga = px.histogram(
            plot_sample, x="discounted_price", nbins=30, title="Distribusi Harga Diskon"
        )
        fig_harga.update_layout(
            xaxis_title="Harga Diskon (Rp)",
            yaxis_title="Jumlah",
            xaxis_tickformat=",.0f",
        )
        fig_harga.update_traces(hovertemplate="Rp %{x:,.0f}<extra></extra>")
        st.plotly_chart(fig_harga, use_container_width=True, key="fig_harga")
        st.caption(
            f"Harga diskon rata-rata: {format_rupiah(plot_sample['discounted_price'].mean())}"
        )
    if "rating" in plot_sample.columns:
        st.markdown(
            "**Histogram Distribusi Rating Produk**\nGrafik ini memperlihatkan sebaran rating produk. Anda bisa melihat berapa banyak produk yang mendapat rating tinggi atau rendah."
        )
        fig_rating = px.histogram(
            plot_sample, x="rating", nbins=20, title="Distribusi Rating Produk"
        )
        fig_rating.update_layout(xaxis_title="Rating", yaxis_title="Jumlah")
        st.plotly_chart(fig_rating, use_container_width=True, key="fig_rating")
        st.caption(f"Rata-rata rating produk: {plot_sample['rating'].mean():.2f}")
    st.info(
        "Gunakan tab Prediksi Rating untuk mencoba prediksi rating produk berdasarkan fitur sederhana."
    )
    if "discounted_price" in plot_sample.columns and "rating" in plot_sample.columns:
        st.markdown(
            "**Scatter Plot Harga Diskon vs Rating**\nTitik-titik pada grafik ini menunjukkan hubungan antara harga diskon dan rating produk. Pola tertentu bisa menunjukkan pengaruh harga terhadap rating."
        )
        fig_scatter = px.scatter(
            plot_sample,
            x="discounted_price",
            y="rating",
            title="Harga Diskon vs Rating",
            opacity=0.5,
        )
        fig_scatter.update_layout(
            xaxis_title="Harga Diskon (Rp)",
            yaxis_title="Rating",
            xaxis_tickformat=",.0f",
        )
        fig_scatter.update_traces(hovertemplate="Harga: Rp %{x:,.0f}<br>Rating: %{y}")
        st.plotly_chart(fig_scatter, use_container_width=True, key="fig_scatter")
        st.caption("Visualisasi hubungan harga diskon dan rating produk.")
    if "review_content" in plot_sample.columns:
        st.markdown(
            "**Word Cloud Review Produk**\nKata-kata yang sering muncul pada ulasan produk akan tampak lebih besar. Ini membantu mengetahui apa yang sering dibahas pembeli."
        )
        try:
            text = " ".join(plot_sample["review_content"].dropna().astype(str))
            wordcloud = WordCloud(
                width=800, height=400, background_color="white"
            ).generate(text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
            st.caption("Word cloud: kata yang sering muncul pada review produk.")
        except Exception:
            pass
    if (
        "discounted_price" in plot_sample.columns
        and "product_name" in plot_sample.columns
    ):
        st.markdown(
            "**Top 5 Produk dengan Diskon Tertinggi**\nTabel ini menampilkan lima produk dengan potongan harga terbesar."
        )
        top5_diskon = plot_sample.copy()
        if "actual_price" in top5_diskon.columns:
            top5_diskon["diskon_rupiah"] = top5_diskon["actual_price"].astype(
                float
            ) - top5_diskon["discounted_price"].astype(float)
            top5_diskon = top5_diskon.sort_values(
                "diskon_rupiah", ascending=False
            ).head(5)
            top5_diskon_show = top5_diskon[
                ["product_name", "actual_price", "discounted_price", "diskon_rupiah"]
            ].copy()
            for col in ["actual_price", "discounted_price", "diskon_rupiah"]:
                top5_diskon_show[col] = top5_diskon_show[col].apply(format_rupiah)
            st.dataframe(
                top5_diskon_show.rename(
                    columns={
                        "product_name": "Produk",
                        "actual_price": "Harga Asli",
                        "discounted_price": "Harga Diskon",
                        "diskon_rupiah": "Diskon (Rp)",
                    }
                ),
                use_container_width=True,
            )
    if "rating" in plot_sample.columns and "product_name" in plot_sample.columns:
        st.markdown(
            "**Top 5 Produk dengan Rating Tertinggi**\nTabel ini menampilkan lima produk dengan rating tertinggi."
        )
        top5_rating = plot_sample.sort_values("rating", ascending=False).head(5)
        top5_rating_show = top5_rating[
            ["product_name", "rating", "discounted_price"]
        ].copy()
        top5_rating_show["discounted_price"] = top5_rating_show[
            "discounted_price"
        ].apply(format_rupiah)
        st.dataframe(
            top5_rating_show.rename(
                columns={
                    "product_name": "Produk",
                    "rating": "Rating",
                    "discounted_price": "Harga Diskon",
                }
            ),
            use_container_width=True,
        )


def show_predict_tab(df: pd.DataFrame, df_prep: pd.DataFrame):
    """Tampilkan tab prediksi rating."""
    st.header("Prediksi Rating Produk Amazon")
    st.markdown(
        "**Prediksi Rating Produk**\nMasukkan fitur produk, lalu klik Prediksi Rating untuk melihat perkiraan rating produk Anda berdasarkan data yang ada. Semakin besar diskon dan semakin panjang review, biasanya rating produk akan lebih tinggi."
    )
    feature_cols = [
        c
        for c in [
            "discounted_price",
            "diskon_rupiah",
            "panjang_review",
            "jumlah_kata_review",
        ]
        if c in df_prep.columns
    ]
    if "rating" not in df_prep.columns or len(feature_cols) < 1:
        st.warning(
            "Maaf, data yang tersedia belum cukup untuk melakukan prediksi rating."
        )
    else:
        X = df_prep[feature_cols]
        y = df_prep["rating"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        rf = train_random_forest(X_train, y_train)
        with st.form("form_prediksi_rating"):
            harga_diskon = st.number_input(
                "Harga Diskon (Rp)", min_value=0, value=100000
            )
            diskon_rp = st.number_input("Diskon (Rp)", min_value=0, value=10000)
            panjang_review = st.number_input(
                "Panjang Review (karakter)", min_value=0, value=50
            )
            jumlah_kata = st.number_input("Jumlah Kata Review", min_value=0, value=10)
            submitted = st.form_submit_button("Prediksi Rating")
        if submitted:
            fitur_pred = pd.DataFrame(
                {
                    "discounted_price": [harga_diskon],
                    "diskon_rupiah": [diskon_rp],
                    "panjang_review": [panjang_review],
                    "jumlah_kata_review": [jumlah_kata],
                }
            )
            pred_rating = rf.predict(fitur_pred)[0]
            st.success(
                f"Perkiraan rating produk Anda adalah: {pred_rating:.2f} dari 5. Hasil ini bersifat prediksi dan dapat berbeda dengan kenyataan."
            )
        del X, y, X_train, X_test, y_train, y_test, rf


def show_compare_tab(df: pd.DataFrame, df_prep: pd.DataFrame):
    """Tampilkan tab perbandingan model prediksi."""
    st.header("Perbandingan Model Prediksi")
    st.markdown(
        """
        Kami membandingkan dua model yang sering digunakan untuk memprediksi rating produk: Model Random Forest dan Model XGBoost.\nGrafik-grafik di bawah ini membantu Anda melihat seberapa baik kedua model dalam memprediksi rating produk.
        """
    )
    st.markdown(
        "**Bar Chart Perbandingan Error**\nGrafik ini membandingkan rata-rata kesalahan prediksi dua model. Nilai lebih kecil berarti prediksi lebih akurat."
    )
    feature_cols = [
        c
        for c in [
            "discounted_price",
            "diskon_rupiah",
            "panjang_review",
            "jumlah_kata_review",
        ]
        if c in df_prep.columns
    ]
    if "rating" not in df_prep.columns or len(feature_cols) < 1:
        st.warning("Maaf, data yang tersedia belum cukup untuk membandingkan model.")
    else:
        X = df_prep[feature_cols]
        y = df_prep["rating"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # Random Forest
        rf = train_random_forest(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        error_rf = mean_absolute_error(y_test, y_pred_rf)
        # XGBoost
        xgbr = xgb.XGBRegressor(
            n_estimators=50, random_state=42, n_jobs=-1, verbosity=0
        )
        xgbr.fit(X_train, y_train)
        y_pred_xgb = xgbr.predict(X_test)
        error_xgb = mean_absolute_error(y_test, y_pred_xgb)
        # Bar chart perbandingan error
        fig_bar = go.Figure(
            data=[
                go.Bar(name="Random Forest", x=["Random Forest"], y=[error_rf]),
                go.Bar(name="XGBoost", x=["XGBoost"], y=[error_xgb]),
            ]
        )
        fig_bar.update_layout(
            title="Rata-rata Selisih Prediksi vs Rating Sebenarnya",
            yaxis_title="Rata-rata Selisih (Semakin kecil semakin baik)",
            xaxis_title="Model",
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption(
            "Model dengan rata-rata selisih prediksi yang lebih kecil dianggap lebih akurat."
        )
        # Scatter plot Prediksi vs Aktual
        st.markdown(
            "**Scatter Plot Prediksi vs Aktual**\nTitik-titik yang dekat dengan garis putus-putus berarti prediksi model mendekati nilai sebenarnya."
        )
        col1, col2 = st.columns(2)
        with col1:
            fig_rf = px.scatter(
                x=y_test,
                y=y_pred_rf,
                labels={"x": "Rating Sebenarnya", "y": "Prediksi"},
                title="Random Forest: Prediksi vs Rating Sebenarnya",
            )
            fig_rf.add_shape(
                type="line",
                x0=y_test.min(),
                y0=y_test.min(),
                x1=y_test.max(),
                y1=y_test.max(),
                line=dict(color="green", dash="dash"),
            )
            st.plotly_chart(fig_rf, use_container_width=True)
            st.caption(
                "Titik-titik yang semakin dekat ke garis putus-putus berarti prediksi model semakin akurat."
            )
        with col2:
            fig_xgb = px.scatter(
                x=y_test,
                y=y_pred_xgb,
                labels={"x": "Rating Sebenarnya", "y": "Prediksi"},
                title="XGBoost: Prediksi vs Rating Sebenarnya",
            )
            fig_xgb.add_shape(
                type="line",
                x0=y_test.min(),
                y0=y_test.min(),
                x1=y_test.max(),
                y1=y_test.max(),
                line=dict(color="green", dash="dash"),
            )
            st.plotly_chart(fig_xgb, use_container_width=True)
            st.caption(
                "Titik-titik yang semakin dekat ke garis putus-putus berarti prediksi model semakin akurat."
            )
        # Histogram distribusi error
        st.markdown(
            "**Histogram Distribusi Error**\nGrafik ini menunjukkan sebaran kesalahan prediksi. Semakin banyak di sekitar nol, semakin baik modelnya."
        )
        col3, col4 = st.columns(2)
        with col3:
            error_rf_dist = y_test - y_pred_rf
            fig_hist_rf = px.histogram(
                error_rf_dist,
                nbins=20,
                title="Distribusi Error Random Forest",
                labels={"value": "Error (Aktual - Prediksi)"},
            )
            st.plotly_chart(fig_hist_rf, use_container_width=True)
            st.caption("Semakin banyak error di sekitar nol, semakin baik model.")
        with col4:
            error_xgb_dist = y_test - y_pred_xgb
            fig_hist_xgb = px.histogram(
                error_xgb_dist,
                nbins=20,
                title="Distribusi Error XGBoost",
                labels={"value": "Error (Aktual - Prediksi)"},
            )
            st.plotly_chart(fig_hist_xgb, use_container_width=True)
            st.caption("Semakin banyak error di sekitar nol, semakin baik model.")
        # Insight sederhana
        if error_xgb < error_rf:
            st.info(
                "Pada data ini, model XGBoost menghasilkan prediksi yang lebih mendekati rating sebenarnya dibanding Random Forest."
            )
        else:
            st.info(
                "Pada data ini, model Random Forest menghasilkan prediksi yang lebih mendekati rating sebenarnya dibanding XGBoost."
            )
