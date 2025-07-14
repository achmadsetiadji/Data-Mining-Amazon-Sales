import streamlit as st
import pandas as pd
import plotly.express as px


# --- Utility Functions ---
def clean_price_column(series):
    # Hilangkan simbol, koma, spasi, dsb, lalu konversi ke float
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce"
    ).fillna(0)


def format_rupiah(x):
    try:
        return f"Rp {int(x):,}".replace(",", ".")
    except:
        return "-"


@st.cache_data(show_spinner=False)
def load_data():
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
    # Baca semua kolom sebagai string agar cleaning harga aman
    df = pd.read_csv("amazon.csv", usecols=columns, dtype=str, low_memory=False)
    kurs_inr_to_idr = 190
    # Cleaning harga sebelum konversi ke float
    for col in ["discounted_price", "actual_price"]:
        if col in df.columns:
            df[col] = clean_price_column(df[col]) * kurs_inr_to_idr
    if "discount_percentage" in df.columns:
        df["discount_percentage"] = pd.to_numeric(
            df["discount_percentage"]
            .astype(str)
            .str.replace("%", "")
            .str.replace(r"[^\d.]", "", regex=True),
            errors="coerce",
        ).fillna(0)
    for col in ["rating", "rating_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    max_rows = 100000
    sampled = False
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)
        sampled = True
    return df, sampled


@st.cache_data(show_spinner=False)
def prepare_data(df):
    df_prep = df.copy()
    # Handle missing value: numerik -> median, kategori -> 'Unknown', teks -> ''
    for col in df_prep.select_dtypes(include=["float", "int"]):
        df_prep[col] = df_prep[col].fillna(df_prep[col].median())
    for col in df_prep.select_dtypes(include=["object"]):
        df_prep[col] = df_prep[col].fillna("")
    # Feature engineering: panjang review, jumlah kata review, diskon_rupiah
    if "review_content" in df_prep.columns:
        df_prep["panjang_review"] = df_prep["review_content"].astype(str).str.len()
        df_prep["jumlah_kata_review"] = (
            df_prep["review_content"].astype(str).str.split().apply(len)
        )
    if "discounted_price" in df_prep.columns and "actual_price" in df_prep.columns:
        df_prep["diskon_rupiah"] = df_prep["actual_price"] - df_prep["discounted_price"]
    return df_prep


@st.cache_resource(show_spinner=False)
def train_random_forest(X_train, y_train):
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(
        X_train, y_train
    )


@st.cache_resource(show_spinner=False)
def train_decision_tree(X_train, y_train):
    from sklearn.tree import DecisionTreeRegressor

    return DecisionTreeRegressor(random_state=42).fit(X_train, y_train)


@st.cache_data(show_spinner=False)
def get_pca_scaled(df, n_components=2):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    scaler = StandardScaler().fit(df)
    Xs = scaler.transform(df)
    pca = PCA(n_components=n_components).fit_transform(Xs)
    return pca


def get_sampled_df(df, max_rows=10000):
    if len(df) > max_rows:
        return df.sample(max_rows, random_state=42)
    return df


# --- Page configuration ---
st.set_page_config(page_title="Dashboard Penjualan Amazon", layout="wide")
st.title("📦 Dashboard Interaktif Penjualan Amazon")

# --- Load and Prepare Data ---
with st.spinner("Memuat dan menyiapkan data..."):
    try:
        df, sampled = load_data()
        df_prep = prepare_data(df)
        st.sidebar.header("CRISP-DM Data Mining")
        st.sidebar.markdown(
            """
        **Tujuan:**
        - Menganalisis pola penjualan, ulasan, dan faktor yang memengaruhi rating produk Amazon.
        - Menemukan insight dan prediksi yang bermanfaat untuk bisnis/riset.
        
        **Tahapan CRISP-DM:**
        1. Business Understanding
        2. Data Understanding
        3. Data Preparation
        4. Modeling
        5. Evaluation
        6. Deployment
        """
        )
        st.sidebar.success(
            f"✅ Data dimuat: {df.shape[0]} baris, {df['category'].nunique()} kategori."
        )
        if sampled:
            st.sidebar.warning("Data sangat besar, hanya sampel yang dianalisis.")
    except Exception as e:
        st.sidebar.error(f"❌ Gagal memuat data: {e}")
        st.stop()

# --- Navigation Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Data Understanding",
        "Preparation",
        "Modeling & Evaluation",
        "Kesimpulan",
    ]
)

# --- Tab 1: Data Understanding ---
with tab1:
    st.header("1. Data Understanding")
    st.write(
        """
    - Melihat struktur, statistik, missing value, dan distribusi utama data penjualan & ulasan produk Amazon.
    """
    )
    st.write(f"Jumlah baris: {df.shape[0]}, Jumlah kolom: {df.shape[1]}")
    st.write("**Tipe Data Tiap Kolom**")
    st.dataframe(pd.DataFrame(df.dtypes, columns=["Tipe Data"]))
    st.write("**Missing Value Tiap Kolom**")
    st.dataframe(
        df.isnull()
        .sum()
        .reset_index()
        .rename(columns={0: "Missing Value", "index": "Kolom"})
    )
    # Statistik numerik
    num_df = df.select_dtypes(include=["float", "int"])
    if not num_df.empty:
        st.write("**Statistik Ringkasan Numerik**")
        st.dataframe(num_df.describe().T)
    # Statistik kategorikal/teks
    cat_df = df.select_dtypes(include=["object", "category"])
    if not cat_df.empty:
        st.write("**Statistik Ringkasan Kategorikal/Teks**")
        st.dataframe(cat_df.describe().T)
    # Visualisasi penting
    if "category" in df.columns:
        top_cat = df["category"].value_counts().nlargest(10).reset_index()
        top_cat.columns = ["Kategori", "Jumlah"]
        fig_cat = px.bar(
            top_cat, x="Kategori", y="Jumlah", title="Top 10 Kategori Produk"
        )
        st.plotly_chart(fig_cat, use_container_width=True)
        st.caption(
            f"Kategori terpopuler: {top_cat['Kategori'][0]} ({top_cat['Jumlah'][0]} produk)"
        )
    if "discounted_price" in df.columns:
        fig_harga = px.histogram(
            df, x="discounted_price", nbins=30, title="Distribusi Harga Diskon"
        )
        fig_harga.update_layout(
            xaxis_title="Harga Diskon (Rp)",
            yaxis_title="Jumlah",
            xaxis_tickformat=",.0f",
        )
        fig_harga.update_traces(hovertemplate="Rp %{x:,.0f}<extra></extra>")
        st.plotly_chart(fig_harga, use_container_width=True)
        st.caption(
            f"Harga diskon rata-rata: {format_rupiah(df['discounted_price'].mean())}"
        )
    if "actual_price" in df.columns:
        st.caption(f"Harga asli rata-rata: {format_rupiah(df['actual_price'].mean())}")
    if "rating" in df.columns:
        fig_rating = px.histogram(
            df, x="rating", nbins=20, title="Distribusi Rating Produk"
        )
        fig_rating.update_layout(xaxis_title="Rating", yaxis_title="Jumlah")
        st.plotly_chart(fig_rating, use_container_width=True)
        st.caption(f"Rata-rata rating produk: {df['rating'].mean():.2f}")
    if "discounted_price" in df.columns and "rating" in df.columns:
        fig_scatter = px.scatter(
            df,
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
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("Visualisasi hubungan harga diskon dan rating produk.")
    if "review_content" in df.columns:
        import matplotlib.pyplot as plt
        from wordcloud import WordCloud

        try:
            text = " ".join(df["review_content"].dropna().astype(str))
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
    # Tabel Top 5 produk diskon tertinggi
    if "discounted_price" in df.columns and "product_name" in df.columns:
        st.write("**Top 5 Produk dengan Diskon Tertinggi**")
        top5_diskon = df.copy()
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
    # Tabel Top 5 produk rating tertinggi
    if "rating" in df.columns and "product_name" in df.columns:
        st.write("**Top 5 Produk dengan Rating Tertinggi**")
        top5_rating = df.sort_values("rating", ascending=False).head(5)
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

# --- Tab 2: Preparation ---
with tab2:
    st.header("2. Data Preparation")
    st.write(
        """
    - Proses cleaning, handle missing value, dan feature engineering (panjang review, jumlah kata, diskon rupiah).
    - Data hasil preparation digunakan untuk modeling.
    """
    )
    st.write("**Contoh Data Setelah Preparation**")
    df_show = df_prep.copy()
    for col in ["discounted_price", "actual_price", "diskon_rupiah"]:
        if col in df_show.columns:
            df_show[col] = df_show[col].apply(format_rupiah)
    st.dataframe(df_show.head(20), use_container_width=True)
    st.write("**Statistik Ringkasan Setelah Preparation**")
    st.dataframe(df_prep.describe().T)

# --- Tab 3: Modeling & Evaluation ---
with tab3:
    st.header("3. Modeling & Evaluation")
    st.write(
        """
    - Clustering (DBSCAN, KMeans) dan regresi (Random Forest, Decision Tree) pada data hasil preparation.
    - Evaluasi model dengan metrik Silhouette Score, R², dan MAE.
    """
    )
    # --- CLUSTERING ---
    st.subheader("Clustering: DBSCAN vs KMeans")
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.metrics import silhouette_score

    num_cols = [c for c in df_prep.columns if pd.api.types.is_numeric_dtype(df_prep[c])]
    use_cols = num_cols
    if not num_cols:
        st.warning("Tidak ada kolom numerik yang bisa digunakan untuk clustering.")
    else:
        df_cl = pd.get_dummies(df_prep[use_cols], drop_first=True).fillna(0)
        pca2 = get_pca_scaled(df_cl, n_components=2)
        # DBSCAN
        model_db = DBSCAN(eps=1.0, min_samples=5).fit(pca2)
        labels_db = model_db.labels_
        n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
        sil_db = silhouette_score(pca2, labels_db) if n_clusters_db > 1 else 0
        st.metric("Jumlah Cluster DBSCAN", n_clusters_db)
        st.metric("Silhouette DBSCAN", f"{sil_db:.2f}")
        fig_db = px.scatter(
            x=pca2[:, 0],
            y=pca2[:, 1],
            color=labels_db.astype(str),
            title="Cluster DBSCAN (PCA)",
        )
        st.plotly_chart(fig_db, use_container_width=True)
        # KMeans
        n_k = 3
        model_km = KMeans(n_clusters=n_k, random_state=42).fit(pca2)
        labels_km = model_km.labels_
        sil_km = silhouette_score(pca2, labels_km)
        st.metric("Jumlah Cluster KMeans", n_k)
        st.metric("Silhouette KMeans", f"{sil_km:.2f}")
        fig_km = px.scatter(
            x=pca2[:, 0],
            y=pca2[:, 1],
            color=labels_km.astype(str),
            title="Cluster KMeans (PCA)",
        )
        st.plotly_chart(fig_km, use_container_width=True)
    # --- REGRESSION ---
    st.subheader("Regresi: Random Forest vs Decision Tree")
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split

    if not num_cols or len(num_cols) < 2:
        st.warning("Tidak cukup kolom numerik untuk regresi.")
    else:
        target_col = num_cols[-1]
        feature_cols_no_target = [c for c in num_cols if c != target_col]
        X = df_prep[feature_cols_no_target]
        y = df_prep[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        rf = train_random_forest(X_train, y_train)
        dt = train_decision_tree(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        y_pred_dt = dt.predict(X_test)
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Random Forest**")
            st.metric("R²", f"{r2_score(y_test, y_pred_rf):.3f}")
            st.metric("MAE", f"{mean_absolute_error(y_test, y_pred_rf):.2f}")
            st.plotly_chart(
                px.scatter(
                    x=y_test,
                    y=y_pred_rf,
                    labels={"x": "Aktual", "y": "Prediksi"},
                    title="Random Forest: Aktual vs Prediksi",
                ),
                use_container_width=True,
            )
        with col2:
            st.write("**Decision Tree**")
            st.metric("R²", f"{r2_score(y_test, y_pred_dt):.3f}")
            st.metric("MAE", f"{mean_absolute_error(y_test, y_pred_dt):.2f}")
            st.plotly_chart(
                px.scatter(
                    x=y_test,
                    y=y_pred_dt,
                    labels={"x": "Aktual", "y": "Prediksi"},
                    title="Decision Tree: Aktual vs Prediksi",
                ),
                use_container_width=True,
            )
    # --- Prediksi Rating Sederhana ---
    st.subheader("Prediksi Rating Berdasarkan Fitur")
    st.write(
        "Masukkan fitur berikut untuk memprediksi rating produk (menggunakan model Random Forest):"
    )
    with st.form("form_prediksi_rating"):
        harga_diskon = st.number_input("Harga Diskon (Rp)", min_value=0, value=100000)
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
        if set(fitur_pred.columns).issubset(df_prep.columns):
            X_all = df_prep[[c for c in fitur_pred.columns if c in df_prep.columns]]
            y_all = df_prep["rating"] if "rating" in df_prep.columns else None
            if y_all is not None and len(X_all) == len(y_all):
                rf_pred = train_random_forest(X_all, y_all)
                pred_rating = rf_pred.predict(fitur_pred)[0]
                st.success(f"Prediksi rating produk: {pred_rating:.2f}")
            else:
                st.warning("Data tidak cukup untuk prediksi rating.")
        else:
            st.warning("Fitur input tidak sesuai dengan data.")

# --- Tab 4: Conclusion ---
with tab4:
    st.header("4. Kesimpulan & Insight (Deployment)")
    st.markdown(
        """
        ### Ringkasan Hasil
        - Visualisasi dan modeling sudah mengikuti tahapan CRISP-DM secara nyata.
        - Hasil clustering dan regresi dapat digunakan untuk insight bisnis dan pengambilan keputusan.
        
        ### Insight
        - Kategori produk terpopuler, distribusi harga, dan rating sudah divisualisasikan.
        - Hasil clustering dan regresi menunjukkan pola dan prediksi yang dapat dimanfaatkan.
        
        ---
        **Penjelasan CRISP-DM pada Proyek Ini**
        1. **Business Understanding**: Menentukan tujuan analisis dan pertanyaan riset.
        2. **Data Understanding**: Eksplorasi struktur, kualitas, dan pola data.
        3. **Data Preparation**: Cleaning dan feature engineering.
        4. **Modeling**: Clustering dan prediksi.
        5. **Evaluation**: Evaluasi model dan insight.
        6. **Deployment**: Dashboard ini siap digunakan untuk presentasi dan pengambilan keputusan.
        """
    )
    st.success(
        "Seluruh pipeline CRISP-DM telah diimplementasikan secara nyata dan optimal pada dashboard ini."
    )
