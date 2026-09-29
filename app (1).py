```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="MarketBasket AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #f7f8fc;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(135deg, #111827, #312e81);
    padding: 30px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 17px;
    color: #dbeafe;
}

.kpi {
    background: white;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    text-align: center;
}

.kpi-title {
    color: #6b7280;
    font-size: 13px;
    font-weight: 600;
}

.kpi-value {
    color: #111827;
    font-size: 28px;
    font-weight: 700;
    margin-top: 5px;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    color: #111827;
    margin-top: 25px;
    margin-bottom: 15px;
}

.insight {
    background: #eef2ff;
    padding: 18px;
    border-radius: 14px;
    border-left: 5px solid #4f46e5;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("Groceries_dataset.csv")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Clean product names
    df["itemDescription"] = (
        df["itemDescription"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Convert date
    df["Date"] = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        errors="coerce"
    )

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove missing values
    df = df.dropna(
        subset=[
            "Member_number",
            "Date",
            "itemDescription"
        ]
    )

    return df


try:

    df = load_data()

except Exception as e:

    st.error("Unable to load Groceries_dataset.csv")

    st.write("Make sure the CSV file is present in the GitHub repository.")

    st.stop()


# =========================================================
# BASIC DATA INFORMATION
# =========================================================

total_transactions = len(df)

total_customers = df["Member_number"].nunique()

total_products = df["itemDescription"].nunique()


# =========================================================
# CREATE CUSTOMER-PRODUCT MATRIX
# =========================================================

@st.cache_data
def create_basket(data):

    basket = data.groupby(
        [
            "Member_number",
            "itemDescription"
        ]
    ).size().unstack(fill_value=0)

    # Convert to presence / absence
    basket = (basket > 0).astype(int)

    return basket


basket = create_basket(df)


# =========================================================
# PRODUCT CLUSTERING
# =========================================================

@st.cache_data
def perform_clustering(basket):

    # Products become rows
    product_data = basket.T

    # Scale data
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(product_data)

    # K-Means
    kmeans = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X_scaled)

    # Cluster result
    cluster_data = pd.DataFrame({
        "Product": product_data.index,
        "Cluster": clusters
    })

    # Silhouette score
    score = silhouette_score(
        X_scaled,
        clusters
    )

    # PCA
    pca = PCA(
        n_components=2,
        random_state=42
    )

    X_pca = pca.fit_transform(X_scaled)

    cluster_data["PC1"] = X_pca[:, 0]
    cluster_data["PC2"] = X_pca[:, 1]

    return cluster_data, score


cluster_data, silhouette = perform_clustering(basket)


# =========================================================
# FREQUENT PRODUCT PAIRS
# =========================================================

@st.cache_data
def create_pairs(basket):

    # Product-product co-occurrence matrix
    co_matrix = basket.T.dot(basket)

    products = list(co_matrix.index)

    pairs = []

    # Only upper triangle is checked.
    # Therefore diagonal values are never included.
    for i in range(len(products)):

        for j in range(i + 1, len(products)):

            count = co_matrix.iloc[i, j]

            if count > 0:

                pairs.append(
                    (
                        products[i],
                        products[j],
                        int(count)
                    )
                )

    pair_df = pd.DataFrame(
        pairs,
        columns=[
            "Product 1",
            "Product 2",
            "Purchase Together"
        ]
    )

    if not pair_df.empty:

        pair_df = pair_df.sort_values(
            "Purchase Together",
            ascending=False
        ).reset_index(drop=True)

    return pair_df


pair_df = create_pairs(basket)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    """
    # 🛒 MarketBasket AI
    ### Retail Purchase Intelligence
    """
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "📊 Purchase Analytics",
        "🔗 Frequently Bought Together",
        "🧩 Product Clustering",
        "🔍 Product Explorer"
    ]
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    **Machine Learning**

    Algorithm: K-Means

    Clusters: 5

    Dataset: Groceries Dataset
    """
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">

        <h1>🛒 MarketBasket AI</h1>

        <p>
        Retail Market Basket Clustering &
        Purchase Intelligence Dashboard
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# KPI CARDS
# =========================================================

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.markdown(
        f"""
        <div class="kpi">

            <div class="kpi-title">
            TOTAL TRANSACTIONS
            </div>

            <div class="kpi-value">
            {total_transactions:,}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c2:

    st.markdown(
        f"""
        <div class="kpi">

            <div class="kpi-title">
            CUSTOMERS
            </div>

            <div class="kpi-value">
            {total_customers:,}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c3:

    st.markdown(
        f"""
        <div class="kpi">

            <div class="kpi-title">
            PRODUCTS
            </div>

            <div class="kpi-value">
            {total_products:,}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c4:

    st.markdown(
        f"""
        <div class="kpi">

            <div class="kpi-title">
            SILHOUETTE SCORE
            </div>

            <div class="kpi-value">
            {silhouette:.3f}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# PAGE 1 — OVERVIEW
# =========================================================

if page == "🏠 Overview":

    st.markdown(
        '<div class="section-title">📈 Retail Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)


    # ---------------- TOP PRODUCTS ----------------

    with col1:

        top_products = (
            df["itemDescription"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        top_products.columns = [
            "Product",
            "Purchases"
        ]

        top_products["Product"] = (
            top_products["Product"]
            .str.title()
        )

        fig = px.bar(
            top_products,
            x="Purchases",
            y="Product",
            orientation="h",
            title="Top 10 Most Purchased Products"
        )

        fig.update_layout(
            height=450,
            yaxis=dict(
                categoryorder="total ascending"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ---------------- MONTHLY TREND ----------------

    with col2:

        monthly = (
            df.assign(
                Month=df["Date"]
                .dt.to_period("M")
                .astype(str)
            )
            .groupby("Month")
            .size()
            .reset_index(
                name="Transactions"
            )
        )

        fig = px.line(
            monthly,
            x="Month",
            y="Transactions",
            markers=True,
            title="Monthly Purchase Trend"
        )

        fig.update_layout(
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ---------------- INSIGHT ----------------

    top_product = (
        df["itemDescription"]
        .value_counts()
        .index[0]
    )

    top_count = (
        df["itemDescription"]
        .value_counts()
        .iloc[0]
    )

    st.markdown(
        f"""
        <div class="insight">

        💡 <b>Key Insight:</b>

        <b>{top_product.title()}</b>
        is the most purchased product with
        <b>{top_count:,}</b> recorded purchases.

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# PAGE 2 — PURCHASE ANALYTICS
# =========================================================

elif page == "📊 Purchase Analytics":

    st.markdown(
        '<div class="section-title">📊 Purchase Analytics</div>',
        unsafe_allow_html=True
    )

    top_n = st.slider(
        "Number of products to display",
        min_value=5,
        max_value=20,
        value=10
    )

    top_products = (
        df["itemDescription"]
        .value_counts()
        .head(top_n)
        .reset_index()
    )

    top_products.columns = [
        "Product",
        "Purchases"
    ]

    top_products["Product"] = (
        top_products["Product"]
        .str.title()
    )

    fig = px.bar(
        top_products,
        x="Product",
        y="Purchases",
        title=f"Top {top_n} Products"
    )

    fig.update_layout(
        height=500,
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # Monthly trend

    monthly = (
        df.assign(
            Month=df["Date"]
            .dt.to_period("M")
            .astype(str)
        )
        .groupby("Month")
        .size()
        .reset_index(
            name="Transactions"
        )
    )

    fig = px.area(
        monthly,
        x="Month",
        y="Transactions",
        title="Monthly Transaction Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# PAGE 3 — FREQUENTLY BOUGHT TOGETHER
# =========================================================

elif page == "🔗 Frequently Bought Together":

    st.markdown(
        '<div class="section-title">'
        '🔗 Frequently Bought Together'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Products that frequently appear in the "
        "same customer's purchase history."
    )


    if pair_df.empty:

        st.warning(
            "No product-pair information available."
        )

    else:

        top_pairs = pair_df.head(15).copy()

        top_pairs["Pair"] = (
            top_pairs["Product 1"]
            .str.title()
            + " + "
            + top_pairs["Product 2"]
            .str.title()
        )

        fig = px.bar(
            top_pairs,
            x="Purchase Together",
            y="Pair",
            orientation="h",
            title="Top Frequently Purchased Product Pairs"
        )

        fig.update_layout(
            height=600,
            yaxis=dict(
                categoryorder="total ascending"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        st.subheader(
            "🛒 Product Pair Details"
        )

        display_pairs = pair_df.head(20).copy()

        display_pairs["Product 1"] = (
            display_pairs["Product 1"]
            .str.title()
        )

        display_pairs["Product 2"] = (
            display_pairs["Product 2"]
            .str.title()
        )

        st.dataframe(
            display_pairs,
            use_container_width=True,
            hide_index=True
        )


        # Download

        csv = pair_df.to_csv(
            index=False
        )

        st.download_button(
            label="⬇️ Download Product Pairs",
            data=csv,
            file_name="frequent_product_pairs.csv",
            mime="text/csv"
        )


# =========================================================
# PAGE 4 — PRODUCT CLUSTERING
# =========================================================

elif page == "🧩 Product Clustering":

    st.markdown(
        '<div class="section-title">'
        '🧩 Product Clustering'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(2)


    # ---------------- CLUSTER DISTRIBUTION ----------------

    with col1:

        cluster_counts = (
            cluster_data["Cluster"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        cluster_counts.columns = [
            "Cluster",
            "Products"
        ]

        fig = px.bar(
            cluster_counts,
            x="Cluster",
            y="Products",
            title="Products per Cluster"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ---------------- PCA ----------------

    with col2:

        fig = px.scatter(
            cluster_data,
            x="PC1",
            y="PC2",
            color="Cluster",
            hover_name="Product",
            title="Product Cluster Map"
        )

        fig.update_layout(
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ---------------- CLUSTER SELECTOR ----------------

    st.subheader(
        "🔎 Explore a Cluster"
    )

    selected_cluster = st.selectbox(
        "Select Cluster",
        sorted(
            cluster_data["Cluster"]
            .unique()
        )
    )

    selected_products = (
        cluster_data[
            cluster_data["Cluster"]
            == selected_cluster
        ][
            ["Product", "Cluster"]
        ]
        .copy()
    )

    selected_products["Product"] = (
        selected_products["Product"]
        .str.title()
    )

    st.write(
        f"Products in Cluster {selected_cluster}: "
        f"**{len(selected_products)}**"
    )

    st.dataframe(
        selected_products,
        use_container_width=True,
        hide_index=True
    )


    # Download

    csv = cluster_data.to_csv(
        index=False
    )

    st.download_button(
        label="⬇️ Download Cluster Results",
        data=csv,
        file_name="product_clusters.csv",
        mime="text/csv"
    )


# =========================================================
# PAGE 5 — PRODUCT EXPLORER
# =========================================================

elif page == "🔍 Product Explorer":

    st.markdown(
        '<div class="section-title">'
        '🔍 Product Explorer'
        '</div>',
        unsafe_allow_html=True
    )


    selected_product = st.selectbox(
        "Choose a product",
        sorted(
            df["itemDescription"]
            .unique()
        )
    )


    # ---------------- PRODUCT CLUSTER ----------------

    product_cluster = cluster_data[
        cluster_data["Product"]
        == selected_product
    ]


    if not product_cluster.empty:

        cluster_number = int(
            product_cluster[
                "Cluster"
            ].iloc[0]
        )

        st.success(
            f"🧩 {selected_product.title()} "
            f"belongs to Cluster {cluster_number}"
        )


    # ---------------- RELATED PRODUCTS ----------------

    related = pair_df[
        (
            pair_df["Product 1"]
            == selected_product
        )
        |
        (
            pair_df["Product 2"]
            == selected_product
        )
    ].copy()


    if not related.empty:

        related["Related Product"] = np.where(
            related["Product 1"]
            == selected_product,

            related["Product 2"],

            related["Product 1"]
        )


        related = related[
            [
                "Related Product",
                "Purchase Together"
            ]
        ].sort_values(
            "Purchase Together",
            ascending=False
        )


        related_display = (
            related.head(10)
            .copy()
        )

        related_display["Related Product"] = (
            related_display["Related Product"]
            .str.title()
        )


        fig = px.bar(
            related_display,
            x="Purchase Together",
            y="Related Product",
            orientation="h",
            title=(
                "Products Commonly Purchased With "
                + selected_product.title()
            )
        )

        fig.update_layout(
            height=450,
            yaxis=dict(
                categoryorder="total ascending"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        st.subheader(
            "📋 Related Product Details"
        )

        st.dataframe(
            related_display,
            use_container_width=True,
            hide_index=True
        )


    else:

        st.warning(
            "No co-purchase information is available "
            "for this product."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#6b7280;">

    🛒 <b>MarketBasket AI</b>

    <br>

    Retail Market Basket Clustering

    <br><br>

    Built with Python • Pandas • Scikit-learn • Plotly • Streamlit

    </div>
    """,
    unsafe_allow_html=True
)
```
