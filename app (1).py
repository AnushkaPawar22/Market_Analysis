
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MarketBasket AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
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
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.06);
    border: 1px solid #eeeeee;
}

.kpi-title {
    color: #6b7280;
    font-size: 14px;
}

.kpi-value {
    font-size: 28px;
    font-weight: 700;
    color: #111827;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    color: #111827;
    margin-top: 20px;
}

.insight {
    background: #eef2ff;
    padding: 18px;
    border-radius: 14px;
    border-left: 5px solid #4f46e5;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("Groceries_dataset.csv")

    df.columns = df.columns.str.strip()

    # Clean columns
    df["itemDescription"] = (
        df["itemDescription"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        errors="coerce"
    )

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove invalid rows
    df = df.dropna(
        subset=["Member_number", "Date", "itemDescription"]
    )

    return df


df = load_data()

# ============================================================
# BASIC INFORMATION
# ============================================================

total_transactions = len(df)
total_customers = df["Member_number"].nunique()
total_products = df["itemDescription"].nunique()

# ============================================================
# TRANSACTION MATRIX
# ============================================================

@st.cache_data
def create_basket(data):

    basket = data.groupby(
        ["Member_number", "itemDescription"]
    ).size().unstack(fill_value=0)

    basket = (basket > 0).astype(int)

    return basket


basket = create_basket(df)

# ============================================================
# PRODUCT CLUSTERING
# ============================================================

@st.cache_data
def perform_clustering(basket):

    product_data = basket.T

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(product_data)

    # K = 5
    kmeans = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X_scaled)

    result = pd.DataFrame({
        "Product": product_data.index,
        "Cluster": clusters
    })

    score = silhouette_score(
        X_scaled,
        clusters
    )

    # PCA
    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X_scaled)

    result["PC1"] = X_pca[:, 0]
    result["PC2"] = X_pca[:, 1]

    return result, score


cluster_data, silhouette = perform_clustering(basket)

# ============================================================
# CO-PURCHASE ANALYSIS
# ============================================================

@st.cache_data
def create_pairs(basket):

    co_matrix = basket.T.dot(basket)

    np.fill_diagonal(
        co_matrix.values,
        0
    )

    products = co_matrix.index

    pairs = []

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

    pair_df = pair_df.sort_values(
        "Purchase Together",
        ascending=False
    )

    return pair_df


pair_df = create_pairs(basket)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 🛒 MarketBasket AI")

st.sidebar.markdown(
    "### Retail Purchase Intelligence"
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

st.sidebar.info(
    "Dataset: Groceries Dataset\n\n"
    "Model: K-Means\n\n"
    "Clusters: 5"
)

# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

<h1>🛒 MarketBasket AI</h1>

<p>
Retail Market Basket Clustering & Purchase Intelligence
</p>

</div>
""", unsafe_allow_html=True)

# ============================================================
# KPI CARDS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="kpi">
        <div class="kpi-title">TOTAL TRANSACTIONS</div>
        <div class="kpi-value">{total_transactions:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f"""
        <div class="kpi">
        <div class="kpi-title">CUSTOMERS</div>
        <div class="kpi-value">{total_customers:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f"""
        <div class="kpi">
        <div class="kpi-title">PRODUCTS</div>
        <div class="kpi-value">{total_products:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        f"""
        <div class="kpi">
        <div class="kpi-title">SILHOUETTE SCORE</div>
        <div class="kpi-value">{silhouette:.3f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")

# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.markdown(
        '<div class="section-title">📈 Retail Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # TOP PRODUCTS
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

        fig = px.bar(
            top_products,
            x="Purchases",
            y="Product",
            orientation="h",
            title="Top 10 Most Purchased Products"
        )

        fig.update_layout(
            height=450,
            yaxis=dict(categoryorder="total ascending")
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # MONTHLY TREND
    with col2:

        monthly = (
            df.assign(
                Month=df["Date"].dt.to_period("M").astype(str)
            )
            .groupby("Month")
            .size()
            .reset_index(name="Transactions")
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

    st.markdown(
        '<div class="section-title">💡 Quick Insight</div>',
        unsafe_allow_html=True
    )

    top_product = (
        df["itemDescription"]
        .value_counts()
        .index[0]
    )

    st.markdown(
        f"""
        <div class="insight">

        <b>{top_product.title()}</b> is the most frequently
        purchased product in the dataset.

        The system also identifies product groups using
        K-Means clustering and discovers frequently
        co-purchased product pairs.

        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# PURCHASE ANALYTICS
# ============================================================

elif page == "📊 Purchase Analytics":

    st.markdown(
        '<div class="section-title">📊 Purchase Analytics</div>',
        unsafe_allow_html=True
    )

    top_n = st.slider(
        "Number of products to display",
        5,
        20,
        10
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

    fig = px.bar(
        top_products,
        x="Product",
        y="Purchases",
        title=f"Top {top_n} Products"
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # MONTHLY
    monthly = (
        df.assign(
            Month=df["Date"].dt.to_period("M").astype(str)
        )
        .groupby("Month")
        .size()
        .reset_index(name="Transactions")
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

# ============================================================
# FREQUENTLY BOUGHT TOGETHER
# ============================================================

elif page == "🔗 Frequently Bought Together":

    st.markdown(
        '<div class="section-title">🔗 Frequently Bought Together</div>',
        unsafe_allow_html=True
    )

    st.write(
        "These product pairs occur together in the same "
        "customer purchase history."
    )

    top_pairs = pair_df.head(15).copy()

    top_pairs["Pair"] = (
        top_pairs["Product 1"].str.title()
        + " + "
        + top_pairs["Product 2"].str.title()
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
        yaxis=dict(categoryorder="total ascending")
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Top Product Pairs")

    st.dataframe(
        pair_df.head(20),
        use_container_width=True,
        hide_index=True
    )

    csv = pair_df.to_csv(index=False)

    st.download_button(
        "⬇️ Download Product Pairs",
        csv,
        "frequent_product_pairs.csv",
        "text/csv"
    )

# ============================================================
# PRODUCT CLUSTERING
# ============================================================

elif page == "🧩 Product Clustering":

    st.markdown(
        '<div class="section-title">🧩 Product Clustering</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

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

    with col2:

        fig = px.scatter(
            cluster_data,
            x="PC1",
            y="PC2",
            color="Cluster",
            hover_name="Product",
            title="Product Cluster Map",
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Cluster Explorer")

    selected_cluster = st.selectbox(
        "Select Cluster",
        sorted(
            cluster_data["Cluster"].unique()
        )
    )

    selected_products = cluster_data[
        cluster_data["Cluster"] == selected_cluster
    ][["Product", "Cluster"]]

    st.dataframe(
        selected_products,
        use_container_width=True,
        hide_index=True
    )

    csv = cluster_data.to_csv(index=False)

    st.download_button(
        "⬇️ Download Cluster Results",
        csv,
        "product_clusters.csv",
        "text/csv"
    )

# ============================================================
# PRODUCT EXPLORER
# ============================================================

elif page == "🔍 Product Explorer":

    st.markdown(
        '<div class="section-title">🔍 Product Explorer</div>',
        unsafe_allow_html=True
    )

    selected_product = st.selectbox(
        "Choose a product",
        sorted(
            df["itemDescription"].unique()
        )
    )

    # Product cluster
    product_cluster = cluster_data[
        cluster_data["Product"] == selected_product
    ]

    if len(product_cluster) > 0:

        cluster_number = int(
            product_cluster["Cluster"].iloc[0]
        )

        st.success(
            f"🧩 {selected_product.title()} belongs to Cluster {cluster_number}"
        )

    # Related products
    related = pair_df[
        (pair_df["Product 1"] == selected_product)
        |
        (pair_df["Product 2"] == selected_product)
    ].copy()

    if len(related) > 0:

        related["Related Product"] = np.where(
            related["Product 1"] == selected_product,
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

        st.subheader(
            f"🛒 Products commonly purchased with {selected_product.title()}"
        )

        fig = px.bar(
            related.head(10),
            x="Purchase Together",
            y="Related Product",
            orientation="h",
            title="Related Products"
        )

        fig.update_layout(
            height=450,
            yaxis=dict(categoryorder="total ascending")
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            related.head(15),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "No co-purchase information available for this product."
        )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <center>
    <small>
    🛒 MarketBasket AI | Retail Market Basket Clustering
    <br>
    Built using Python • Pandas • Scikit-learn • Plotly • Streamlit
    </small>
    </center>
    """,
    unsafe_allow_html=True
)
