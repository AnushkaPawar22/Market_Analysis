```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="MarketBasket AI",
    page_icon="🛒",
    layout="wide"
)


# =========================================================
# SIMPLE STREAMLIT CSS
# =========================================================

st.markdown(
    """
    <style>
    
    .main {
        background-color: #f5f7fb;
    }

    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e5e7eb;
        padding: 15px;
        border-radius: 12px;
    }

    .stButton button {
        border-radius: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("Groceries_dataset.csv")

    df.columns = df.columns.str.strip()

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

    df = df.drop_duplicates()

    df = df.dropna(
        subset=[
            "Member_number",
            "Date",
            "itemDescription"
        ]
    )

    return df


# =========================================================
# ERROR HANDLING
# =========================================================

try:

    df = load_data()

except Exception as e:

    st.error("Dataset could not be loaded.")

    st.write(
        "Make sure Groceries_dataset.csv is in the same "
        "GitHub repository as app.py."
    )

    st.stop()


# =========================================================
# BASIC INFORMATION
# =========================================================

total_transactions = len(df)

total_customers = df["Member_number"].nunique()

total_products = df["itemDescription"].nunique()


# =========================================================
# CUSTOMER-PRODUCT MATRIX
# =========================================================

@st.cache_data
def create_basket(data):

    basket = data.groupby(
        ["Member_number", "itemDescription"]
    ).size().unstack(fill_value=0)

    basket = (basket > 0).astype(int)

    return basket


basket = create_basket(df)


# =========================================================
# K-MEANS CLUSTERING
# =========================================================

@st.cache_data
def create_clusters(basket):

    product_data = basket.T

    scaler = StandardScaler()

    X = scaler.fit_transform(product_data)

    model = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    result = pd.DataFrame({
        "Product": product_data.index,
        "Cluster": labels
    })

    score = silhouette_score(
        X,
        labels
    )

    pca = PCA(
        n_components=2,
        random_state=42
    )

    components = pca.fit_transform(X)

    result["PC1"] = components[:, 0]

    result["PC2"] = components[:, 1]

    return result, score


cluster_data, silhouette = create_clusters(basket)


# =========================================================
# PRODUCT CO-PURCHASE
# =========================================================

@st.cache_data
def create_product_pairs(basket):

    matrix = basket.T.dot(basket)

    products = list(matrix.index)

    pairs = []

    for i in range(len(products)):

        for j in range(i + 1, len(products)):

            value = matrix.iloc[i, j]

            if value > 0:

                pairs.append(
                    [
                        products[i],
                        products[j],
                        int(value)
                    ]
                )

    result = pd.DataFrame(
        pairs,
        columns=[
            "Product 1",
            "Product 2",
            "Purchased Together"
        ]
    )

    if not result.empty:

        result = result.sort_values(
            "Purchased Together",
            ascending=False
        ).reset_index(drop=True)

    return result


pair_data = create_product_pairs(basket)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🛒 MarketBasket AI")

st.sidebar.write(
    "Retail Purchase Intelligence"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Purchase Analysis",
        "Product Pairs",
        "Clustering",
        "Product Explorer"
    ]
)

st.sidebar.divider()

st.sidebar.write("### Project")

st.sidebar.write(
    "Retail Market Basket Clustering"
)

st.sidebar.write(
    "Machine Learning: K-Means"
)

st.sidebar.write(
    "Number of clusters: 5"
)


# =========================================================
# MAIN TITLE
# =========================================================

st.title("🛒 MarketBasket AI")

st.caption(
    "Retail Market Basket Clustering & Purchase Intelligence"
)

st.divider()


# =========================================================
# KPI SECTION
# =========================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "🧾 Transactions",
    f"{total_transactions:,}"
)

col2.metric(
    "👥 Customers",
    f"{total_customers:,}"
)

col3.metric(
    "📦 Products",
    f"{total_products:,}"
)

col4.metric(
    "🎯 Silhouette Score",
    f"{silhouette:.3f}"
)


st.write("")


# =========================================================
# DASHBOARD PAGE
# =========================================================

if page == "Dashboard":

    st.header("📊 Retail Overview")

    left, right = st.columns(2)


    # ---------------- TOP PRODUCTS ----------------

    with left:

        top = (
            df["itemDescription"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        top.columns = [
            "Product",
            "Purchases"
        ]

        top["Product"] = (
            top["Product"]
            .str.title()
        )

        fig = px.bar(
            top,
            x="Purchases",
            y="Product",
            orientation="h",
            title="Top 10 Purchased Products"
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

    with right:

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


    # ---------------- QUICK INSIGHTS ----------------

    st.header("💡 Quick Insights")

    most_purchased = (
        df["itemDescription"]
        .value_counts()
        .index[0]
    )

    purchase_count = (
        df["itemDescription"]
        .value_counts()
        .iloc[0]
    )

    a, b, c = st.columns(3)

    a.info(
        f"Most purchased product:\n\n"
        f"**{most_purchased.title()}**"
    )

    b.info(
        f"Number of products:\n\n"
        f"**{total_products}**"
    )

    c.info(
        f"Product clusters:\n\n"
        f"**5**"
    )


# =========================================================
# PURCHASE ANALYSIS
# =========================================================

elif page == "Purchase Analysis":

    st.header("📈 Purchase Analysis")

    number = st.slider(
        "Number of products",
        5,
        20,
        10
    )

    top = (
        df["itemDescription"]
        .value_counts()
        .head(number)
        .reset_index()
    )

    top.columns = [
        "Product",
        "Purchases"
    ]

    top["Product"] = (
        top["Product"]
        .str.title()
    )

    fig = px.bar(
        top,
        x="Product",
        y="Purchases",
        title=f"Top {number} Products"
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.subheader("📅 Monthly Transaction Trend")

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
        y="Transactions"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# PRODUCT PAIRS
# =========================================================

elif page == "Product Pairs":

    st.header("🔗 Frequently Bought Together")

    st.write(
        "These products frequently occur together "
        "in customer purchase histories."
    )

    if pair_data.empty:

        st.warning(
            "No product pair information found."
        )

    else:

        top_pairs = pair_data.head(15).copy()

        top_pairs["Pair"] = (
            top_pairs["Product 1"].str.title()
            + " + "
            + top_pairs["Product 2"].str.title()
        )

        fig = px.bar(
            top_pairs,
            x="Purchased Together",
            y="Pair",
            orientation="h",
            title="Top 15 Product Combinations"
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


        st.subheader("Product Pair Table")

        display = pair_data.head(20).copy()

        display["Product 1"] = (
            display["Product 1"].str.title()
        )

        display["Product 2"] = (
            display["Product 2"].str.title()
        )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )


        st.download_button(
            "⬇️ Download Product Pairs",
            pair_data.to_csv(index=False),
            "frequent_product_pairs.csv",
            "text/csv"
        )


# =========================================================
# CLUSTERING
# =========================================================

elif page == "Clustering":

    st.header("🧩 Product Clustering")

    st.write(
        "K-Means groups products according to "
        "similar purchasing patterns."
    )


    left, right = st.columns(2)


    # ---------------- CLUSTER SIZE ----------------

    with left:

        counts = (
            cluster_data["Cluster"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        counts.columns = [
            "Cluster",
            "Products"
        ]

        fig = px.bar(
            counts,
            x="Cluster",
            y="Products",
            title="Products in Each Cluster"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ---------------- PCA ----------------

    with right:

        fig = px.scatter(
            cluster_data,
            x="PC1",
            y="PC2",
            color="Cluster",
            hover_name="Product",
            title="Product Cluster Map"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ---------------- CLUSTER SELECTOR ----------------

    st.subheader("🔎 Explore Cluster")

    selected = st.selectbox(
        "Choose cluster",
        sorted(
            cluster_data["Cluster"].unique()
        )
    )

    products = cluster_data[
        cluster_data["Cluster"] == selected
    ][
        ["Product", "Cluster"]
    ].copy()

    products["Product"] = (
        products["Product"]
        .str.title()
    )

    st.write(
        f"Cluster {selected} contains "
        f"**{len(products)} products**."
    )

    st.dataframe(
        products,
        use_container_width=True,
        hide_index=True
    )


    st.download_button(
        "⬇️ Download Cluster Data",
        cluster_data.to_csv(index=False),
        "product_clusters.csv",
        "text/csv"
    )


# =========================================================
# PRODUCT EXPLORER
# =========================================================

elif page == "Product Explorer":

    st.header("🔍 Product Explorer")

    selected_product = st.selectbox(
        "Select a product",
        sorted(
            df["itemDescription"]
            .unique()
        )
    )


    # Find cluster

    product_info = cluster_data[
        cluster_data["Product"]
        == selected_product
    ]


    if not product_info.empty:

        cluster_number = int(
            product_info["Cluster"].iloc[0]
        )

        st.success(
            f"🧩 {selected_product.title()} "
            f"belongs to Cluster {cluster_number}"
        )


    # Find related products

    related = pair_data[
        (
            pair_data["Product 1"]
            == selected_product
        )
        |
        (
            pair_data["Product 2"]
            == selected_product
        )
    ].copy()


    if related.empty:

        st.warning(
            "No frequently co-purchased products "
            "were found."
        )

    else:

        related["Related Product"] = np.where(
            related["Product 1"]
            == selected_product,

            related["Product 2"],

            related["Product 1"]
        )

        related = related[
            [
                "Related Product",
                "Purchased Together"
            ]
        ].sort_values(
            "Purchased Together",
            ascending=False
        )


        st.subheader(
            "🛍️ Commonly Purchased With"
        )


        chart_data = related.head(10).copy()

        chart_data["Related Product"] = (
            chart_data["Related Product"]
            .str.title()
        )


        fig = px.bar(
            chart_data,
            x="Purchased Together",
            y="Related Product",
            orientation="h",
            title=(
                "Products Purchased With "
                + selected_product.title()
            )
        )

        fig.update_layout(
            yaxis=dict(
                categoryorder="total ascending"
            ),
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        st.dataframe(
            chart_data,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "MarketBasket AI • Retail Market Basket Clustering "
    "• Python • Pandas • Scikit-learn • Plotly • Streamlit"
)
```
