import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="Hybrid Recommender — All_Beauty", layout="wide")
st.title("Hybrid Recommender — All_Beauty")

st.markdown(
    "🚀 This app shows **Popular products**, **Content-based**, **Collaborative**, and a **Hybrid** blend. "
    "Files required: `catalog_Price.pkl`, `tfidf.joblib`, `X_item_user.joblib`, `item_index.joblib`."
)

# ---------- Helpers ----------
def clean_price(x):
    """Ensure price is a clean float"""
    if isinstance(x, list) and len(x) > 0:
        return x[0]
    try:
        return float(x)
    except:
        return np.nan

def get_image_url(images):
    if isinstance(images, list) and len(images) > 0:
        return images[0].get("thumb")
    return None

def show_product(row, show_score=False):
    cols = st.columns([1, 4])
    with cols[0]:
        img = get_image_url(row.get("images"))
        if img:
            st.image(img, width=120)
        else:
            st.write("📷")
    with cols[1]:
        st.write(f"**{row['title']}**")

        # ✅ Always show price
        if pd.notna(row.get("price")):
            st.write(f"💲 {row['price']:.2f}")

        # ✅ Show ratings if available
        if "average_rating" in row and pd.notna(row["average_rating"]):
            stars = "⭐" * int(round(row["average_rating"]))
            st.write(f"{stars} ({row.get('rating_number', 0)} reviews)")

        if show_score and "score" in row:
            st.caption(f"score: {row['score']:.3f}")

# ---------- Loaders ----------
@st.cache_resource
def load_catalog():
    catalog = pd.read_pickle('catalog_Price.pkl')
    catalog['parent_asin'] = catalog['parent_asin'].astype(str)
    catalog['price'] = catalog['price'].apply(clean_price)   # ✅ fix prices
    return catalog, {asin: i for i, asin in enumerate(catalog['parent_asin'].tolist())}

@st.cache_resource
def load_content_vectors(catalog):
    tfidf = joblib.load('tfidf.joblib')
    X_tfidf = tfidf.transform(catalog['content_text'].fillna(""))
    return tfidf, X_tfidf

@st.cache_resource
def load_collab_matrix():
    X_item_user = joblib.load('X_item_user.joblib')
    item_index = joblib.load('item_index.joblib')
    inv_item_index = [None] * X_item_user.shape[0]
    for a, i in item_index.items():
        if i < len(inv_item_index):
            inv_item_index[i] = str(a)
    return X_item_user, item_index, np.array(inv_item_index, dtype=object)

# ---------- Recommendation functions ----------
def get_popular(catalog, min_ratings=20, topn=200):
    if 'rating_number' in catalog.columns:
        pop = catalog[catalog['rating_number'].fillna(0).astype(int) >= min_ratings]
        pop = pop.sort_values(['average_rating','rating_number'], ascending=[False, False])
    else:
        pop = catalog.copy()
    if pop.empty:
        pop = catalog.copy()
    # ✅ shuffle so results look fresh each time
    return pop.sample(n=min(topn, len(pop)), random_state=42).reset_index(drop=True)

def get_content_recs(asin, catalog, catalog_idx_map, X_tfidf, k=10):
    i = catalog_idx_map.get(str(asin))
    if i is None: return pd.DataFrame()
    sims = linear_kernel(X_tfidf[i], X_tfidf).ravel()
    order = np.argsort(sims)[::-1]
    order = [o for o in order if o != i][:k]
    recs = catalog.iloc[order].copy()
    recs['score'] = sims[order]
    return recs

def get_collab_recs(asin, catalog, X_item_user, item_index, inv_item_index, k=10):
    i = item_index.get(str(asin))
    if i is None: return pd.DataFrame()
    sims = X_item_user[i].dot(X_item_user.T).toarray().ravel()
    order = np.argsort(sims)[::-1]
    order = [o for o in order if o != i][:k]
    asins = [inv_item_index[o] for o in order if inv_item_index[o] is not None]
    recs = catalog[catalog['parent_asin'].isin(asins)].copy()
    score_map = {inv_item_index[o]: sims[o] for o in order if inv_item_index[o] is not None}
    recs['score'] = recs['parent_asin'].map(score_map)
    recs = recs.sort_values('score', ascending=False)
    return recs

def get_hybrid_recs(asin, catalog, catalog_idx_map, X_tfidf, X_item_user, item_index, inv_item_index, k=10, alpha=0.5):
    n = catalog.shape[0]
    content_scores = np.zeros(n)
    collab_scores = np.zeros(n)

    i = catalog_idx_map.get(str(asin))
    if i is not None:
        content_scores = linear_kernel(X_tfidf[i], X_tfidf).ravel()

    ii = item_index.get(str(asin))
    if ii is not None:
        collab_vec = X_item_user[ii].dot(X_item_user.T).toarray().ravel()
        score_by_catalog_index = np.zeros(n)
        asin_to_catalog = {a: j for j, a in enumerate(catalog['parent_asin'].tolist())}
        for row_idx, a in enumerate(inv_item_index):
            if a is None: continue
            cj = asin_to_catalog.get(a)
            if cj is not None:
                score_by_catalog_index[cj] = collab_vec[row_idx]
        collab_scores = score_by_catalog_index

    def norm(a):
        amax, amin = a.max(), a.min()
        return np.zeros_like(a) if amax - amin < 1e-12 else (a - amin) / (amax - amin)

    cs, rs = norm(content_scores), norm(collab_scores)
    final = alpha * cs + (1 - alpha) * rs
    if i is not None:
        final[i] = -1
    order = np.argsort(final)[::-1][:k]
    recs = catalog.iloc[order].copy()
    recs['score'] = final[order]
    return recs

# ---------- UI ----------
catalog, catalog_idx_map = load_catalog()
tfidf, X_tfidf = load_content_vectors(catalog)
X_item_user, item_index, inv_item_index = load_collab_matrix()

st.sidebar.header("Controls")
min_r = st.sidebar.slider("Min #ratings for Popular", 0, 1000, 20, step=10)
topn = st.sidebar.slider("How many Popular items", 10, 500, 200, step=10)
k = st.sidebar.slider("Recommendations per section", 1, 20, 6, step=1)
alpha = st.sidebar.slider("Hybrid: content weight (alpha)", 0.0, 1.0, 0.5, step=0.05)

# ✅ Search entire catalog
search = st.text_input("🔍 Search products by title:", "")
if search:
    results = catalog[catalog['title'].str.contains(search, case=False, na=False)].head(20)
    if not results.empty:
        st.subheader("Search results")
        for _, r in results.iterrows():
            show_product(r)
        st.markdown("---")

popular = get_popular(catalog, min_ratings=min_r, topn=topn)

st.subheader("🔥 Popular Products")
options = list(popular.index)
sel_idx = st.selectbox("Select a product:", options=options, format_func=lambda i: popular.loc[i, 'title'])
sel_row = popular.loc[sel_idx]
sel_asin = str(sel_row['parent_asin'])

st.write("**Selected:**", sel_row['title'])
img = get_image_url(sel_row['images'])
if img:
    st.image(img, width=200)
if pd.notna(sel_row.get("price")):
    st.write(f"💲 {sel_row['price']:.2f}")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🧩 Content-based")
    for _, r in get_content_recs(sel_asin, catalog, catalog_idx_map, X_tfidf, k=k).iterrows():
        show_product(r, show_score=True)

with col2:
    st.markdown("### 👥 Collaborative")
    for _, r in get_collab_recs(sel_asin, catalog, X_item_user, item_index, inv_item_index, k=k).iterrows():
        show_product(r, show_score=True)

st.markdown("---")
st.markdown("### ⚡ Hybrid Blend")
for _, r in get_hybrid_recs(sel_asin, catalog, catalog_idx_map, X_tfidf, X_item_user, item_index, inv_item_index, k=k, alpha=alpha).iterrows():
    show_product(r, show_score=True)
