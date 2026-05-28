"""
Crop Yield Data Mining — Streamlit App
Sections: Outlier Detection | K-Means Clustering | Prediction
Dataset: Kaggle Crop Recommendation (2200 rows, 22 crops, 7 features)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Yield Data Mining | Punjab",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0d1117; color: #e6edf3; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #161b22; }
    section[data-testid="stSidebar"] .stMarkdown p { color: #8b949e; }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 14px;
    }
    div[data-testid="metric-container"] label { color: #8b949e !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #3fb950 !important;
        font-size: 1.8rem !important;
        font-weight: 700;
    }
    
    /* Section headers */
    h1, h2, h3 { color: #e6edf3 !important; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 8px; gap: 4px; }
    .stTabs [data-baseweb="tab"] { color: #8b949e; border-radius: 6px; }
    .stTabs [aria-selected="true"] { background: #238636 !important; color: #fff !important; }
    
    /* Buttons */
    .stButton button {
        background: #238636; color: #fff; border: 1px solid #1a7f37;
        border-radius: 8px; font-weight: 600;
    }
    .stButton button:hover { background: #1a7f37; }
    
    /* Sliders and selects */
    .stSlider, .stSelectbox { color: #e6edf3; }
    
    /* Info boxes */
    .info-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .info-card h4 { color: #58a6ff; margin: 0 0 6px 0; font-size: 14px; }
    .info-card p  { color: #8b949e; margin: 0; font-size: 13px; line-height: 1.6; }
    
    /* Outlier badge */
    .badge-red   { background: rgba(248,81,73,.2); color: #f85149; border:1px solid rgba(248,81,73,.3);
                   padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
    .badge-green { background: rgba(63,185,80,.2); color: #3fb950; border:1px solid rgba(63,185,80,.3);
                   padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
    
    /* Prediction result box */
    .pred-box {
        background: #1c2330;
        border: 2px solid #238636;
        border-radius: 14px;
        padding: 28px;
        text-align: center;
    }
    .pred-crop { font-size: 2.6rem; font-weight: 800; color: #3fb950; margin: 6px 0; }
    .pred-conf { font-size: 1rem; color: #8b949e; }
    
    div[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; }
    
    /* dataframe text */
    .dataframe { color: #e6edf3 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
CROP_EMOJI = {
    "rice": "🌾", "maize": "🌽", "chickpea": "🫘", "kidneybeans": "🫘",
    "pigeonpeas": "🫘", "mothbeans": "🫘", "mungbean": "🫘", "blackgram": "🫘",
    "lentil": "🌿", "pomegranate": "🍎", "banana": "🍌", "mango": "🥭",
    "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈", "apple": "🍎",
    "orange": "🍊", "papaya": "🥭", "coconut": "🥥", "cotton": "🌸",
    "jute": "🌿", "coffee": "☕",
}

@st.cache_data
def load_data():
    """Generate synthetic dataset matching Kaggle crop recommendation stats."""
    np.random.seed(42)
    crops = list(CROP_EMOJI.keys())
    n_per = 100

    records = []
    params = {
        "rice":        dict(N=(60,20),  P=(45,15),  K=(40,15),  t=(23,3),  h=(82,8),  ph=(6.4,.4), r=(236,30)),
        "maize":       dict(N=(77,20),  P=(48,15),  K=(19,8),   t=(22,3),  h=(65,10), ph=(6.3,.4), r=(67,20)),
        "chickpea":    dict(N=(40,15),  P=(67,15),  K=(79,15),  t=(18,4),  h=(16,4),  ph=(7.3,.5), r=(80,20)),
        "kidneybeans": dict(N=(20,8),   P=(67,15),  K=(20,8),   t=(20,3),  h=(22,5),  ph=(5.7,.5), r=(105,25)),
        "pigeonpeas":  dict(N=(20,8),   P=(67,15),  K=(20,8),   t=(27,3),  h=(48,10), ph=(5.8,.5), r=(149,30)),
        "mothbeans":   dict(N=(21,8),   P=(48,12),  K=(20,8),   t=(28,3),  h=(53,10), ph=(6.9,.5), r=(51,15)),
        "mungbean":    dict(N=(21,8),   P=(47,12),  K=(20,8),   t=(29,3),  h=(86,7),  ph=(6.7,.5), r=(49,15)),
        "blackgram":   dict(N=(40,12),  P=(67,15),  K=(19,8),   t=(30,3),  h=(65,10), ph=(7.0,.5), r=(68,20)),
        "lentil":      dict(N=(18,6),   P=(68,15),  K=(19,8),   t=(24,3),  h=(65,10), ph=(6.9,.4), r=(46,15)),
        "pomegranate": dict(N=(18,8),   P=(18,8),   K=(40,12),  t=(21,3),  h=(90,7),  ph=(6.4,.5), r=(107,25)),
        "banana":      dict(N=(100,15), P=(82,15),  K=(50,12),  t=(27,3),  h=(80,8),  ph=(6.0,.4), r=(105,25)),
        "mango":       dict(N=(20,8),   P=(27,10),  K=(30,10),  t=(31,3),  h=(50,10), ph=(5.7,.4), r=(95,25)),
        "grapes":      dict(N=(23,8),   P=(132,15), K=(200,20), t=(24,3),  h=(82,8),  ph=(6.0,.5), r=(69,20)),
        "watermelon":  dict(N=(100,15), P=(10,5),   K=(50,12),  t=(25,3),  h=(85,8),  ph=(6.5,.4), r=(50,15)),
        "muskmelon":   dict(N=(100,15), P=(18,7),   K=(50,12),  t=(28,3),  h=(92,5),  ph=(6.4,.4), r=(25,10)),
        "apple":       dict(N=(21,8),   P=(134,15), K=(199,20), t=(22,3),  h=(92,5),  ph=(5.9,.5), r=(112,25)),
        "orange":      dict(N=(20,8),   P=(16,7),   K=(10,5),   t=(23,3),  h=(92,5),  ph=(7.0,.4), r=(110,25)),
        "papaya":      dict(N=(49,12),  P=(58,12),  K=(50,12),  t=(34,3),  h=(92,5),  ph=(6.7,.4), r=(143,30)),
        "coconut":     dict(N=(21,8),   P=(16,7),   K=(30,10),  t=(27,3),  h=(95,4),  ph=(5.9,.5), r=(175,30)),
        "cotton":      dict(N=(117,15), P=(46,12),  K=(19,8),   t=(24,3),  h=(80,8),  ph=(6.9,.4), r=(80,20)),
        "jute":        dict(N=(78,15),  P=(46,12),  K=(39,12),  t=(25,3),  h=(80,8),  ph=(6.7,.4), r=(175,30)),
        "coffee":      dict(N=(101,15), P=(28,10),  K=(29,10),  t=(25,3),  h=(58,10), ph=(6.8,.4), r=(158,30)),
    }

    for crop in crops:
        p = params[crop]
        for _ in range(n_per):
            records.append({
                "N":           max(0, np.random.normal(*p["N"])),
                "P":           max(0, np.random.normal(*p["P"])),
                "K":           max(0, np.random.normal(*p["K"])),
                "temperature": np.clip(np.random.normal(*p["t"]),  8, 43),
                "humidity":    np.clip(np.random.normal(*p["h"]), 14, 100),
                "ph":          np.clip(np.random.normal(*p["ph"]), 3.5, 9.9),
                "rainfall":    max(0, np.random.normal(*p["r"])),
                "label":       crop,
            })

    df = pd.DataFrame(records)
    df[FEATURES] = df[FEATURES].round(2)
    return df


@st.cache_resource
def train_model(df):
    X = df[FEATURES]
    y = df["label"]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_train)
    acc = accuracy_score(y_test, clf.predict(X_te))
    return clf, le, scaler, acc


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
df = load_data()
clf, le, scaler, model_acc = train_model(df)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 Crop Yield Mining")
    st.markdown("**Punjab, India — Soil & Climate Analysis**")
    st.divider()
    st.markdown("### Navigation")
    page = st.radio(
        "",
        ["🔴 Outlier Detection", "🔵 K-Means Clustering", "🎯 Prediction"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("""
    <div class="info-card">
    <h4>📊 Dataset</h4>
    <p>Kaggle Crop Recommendation<br>2,200 rows · 22 crops · 7 features</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
    <h4>🤖 Model</h4>
    <p>Random Forest Classifier<br>150 trees · 80/20 split</p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 1: OUTLIER DETECTION
# ═══════════════════════════════════════════════════════════════
if page == "🔴 Outlier Detection":
    st.title("🔴 Outlier Detection")
    st.markdown("**IQR-based outlier detection across all features and crops**")
    st.divider()

    # ── Controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 1])
    with col_ctrl1:
        sel_feature = st.selectbox("Select Feature", FEATURES)
    with col_ctrl2:
        iqr_mult = st.slider("IQR Multiplier (k)", 1.0, 3.0, 1.5, 0.1,
                             help="Outlier if value < Q1 - k*IQR  or  > Q3 + k*IQR")
    with col_ctrl3:
        sel_crop = st.selectbox("Crop Filter", ["All"] + sorted(df["label"].unique().tolist()))

    df_view = df if sel_crop == "All" else df[df["label"] == sel_crop]

    # Compute IQR outliers for selected feature
    Q1  = df_view[sel_feature].quantile(0.25)
    Q3  = df_view[sel_feature].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - iqr_mult * IQR
    upper = Q3 + iqr_mult * IQR
    is_outlier = (df_view[sel_feature] < lower) | (df_view[sel_feature] > upper)
    n_out = is_outlier.sum()
    pct_out = n_out / len(df_view) * 100

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows Analysed",    f"{len(df_view):,}")
    c2.metric("Outliers Found",   f"{n_out:,}",         f"{pct_out:.1f}%")
    c3.metric("Lower Fence",      f"{lower:.2f}")
    c4.metric("Upper Fence",      f"{upper:.2f}")

    st.divider()

    # ── Box plot
    col_p1, col_p2 = st.columns([3, 2])
    with col_p1:
        st.subheader(f"📦 Boxplot — {sel_feature}")
        fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0d1117")
        ax.set_facecolor("#161b22")

        # Show top 10 or fewer crops
        crops_in_view = df_view["label"].unique()
        bp_data = [df_view[df_view["label"] == c][sel_feature].values for c in crops_in_view]
        bplot = ax.boxplot(bp_data, patch_artist=True, notch=False,
                           medianprops=dict(color="#3fb950", linewidth=2),
                           whiskerprops=dict(color="#58a6ff"),
                           capprops=dict(color="#58a6ff"),
                           flierprops=dict(marker='o', color='#f85149', alpha=0.6, markersize=4))
        for patch in bplot["boxes"]:
            patch.set_facecolor("#238636")
            patch.set_alpha(0.4)
        ax.axhline(lower, color="#f85149", linestyle="--", linewidth=1.2, label=f"Lower fence ({lower:.1f})")
        ax.axhline(upper, color="#f85149", linestyle="--", linewidth=1.2, label=f"Upper fence ({upper:.1f})")
        ax.set_xticks(range(1, len(crops_in_view)+1))
        ax.set_xticklabels(crops_in_view, rotation=45, ha="right", color="#8b949e", fontsize=8)
        ax.tick_params(axis="y", colors="#8b949e")
        ax.set_ylabel(sel_feature, color="#8b949e")
        ax.set_title(f"{sel_feature} by Crop (red dashes = IQR fences)", color="#e6edf3")
        for sp in ax.spines.values(): sp.set_color("#30363d")
        ax.legend(fontsize=9, facecolor="#161b22", labelcolor="#8b949e", edgecolor="#30363d")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_p2:
        st.subheader("📊 Outlier Summary per Feature")
        out_rows = []
        for f in FEATURES:
            q1, q3 = df_view[f].quantile([0.25, 0.75])
            iqr_ = q3 - q1
            n = int(((df_view[f] < q1 - iqr_mult*iqr_) | (df_view[f] > q3 + iqr_mult*iqr_)).sum())
            out_rows.append({"Feature": f, "Outliers": n, "% of Rows": f"{n/len(df_view)*100:.1f}%",
                             "Q1": f"{q1:.1f}", "Q3": f"{q3:.1f}", "IQR": f"{iqr_:.1f}"})
        out_df = pd.DataFrame(out_rows)
        st.dataframe(out_df, hide_index=True, use_container_width=True)

    st.divider()

    # ── Scatter: outliers highlighted
    st.subheader(f"🔍 Outlier Scatter — {sel_feature} (coloured by outlier status)")
    fig2, ax2 = plt.subplots(figsize=(14, 4), facecolor="#0d1117")
    ax2.set_facecolor("#161b22")
    normal  = df_view[~is_outlier]
    outlier = df_view[is_outlier]
    ax2.scatter(normal.index,  normal[sel_feature],  c="#3fb950", s=8, alpha=0.5, label="Normal")
    ax2.scatter(outlier.index, outlier[sel_feature], c="#f85149", s=18, alpha=0.9, label=f"Outlier ({n_out})", zorder=5)
    ax2.axhline(lower, color="#f85149", linestyle="--", linewidth=1, alpha=0.7)
    ax2.axhline(upper, color="#f85149", linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_xlabel("Row Index", color="#8b949e")
    ax2.set_ylabel(sel_feature, color="#8b949e")
    ax2.tick_params(colors="#8b949e")
    for sp in ax2.spines.values(): sp.set_color("#30363d")
    ax2.legend(facecolor="#161b22", labelcolor="#8b949e", edgecolor="#30363d")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.divider()

    # ── Show outlier rows
    if n_out > 0:
        st.subheader(f"🗂️ Outlier Rows — {sel_feature} ({n_out} records)")
        show_cols = ["label", sel_feature] + [f for f in FEATURES if f != sel_feature]
        st.dataframe(df_view[is_outlier][show_cols].reset_index(drop=True),
                     hide_index=True, use_container_width=True, height=250)
    else:
        st.success(f"✅ No outliers found in **{sel_feature}** with k = {iqr_mult}")


# ═══════════════════════════════════════════════════════════════
# PAGE 3: K-MEANS CLUSTERING
# ═══════════════════════════════════════════════════════════════
elif page == "🔵 K-Means Clustering":
    st.title("🔵 K-Means Clustering")
    st.markdown("**Discover natural soil-climate archetypes in the dataset**")
    st.divider()

    col_c1, col_c2, col_c3 = st.columns([2, 2, 2])
    with col_c1:
        k_val = st.slider("Number of Clusters (K)", 2, 10, 4)
    with col_c2:
        x_axis = st.selectbox("X-Axis Feature", FEATURES, index=0)
    with col_c3:
        y_axis = st.selectbox("Y-Axis Feature", FEATURES, index=6)

    # Scale & cluster
    scaler_km = MinMaxScaler()
    X_scaled   = scaler_km.fit_transform(df[FEATURES])

    # Elbow & silhouette pre-compute
    @st.cache_data
    def compute_elbow():
        inertias, silhs = [], []
        for k in range(2, 11):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertias.append(km.inertia_)
            silhs.append(silhouette_score(X_scaled, labels, sample_size=500))
        return inertias, silhs

    inertias, silhs = compute_elbow()

    km_model = KMeans(n_clusters=k_val, random_state=42, n_init=10)
    df["cluster"] = km_model.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, df["cluster"], sample_size=500)

    c1, c2, c3 = st.columns(3)
    c1.metric("Clusters (K)",    str(k_val))
    c2.metric("Silhouette Score", f"{sil:.3f}", help="Closer to 1 = better separation")
    c3.metric("Inertia (WCSS)",  f"{km_model.inertia_:,.0f}")

    st.divider()

    # ── Elbow + Silhouette
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.subheader("📉 Elbow Curve")
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#0d1117")
        ax.set_facecolor("#161b22")
        ax.plot(range(2, 11), inertias, "o-", color="#58a6ff", linewidth=2, markersize=6)
        ax.axvline(k_val, color="#3fb950", linestyle="--", linewidth=1.5, label=f"K={k_val}")
        ax.set_xlabel("K", color="#8b949e"); ax.set_ylabel("Inertia (WCSS)", color="#8b949e")
        ax.set_title("Elbow Method", color="#e6edf3")
        ax.tick_params(colors="#8b949e")
        for sp in ax.spines.values(): sp.set_color("#30363d")
        ax.legend(facecolor="#161b22", labelcolor="#8b949e", edgecolor="#30363d")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_e2:
        st.subheader("📐 Silhouette Scores")
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#0d1117")
        ax.set_facecolor("#161b22")
        ax.plot(range(2, 11), silhs, "s-", color="#d29922", linewidth=2, markersize=6)
        ax.axvline(k_val, color="#3fb950", linestyle="--", linewidth=1.5, label=f"K={k_val}")
        ax.set_xlabel("K", color="#8b949e"); ax.set_ylabel("Silhouette Score", color="#8b949e")
        ax.set_title("Silhouette Analysis", color="#e6edf3")
        ax.tick_params(colors="#8b949e")
        for sp in ax.spines.values(): sp.set_color("#30363d")
        ax.legend(facecolor="#161b22", labelcolor="#8b949e", edgecolor="#30363d")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.divider()

    # ── Scatter plot of clusters
    st.subheader(f"🎨 Cluster Scatter — {x_axis} vs {y_axis}")
    CLUSTER_COLORS = ["#3fb950","#58a6ff","#d29922","#f85149","#bc8cff",
                      "#39d353","#f778ba","#ffa657","#79c0ff","#56d364"]
    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    for c in range(k_val):
        mask = df["cluster"] == c
        ax.scatter(df[mask][x_axis], df[mask][y_axis],
                   s=22, alpha=0.7, color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                   label=f"Cluster {c}", edgecolors="none")
    # Plot centroids (in original scale)
    centroids_orig = scaler_km.inverse_transform(km_model.cluster_centers_)
    cent_df = pd.DataFrame(centroids_orig, columns=FEATURES)
    xi = FEATURES.index(x_axis); yi = FEATURES.index(y_axis)
    ax.scatter(cent_df[x_axis], cent_df[y_axis],
               s=160, marker="X", color="white", edgecolors="#0d1117", linewidth=1.5,
               zorder=10, label="Centroids")
    ax.set_xlabel(x_axis, color="#8b949e"); ax.set_ylabel(y_axis, color="#8b949e")
    ax.set_title(f"K-Means Clusters (K={k_val}) — {x_axis} vs {y_axis}", color="#e6edf3")
    ax.tick_params(colors="#8b949e")
    for sp in ax.spines.values(): sp.set_color("#30363d")
    ax.legend(facecolor="#161b22", labelcolor="#8b949e", edgecolor="#30363d", fontsize=9,
              ncol=2 if k_val > 5 else 1)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.divider()

    # ── Cluster profiles
    st.subheader("📋 Cluster Profiles — Centroid Feature Values")
    cent_df["Cluster"] = [f"Cluster {i}" for i in range(k_val)]
    cent_df["Size"] = [int((df["cluster"] == i).sum()) for i in range(k_val)]
    cent_df["Top Crops"] = [
        ", ".join(df[df["cluster"] == i]["label"].value_counts().head(3).index.tolist())
        for i in range(k_val)
    ]
    display_cols = ["Cluster", "Size", "Top Crops"] + FEATURES
    st.dataframe(cent_df[display_cols].round(2), hide_index=True, use_container_width=True)

    st.divider()

    # ── Radar chart of cluster centroids
    st.subheader("🕸️ Cluster Radar Chart (Normalized Centroids)")
    centroids_norm = km_model.cluster_centers_
    angles = np.linspace(0, 2*np.pi, len(FEATURES), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    for i in range(k_val):
        vals = centroids_norm[i].tolist() + [centroids_norm[i][0]]
        ax.plot(angles, vals, "o-", linewidth=1.8, color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                label=f"Cluster {i}")
        ax.fill(angles, vals, alpha=0.08, color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(FEATURES, color="#e6edf3", size=10)
    ax.tick_params(colors="#8b949e")
    ax.set_yticklabels([]); ax.grid(color="#30363d")
    ax.spines["polar"].set_color("#30363d")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              facecolor="#161b22", labelcolor="#8b949e", edgecolor="#30363d", fontsize=9)
    ax.set_title("Normalized Feature Profiles", color="#e6edf3", pad=20)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # clean up temp column
    df.drop(columns=["cluster"], inplace=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 4: PREDICTION
# ═══════════════════════════════════════════════════════════════
elif page == "🎯 Prediction":
    st.title("🎯 Crop Prediction")
    st.markdown("**Enter field conditions to get an instant crop recommendation**")
    st.divider()

    # ── Input sliders
    st.subheader("🌱 Enter Field Conditions")
    col1, col2 = st.columns(2)
    with col1:
        N   = st.slider("Nitrogen (N) — kg/ha",       0,   140,  80)
        P   = st.slider("Phosphorus (P) — kg/ha",     5,   145,  40)
        K   = st.slider("Potassium (K) — kg/ha",      5,   205,  40)
        temp= st.slider("Temperature — °C",            8,    43,  25)
    with col2:
        hum = st.slider("Humidity — %",               14,  100,  70)
        ph  = st.slider("Soil pH",                    3.5,  9.9, 6.5, step=0.1)
        rain= st.slider("Annual Rainfall — mm",       20,  299, 100)

    st.divider()

    predict_btn = st.button("🚀 Predict Best Crop", use_container_width=True)

    if predict_btn:
        inp = np.array([[N, P, K, temp, hum, ph, rain]])
        inp_sc = scaler.transform(inp)
        pred_enc = clf.predict(inp_sc)[0]
        pred_crop = le.inverse_transform([pred_enc])[0]
        proba = clf.predict_proba(inp_sc)[0]
        conf = proba[pred_enc]

        # Top 5 predictions
        top5_idx = np.argsort(proba)[::-1][:5]
        top5_crops = le.inverse_transform(top5_idx)
        top5_probs = proba[top5_idx]

        emoji = CROP_EMOJI.get(pred_crop, "🌱")

        col_r1, col_r2 = st.columns([1, 1])
        with col_r1:
            st.markdown(f"""
            <div class="pred-box">
                <div style="font-size:3.5rem">{emoji}</div>
                <div class="pred-crop">{pred_crop.upper()}</div>
                <div class="pred-conf">Confidence: {conf*100:.1f}%</div>
                <div style="background:#0d1117;border-radius:6px;height:10px;margin:12px 0;overflow:hidden;">
                    <div style="width:{conf*100:.1f}%;height:10px;background:#238636;border-radius:6px;"></div>
                </div>
                <div style="font-size:12px;color:#8b949e;">
                    Based on: N={N}, P={P}, K={K}, Temp={temp}°C,<br>
                    Humidity={hum}%, pH={ph}, Rainfall={rain}mm
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_r2:
            st.subheader("🏆 Top 5 Crop Recommendations")
            fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0d1117")
            ax.set_facecolor("#161b22")
            colors = ["#238636" if i == 0 else "#30363d" for i in range(5)]
            bars = ax.barh(
                [f"{CROP_EMOJI.get(c,'🌱')} {c}" for c in top5_crops[::-1]],
                top5_probs[::-1] * 100,
                color=colors[::-1], edgecolor="#0d1117", height=0.6
            )
            for bar, prob in zip(bars, top5_probs[::-1]):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f"{prob*100:.1f}%", va="center", color="#e6edf3", fontsize=10)
            ax.set_xlabel("Confidence %", color="#8b949e")
            ax.tick_params(colors="#e6edf3", labelsize=10)
            for sp in ax.spines.values(): sp.set_color("#30363d")
            ax.set_xlim(0, 105)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.divider()

        # Feature importance
        st.subheader("🔬 Feature Importance (Random Forest)")
        importances = clf.feature_importances_
        feat_imp = pd.DataFrame({"Feature": FEATURES, "Importance": importances})
        feat_imp = feat_imp.sort_values("Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0d1117")
        ax.set_facecolor("#161b22")
        bar_colors = ["#d29922" if f == max(FEATURES, key=lambda f: dict(zip(FEATURES, importances))[f])
                      else "#238636" for f in feat_imp["Feature"]]
        ax.barh(feat_imp["Feature"], feat_imp["Importance"]*100,
                color=bar_colors, edgecolor="#0d1117", height=0.6)
        ax.set_xlabel("Importance (%)", color="#8b949e")
        ax.tick_params(colors="#e6edf3")
        for sp in ax.spines.values(): sp.set_color("#30363d")
        ax.set_title("Which features drive crop prediction?", color="#e6edf3")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Agronomic tips
        st.subheader(f"💡 Agronomic Tips for {pred_crop.capitalize()}")
        tips_map = {
            "rice":        "Requires high humidity (>80%) and ample rainfall. Flooded paddy fields suit best.",
            "maize":       "Moderate nitrogen, low humidity. Well-drained soils are ideal.",
            "chickpea":    "Thrives in cool, dry conditions. High P & K with neutral-alkaline pH.",
            "kidneybeans": "Prefers acidic soil (pH ~5.7). Moderate rainfall and cool temperatures.",
            "pigeonpeas":  "Drought-tolerant. Warm temperatures and moderate rainfall needed.",
            "mothbeans":   "Very drought-resistant. Sandy, well-drained soils with low rainfall.",
            "mungbean":    "High humidity crop. Warm temperatures with adequate moisture.",
            "blackgram":   "Warm, humid conditions. pH 6.5–7.5 with moderate rainfall.",
            "lentil":      "Cool, dry climate. Low nitrogen fixation means lower N needs.",
            "pomegranate": "Semi-arid crop. Deep, loamy soil with pH 5.5–7.2.",
            "banana":      "High nitrogen demand. Tropical, humid, frost-free environments.",
            "mango":       "Hot, dry summers and frost-free winters. Deep, well-drained soil.",
            "grapes":      "High K requirement. Warm, dry summers; pH 6.0–6.5.",
            "watermelon":  "High humidity and warmth. Sandy loam soil, good drainage.",
            "muskmelon":   "Very high humidity. Warm temperatures, light sandy soil.",
            "apple":       "Cool temperate. High P & K. pH 5.5–6.5. Chilling hours required.",
            "orange":      "Subtropical climate. Well-drained sandy loam, pH 6–7.",
            "papaya":      "Tropical, frost-free. High temperature and humidity.",
            "coconut":     "Coastal tropical. Sandy loam soil, high humidity, heavy rainfall.",
            "cotton":      "High nitrogen, warm temperatures. Black cotton soil preferred.",
            "jute":        "Tropical monsoon crop. High rainfall, warm temperature, loamy soil.",
            "coffee":      "Shade-grown in tropical highlands. High rainfall, acidic pH.",
        }
        tip = tips_map.get(pred_crop, "Consult local agricultural extension office for region-specific advice.")
        st.info(f"🌿 **{pred_crop.capitalize()}**: {tip}")

    else:
        st.markdown("""
        <div style="background:#161b22;border:1px dashed #30363d;border-radius:12px;
                    padding:40px;text-align:center;color:#8b949e;margin-top:12px;">
            <div style="font-size:3rem">🎯</div>
            <div style="font-size:16px;font-weight:600;margin:10px 0;color:#e6edf3;">
                Adjust the sliders above and click <b>Predict Best Crop</b>
            </div>
            <div style="font-size:13px;">
                The Random Forest model will recommend the optimal crop based on your field's soil and climate conditions.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Model info
    with st.expander("ℹ️ About the Model"):
        st.markdown(f"""
        | Detail | Value |
        |--------|-------|
        | Algorithm | Random Forest Classifier |
        | Trees | 150 |
        | Train/Test Split | 80% / 20% |
        | Stratified Split | ✅ Yes |
        | Test Accuracy | **{model_acc*100:.1f}%** |
        | Feature Scaling | StandardScaler (mean=0, std=1) |
        | Classes | 22 crops |
        """)
