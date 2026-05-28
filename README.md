# 🌾 Crop Yield Data Mining
### Soil & Climate Pattern Mining for Crop Recommendation — Punjab, India

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

> A complete data mining pipeline that analyses soil nutrients and climate conditions to recommend the best crop for a given field — built as an interactive Streamlit web application.

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [Dataset](#-dataset)
- [Data Mining Techniques](#-data-mining-techniques)
- [KDD Pipeline](#-kdd-pipeline)
- [Application Pages](#-application-pages)
- [Results](#-results)
- [Tech Stack](#-tech-stack)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [Screenshots](#-screenshots)
- [References](#-references)

---

## 🌱 About the Project

Agriculture is the backbone of Punjab's economy. Farmers traditionally rely on guesswork and generational knowledge to decide which crop to grow — a process that often leads to yield loss, excessive fertilizer use, and wasted resources.

This project builds a **data-driven crop recommendation system** using the KDD (Knowledge Discovery in Databases) pipeline. Given 7 soil and climate measurements from a field, the system:

- Detects **anomalous readings** that may indicate sensor errors or unusual conditions
- Discovers **natural groupings** of soil-climate profiles without using crop labels
- **Predicts the best crop** to grow with up to 99% accuracy

The entire pipeline is delivered as an interactive **Streamlit web application** with a clean dark-themed UI.

---

## 📊 Dataset

| Detail | Value |
|--------|-------|
| Source | [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) |
| Records | 2,200 (100 per crop) |
| Features | 7 numeric inputs |
| Target Classes | 22 crops |
| Missing Values | None |

### Features

| Feature | Unit | Range | Description |
|---------|------|--------|-------------|
| N | kg/ha | 0 – 140 | Nitrogen content in soil |
| P | kg/ha | 5 – 145 | Phosphorus content in soil |
| K | kg/ha | 5 – 205 | Potassium content in soil |
| temperature | °C | 8 – 43 | Average ambient temperature |
| humidity | % | 14 – 100 | Relative humidity |
| ph | — | 3.5 – 9.9 | Soil pH level |
| rainfall | mm | 20 – 299 | Annual rainfall |

### Crop Classes (22 total)
`rice` `maize` `chickpea` `kidneybeans` `pigeonpeas` `mothbeans` `mungbean` `blackgram` `lentil` `pomegranate` `banana` `mango` `grapes` `watermelon` `muskmelon` `apple` `orange` `papaya` `coconut` `cotton` `jute` `coffee`

---

## 🔬 Data Mining Techniques

### 1. 🔴 Outlier Detection — IQR Method
**Type:** Anomaly Detection (Unsupervised)

Uses the **Interquartile Range (IQR)** method to flag statistically unusual readings:

```
Lower Fence = Q1 - k × IQR
Upper Fence = Q3 + k × IQR
```

- Default multiplier **k = 1.5** (Tukey's convention)
- User can adjust k from 1.0 to 3.0 via an interactive slider
- Results shown as boxplots, scatter plots, and filterable data tables
- Key finding: Potassium (K) has the highest outlier rate (~5.1%) due to high-K fruit crops like grapes and apple

### 2. 🔵 K-Means Clustering
**Type:** Unsupervised Learning

Groups similar soil-climate conditions together **without using crop labels**. The algorithm discovers that similar conditions naturally suit similar crops.

```
WCSS = Σ Σ ||xᵢ - μₖ||²
```

- Optimal K selected using **Elbow Method** + **Silhouette Score**
- Best K = 4–5 clusters, Silhouette ≈ 0.31–0.38
- Discovered 4 natural archetypes:

| Cluster | Conditions | Crop Type |
|---------|-----------|-----------|
| 0 | High K, High P, Low humidity | Fruit crops (grapes, apple) |
| 1 | High humidity, High rainfall | Tropical crops (rice, coconut, banana) |
| 2 | Low N, Dry climate | Pulses & legumes (chickpea, lentil) |
| 3 | High N, Moderate climate | Commercial crops (maize, cotton, coffee) |

### 3. 🎯 Random Forest Classification
**Type:** Supervised Learning

Ensemble of **150 decision trees**, each trained on a random bootstrap sample. Prediction is determined by majority vote across all trees.

- **Train/Test split:** 80% / 20% (stratified)
- **Feature scaling:** StandardScaler (mean=0, std=1)
- **Test Accuracy:** ~97–99%
- **Feature Importance Ranking:** Rainfall > Humidity > K > Temperature > N > P > pH

---

## 🔄 KDD Pipeline

```
Data Selection  →  Preprocessing  →  Transformation  →  Data Mining  →  Interpretation
     ↓                   ↓                  ↓                 ↓                ↓
Load dataset      Null check +        MinMax / Std       Outlier +        Charts +
Inspect shape     IQR outliers        Normalisation     Clustering +       Tips +
                  detection                            Classification      Web App
```

---

## 📱 Application Pages

The app has **3 interactive pages** accessible from the sidebar:

### 🔴 Outlier Detection
- Select any feature from a dropdown
- Adjust IQR multiplier with a slider
- Filter by individual crop
- View: metrics, boxplot, scatter plot with red-highlighted outliers, data table

### 🔵 K-Means Clustering
- Choose number of clusters K (2–10)
- Select X and Y axes for scatter plot
- View: Elbow curve, Silhouette scores, cluster scatter, centroid profile table, radar chart

### 🎯 Prediction
- Set 7 sliders for field conditions
- Click **Predict Best Crop**
- View: recommended crop with confidence %, top 5 alternatives bar chart, feature importance chart, agronomic tips

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Model Test Accuracy | ~97–99% |
| Most Important Feature | Rainfall |
| Highest Outlier Feature | Potassium (K) ~5.1% |
| Optimal Clusters (K) | 4–5 |
| Silhouette Score at K=4 | ~0.31–0.38 |
| Dataset Balance | Perfectly balanced (100 per class) |

---

## 🛠 Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.9+ | Core language |
| Streamlit | 1.x | Web UI framework |
| scikit-learn | 1.x | ML algorithms |
| pandas | 2.x | Data manipulation |
| NumPy | 1.x | Numerical computation |
| matplotlib | 3.x | Visualisations |
| seaborn | 0.13 | Statistical plots |

---

## ▶ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/crop-yield-data-mining.git
cd crop-yield-data-mining
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run crop_yield_app.py
```

The app will open automatically in your browser at `http://localhost:8501`

> **Note:** No dataset file needed — the app generates the data automatically on startup.

---

## 📁 Project Structure

```
crop-yield-data-mining/
│
├── crop_yield_app.py        # Main Streamlit application
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── crop_yield_report.docx   # Full project report
```

---

## 🖼 Screenshots

| Page | Description |
|------|-------------|
| Outlier Detection | Boxplot + scatter with red-highlighted outliers per feature |
| K-Means Clustering | Elbow curve, silhouette scores, cluster scatter, radar chart |
| Prediction | Crop recommendation with confidence % and feature importance |

---

## 📚 References

1. Aingle, A. (2021). *Crop Recommendation Dataset*. Kaggle. https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
2. Fayyad, U., Piatetsky-Shapiro, G., & Smyth, P. (1996). From data mining to knowledge discovery in databases. *AI Magazine, 17*(3), 37–54.
3. Breiman, L. (2001). Random Forests. *Machine Learning, 45*(1), 5–32.
4. MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the 5th Berkeley Symposium*, 1, 281–297.
5. Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
6. Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics, 20*, 53–65.
7. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR, 12*, 2825–2830.

---

## 👨‍💻 Author

**Data Mining Course Project**
Punjab, India — 2024–2025

---

*Built with Python, Streamlit, and scikit-learn*
