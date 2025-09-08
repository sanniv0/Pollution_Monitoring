# app.py
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
import streamlit.components.v1 as components

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="Pollution Dashboard", layout="wide")
st.title("🌍 Pollution Prediction & Insights Dashboard")

# -------------------------------
# Load dataset safely
# -------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Dataset_Cities.csv")
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        st.error("❌ Dataset_Cities.csv not found. Please place it in the app folder.")
        st.stop()

data = load_data()

# -------------------------------
# Detect city column
# -------------------------------
city_col = None
for col in data.columns:
    if col.lower() == "city":
        city_col = col
        break

if not city_col:
    st.error("❌ No 'city' column found in dataset!")
    st.stop()

# -------------------------------
# Sidebar for city selection
# -------------------------------
st.sidebar.header("City Selection")
city = st.sidebar.selectbox("Select a City", data[city_col].unique())

# Filter city data
city_data = data[data[city_col] == city]
if city_data.empty:
    st.warning(f"No data available for {city}")
    st.stop()

# -------------------------------
# Load pre-trained model safely
# -------------------------------
@st.cache_resource
def load_model():
    try:
        loaded = joblib.load("model.pkl")
    except FileNotFoundError:
        st.error("❌ model.pkl not found. Please place it in the app folder.")
        st.stop()

    if isinstance(loaded, dict):
        for key in ["model", "best_model", "rf_model"]:
            if key in loaded:
                return loaded[key]
        return list(loaded.values())[0]
    return loaded

model = load_model()

# -------------------------------
# Prediction
# -------------------------------
try:
    features = city_data.drop([city_col], axis=1)
    pred = model.predict(features)
    avg_pred = round(pred.mean(), 2)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# -------------------------------
# Dashboard Tabs
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📊 City Data & Prediction", "🔎 SHAP Insights", "🌍 Pollution Heatmap"])

# Tab 1: City data and prediction
with tab1:
    st.metric(label="Predicted Pollution Level (avg)", value=avg_pred)
    st.subheader(f"City Data: {city}")
    st.dataframe(city_data, use_container_width=True)

# Tab 2: SHAP Insights
with tab2:
    st.subheader("SHAP Feature Contribution")

    try:
        if "prep" not in model.named_steps or "rf" not in model.named_steps:
            st.warning("⚠️ SHAP explanations only available for models with a 'prep' and 'rf' step.")
        else:
            # Transform features
            X_transformed = model.named_steps['prep'].transform(features)

            # Extract feature names
            transformers = model.named_steps['prep'].transformers_
            num_cols, cat_cols = [], []

            for name, trans, cols in transformers:
                if name == "num":
                    num_cols = cols
                elif name == "cat":
                    cat_cols = cols

            if cat_cols:
                ohe = model.named_steps['prep'].named_transformers_['cat']
                cat_feat_names = ohe.get_feature_names_out(cat_cols).tolist()
            else:
                cat_feat_names = []

            feature_names = list(num_cols) + cat_feat_names

            # Run SHAP
            explainer = shap.TreeExplainer(model.named_steps['rf'])
            shap_values = explainer.shap_values(X_transformed)

            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
            st.pyplot(fig)

    except Exception as shap_err:
        st.error(f"❌ SHAP explanation not available: {shap_err}")

# Tab 3: Pollution Heatmap
with tab3:
    st.subheader("Interactive Pollution Heatmap")
    try:
        with open("pollution_heatmap.html", "r", encoding="utf-8") as f:
            heatmap_html = f.read()
        components.html(heatmap_html, height=800, scrolling=False)
    except FileNotFoundError:
        st.error("❌ pollution_heatmap.html not found.")
        st.info("👉 Generate the heatmap from your notebook (e.g., using Folium) and save it as 'pollution_heatmap.html' in the app folder.")
