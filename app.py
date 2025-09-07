# app.py
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
import streamlit.components.v1 as components

st.set_page_config(page_title="Pollution Dashboard", layout="wide")

st.title("🌍 Pollution Prediction & Insights Dashboard")

# Load dataset
data = pd.read_csv("Dataset_Cities.csv")
data.columns = data.columns.str.strip()

# Detect city column
city_col = None
for col in data.columns:
    if col.lower() == "city":
        city_col = col
        break

if not city_col:
    st.error("No 'city' column found in dataset!")
    st.stop()

# Sidebar for city selection
st.sidebar.header("City Selection")
city = st.sidebar.selectbox("Select a City", data[city_col].unique())

# Filter city data
city_data = data[data[city_col] == city]

# Load pre-trained model
@st.cache_resource
def load_model():
    loaded = joblib.load("model.pkl")
    if isinstance(loaded, dict):
        for key in ["model", "best_model", "rf_model"]:
            if key in loaded:
                return loaded[key]
        return list(loaded.values())[0]
    return loaded

# Try prediction
try:
    model = load_model()
    features = city_data.drop([city_col], axis=1)
    pred = model.predict(features)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# Create tabs for dashboard
tab1, tab2, tab3 = st.tabs(["📊 City Data & Prediction", "🔎 SHAP Insights", "🌍 Pollution Heatmap"])

with tab1:
    st.metric(label="Predicted Pollution Level", value=pred[0])
    st.subheader(f"City Data: {city}")
    st.dataframe(city_data, use_container_width=True)


with tab2:
    st.subheader("SHAP Feature Contribution")

    try:
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
        st.error(f"SHAP explanation not available: {shap_err}")

with tab3:
    st.subheader("Interactive Pollution Heatmap")
    try:
        with open("pollution_heatmap.html", "r", encoding="utf-8") as f:
            heatmap_html = f.read()
        components.html(heatmap_html, height=800, scrolling=False)  # no scroll needed
    except FileNotFoundError:
        st.error("pollution_heatmap.html not found. Please generate it from the notebook and place it in the app folder.")
