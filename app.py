import os

import joblib
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Insurance Charges Predictor", page_icon="🏥", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load("insurance_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv") if os.path.exists("insurance.csv") else None


model = load_model()
data = load_data()

st.title("🏥 Medical Insurance Charges Predictor")
st.caption("Linear Regression model (one-hot encoding + interaction features)")

tab_predict, tab_explore = st.tabs(["🔮 Predict", "📊 Explore Data"])

# ---------------- Prediction tab ----------------
with tab_predict:
    st.subheader("Fill Your details")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 100, 30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        children = st.number_input("Children", min_value=0, max_value=10, value=0, step=1)
    with col2:
        sex = st.selectbox("Sex", ["female", "male"])
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    if st.button("Predict Charges", type="primary"):
        # Same encoding as the notebook
        sex_val = 1 if sex == "female" else 0
        smoker_val = 1 if smoker == "yes" else 0

        row = {
            "age": age,
            "sex": sex_val,
            "bmi": bmi,
            "children": children,
            "smoker": smoker_val,
            "region_northeast": int(region == "northeast"),
            "region_northwest": int(region == "northwest"),
            "region_southeast": int(region == "southeast"),
            "region_southwest": int(region == "southwest"),
            "age_smoker": age * smoker_val,
            "bmi_smoker": bmi * smoker_val,
        }

        # Column order must match training
        input_df = pd.DataFrame([row])[list(model.feature_names_in_)]
        prediction = model.predict(input_df)[0]

        st.success(f"Estimated Insurance Charges: **${max(prediction, 0):,.2f}**")

    with st.expander("About to Model"):
        st.write(
            "- Features: age, sex, bmi, children, smoker, region (one-hot), "
            "age×smoker, bmi×smoker\n"
            "- Test R² score: ~0.865 (With interaction features )\n"
            "- Train R² ~0.834, Test R² ~0.865 → Not show any issues underfitting/overfitting\n"
            "- Bmi formula  =  weight(kg)/height(m)^2 , 1 foot = 0.3048  , 1 inch = 0.0254"
        )

# ---------------- Explore tab ----------------
with tab_explore:
    if data is None:
        st.info("`insurance.csv` app.py ke same folder mein rakho to yahan data explore kar sakte ho.")
    else:
        st.subheader("Dataset")
        st.dataframe(data.head(20), use_container_width=True)
        st.write(f"Rows: {data.shape[0]} | Columns: {data.shape[1]}")

        st.subheader("BMI vs Charges (smoker ke hisaab se)")
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x="bmi", y="charges", hue="smoker", ax=ax)
        st.pyplot(fig)