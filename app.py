import streamlit as st
import numpy as np
from joblib import load

# =====================================
# Load trained Random Forest model
# =====================================
model = load("stellar_random_forest.joblib")

st.set_page_config(page_title="Stellar Classification App")

# =====================================
# App UI
# =====================================
st.title("🌌 Stellar Classification App")
st.write(
    "Predict whether a celestial object is a **Star, Galaxy, or Quasar** "
    "using photometric and spectroscopic features."
)

st.header("🔭 Enter Object Details")

# =====================================
# Input fields (MATCH TRAINING FEATURES)
# Order MUST match:
# ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'cam_col', 'redshift', 'plate', 'MJD']
# =====================================
alpha = st.number_input("Right Ascension (alpha)", value=180.0)
delta = st.number_input("Declination (delta)", value=0.0)

u = st.number_input("u magnitude", value=18.0)
g = st.number_input("g magnitude", value=17.5)
r = st.number_input("r magnitude", value=17.0)
i = st.number_input("i magnitude", value=16.8)
z = st.number_input("z magnitude", value=16.6)

cam_col = st.number_input("Camera Column (cam_col)", value=1, step=1)

redshift = st.number_input("Redshift", value=0.1, format="%.5f")

plate = st.number_input("Plate", value=1000, step=1)
mjd = st.number_input("MJD", value=55000, step=1)

# =====================================
# Prediction
# =====================================
if st.button("🔮 Predict"):

    # Input array MUST match training order
    input_data = np.array([[
        alpha, delta, u, g, r, i, z, cam_col, redshift, plate, mjd
    ]])

    prediction = model.predict(input_data)[0]

    st.subheader("📌 Prediction Result")

    # Label mapping from your training code
    # {'GALAXY': 0, 'QSO': 1, 'STAR': 2}
    if prediction == 0:
        st.success("🌌 Galaxy")
    elif prediction == 1:
        st.warning("✨ Quasar (QSO)")
    elif prediction == 2:
        st.info("⭐ Star")
    else:
        st.error("Unknown class")