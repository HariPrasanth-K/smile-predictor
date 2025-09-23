import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="Say Cheese!",
    page_icon="😊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for styling ---
def set_background():
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f4f8;
        }
        .title {
            color: #ff4b4b;
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
        }
        .footer {
            color: #aaa;
            font-size: 12px;
            text-align: center;
            margin-top: 30px;
        }
        .stButton > button {
            color: white;
            background: linear-gradient(90deg, #ff4b4b, #ff914d);
            border: none;
        }
        .stSlider > div {
            background: #ffe5d9;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Load Model and Scaler ---
@st.cache_resource
def load_model():
    model = joblib.load("smile_stalker.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# --- Preprocessing Function ---
def preprocess_image(image):
    image = image.convert('L')  # Grayscale
    image = image.resize((64, 64))  # Resize
    image = np.array(image).flatten().reshape(1, -1)  # Flatten
    image = scaler.transform(image)  # Scale
    return image

# --- Set Background and Title ---
set_background()
st.markdown('<div class="title">😄 Say Cheese !</div>', unsafe_allow_html=True)
st.subheader("Detect Smiles on your Face")
st.markdown("Upload an image and let the app determine if someone is smiling or not!")

# --- Upload Section ---
uploaded_file = st.file_uploader("📸 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing the image..."):
        processed_img = preprocess_image(img)
        prediction_proba = model.predict_proba(processed_img)[0][1]
        prediction = model.predict(processed_img)

    smile_score = int(prediction_proba * 100)
    st.slider("Smile Score", 0, 100, smile_score, disabled=True)

    if prediction[0] == 1:
        st.success(f"😊 The person is smiling! (Confidence: {smile_score}%)")
    else:
        st.warning(f"😐 The person is not smiling. (Confidence: {smile_score}%)")

# --- Footer ---
st.markdown('<div class="footer">Let your smile change the world ❤️ </div>', unsafe_allow_html=True)
