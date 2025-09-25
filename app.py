import streamlit as st
import numpy as np
import joblib
from PIL import Image
import cv2

# --- Page Configuration ---
st.set_page_config(
    page_title="Say Cheese!",
    page_icon="😊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for Black Theme Styling + Darker Drag & Drop ---
def set_dark_theme():
    st.markdown(
        """
        <style>
        body, .stApp {
            background-color: #0d1117;
            color: #f0f6fc;
        }

        /* File Uploader Override */
        .stFileUploader label {
            color: #f0f6fc !important;
            font-weight: 600;
            font-size: 16px;
        }

        /* Force dark background for the whole dropzone area */
        .stFileUploader > section[data-testid="stFileUploaderDropzone"] {
            background-color: #121823 !important;
            border: 2px dashed #58a6ff !important;
            border-radius: 10px !important;
            padding: 25px !important;
            transition: background-color 0.3s ease !important;
        }

        .stFileUploader > section[data-testid="stFileUploaderDropzone"]:hover {
            background-color: #1f2a4d !important;
        }

        /* Drag and drop + limit text */
        .stFileUploader span,
        .stFileUploader p {
            color: #e6edf3 !important;
            font-weight: 500 !important;
        }

        /* Browse files button */
        .stFileUploader button {
            background-color: #1f2937 !important;
            color: #f0f6fc !important;
            border: 1px solid #30363d !important;
            border-radius: 8px !important;
            padding: 6px 14px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }

        .stFileUploader button:hover {
            background-color: #2d333b !important;
            border-color: #58a6ff !important;
            color: #58a6ff !important;
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

# --- Face Detection Function ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(pil_image):
    # Convert PIL image to OpenCV BGR format
    open_cv_image = np.array(pil_image.convert('RGB'))[:, :, ::-1].copy()
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None  # No face detected
    
    # Take the first detected face
    (x, y, w, h) = faces[0]
    
    # Crop and return the face as PIL image
    face_img = pil_image.crop((x, y, x + w, y + h))
    return face_img

# --- Image Preprocessing ---
def preprocess_image(image):
    image = image.convert('L')  # Grayscale
    image = image.resize((64, 64))
    image = np.array(image).flatten().reshape(1, -1)
    image = scaler.transform(image)
    return image

# --- Apply Styling ---
set_dark_theme()

# --- Title ---
st.markdown('<div class="title">📸 Say Cheese!</div>', unsafe_allow_html=True)
st.markdown("### Detect Smiles With a Click 😄")
st.write("Upload a face photo and let the app detect if the person is smiling.")

# --- Image Upload with max file size limit ---
uploaded_file = st.file_uploader(
    "🖼️ Upload an Image (Max 200MB)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=False,
    help="Drag & drop or browse image files"
)

if uploaded_file:
    # Check file size limit (200MB)
    uploaded_file.seek(0, 2)  # Seek to end of file
    size = uploaded_file.tell()
    uploaded_file.seek(0)  # Reset pointer to start
    
    max_size = 200 * 1024 * 1024  # 200 MB in bytes
    if size > max_size:
        st.error("❌ File size exceeds 200MB limit. Please upload a smaller file.")
    else:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        face_img = detect_face(img)
        
        if face_img is None:
            st.error("⚠️ No face detected! Please upload a clear face photo.")
        else:
            st.image(face_img, caption="Detected Face", use_column_width=True)
            
            with st.spinner("🔍 Analyzing the face..."):
                processed_img = preprocess_image(face_img)
                prediction_proba = model.predict_proba(processed_img)[0][1]
                prediction = model.predict(processed_img)

            smile_score = int(prediction_proba * 100)
            st.slider("Smile Score 😊", 0, 100, smile_score, disabled=True)

            if prediction[0] == 1:
                st.success(f"😄 The person is **smiling!** (Confidence: {smile_score}%)")
            else:
                st.warning(f"😐 The person is **not smiling.** (Confidence: {smile_score}%)")

# --- Footer ---
st.markdown('<div class="footer">Let your smile change the world ❤️</div>', unsafe_allow_html=True)

