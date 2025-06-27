import streamlit as st
from deepface import DeepFace
import cv2
import pandas as pd
import tempfile
import os
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# CSV file for logging
LOG_FILE = "emotion_log.csv"

# Page configuration
st.set_page_config(page_title="🧠 StressSense - Emotion Detection", layout="wide")
st.title("🧠 StressSense - Emotion Detection App")

# Initialize log file if it doesn't exist
if not os.path.exists(LOG_FILE):
    df_init = pd.DataFrame(columns=["timestamp", "source", "emotion"])
    df_init.to_csv(LOG_FILE, index=False)

# Function to detect emotion and log it
def analyze_emotion(image, source="Image Upload"):
    try:
        result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_df = pd.DataFrame([[timestamp, source, emotion]], columns=["timestamp", "source", "emotion"])
        log_df.to_csv(LOG_FILE, mode='a', header=False, index=False)
        return emotion
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Tabs for app sections
tab1, tab2, tab3 = st.tabs(["📁 Upload Image", "🎥 Live Webcam Detection", "📊 Emotion Report"])

# --- TAB 1: Upload Image ---
with tab1:
    st.subheader("Upload an image to detect emotion")
    uploaded_img = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"], key="upload")

    if uploaded_img is not None:
        file_bytes = uploaded_img.read()
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        img = cv2.imread(tmp_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

        if st.button("🧠 Detect Emotion", key="detect_image_btn"):
            emotion = analyze_emotion(img)
            if emotion:
                st.success(f"Detected Emotion: **{emotion.upper()}**")

# --- TAB 2: Live Webcam Detection ---
with tab2:
    st.subheader("Live webcam feed emotion detection")

    class EmotionDetector(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            try:
                result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
                dominant_emotion = result[0]['dominant_emotion']
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_df = pd.DataFrame([[timestamp, "Webcam", dominant_emotion]], columns=["timestamp", "source", "emotion"])
                log_df.to_csv(LOG_FILE, mode='a', header=False, index=False)
                cv2.putText(img, f'Emotion: {dominant_emotion}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)
            except:
                pass
            return img

    webrtc_streamer(key="webcam_feed", video_transformer_factory=EmotionDetector)

# --- TAB 3: Emotion Report ---
with tab3:
    st.subheader("📈 Emotion Detection Report")
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        st.dataframe(df.tail(20), use_container_width=True)
        st.download_button("📥 Download Full CSV", data=df.to_csv(index=False),
                           file_name="emotion_log.csv", mime="text/csv")
    else:
        st.info("No emotion data logged yet.")
