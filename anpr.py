import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import sqlite3
import json
from datetime import datetime
from ultralytics import YOLOv10
from paddleocr import PaddleOCR
import re
import pandas as pd
import tempfile
import base64

# Initialize models
model = YOLOv10('yolov10/weights/yolov10n.pt')
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# --------------------------
# CUSTOM STYLING
# --------------------------

def set_background(image_file):
    """
    Set background image from local file
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# --------------------------
# HELPER FUNCTIONS
# --------------------------

def normalize_license_plate(plate):
    """Normalize the license plate text"""
    return plate.strip().upper()

def paddle_ocr(frame, x1, y1, x2, y2):
    """Perform OCR on license plate region"""
    cropped_frame = frame[y1:y2, x1:x2]

    if len(cropped_frame.shape) == 2:  # Grayscale
        cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_GRAY2RGB)
    elif cropped_frame.shape[2] == 4:  # RGBA
        cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_RGBA2RGB)

    result = ocr.ocr(cropped_frame, cls=True)

    text = ""
    if result and result[0]:
        max_conf = 0
        for line in result[0]:
            detected_text = line[1][0]
            confidence = line[1][1]
            if confidence > max_conf and confidence > 0.6:
                text = detected_text
                max_conf = confidence

    text = re.sub(r'[\W]', '', text)
    text = text.replace("???", "").replace("O", "0").replace("粤", "")
    return normalize_license_plate(text)

def save_json(license_plates, startTime, endTime):
    """Save results to JSON file"""
    interval_data = {
        "Start Time": startTime.isoformat(),
        "End Time": endTime.isoformat(),
        "License Plate": list(license_plates)
    }
    interval_file_path = f"json/output_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    with open(interval_file_path, 'w') as f:
        json.dump(interval_data, f, indent=2)

    cumulative_file_path = "json/LicensePlateData.json"
    existing_data = []
    if os.path.exists(cumulative_file_path):
        with open(cumulative_file_path, 'r') as f:
            existing_data = json.load(f)

    existing_data.append(interval_data)
    with open(cumulative_file_path, 'w') as f:
        json.dump(existing_data, f, indent=2)

    save_to_database(license_plates, startTime, endTime)

def save_to_database(license_plates, start_time, end_time):
    """Save results to SQLite database"""
    conn = sqlite3.connect('licensePlatesDatabase.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS LicensePlates(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT,
            end_time TEXT,
            license_plate TEXT UNIQUE
        )
    ''')

    for plate in license_plates:
        try:
            cursor.execute('''
                INSERT INTO LicensePlates(start_time, end_time, license_plate)
                VALUES (?, ?, ?)
            ''', (start_time.isoformat(), end_time.isoformat(), plate))
        except sqlite3.IntegrityError:
            st.warning(f"Duplicate license plate skipped: {plate}")

    conn.commit()
    conn.close()

def process_image(image):
    """Process single image for license plates"""
    frame = np.array(image)
    if len(frame.shape) == 2:  # Grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:  # RGBA
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    results = model.predict(frame, conf=0.45)
    detected_plates = set()

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 3)  # Orange border
            label = paddle_ocr(frame, x1, y1, x2, y2)

            if label:
                detected_plates.add(label)
                textSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                c2 = x1 + textSize[0], y1 - textSize[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, (0, 200, 255), -1)  # Orange background
                cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                           0.8, (255, 255, 255), 2, cv2.LINE_AA)
                st.success(f"Detected License Plate: {label}")

    if detected_plates:
        save_json(detected_plates, datetime.now(), datetime.now())

    return frame

def process_video(uploaded_file):
    """Process video file for license plates"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    cap = cv2.VideoCapture(video_path)
    startTime = datetime.now()
    license_plates = set()

    stframe = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            results = model.predict(frame, conf=0.45)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 3)
                    label = paddle_ocr(frame, x1, y1, x2, y2)

                    if label:
                        license_plates.add(label)
                        textSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                        c2 = x1 + textSize[0], y1 - textSize[1] - 3
                        cv2.rectangle(frame, (x1, y1), c2, (0, 200, 255), -1)
                        cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.8, (255, 255, 255), 2, cv2.LINE_AA)
                        status_text.success(f"Detected: {label}")

            stframe.image(frame, channels="BGR", use_container_width=True)
            progress_bar.progress(frame_count / total_frames)

    finally:
        cap.release()
        os.unlink(video_path)

        if license_plates:
            save_json(license_plates, startTime, datetime.now())
            st.success(f"Processing complete! Found {len(license_plates)} unique plates")

def process_webcam():
    """Process live webcam feed"""
    cap = cv2.VideoCapture(0)
    startTime = datetime.now()
    license_plates = set()
    stop_button_pressed = False

    stframe = st.empty()
    stop_button = st.button("Stop Webcam", key="stop_webcam")

    while cap.isOpened() and not stop_button_pressed:
        if stop_button:
            stop_button_pressed = True
            break

        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.45)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 3)
                label = paddle_ocr(frame, x1, y1, x2, y2)

                if label:
                    license_plates.add(label)
                    textSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    c2 = x1 + textSize[0], y1 - textSize[1] - 3
                    cv2.rectangle(frame, (x1, y1), c2, (0, 200, 255), -1)
                    cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                               0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    st.success(f"Detected: {label}")

        stframe.image(frame, channels="BGR", use_container_width=True)

    cap.release()
    if license_plates:
        save_json(license_plates, startTime, datetime.now())

def show_database():
    """Display database records"""
    st.markdown("## 📋 Database Records")
    conn = sqlite3.connect('licensePlatesDatabase.db')
    query = "SELECT * FROM LicensePlates ORDER BY start_time DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()

    if not df.empty:
        st.dataframe(df, use_container_width=True)

        if st.button("📥 Export to CSV", key="export_csv"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name='license_plates.csv',
                mime='text/csv'
            )
    else:
        st.warning("No records found in the database")

def show_home_page():
    """Display the front page"""
    st.markdown("""
    <div style="
        background: rgba(0, 0, 0, 0.7);
        padding: 3rem;
        border-radius: 15px;
        margin: 2rem 0;
        color: white;
    ">
        <h1 style="color: #FFA500; text-align: center; font-size: 3rem;">🚗 Automatic Car Number Plate Recognition</h1>
        <p style="text-align: center; font-size: 1.2rem;">
            Advanced AI-powered system for detecting and recognizing license plates in real-time
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="
            background: rgba(255, 165, 0, 0.1);
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #FFA500;
            margin: 1rem 0;
        ">
            <h3>📹 Video Processing</h3>
            <p>Upload videos to detect license plates across multiple frames</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="
            background: rgba(0, 150, 255, 0.1);
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #0096FF;
            margin: 1rem 0;
        ">
            <h3>🖼️ Image Analysis</h3>
            <p>Process single images with high accuracy detection</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="
            background: rgba(50, 205, 50, 0.1);
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #32CD32;
            margin: 1rem 0;
        ">
            <h3>🎥 Live Camera</h3>
            <p>Real-time processing from your webcam feed</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; margin: 3rem 0;">
        <a href="#upload-video" class="nav-button">Get Started →</a>
    </div>
    """, unsafe_allow_html=True)

# --------------------------
# MAIN APP
# --------------------------

def main():
    # Set background image (replace with your own image path)
    set_background("yolov10/static/back1.png")  # Create this image in your directory

    # Custom CSS
    local_css("yolov10/static/style.css")

    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "Home"

    # Sidebar navigation
    with st.sidebar:
        st.markdown("""
        <div style="
            background: rgba(0, 0, 0, 0.7);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        ">
            <h2 style="color: white; text-align: center;">🔍 Navigation</h2>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "",
            ["🏠 Home", "📹 Upload Video", "🖼️ Upload Image", "🎥 Webcam", "📋 Database", "🧹 Clear DB"],
            index=0,
            label_visibility="collapsed"
        )

        st.markdown("""
        <div style="
            background: rgba(0, 0, 0, 0.7);
            padding: 1rem;
            border-radius: 10px;
            margin-top: 2rem;
            color: white;
        ">
            <h3 style="color: #FFA500;">ℹ️ About</h3>
            <p>This app uses:</p>
            <ul>
                <li>YOLOv10 for detection</li>
                <li>PaddleOCR for text recognition</li>
                <li>Streamlit for UI</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "📹 Upload Video":
        st.markdown("## 📹 Video Processing")
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4"], key="video_upload")
        if uploaded_file is not None:
            process_video(uploaded_file)
    elif page == "🖼️ Upload Image":
        st.markdown("## 🖼️ Image Processing")
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png"], key="image_upload")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            with col2:
                processed_image = process_image(image)
                st.image(processed_image, caption="Processed Image", use_container_width=True)
    elif page == "🎥 Webcam":
        st.markdown("## 🎥 Live Webcam")
        if st.button("Start Webcam", key="start_webcam"):
            process_webcam()
    elif page == "📋 Database":
        show_database()
    elif page == "🧹 Clear DB":
        st.markdown("## 🧹 Clear Database")
        if st.button("Clear All Records", key="clear_db"):
            conn = sqlite3.connect('licensePlatesDatabase.db')
            cursor = conn.cursor()
            cursor.execute('DELETE FROM LicensePlates')
            conn.commit()
            conn.close()
            st.success("Database cleared successfully!")

if __name__ == "__main__":
    main()