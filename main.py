import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import os
from PIL import Image

st.set_page_config(page_title="Face Recognition App", layout="wide")

known_faces_path = "known_faces"
known_embeddings = []
known_names = []

def load_known_faces():
    known_embeddings.clear()
    known_names.clear()
    for root, dirs, files in os.walk(known_faces_path):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, filename)
                try:
                    embedding = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                    known_embeddings.append(embedding)
                    name = os.path.splitext(filename)[0]
                    known_names.append(name)
                except Exception as e:
                    st.error(f"Error loading {filename}: {e}")

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

load_known_faces()

st.title("🔍 Real-Time Face Recognition")
run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

if run:
    st.info("Press 'Stop Camera' to stop the feed.")
    while run:
        success, frame = camera.read()
        if not success:
            st.error("Failed to capture video")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        cv2.imwrite("current.jpg", small_frame)

        try:
            embedding = DeepFace.represent(img_path="current.jpg", model_name="Facenet", enforce_detection=False)[0]["embedding"]
            similarities = [cosine_similarity(embedding, known_emb) for known_emb in known_embeddings]

            name = "Unknown"
            if similarities:
                best_idx = np.argmax(similarities)
                best_score = similarities[best_idx]
                if best_score > 0.7:
                    name = known_names[best_idx]

            cv2.putText(frame, f"{name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)

        except Exception as e:
            st.warning(f"Face not detected: {e}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

else:
    st.warning("Camera stopped.")
    camera.release()

# ---------------- Register New Face ---------------- #
st.header("📸 Register a New Face")

name_input = st.text_input("Enter name for registration")
uploaded_img = st.file_uploader("Upload an image (.jpg/.png)", type=["jpg", "jpeg", "png"])

if name_input and uploaded_img:
    save_path = os.path.join(known_faces_path, f"{name_input}.jpg")
    with open(save_path, "wb") as f:
        f.write(uploaded_img.read())

    try:
        embedding = DeepFace.represent(img_path=save_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
        known_embeddings.append(embedding)
        known_names.append(name_input)
        st.success(f"{name_input} registered successfully!")
    except Exception as e:
        st.error(f"Error registering {name_input}: {e}")
