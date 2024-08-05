
import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip
from tensorflow.keras.models import load_model

# Function to resize a video to a target resolution
def resize_video(input_path, output_path, target_resolution=(640, 480)):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), target_resolution)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, target_resolution)
        out.write(frame_resized)

    cap.release()
    out.release()

# Function to extract frames from a video at a specified frame rate
def extract_frames(video_path, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_rate == 0:
            frames.append(frame)
        count += 1

    cap.release()
    return frames

# Function to preprocess a frame (assuming some preprocessing steps)
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))  # Example resize
    frame = frame / 255.0  # Normalize
    return frame

# Function to predict the exercise from a frame
def predict_exercise(model, frame):
    input_data = np.expand_dims(preprocess_frame(frame), axis=0)
    predictions = model.predict(input_data)
    predicted_index = np.argmax(predictions)
    exercise_types = ['PushUp', 'Squat', 'Hammer Curl', 'Russian Twist', 'Lateral Raise']  # Example exercise types
    exercise = exercise_types[predicted_index]
    confidence = predictions[0][predicted_index] * 100
    return exercise, confidence

# Load the pre-trained model (replace 'model_path' with your model's path)
model_path = "exercise_classification_model.h5"
model = load_model(model_path)

# Streamlit UI
st.title("Fitness Exercise and Pose Estimation")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)

    with open("uploaded_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Video uploaded successfully!")

    if st.button("Predict Exercise"):
        # Resize the video
        resize_video("uploaded_video.mp4", "resized_video.mp4")

        # Extract frames
        frames = extract_frames("resized_video.mp4")

        # Predict exercise for each frame and aggregate results
        predicted_exercises = []
        total_confidence = 0.0
        frame_count = 0

        for frame in frames:
            exercise, confidence = predict_exercise(model, frame)
            predicted_exercises.append(exercise)
            total_confidence += confidence
            frame_count += 1

        average_confidence = total_confidence / frame_count
        most_common_exercise = max(set(predicted_exercises), key=predicted_exercises.count)

        st.write(f"Predicted Exercise: {most_common_exercise}")
        st.write(f"Accuracy: {average_confidence:.2f}%")

        st.success("Prediction completed successfully!")
