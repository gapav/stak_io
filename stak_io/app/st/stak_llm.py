import os
import tempfile
from matplotlib import pyplot as plt
import numpy as np
import streamlit as st
import sys
import cv2
import mediapipe as mp
import tempfile
import cv2
import mediapipe as mp
import plotly.graph_objects as go

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def visualize_landmarks_plotly(image, landmarks, shoulder_idx, hip_idx, knee_idx, angle):
    """Visualize the hip angle and landmarks using Plotly."""
    # Convert landmarks to pixel coordinates
    height, width, _ = image.shape
    shoulder = (int(landmarks[shoulder_idx].x * width), int(landmarks[shoulder_idx].y * height))
    hip = (int(landmarks[hip_idx].x * width), int(landmarks[hip_idx].y * height))
    knee = (int(landmarks[knee_idx].x * width), int(landmarks[knee_idx].y * height))
    
    # Convert the image to RGB format for visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create the figure
    fig = go.Figure()

    # Add the image as a background
    fig.add_trace(
        go.Image(z=image_rgb)
    )

    # Add the lines connecting the landmarks
    fig.add_trace(
        go.Scatter(
            x=[shoulder[0], hip[0], knee[0]],
            y=[shoulder[1], hip[1], knee[1]],
            mode='lines+markers',
            line=dict(color="blue", width=3),
            marker=dict(size=10, color="red"),
            name="Landmarks",
        )
    )

    # Add the angle text annotation
    fig.add_trace(
        go.Scatter(
            x=[hip[0]],
            y=[hip[1] - 20],
            mode='text',
            text=[f"{angle:.2f}°"],
            textfont=dict(size=16, color="green", family="Arial Black"),
            name="Angle",
        )
    )

    # Flip the y-axis to match the image coordinates
    fig.update_yaxes(
        autorange="reversed",
        scaleanchor="x",  # Lock the aspect ratio
        scaleratio=1,
    )

    # Remove axis lines and ticks for cleaner visualization
    fig.update_layout(
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False, visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    # Save the plot as an image
    output_path = "max_hip_angle_visualization_plotly.html"
    fig.write_html(output_path)

    return output_path


def visualize_landmarks(image, landmarks, shoulder_idx, hip_idx, knee_idx, angle):
    """Visualize the hip angle and landmarks using Matplotlib."""
    # Convert landmarks to pixel coordinates
    height, width, _ = image.shape
    shoulder = (int(landmarks[shoulder_idx].x * width), int(landmarks[shoulder_idx].y * height))
    hip = (int(landmarks[hip_idx].x * width), int(landmarks[hip_idx].y * height))
    knee = (int(landmarks[knee_idx].x * width), int(landmarks[knee_idx].y * height))
    
    # Plot the image and overlay landmarks
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.plot([shoulder[0], hip[0]], [shoulder[1], hip[1]], color="blue", linewidth=2)
    ax.plot([hip[0], knee[0]], [hip[1], knee[1]], color="blue", linewidth=2)
    ax.scatter([shoulder[0], hip[0], knee[0]], [shoulder[1], hip[1], knee[1]], color="red", s=50)

    # Annotate the angle
    ax.text(hip[0], hip[1] - 20, f"{angle:.2f}°", color="green", fontsize=12, weight="bold", ha="center")

    # Hide axes
    ax.axis("off")

    # Save and return the figure
    output_path = "max_hip_angle_visualization.png"
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output_path

def calculate_angle(landmark1, landmark2, landmark3):
    """Calculate the angle between three points."""
    a = np.array([landmark1.x, landmark1.y]) - np.array([landmark2.x, landmark2.y])
    b = np.array([landmark3.x, landmark3.y]) - np.array([landmark2.x, landmark2.y])
    cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def process_video_for_max_hip_angle(video_file_path: str):
    """Process the video to find the frame with the maximum hip angle."""
    cap = cv2.VideoCapture(video_file_path)
    max_angle = 0
    frame_with_max_angle = None
    landmarks_with_max_angle = None

    with mp_pose.Pose(min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                shoulder_idx, hip_idx, knee_idx = 11, 23, 25  # Right side landmarks
                angle = calculate_angle(landmarks[shoulder_idx], landmarks[hip_idx], landmarks[knee_idx])

                if angle > max_angle:
                    max_angle = angle
                    frame_with_max_angle = frame.copy()
                    landmarks_with_max_angle = landmarks

    cap.release()

    if frame_with_max_angle is not None and landmarks_with_max_angle is not None:
        # Visualize and save the frame with landmarks and angle
        visualization_path = visualize_landmarks(
            frame_with_max_angle,
            landmarks_with_max_angle,
            shoulder_idx=11,
            hip_idx=23,
            knee_idx=25,
            angle=max_angle,
        )
        return visualization_path, max_angle
    return None, 0
# Main chatbot interface


# Sidebar with instructions and source links
with st.sidebar:
    st.title("Ski Technique Analysis")
    st.caption("Upload a video to analyze your skiing technique.")
    "[View the source code](https://github.com/your-repo/stakio)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/your-repo/stakio?quickstart=1)"

st.title("🎿 Ski Technique Chatbot")
st.caption("🤖 A chatbot to analyze your skiing technique.")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Upload your skiing video, and I'll analyze it for you!"}]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "image_path" in msg:
            st.image(msg["image_path"], caption=msg["content"])
        else:
            st.write(msg["content"])

# File upload section
uploaded_video = st.file_uploader("Upload your skiing video (mp4, mov, avi, etc.)", type=["mp4", "mov", "avi", "asf", "m4v"])
if uploaded_video:
    # Append user message to the conversation
    st.session_state.messages.append({"role": "user", "content": "Here is my skiing video!"})
    with st.chat_message("user"):
        st.write("Here is my skiing video!")

    # Process video
    with st.spinner("Analyzing your video..."):
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
                tmp_video.write(uploaded_video.read())
                tmp_video_path = tmp_video.name

            # Run video processing function
            image_path, max_angle = process_video_for_max_hip_angle(tmp_video_path)

            # Append assistant response with image
            if image_path:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"The greatest hip angle in your video is {max_angle:.2f}°.",
                    "image_path": image_path,
                })
                with st.chat_message("assistant"):
                    st.image(image_path, caption=f"The greatest hip angle is {max_angle:.2f}°.")
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I couldn't analyze the video. Please try again with a different one."
                })
                with st.chat_message("assistant"):
                    st.write("I couldn't analyze the video. Please try again with a different one.")

        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"An error occurred: {str(e)}"
            })
            with st.chat_message("assistant"):
                st.write(f"An error occurred: {str(e)}")

        finally:
            # Clean up temporary files
            if os.path.exists(tmp_video_path):
                os.remove(tmp_video_path)
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
