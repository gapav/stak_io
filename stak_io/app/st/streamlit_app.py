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

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def setup_video(video_file_path: str, frames_per_second: int):
    """Set up video capture and writer."""
    cap = cv2.VideoCapture(video_file_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / frames_per_second)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # Change FourCC code to 'mp4v' (lowercase)
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frames_per_second, (frame_width, frame_height))
    return cap, out, frame_interval

def process_frame(frame, pose):
    """Process a single frame."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    annotated_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
    return annotated_frame

def main_video_processing(video_file_path: str, frames_per_second: int = 4):
    """Main processing loop."""
    cap, out, frame_interval = setup_video(video_file_path, frames_per_second)
    frame_count = 0
    output_video_path = 'output.mp4'  # Define output path at the start

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                annotated_frame = process_frame(frame, pose)
                out.write(annotated_frame)
            frame_count += 1

    cap.release()
    out.release()
    return output_video_path  # Return the output path

    
def save_frame_as_image(video_file_path: str, frame_rate_divisor: int = 4):
    """Saves a single frame from the video as an image."""
    cap = cv2.VideoCapture(video_file_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / frame_rate_divisor)
    frame_count = 0
    saved_image_path = 'output_frame.jpg'  # Define the path for the output image

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:  # Process only every Nth frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                if results.pose_landmarks:
                    annotated_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(
                        annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
                    cv2.imwrite(saved_image_path, annotated_frame)  # Save the frame as an image
                    break  # Save only the first processed frame and exit
            frame_count += 1

    cap.release()
    return saved_image_path


def calculate_angle(landmark1, landmark2, landmark3):
    """Calculate the angle between three points."""
    a = np.array([landmark1.x, landmark1.y]) - np.array([landmark2.x, landmark2.y])
    b = np.array([landmark3.x, landmark3.y]) - np.array([landmark2.x, landmark2.y])
    cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

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

# Example Streamlit Integration
def main():
    st.title('Ski Technique Analysis')
    st.header('Upload a video of your skiing and we will analyze it for you!')

    video_file = st.file_uploader("Upload a video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    if video_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
            tmp_video.write(video_file.read())
            tmp_video_path = tmp_video.name

        image_path, max_angle = process_video_for_max_hip_angle(tmp_video_path)

        if image_path and os.path.exists(image_path):
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_path, caption=f'Max Hip Angle Visualization')
            with col2:
                st.metric(label="Max Hip Joint Angle", value=f"{max_angle:.2f} degrees")

        # Clean up the temporary files if needed
        os.remove(tmp_video_path)
        if image_path and os.path.exists(image_path):
            os.remove(image_path)  # Optionally remove the image file after displaying

if __name__ == "__main__":
    main()
