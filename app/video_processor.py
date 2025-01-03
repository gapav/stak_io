from io import BytesIO
from typing import Tuple, Optional, List
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


mp_pose = mp.solutions.pose


# Utility Functions


def calculate_angle(landmark1, landmark2, landmark3) -> float:
    """Calculate the angle between three points."""
    a = np.array([landmark1.x - landmark2.x, landmark1.y - landmark2.y])
    b = np.array([landmark3.x - landmark2.x, landmark3.y - landmark2.y])
    cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def calculate_forward_lean_angle(shoulder, foot) -> float:
    """Calculate the forward lean angle based on vertical."""
    vector = np.array([foot[0] - shoulder[0], foot[1] - shoulder[1]])
    vertical = np.array([0, 1])
    cosine_angle = np.dot(vector, vertical) / (
        np.linalg.norm(vector) * np.linalg.norm(vertical)
    )
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


# Visualization Functions


def save_pillow_visualization(image, points, lines, output_path: str):
    """Save visualization with Pillow."""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    for start, end in lines:
        draw.line([points[start], points[end]], fill=(0, 0, 255), width=5)

    for point in points:
        draw.ellipse(
            [
                (point[0] - 10, point[1] - 10),
                (point[0] + 10, point[1] + 10),
            ],
            fill=(255, 0, 0),
        )

    pil_image.save(output_path, format="PNG")


def visualize_landmarks(image, points, lines, output_path: str):
    """Visualize landmarks using Matplotlib."""
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for start, end in lines:
        ax.plot(
            [points[start][0], points[end][0]],
            [points[start][1], points[end][1]],
            color="greenyellow",
            linewidth=2,
            alpha=0.8,
        )

    x_coords, y_coords = zip(*points)
    ax.scatter(
        x_coords, y_coords, edgecolors="black", color="dimgray", s=150, alpha=0.8
    )

    ax.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# Processing Functions


def extract_landmark_points(landmarks, indices, image_shape):
    """Extract landmark points from Mediapipe landmarks."""
    height, width, _ = image_shape
    return [
        (int(landmarks[idx].x * width), int(landmarks[idx].y * height))
        for idx in indices
    ]


def process_frame(
    frame, pose
) -> Tuple[float, float, Optional[np.ndarray], Optional[list]]:
    """Process a single frame to compute angles and extract landmarks."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        hip_angle = calculate_angle(
            landmarks[12], landmarks[24], landmarks[26]
        )  # Shoulder, hip, knee (right side)
        forward_lean_angle = calculate_forward_lean_angle(
            (
                landmarks[12].x * frame.shape[1],
                landmarks[12].y * frame.shape[0],
            ),
            (
                landmarks[32].x * frame.shape[1],
                landmarks[32].y * frame.shape[0],
            ),
        )  # Shoulder, foot (right side)

        return hip_angle, forward_lean_angle, frame.copy(), landmarks

    return 0, 0, None, None


# Helper functions
def capture_video_frames(video_file_path: str):
    cap = cv2.VideoCapture(video_file_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


def initialize_pose_detector():
    return mp.solutions.pose.Pose(
        min_detection_confidence=0.5, enable_segmentation=True
    )


def process_frame(frame, pose):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pose.process(frame_rgb)


def calculate_max_hip_angle(
    landmarks, max_hip_angle, frame, landmarks_with_max_hip_angle
):
    hip_angle = calculate_angle(
        landmarks[12],  # Shoulder
        landmarks[24],  # Hip
        landmarks[26],  # Knee
    )
    if hip_angle > max_hip_angle:
        max_hip_angle = hip_angle
        return hip_angle, frame.copy(), landmarks
    return max_hip_angle, None, None


def overlay_landmarks(frame, landmarks, pose_connections):
    frame_with_landmarks = frame.copy()
    mp.solutions.drawing_utils.draw_landmarks(
        frame_with_landmarks,
        landmarks,
        pose_connections,
        landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
    )
    return frame_with_landmarks


def generate_visualizations(
    frame, landmarks, shape, points_indices, connections, output_path
):
    points = extract_landmark_points(landmarks, points_indices, shape)
    visualize_landmarks(frame, points, connections, output_path)
    return output_path


def calculate_max_forward_lean(landmarks, frame):
    return calculate_forward_lean_angle(
        (
            landmarks[12].x * frame.shape[1],
            landmarks[12].y * frame.shape[0],
        ),
        (
            landmarks[32].x * frame.shape[1],
            landmarks[32].y * frame.shape[0],
        ),
    )


def process_video_angles(
    video_file_path: str,
) -> Tuple[Optional[str], Optional[str], float, float, List, List]:
    cap_frames = capture_video_frames(video_file_path)

    max_hip_angle = 0
    frame_with_max_hip_angle = None
    landmarks_with_max_hip_angle = None
    frames_with_landmarks = []
    frames_without_landmarks = []
    frame_count = 0
    first_iteration = True  # Flag to track the first iteration

    with initialize_pose_detector() as pose:
        for frame in cap_frames:
            frame_count += 1

            results = process_frame(frame, pose)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                max_hip_angle, max_frame, max_landmarks = calculate_max_hip_angle(
                    landmarks, max_hip_angle, frame, landmarks_with_max_hip_angle
                )
                if max_frame is not None:
                    frame_with_max_hip_angle = max_frame
                    landmarks_with_max_hip_angle = max_landmarks

                # Append frames only during the first iteration
                if first_iteration:
                    frames_with_landmarks.append(
                        overlay_landmarks(
                            frame,
                            results.pose_landmarks,
                            mp.solutions.pose.POSE_CONNECTIONS,
                        )
                    )
                    frames_without_landmarks.append(frame.copy())
                    first_iteration = False  # Ensure it only happens once

    max_forward_lean_angle = 0
    if (
        frame_with_max_hip_angle is not None
        and landmarks_with_max_hip_angle is not None
    ):
        max_forward_lean_angle = calculate_max_forward_lean(
            landmarks_with_max_hip_angle, frame_with_max_hip_angle
        )

    hip_visualization_path = None
    lean_visualization_path = None

    if frame_with_max_hip_angle is not None:
        hip_visualization_path = generate_visualizations(
            frame_with_max_hip_angle,
            landmarks_with_max_hip_angle,
            frame_with_max_hip_angle.shape,
            [12, 24, 26],
            [(0, 1), (1, 2)],
            "/opt/stak_io/app/model_output/max_hip_angle_visualization.png",
        )

        lean_visualization_path = generate_visualizations(
            frame_with_max_hip_angle,
            landmarks_with_max_hip_angle,
            frame_with_max_hip_angle.shape,
            [12, 32],
            [(0, 1)],
            "/opt/stak_io/app/model_output/forward_lean_visualization.png",
        )

    return (
        hip_visualization_path,
        lean_visualization_path,
        max_hip_angle,
        max_forward_lean_angle,
        frames_with_landmarks,
        frames_without_landmarks,
    )
