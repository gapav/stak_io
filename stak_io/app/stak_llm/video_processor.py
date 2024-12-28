from io import BytesIO
from typing import Tuple, Optional
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


def process_video_angles(
    video_file_path: str,
) -> Tuple[Optional[str], Optional[str], float, float]:
    """Process a video to find the maximum hip angle and corresponding forward lean angle."""
    cap = cv2.VideoCapture(video_file_path)
    max_hip_angle = 0
    frame_with_max_hip_angle = None
    landmarks_with_max_hip_angle = None

    with mp_pose.Pose(min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Calculate hip angle
                hip_angle = calculate_angle(
                    landmarks[12],  # Shoulder
                    landmarks[24],  # Hip
                    landmarks[26],  # Knee
                )

                # Update the frame with the max hip angle
                if hip_angle > max_hip_angle:
                    max_hip_angle = hip_angle
                    frame_with_max_hip_angle = frame.copy()
                    landmarks_with_max_hip_angle = landmarks

    cap.release()

    # Calculate forward lean for the frame with the maximum hip angle
    max_forward_lean_angle = 0
    if frame_with_max_hip_angle is not None and landmarks_with_max_hip_angle:
        max_forward_lean_angle = calculate_forward_lean_angle(
            (
                landmarks_with_max_hip_angle[12].x * frame_with_max_hip_angle.shape[1],
                landmarks_with_max_hip_angle[12].y * frame_with_max_hip_angle.shape[0],
            ),
            (
                landmarks_with_max_hip_angle[32].x * frame_with_max_hip_angle.shape[1],
                landmarks_with_max_hip_angle[32].y * frame_with_max_hip_angle.shape[0],
            ),
        )

    # Generate visualizations
    hip_visualization_path = None
    lean_visualization_path = None

    if frame_with_max_hip_angle is not None:
        hip_points = extract_landmark_points(
            landmarks_with_max_hip_angle, [12, 24, 26], frame_with_max_hip_angle.shape
        )
        hip_visualization_path = "max_hip_angle_visualization.png"
        visualize_landmarks(
            frame_with_max_hip_angle,
            hip_points,
            [(0, 1), (1, 2)],
            hip_visualization_path,
        )

        lean_points = extract_landmark_points(
            landmarks_with_max_hip_angle, [12, 32], frame_with_max_hip_angle.shape
        )
        lean_visualization_path = "forward_lean_visualization.png"
        visualize_landmarks(
            frame_with_max_hip_angle, lean_points, [(0, 1)], lean_visualization_path
        )

    return (
        hip_visualization_path,
        lean_visualization_path,
        max_hip_angle,
        max_forward_lean_angle,
    )


mp_selfie_segmentation = mp.solutions.selfie_segmentation


def process_video_in_memory(video_file_path: str) -> BytesIO:
    """Process a video and return it as a BytesIO object."""
    cap = cv2.VideoCapture(video_file_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_stream = BytesIO()

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        with mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1
        ) as selfie_segmentation:
            height, width = (
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            )
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Use VideoWriter to write to a buffer
            out = cv2.VideoWriter(video_stream, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                segmentation_results = selfie_segmentation.process(frame_rgb)

                mask = segmentation_results.segmentation_mask
                mask = (mask > 0.5).astype(np.uint8) * 255
                mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(frame, 0.6, mask_colored, 0.4, 0)

                out.write(overlay)

            cap.release()
            out.release()

    video_stream.seek(0)
    return video_stream


video_bytes = process_video_in_memory("input_video.mp4")
st.video(video_bytes)


# Example usage:
# process_video_with_segmentation("input_video.mp4", "output_with_segmentation.mp4")
