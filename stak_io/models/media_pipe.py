import cv2
import mediapipe as mp
import tempfile

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def process_video(video_file):
    # Temporary file to store the processed video
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile_name = tfile.name

    cap = cv2.VideoCapture(video_file.name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(tfile_name, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=True, min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the BGR image to RGB, process it with MediaPipe, and render the landmarks.
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

            out.write(frame)

    cap.release()
    out.release()
    return tfile_name

