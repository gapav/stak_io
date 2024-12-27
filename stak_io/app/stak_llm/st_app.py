import tempfile
import streamlit as st

from video_processor import process_video_angles

# Set up page configuration
st.set_page_config(page_title="Stakj.IO", page_icon="⛷️", layout="wide")
# Initialize session state for results
if "results" not in st.session_state:
    st.session_state["results"] = {
        "hip_image": None,
        "lean_image": None,
        "max_hip_angle": None,
        "max_lean_angle": None,
    }

# Title and description
st.title("Skiing Technique Analysis")
st.caption("Analyze and improve your skiing technique across four key areas.")

# Video Upload Section
uploaded_video = st.file_uploader(
    "Upload your skiing video", type=["mp4", "mov", "avi", "asf", "m4v"]
)

if uploaded_video:
    # Inform the user about the upload
    with st.chat_message("user"):
        st.write("Here is my skiing video!")

    # Analyze the video with a spinner
    with st.spinner("Analyzing your video..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(uploaded_video.read())
            tmp_video_path = tmp_video.name

        # Process video and update session state
        hip_image_path, lean_image_path, max_hip_angle, max_lean_angle = (
            process_video_angles(tmp_video_path)
        )
        st.session_state["results"] = {
            "hip_image": hip_image_path,
            "lean_image": lean_image_path,
            "max_hip_angle": max_hip_angle,
            "max_lean_angle": max_lean_angle,
        }

# Tabbed Analysis Interface
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Initial Start of Stroke",
        "End of Stroke",
        "Return to Extended Position",
        "Advanced Techniques",
    ]
)

# Initial Start of Stroke
with tab1:
    st.header("Initial Start of Stroke")
    if st.session_state["results"]["hip_image"]:
        st.image(st.session_state["results"]["hip_image"], caption="Hip Analysis")
        st.write(f"Max Hip Angle: {st.session_state['results']['max_hip_angle']}°")
    else:
        st.write("Upload a video to see the analysis.")

# End of Stroke
with tab2:
    st.header("End of Stroke")
    if st.session_state["results"]["lean_image"]:
        st.image(st.session_state["results"]["lean_image"], caption="Lean Analysis")
        st.write(f"Max Lean Angle: {st.session_state['results']['max_lean_angle']}°")
    else:
        st.write("Upload a video to see the analysis.")

# Return to Extended Position
with tab3:
    st.header("Return to Extended Position")
    st.write("Smoothness and timing analysis coming soon!")

# Advanced Techniques
with tab4:
    st.header("Advanced Techniques")
    st.write("Explore advanced skiing techniques and efficiency optimizations.")
