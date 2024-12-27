import tempfile
import streamlit as st
import time
from video_processor import process_video_angles

# Set up page configuration
st.set_page_config(page_title="Stakj.IO", page_icon="⛷️", layout="wide")

# Initialize session state for results and feedback
if "results" not in st.session_state:
    st.session_state["results"] = {
        "hip_image": None,
        "lean_image": None,
        "max_hip_angle": None,
        "max_lean_angle": None,
    }
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "tab1"
if "tab1_feedback" not in st.session_state:
    st.session_state["tab1_feedback"] = False
if "tab2_feedback" not in st.session_state:
    st.session_state["tab2_feedback"] = False

# Title and description
st.title("Skiing Technique Analysis")
st.caption("Analyze and improve your skiing technique across four key areas.")

# Sidebar for video upload
with st.sidebar:
    st.title("Video Upload")
    uploaded_video = st.file_uploader(
        "Upload your skiing video", type=["mp4", "mov", "avi", "asf", "m4v"]
    )

    # Process the video only if it's uploaded and hasn't been processed yet
    if uploaded_video and "processed_video" not in st.session_state:
        with st.spinner("Processing your video..."):
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
            st.session_state["processed_video"] = True  # Mark video as processed


# Function to generate text dynamically
def response_generator(response: str):
    for word in response.split():
        yield word + " "
        time.sleep(0.03)


# Dropdown menu for tab selection
tabs = {
    "tab1": "Initial Start of Stroke",
    "tab2": "End of Stroke",
    "tab3": "Return to Extended Position",
    "tab4": "Advanced Techniques",
}
active_tab = st.selectbox("Select Analysis Tab", list(tabs.values()))

# Update session state for active tab
if active_tab == tabs["tab1"]:
    if st.session_state["active_tab"] != "tab1":
        st.session_state["active_tab"] = "tab1"
        st.session_state["tab1_feedback"] = False
elif active_tab == tabs["tab2"]:
    if st.session_state["active_tab"] != "tab2":
        st.session_state["active_tab"] = "tab2"
        st.session_state["tab2_feedback"] = False

# Tab 1: Initial Start of Stroke
if active_tab == tabs["tab1"]:
    st.header("Initial Start of Stroke")
    if st.session_state["results"]["hip_image"]:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(st.session_state["results"]["hip_image"], caption="Hip Analysis")
        with col2:
            st.subheader("Feedback")
            feedback1 = (
                f"Great job! Your max hip angle is **{st.session_state['results']['max_hip_angle']:.1f}°**, "
                "which is within an excellent range. Maintaining this level of posture will "
                "help you generate more power and reduce fatigue."
            )
            feedback2 = (
                "To improve further, focus on keeping your core engaged during the stroke. "
                "This will help maintain alignment and prevent over-extension."
            )
            if not st.session_state["tab1_feedback"]:
                with st.chat_message("assistant"):
                    st.write(response_generator(feedback1))
                st.session_state["tab1_feedback"] = True
    else:
        st.write("Upload a video to see the analysis.")

# Tab 2: End of Stroke
elif active_tab == tabs["tab2"]:
    st.header("End of Stroke")
    if st.session_state["results"]["lean_image"]:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(st.session_state["results"]["lean_image"], caption="Lean Analysis")
        with col2:
            st.subheader("Feedback")
            feedback1 = (
                f"Nice work! Your max lean angle is **{st.session_state['results']['max_lean_angle']:.1f}°**, "
                "indicating a solid end position for your stroke. Leaning forward effectively "
                "transfers power into the stroke, improving efficiency."
            )
            feedback2 = (
                "To enhance your performance, make sure to maintain a neutral spine and avoid "
                "over-leaning, which could cause unnecessary strain. "
                "Practice ending your stroke with a controlled forward motion."
            )
            if not st.session_state["tab2_feedback"]:
                with st.chat_message("assistant"):
                    st.write(response_generator(feedback1))
                st.session_state["tab2_feedback"] = True
    else:
        st.write("Upload a video to see the analysis.")

# Tab 3 and Tab 4 placeholders
elif active_tab == tabs["tab3"]:
    st.header("Return to Extended Position")
    st.write("Content for this section coming soon!")

elif active_tab == tabs["tab4"]:
    st.header("Advanced Techniques")
    st.write("Content for this section coming soon!")
