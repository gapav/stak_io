import tempfile
import streamlit as st
import time
from video_processor import process_video_angles

# Constants for paths
SIDEBAR_LOGO_PATH = "/opt/stak_io/app/media/sidebar_logo_blck.png"
HEADER_IMAGE_PATH = "/opt/stak_io/app/media/logo_B.png"
AVATAR_PATH = "/opt/stak_io/app/media/button_alt.png"
PROC_COMPL_PATH = "/opt/stak_io/app/media/proc_compl.png"
UPLOAD_ARROW_PATH = "/opt/stak_io/app/media/upload_to_cont.png"

# Page configuration
st.set_page_config(page_title="Stakj.io", page_icon="⛷️", layout="wide")


# Initialize session state
def initialize_session_state():
    default_results = {
        "hip_image": None,
        "lean_image": None,
        "max_hip_angle": None,
        "max_lean_angle": None,
        "frames_with_landmarks": [],
        "frames_without_landmarks": [],
        "processed_video_path": None,
    }
    st.session_state.setdefault("results", default_results)
    st.session_state.setdefault("active_tab", "landing_page")
    st.session_state.setdefault("tab1_feedback", False)
    st.session_state.setdefault("tab2_feedback", False)


initialize_session_state()


# Sidebar for video upload
def render_sidebar():
    with st.sidebar:
        st.image(SIDEBAR_LOGO_PATH, use_container_width=True)
        st.title("Video Upload")

        file_upload = st.file_uploader(
            "Upload your skiing video", type=["mp4", "mov", "avi", "asf", "m4v"]
        )
        return file_upload


def process_uploaded_video(uploaded_video):
    with st.spinner("Processing your video..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(uploaded_video.read())
            tmp_video_path = tmp_video.name

        # Process video to save random frames with and without landmarks
        (
            hip_image_path,
            lean_image_path,
            max_hip_angle,
            max_lean_angle,
            frames_with_landmarks,
            frames_without_landmarks,
        ) = process_video_angles(tmp_video_path)

        st.session_state["results"].update(
            {
                "hip_image": hip_image_path,
                "lean_image": lean_image_path,
                "max_hip_angle": max_hip_angle,
                "max_lean_angle": max_lean_angle,
                "frames_with_landmarks": frames_with_landmarks,
                "frames_without_landmarks": frames_without_landmarks,
            }
        )
        st.session_state["processed_video"] = True
        st.session_state["active_tab"] = "tab_processing_complete"


# Response generator for chatbot-like feedback
def response_generator(response: str):
    for word in response.split():
        yield word + " "
        time.sleep(0.03)


# Render tab selection dropdown
def render_tab_selector(tabs):
    def update_tab():
        st.session_state["active_tab"] = list(tabs.keys())[
            list(tabs.values()).index(st.session_state["selected_tab"])
        ]

    # Use a dropdown linked to `st.session_state`

    st.selectbox(
        "Select Analysis Tab",
        options=list(tabs.values()),
        index=list(tabs.keys()).index(st.session_state["active_tab"]),
        key="selected_tab",
        on_change=update_tab,  # Callback to update the active tab
    )


def render_landing_page():
    st.image(HEADER_IMAGE_PATH)
    st.subheader("Why Stak.io?")
    st.write(
        "Double-poling (DP) technique on the SkiErg might differ significantly from DP technique on skis. "
        "Achieving impressive numbers on the SkiErg doesn’t always translate to powerful, efficient technique on snow. "
        "Stak.io is here to bridge that gap by providing actionable feedback to help refine your technique. With this app, "
        "you can ensure that your hard-earned hours on the erg contribute to a powerful, economical technique on snow, "
        "maximizing your performance and minimizing energy waste."
    )
    st.subheader("How to Use Stak.io")
    st.markdown(
        """
        **Upload Your Video:** Record your double-poling session on the SkiErg and upload the video directly in the sidebar.
        """
    )
    st.subheader("Contact Us")
    st.write(
        "Have questions or feedback? Reach out at [gardpavels@gmail.com](mailto:gardpavels@gmail.com)"
    )


def render_tab_content(tab_name, image_key, angle_key, feedback_key, feedback_text):
    st.header(tab_name)

    if st.session_state["results"][image_key]:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(
                st.session_state["results"][image_key], caption=f"{tab_name} Analysis"
            )
        with col2:
            if feedback_key not in st.session_state:
                st.session_state[feedback_key] = [feedback_text]
            for message in st.session_state[feedback_key]:
                with st.chat_message("assistant", avatar=AVATAR_PATH):
                    st.write(response_generator(message))
    else:
        st.image(UPLOAD_ARROW_PATH)


def render_tab_processing_complete(tab_name, feedback_key, feedback_text):
    frames_with_landmarks = st.session_state["results"].get("frames_with_landmarks", [])
    frames_without_landmarks = st.session_state["results"].get(
        "frames_without_landmarks", []
    )

    if len(frames_with_landmarks) > 0 and len(frames_without_landmarks) > 0:
        st.image(PROC_COMPL_PATH)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(frames_without_landmarks[0])
        with col2:
            st.image(
                frames_with_landmarks[0],
            )
        with col3:
            if feedback_key not in st.session_state:
                st.session_state[feedback_key] = [feedback_text]
            for message in st.session_state[feedback_key]:
                with st.chat_message("assistant", avatar=AVATAR_PATH):
                    st.write(response_generator(message))
    else:
        st.image(UPLOAD_ARROW_PATH)


# Main app logic
uploaded_video = render_sidebar()
if uploaded_video and "processed_video" not in st.session_state:
    process_uploaded_video(uploaded_video)

TABS = {
    "landing_page": "Welcome!",
    "tab_processing_complete": "Processing steps",
    "tab1": "Key Element #1: The Hips",
    "tab2": "Key Element #2: The Forward Lean",
    "tab3": "Key Element #3: The Return to Extended Position",
    "tab4": "Advanced Techniques",
}
render_tab_selector(TABS)

if st.session_state["active_tab"] == "landing_page":
    render_landing_page()

elif st.session_state["active_tab"] == "tab_processing_complete":
    render_tab_processing_complete(
        tab_name="Processing Steps",
        feedback_key="process_feedback_text",
        feedback_text="Processing complete! Key points have been successfully identified, enabling us to analyze critical biomechanical details such as joint positions and motion patterns. This provides a solid foundation for evaluating and improving your technique.",
    )

elif st.session_state["active_tab"] == "tab1":
    feedback_text = f"Let’s start by analyzing the beginning of your stroke, which is a critical phase for setting up an efficient and powerful double-pole motion. Ideally, your hips should be in a close to fully extended position, allowing for maximum reach and optimal power transfer. Your recorded maximum hip angle is ≈ **{st.session_state['results']['max_hip_angle']:.0f}°**, which falls within an excellent range for effective double-poling. This indicates that you are starting your stroke with strong posture and alignment, which are essential for generating maximum power and minimizing unnecessary energy expenditure. Maintaining this level of extension throughout your technique will not only improve your efficiency but also reduce fatigue during prolonged efforts. This strong starting position sets a solid foundation for the rest of your stroke. Moving forward, we’ll look at how this extension transitions into the next phases of your movement to ensure you’re capitalizing on this great setup for consistent and powerful strokes."
    render_tab_content(
        "The Hips",
        "hip_image",
        "max_hip_angle",
        "tab1_feedback_text",
        feedback_text,
    )
elif st.session_state["active_tab"] == "tab2":
    feedback_text = f"You demonstrate a solid forward lean (≈ **{st.session_state['results']['max_lean_angle']:.0f}°**), reflecting a strong and effective position during the most extended phase of your stroke. This forward lean is crucial for efficiently transferring power from your upper body and core into the poles and, ultimately, into the snow. Maintaining this lean ensures optimal force application and energy efficiency, supporting better endurance performance. As we refine your technique, we’ll focus on keeping this lean integrated with the other stroke phases for smooth and consistent power delivery."

    render_tab_content(
        "The Forward Lean",
        "lean_image",
        "max_lean_angle",
        "tab2_feedback_text",
        feedback_text,
    )
elif st.session_state["active_tab"] == "tab3":
    st.header("The Return to Extended Position")
    st.write("Content for this section coming soon!")
elif st.session_state["active_tab"] == "tab4":
    st.header("Advanced Techniques")
    st.write("Content for this section coming soon!")
