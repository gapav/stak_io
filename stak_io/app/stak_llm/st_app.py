import tempfile
import streamlit as st
import time
from video_processor import process_video_angles

# Constants for paths
LOGO_PATH = "/Users/gardpavels/code/stak_io/stak_io/app/stak_llm/media/logo_alt.png"
HEADER_IMAGE_PATH = (
    "/Users/gardpavels/code/stak_io/stak_io/app/stak_llm/media/header.png"
)
AVATAR_PATH = "/Users/gardpavels/code/stak_io/stak_io/app/stak_llm/media/button_alt.png"

# Page configuration
st.set_page_config(page_title="Stakj.io", page_icon="⛷️", layout="wide")


# Initialize session state
def initialize_session_state():
    default_results = {
        "hip_image": None,
        "lean_image": None,
        "max_hip_angle": None,
        "max_lean_angle": None,
    }
    st.session_state.setdefault("results", default_results)
    st.session_state.setdefault("active_tab", "landing_page")
    st.session_state.setdefault("tab1_feedback", False)
    st.session_state.setdefault("tab2_feedback", False)


initialize_session_state()


# Sidebar for video upload
def render_sidebar():
    with st.sidebar:
        st.image(LOGO_PATH, use_container_width=True)
        st.title("Video Upload")
        return st.file_uploader(
            "Upload your skiing video", type=["mp4", "mov", "avi", "asf", "m4v"]
        )


def process_uploaded_video(uploaded_video):
    with st.spinner("Processing your video..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(uploaded_video.read())
            tmp_video_path = tmp_video.name

        hip_image_path, lean_image_path, max_hip_angle, max_lean_angle = (
            process_video_angles(tmp_video_path)
        )
        st.session_state["results"].update(
            {
                "hip_image": hip_image_path,
                "lean_image": lean_image_path,
                "max_hip_angle": max_hip_angle,
                "max_lean_angle": max_lean_angle,
            }
        )
        st.session_state["processed_video"] = True
        st.session_state["active_tab"] = "tab1"


# Response generator for chatbot-like feedback
def response_generator(response: str):
    for word in response.split():
        yield word + " "
        time.sleep(0.03)


# Render tab selection dropdown
def render_tab_selector(tabs):
    active_tab = st.selectbox(
        "Select Analysis Tab",
        list(tabs.values()),
        index=list(tabs.keys()).index(st.session_state["active_tab"]),
    )
    st.session_state["active_tab"] = list(tabs.keys())[
        list(tabs.values()).index(active_tab)
    ]


def render_landing_page():
    st.image(HEADER_IMAGE_PATH)
    st.title("Welcome to Stak.io!")
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


def render_tab_content(tab_name, image_key, angle_key, feedback_key, feedback_text):
    st.header(tab_name)
    if st.session_state["results"][image_key]:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(
                st.session_state["results"][image_key], caption=f"{tab_name} Analysis"
            )
        with col2:
            st.subheader("Feedback")
            if feedback_key not in st.session_state:
                st.session_state[feedback_key] = [feedback_text]
            for message in st.session_state[feedback_key]:
                with st.chat_message("assistant", avatar=AVATAR_PATH):
                    st.write(response_generator(message))
    else:
        st.write("Upload a video to see the analysis.")


# Main app logic
uploaded_video = render_sidebar()
if uploaded_video and "processed_video" not in st.session_state:
    process_uploaded_video(uploaded_video)

TABS = {
    "landing_page": "Welcome!",
    "tab1": "Initial Start of Stroke",
    "tab2": "Forward Lean",
    "tab3": "Return to Extended Position",
    "tab4": "Advanced Techniques",
}
render_tab_selector(TABS)

if st.session_state["active_tab"] == "landing_page":
    render_landing_page()

elif st.session_state["active_tab"] == "tab1":
    feedback_text = f"Your maximum hip angle is **{st.session_state['results']['max_hip_angle']:.1f}°**, which is excellent."
    render_tab_content(
        "Initial Start of Stroke",
        "hip_image",
        "max_hip_angle",
        "tab1_feedback_text",
        feedback_text,
    )
elif st.session_state["active_tab"] == "tab2":
    feedback_text = f"Your maximum lean angle is **{st.session_state['results']['max_lean_angle']:.1f}°**, which is effective."
    render_tab_content(
        "Forward Lean",
        "lean_image",
        "max_lean_angle",
        "tab2_feedback_text",
        feedback_text,
    )
elif st.session_state["active_tab"] == "tab3":
    st.header("Return to Extended Position")
    st.write("Content for this section coming soon!")
elif st.session_state["active_tab"] == "tab4":
    st.header("Advanced Techniques")
    st.write("Content for this section coming soon!")
