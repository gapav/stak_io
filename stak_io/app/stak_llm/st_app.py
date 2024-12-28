import tempfile
import streamlit as st
import time
from video_processor import process_video_angles

# Set up page configuration
st.set_page_config(page_title="Stakj.io", page_icon="⛷️", layout="wide")

# Initialize session state for results and feedback
if "results" not in st.session_state:
    st.session_state["results"] = {
        "hip_image": None,
        "lean_image": None,
        "max_hip_angle": None,
        "max_lean_angle": None,
    }
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "landing_page"  # Default to Welcome Page
if "tab1_feedback" not in st.session_state:
    st.session_state["tab1_feedback"] = False
if "tab2_feedback" not in st.session_state:
    st.session_state["tab2_feedback"] = False


# Sidebar for video upload
with st.sidebar:
    st.image(
        "/Users/gardpavels/code/stak_io/stak_io/app/stak_llm/media/logo_alt.png",
        use_container_width=True,
    )
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
        st.session_state["active_tab"] = "tab1"  # Automatically switch to Tab 1


# Function to generate text dynamically
def response_generator(response: str):
    for word in response.split():
        yield word + " "
        time.sleep(0.03)


# Dropdown menu for tab selection
tabs = {
    "landing_page": "Welcome!",
    "tab1": "Initial Start of Stroke",
    "tab2": "End of Stroke",
    "tab3": "Return to Extended Position",
    "tab4": "Advanced Techniques",
}

# Use the session state to determine the currently active tab
active_tab = st.selectbox(
    "Select Analysis Tab",
    list(tabs.values()),
    index=list(tabs.keys()).index(st.session_state["active_tab"]),
)

# Update session state for active tab based on dropdown selection
st.session_state["active_tab"] = list(tabs.keys())[
    list(tabs.values()).index(active_tab)
]

# Landing Page content
if st.session_state["active_tab"] == "landing_page":
    # Title and description
    st.image(
        "/Users/gardpavels/code/stak_io/stak_io/app/stak_llm/media/header.png",
    )
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


# Tab 1: Initial Start of Stroke
elif st.session_state["active_tab"] == "tab1":
    st.header("Initial Start of Stroke")
    if st.session_state["results"]["hip_image"]:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(st.session_state["results"]["hip_image"], caption="Hip Analysis")
        with col2:
            st.subheader("Feedback")

            # Check if feedback is already stored in session state
            if "tab1_feedback_text" not in st.session_state:
                feedback1 = (
                    f"Let´s start where the stroke starts: With the body in a (hopefully) fully extended position! "
                    f"Your max hip angle is **{st.session_state['results']['max_hip_angle']:.1f}°**, "
                    "which is within an excellent range. Maintaining this level of posture will "
                    "help you generate more power and reduce fatigue."
                )

                st.session_state["tab1_feedback_text"] = [feedback1]

            # Display chatbot-like feedback
            for message in st.session_state["tab1_feedback_text"]:
                with st.chat_message(
                    "assistant",
                    avatar="/Users/gardpavels/code/stak_io/stak_io/app/stak_llm/media/button_alt.png",
                ):
                    st.write(response_generator(message))
    else:
        st.write("Upload a video to see the analysis.")

# Tab 2: End of Stroke
elif st.session_state["active_tab"] == "tab2":
    st.header("End of Stroke")
    if st.session_state["results"]["lean_image"]:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(st.session_state["results"]["lean_image"], caption="Lean Analysis")
        with col2:
            st.subheader("Feedback")

            # Check if feedback is already stored in session state
            if "tab2_feedback_text" not in st.session_state:
                feedback1 = (
                    f"Nice work! Your max lean angle is **{st.session_state['results']['max_lean_angle']:.1f}°**, "
                    "indicating a solid end position for your stroke. Leaning forward effectively "
                    "transfers power into the stroke, improving efficiency."
                )

                st.session_state["tab2_feedback_text"] = [feedback1]

            # Display chatbot-like feedback
            for message in st.session_state["tab2_feedback_text"]:
                with st.chat_message(
                    "assistant",
                    avatar="/Users/gardpavels/code/stak_io/stak_io/app/stak_llm/media/button_alt.png",
                ):
                    st.write(response_generator(message))
    else:
        st.write("Upload a video to see the analysis.")


# Tab 3 and Tab 4 placeholders
elif st.session_state["active_tab"] == "tab3":
    st.header("Return to Extended Position")
    st.write("Content for this section coming soon!")

elif st.session_state["active_tab"] == "tab4":
    st.header("Advanced Techniques")
    st.write("Content for this section coming soon!")
