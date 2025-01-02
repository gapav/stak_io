import os
import tempfile
import time
import streamlit as st
from video_processor import process_video_angles
from llm import SkiingCoachLLM


def response_generator(response: str):
    for word in response.split():
        yield word + " "
        time.sleep(0.07)


KEY_HARDCODED = (
    "sk-proj-XB0j9Eu4Id_NUr4armpAyIuunI3qwTAc6RwSvr4D40Kg0YUPrYYaTH4EVQ7uRGQGKT"
    "rc4HdlubT3BlbkFJLWpc_9QsSfTuuiUCItAc79IkniX0whmIRRU8_o5eOhYiDV7j0Sj2_GFdIRJ5"
    "K4LQWYtXyS_e8A"
)

# Sidebar
with st.sidebar:
    st.image(
        "/Users/gardpavels/code/stak_io/stak_io/app/stak_llm/media/stakj_header.png",
        use_container_width=True,
    )
    st.caption("Upload a video to analyze your skiing technique.")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    "[View the source code](https://github.com/your-repo/stakio)"
    "[How to record to get best results?](https://github.com/your-repo/stakio)"

st.title("STAKJ.IO")
st.caption("🤖 An app to analyze your double poling technique.")

# Initialize messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Upload your SkiErg video, and we'll analyze it for you!",
        }
    ]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "image_path" in msg:
            st.image(msg["image_path"], caption=msg["content"])
        else:
            st.write(msg["content"])

# Video Upload
uploaded_video = st.file_uploader(
    "Upload your skiing video", type=["mp4", "mov", "avi", "asf", "m4v"]
)

if uploaded_video:
    st.session_state.messages.append(
        {"role": "user", "content": "Here is my Erg video!"}
    )
    with st.chat_message("user"):
        st.write("Here is my skiing video!")

    with st.spinner("Analyzing your video..."):
        # Save temporary video file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(uploaded_video.read())
            tmp_video_path = tmp_video.name

        # Process video
        hip_image_path, lean_image_path, max_hip_angle, max_lean_angle = (
            process_video_angles(tmp_video_path)
        )

    # Generate LLM feedback (placeholder if no API key provided)
    if openai_api_key:
        llm = SkiingCoachLLM(api_key=openai_api_key)
        feedback1 = llm.generate_feedback(max_hip_angle)
        # You can handle multi-part feedback inside LLM or separately
        feedback2 = ""
        feedback3 = ""
    else:
        feedback1 = (
            "Great! Let's work on improving your technique. We'll focus on a few "
            "key points. Let's start with how well you can fully extend your hips:"
        )
        feedback2 = "Your hip extension is just right—exactly where it should be for powerful skiing! "

        feedback3 = (
            "Next, let's focus on your forward lean. This is a key component of "
            "your skiing technique to help generate more power and speed."
        )

    # Define recommended hip angle range
    recommended_min_angle = 160
    recommended_max_angle = 180

    with st.chat_message("assistant"):
        st.write(response_generator(feedback1))

    with st.spinner("Analyzing hip angles...."):
        time.sleep(2)

    # Check if the hip angle is within the recommended range
    if recommended_min_angle <= max_hip_angle <= recommended_max_angle:
        hip_angle_advice = (
            "Your hip extension is just right—exactly where it should be for a "
            "powerful stakj!Proper extension helps you maximize your stroke efficiency, allowing for stronger pushes and longer glides on the snow."
        )
    else:
        hip_angle_advice = (
            "Looks like your hip extension could use a bit more stretch. Aiming "
            "for a fuller extension will help you improve your power."
        )

    with st.chat_message("assistant"):
        st.image(hip_image_path)
        st.write(response_generator(hip_angle_advice))

    with st.chat_message("assistant"):
        st.write(response_generator(feedback3))

    with st.spinner("Analyzing forward lean...."):
        time.sleep(3)
    # Check if the forward lean is within the recommended range
    if max_lean_angle >= 5:
        lean_advice = (
            "Your forward lean is looking great! This position helps you "
            "maintain balance and control while skiing down the slopes."
        )
    else:
        lean_advice = (
            "It seems like you could use a bit more forward lean. This position "
            "helps you maintain balance and control while skiing down the slopes."
        )

    with st.chat_message("assistant"):
        # Use the same feedback2 variable or craft a new one as needed
        st.image(lean_image_path)
        st.write(response_generator(lean_advice))
        st.write(response_generator("Ready to tackle the next technique tips?"))

    # Clean up the lean image
    if lean_image_path:
        os.remove(lean_image_path)

    # Clean up hip image
    if hip_image_path:
        os.remove(hip_image_path)
