import streamlit as st
from ultralytics import YOLO
import cv2

def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    # image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(
        res_plotted,
        caption='Detected Video',
        channels="BGR",
        use_container_width=True
    )


def play_stored_video(conf, model, vid, genre):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.
        vid: video to play and analyse.

    Returns:
        None
    """
    if genre == "Vertical":
        cols = st.columns(2)
        with cols[0]: st.video(vid)
    else:
        st.video(vid)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(vid)
            if genre == "Vertical":
                with cols[1]: st_frame = st.empty()  
            else:
                st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(
                        conf,
                        model,
                        st_frame,
                        image,
                    )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

def play_webcam_feed(conf, model):
    """
    Captures and processes a live webcam feed using YOLOv8.
    """
    st_frame = st.empty()
    vid_cap = cv2.VideoCapture(0)  # 0 is usually the default camera

    while st.session_state["webcam_active"]:
        success, image = vid_cap.read()
        # image = cv2.resize(image, (720, int(720*(9/16))))
        if not success:
            continue

        _display_detected_frames(
            conf,
            model,
            st_frame,
            image,
        )

    vid_cap.release()
    cv2.destroyAllWindows()


########################################################################


st.set_page_config(
    page_title="Honest Mirror",
    page_icon="👩🏻‍🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Pose segmentation for presentation projects.")

# model = YOLO("yolov8m-pose.pt")
model = YOLO("yolov8m.pt")
# model.to('cuda') # Doesnt work yet, as it runs on windows.

# SIDEBAR
st.sidebar.header("Video Config")

# Uploaded video section
with st.sidebar.container():
    st.subheader("Upload Video:")
    uploaded_video = st.file_uploader(
        label="Click here to upload your video!",
        type=["mp4", "mov"],
        help="Accepted input: `.mp4` & `.mov`"
    )

    if uploaded_video is not None:
        genre = st.radio(
            "Video layout",
            ["Vertical", "Horizontal"],
        )

# uploaded_video = st.sidebar.file_uploader(
#     label="Click here to upload your video!",
#     type=["mp4", "mov"],
#     help="Accepted input: `.mp4` & `.mov`"
# )

# genre = st.sidebar.radio(
#     "Video layout",
#     ["Vertical", "Horizontal"],
# )

# Webcam button with toggle functionality and dynamic text
if "webcam_active" not in st.session_state:
    st.session_state["webcam_active"] = False

# Use on_click to toggle the state in a single execution cycle
if uploaded_video is None:
    st.sidebar.subheader("Or use your webcam:")
    st.sidebar.button(
        label="Stop Using Webcam" if st.session_state["webcam_active"] else "Use Webcam", 
        on_click=lambda: st.session_state.update({"webcam_active": not st.session_state["webcam_active"]})
    )

    # Run webcam feed if active
    if st.session_state["webcam_active"]:
        play_webcam_feed(0.5, model)




# Check for upload
if uploaded_video is not None:
    vid = uploaded_video.name
    with open(vid, mode='wb') as f:
        f.write(uploaded_video.read())

    play_stored_video(0.5, model, vid, genre)

