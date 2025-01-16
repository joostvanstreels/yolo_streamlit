import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import os

##########################################################
#                         Config                         #
##########################################################


PREDICTION_CONFIDENCE = 0.5 # above this percentage, the predictions will be plotted.
CAM_INDEX = 0 # index 0 is usually the built in camera of your laptop. 1 is an external webcam
MODELS = (YOLO("yolov8n-pose.pt"), YOLO("yolov8n.pt")) # list of models, using the button on the left you can toggle the model.


##########################################################
#         Functions to run the yolo models with.         #
##########################################################


def _display_detected_frames(
    conf: float, 
    model: YOLO, 
    st_frame: st, 
    image: np.ndarray, 
)-> None:
    """
    Display the detected objects on a video frame using the model.

    @parameters:
    - conf (float): Confidence threshold for object detection.
    - model (YOLO): A YOLO object detection model.
    - st_frame (Streamlit object): 
        A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    """
    res = model.predict(image, conf=conf)

    # plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(
        res_plotted,
        channels="BGR",
        use_column_width=True
    )

def play_stored_video(conf: float, model: YOLO, vid: str, layout: str)-> None:
    """
    Plays a stored video file. 
    
    Detects objects in real-time using the provided model.

    @parameters:
    - conf (float): Confidence threshold for object detection.
    - model (YOLO): A YOLO object detection model.
    - vid (str): Path to video.
    - layout (str): Either Vertical or Horizontal, dictating the layout.
    """
    # make columns if the vertical option is chosen.
    if layout == "Horizontal":
        cols = st.columns(2)
        with cols[0]: st_raw_feed = st.empty()  
        with cols[1]: st_model_feed = st.empty()  
    else:
        st_raw_feed = st.empty()  
        st_model_feed = st.empty() 

    st.sidebar.subheader("Run the model:")
    if st.sidebar.button('  Run  '):
        try:
            vid_cap = cv2.VideoCapture(vid)
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(
                        conf,
                        model,
                        st_model_feed,
                        image,
                    )
                    st_raw_feed.image(
                        image,
                        channels="BGR",
                        use_column_width=True
                    )
                else:
                    vid_cap.release()
                    break
                    
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

def play_webcam_feed(conf: float, model: YOLO, layout: str)-> None:
    """
    Captures and processes a live webcam feed using YOLOv8.

    @parameters:
    - conf (float): Confidence threshold for object detection.
    - model (YOLO): A YOLO object detection model.
    - layout (str): Either Vertical or Horizontal, dictating the layout.
    """
    if layout == "Horizontal":
        cols = st.columns(2)
        with cols[0]: st_raw_feed = st.empty()  
        with cols[1]: st_model_feed = st.empty()  
    else:
        st_raw_feed = st.empty()  
        st_model_feed = st.empty() 

    # connect to webcam
    vid_cap = cv2.VideoCapture(CAM_INDEX)
    
    while st.session_state["webcam_active"]:
        success, image = vid_cap.read()
        if not success:
            continue
        _display_detected_frames(
            conf,
            model,
            st_model_feed,
            image,
        )
        st_raw_feed.image(
            image,
            channels="BGR",
            use_column_width=True
        )

    vid_cap.release()
    cv2.destroyAllWindows()


#########################################################
#   main code. gets run every time something changes.   #
#########################################################


# add option for model_state to toggle between pose and default YOLOv8
# add option for webcam state to toggle cam on or off
if "model_state" not in st.session_state and \
    "webcam_active" not in st.session_state:
    st.session_state["model_state"] = False
    st.session_state["webcam_active"] = False

st.set_page_config(
    page_title="YOLOv8 demo",
    page_icon="👩🏻‍🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Pose segmentation for presentation projects.")

# SIDEBAR
st.sidebar.header("Video Config")

# Uploaded video section (in sidebar)
with st.sidebar.container():
    st.subheader("Upload Video:")
    uploaded_video = st.file_uploader(
        label="Click here to upload your video!",
        type=["mp4", "mov"],
        help="Accepted input: `.mp4` & `.mov`"
    )

    # let you choose between horizontal or vertical if a video has been 
    # selected
    if uploaded_video is not None:
        layout = st.radio(
            "Display layout",
            ["Horizontal", "Vertical"],
        )
        # button to toggle model
        st.sidebar.subheader("Toggle model:")
        st.sidebar.button(
            label="YOLOv8" if st.session_state["model_state"] else "YOLOv8 Pose", 
            on_click=lambda: st.session_state.update({"model_state": not st.session_state["model_state"]}),
        )

# Use on_click to toggle the state in a single execution cycle
if uploaded_video is None:
    # button to toggle webcam
    st.sidebar.subheader("Or use your webcam:")
    with st.sidebar.container():
        st.sidebar.button(
            label="Stop Using Webcam" if st.session_state["webcam_active"] else "Use Webcam", 
            on_click=lambda: st.session_state.update({"webcam_active": not st.session_state["webcam_active"]})
        )

        # let you choose between horizontal or vertical if webcam is 
        # being used
        if st.session_state["webcam_active"]:
            layout = st.radio(
                "Display layout",
                ["Horizontal", "Vertical"],
            )
        
            # button to toggle model
            st.sidebar.subheader("Toggle model:")
            st.sidebar.button(
                label="YOLOv8" if st.session_state["model_state"] else "YOLOv8 Pose", 
                on_click=lambda: st.session_state.update({"model_state": not st.session_state["model_state"]}),
            )

    # Run webcam feed if active
    if st.session_state["webcam_active"]:
        play_webcam_feed(
            PREDICTION_CONFIDENCE, 
            MODELS[st.session_state["model_state"]],
            layout
        )

# if upload_video is not none, use the video feed
else:
    vid = uploaded_video.name
    # cache file so openCV can use it
    with open(vid, mode='wb') as f:
        f.write(uploaded_video.read())

    play_stored_video(
        PREDICTION_CONFIDENCE, 
        MODELS[st.session_state["model_state"]], 
        vid, 
        layout
    )

    # remove cached file
    os.remove(vid)
