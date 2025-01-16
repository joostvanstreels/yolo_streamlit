# YOLO streamlist demo environment

This demo environment has been made by Joris Heemskerk & Joost Vanstreels.\
It serves as a cool demo for any open days at the Hogeschool Utrecht.


## Setup guide:

1. New Conda environment.\
    Make a new Conda environment in any terminal using:
    ```shell
    conda create -n "open_dag" python=3.9.7
    ```
    What exactly you name this environment does not matter.

2. Activate conda environment with:
    ```shell
    conda activate open_dag
    ```

3. Install the requirements using:
    ```shell
    pip install -r requirements.txt
    ```

4. Run the application using:
    ```shell
    streamlit run main.py
    ```

## configuration details:

The following gonfiguration options are available at the top of [main.py](main.py).
```py
PREDICTION_CONFIDENCE = 0.5 
CAM_INDEX = 0 
MODELS = (YOLO("yolov8n-pose.pt"), YOLO("yolov8n.pt"))
```

- The `PREDICTION_CONFIDENCE` dictates above which confidence the predictions get shown.
- The `CAM_INDEX` is the index of the webcam used. When using your main laptop camera, the index is 0. If you use an external webcam, use index 1.
- The `MODELS` variable contains a list of models. The first model should always be the pose variant. The second model should be a default model. For the sake of runtime, the smallest models are picked (the nano models). If your pc supports it, you can use larger models by replacing the `n`'s with one of: `[n, s, m, l, x]`. Currently it is using yolov8. If replaced with 'yolo11' it can also run yolo 11.

