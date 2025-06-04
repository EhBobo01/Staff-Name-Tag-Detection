import os
import streamlit as st
import tempfile
import cv2

st.sidebar.title("Name Tag Detection in Video")

def log_detections(frame_num, box, detection_log):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    label = "name_tag"
    detection_log.append({
        "Frame": frame_num,
        "Label": label,
        "Coordinates": f"({x1}, {y1}, {x2}, {y2})"
    })

uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_file:
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)  # Preview original video

    if st.button("Run Name Tag Detection"):
        st.write("Starting detection process...")

        try:
            from ultralytics import YOLO
        except Exception as e:
            st.error(f"Failed to import YOLO model: {e}")
            st.stop()

        try:
            model = YOLO("model/best.pt")
            st.write("Model loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load model weights: {e}")
            st.stop()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Could not open the video file.")
            st.stop()

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:
            st.warning("FPS info missing or zero; defaulting to 24 FPS.")
            fps = 24

        st.write(f"Video properties: width={width}, height={height}, fps={fps}")

        # Output temporary path (OpenCV output)
        output_temp_path = os.path.join(tempfile.gettempdir(), "output_temp.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_temp_path, fourcc, fps, (width, height))

        frame_count = 0
        detection_count = 0
        detection_log = []

        # Create a placeholder for updating progress message
        progress_text = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                progress_text.text("Finish detecting staff name tag in the video.")
                break

            results = model(frame)[0]

            for box in results.boxes:
                if int(box.cls) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "name_tag", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    detection_count += 1
                    log_detections(frame_count, box, detection_log)

            out.write(frame)
            frame_count += 1

            # Update progress line dynamically
            progress_text.text(f"Processing frame: {frame_count}")

        cap.release()
        out.release()

        # Directly use the OpenCV output video without ffmpeg conversion
        if os.path.exists(output_temp_path):
            st.success(f"Detection complete! Processed {frame_count} frames with {detection_count} detections.")
            st.video(output_temp_path)
            with open(output_temp_path, "rb") as f:
                st.download_button("Download Annotated Video", f, file_name="annotated_output.mp4", mime="video/mp4")

            # Show detection log as a dataframe
            if detection_log:
                st.subheader("Detection Log (Frame, Label, Coordinates)")
                st.table(detection_log)
            else:
                st.info("No detections found in the video.")
        else:
            st.error("Annotated video not found.")

else:
    st.info("Please upload a video to begin.")
