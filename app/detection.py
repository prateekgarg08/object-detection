"""Frame detection and recorded-video processing."""
import json
import os
from pathlib import Path


def process_video(source: Path, output: Path, detections: Path, confidence: float) -> None:
    # Lazy imports keep the API responsive while the model loads in the worker.
    import cv2
    from ultralytics import YOLO

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("Could not open video. Check that it is a supported video file.")
    writer = None
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not (0 < fps < 1000 and width > 0 and height > 0):
            raise ValueError("Video has invalid frame rate or dimensions.")
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Could not create annotated video with the MP4 codec.")
        model = YOLO(os.getenv("DETECTION_MODEL", "yolo26n.pt"))
        count = 0
        with detections.open("w", encoding="utf-8") as result_file:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = model.predict(frame, conf=confidence, verbose=False)[0]
                objects = []
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    objects.append({
                        "class_id": class_id,
                        "label": result.names[class_id],
                        "confidence": round(float(box.conf[0].item()), 5),
                        "bbox_xyxy": [round(float(v), 2) for v in box.xyxy[0].tolist()],
                    })
                # Frame index / FPS is an approximation for variable-frame-rate input.
                result_file.write(json.dumps({"frame_index": count, "timestamp_seconds": round(count / fps, 6), "width": width, "height": height, "objects": objects}) + "\n")
                writer.write(result.plot())
                count += 1
        if count == 0:
            raise ValueError("Video contains no readable frames.")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
