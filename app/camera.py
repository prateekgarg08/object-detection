"""Run local real-time object detection from a laptop camera."""
import argparse
import os
import sys
import time

INDOOR_OBJECTS = [
    "door", "chair", "table", "sofa", "bed", "cabinet", "shelf", "person",
    "backpack", "suitcase", "box", "trash can", "potted plant", "bicycle",
    "staircase", "step", "fire extinguisher"
]


def confidence_value(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("confidence must be a number") from exc
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("confidence must be greater than 0 and at most 1")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect objects from a local camera; press q or Escape to quit.")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--confidence", type=confidence_value, default=0.25)
    parser.add_argument("--model", default=None, help="model checkpoint (overrides DETECTION_MODEL)")
    parser.add_argument("--indoor-obstacles", action="store_true", help="use experimental open-vocabulary indoor object detection")
    parser.add_argument("--size", type=int, default=640, help="model input size (default: 640)")
    args = parser.parse_args()
    if args.camera < 0 or args.size < 32:
        parser.error("camera index must be non-negative and size must be at least 32")

    import cv2
    from ultralytics import YOLO

    try:
        model_path = args.model or os.getenv("DETECTION_MODEL") or (
            "yoloe-26s-seg.pt" if args.indoor_obstacles else "yolo26n.pt"
        )
        model = YOLO(model_path)
        if args.indoor_obstacles:
            if not callable(getattr(model, "set_classes", None)):
                raise ValueError("Indoor obstacle mode requires a text-promptable YOLOE or YOLO-World checkpoint")
            model.set_classes(INDOOR_OBJECTS)
    except Exception as exc:
        print(f"Could not load model: {exc}", file=sys.stderr)
        return 1

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        capture.release()
        print(f"Could not open camera {args.camera}. Check camera permissions and whether another app is using it.", file=sys.stderr)
        return 1

    window = "Live object detection - press Q or Escape to quit"
    print("Camera started. Press Q or Escape in the video window to quit.", flush=True)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Could not read a frame from the camera.", file=sys.stderr)
                return 1
            started = time.perf_counter()
            result = model.predict(frame, conf=args.confidence, imgsz=args.size, verbose=False)[0]
            annotated = result.plot(masks=False)
            fps = 1 / max(time.perf_counter() - started, 1e-9)
            cv2.putText(annotated, f"Inference: {fps:.1f} FPS", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(window, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Camera detection failed: {exc}", file=sys.stderr)
        return 1
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
