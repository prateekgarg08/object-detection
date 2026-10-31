# Object detection backend

A first-phase, local API that detects common objects in recorded video. It produces an annotated MP4 and a JSON Lines file containing a timestamp and bounding boxes for every decoded frame. The default model is the pretrained small YOLO26n model; it recognizes the 80 COCO classes, not every possible everyday object.

## Plan

1. Process recorded videos and evaluate detection quality and processing speed on representative footage (this phase).
2. Benchmark small models on target Android and/or iPhone hardware, including sustained FPS, latency, battery use and accuracy. Export and integrate a compatible on-device model using LiteRT or Core ML as appropriate.
3. Build mobile camera capture and draw detection overlays on the device. Keep only the newest frames when inference cannot keep up.
4. If on-device performance is insufficient, add a server streaming transport with bounded queues, timestamps, and reconnection handling. Compare latency and bandwidth on real networks.

## Run

Use Python 3.10–3.12, then:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

On first use, Ultralytics downloads `yolo26n.pt`; an internet connection is needed. Set `DETECTION_MODEL` to a local model path to use existing weights. API documentation is at http://127.0.0.1:8000/docs.

```sh
curl -F 'file=@/path/to/video.mp4' -F 'confidence=0.25' http://127.0.0.1:8000/jobs
curl http://127.0.0.1:8000/jobs/JOB_ID
curl -o annotated.mp4 http://127.0.0.1:8000/jobs/JOB_ID/video
curl -o detections.jsonl http://127.0.0.1:8000/jobs/JOB_ID/detections
```

Poll until status is `completed` or `failed`. Output has one JSON object per frame, with zero-based `frame_index`, `timestamp_seconds`, frame dimensions, and `objects`. Each object has `class_id`, `label`, `confidence`, and `bbox_xyxy` in pixels: left, top, right, bottom. Timestamps use frame index divided by reported FPS; they are approximate for variable-frame-rate footage. The MP4 uses the `mp4v` codec, whose playback support varies by client; audio is not preserved.

## Live laptop camera

With the environment installed as above, run:

```sh
source .venv/bin/activate
python -m app.camera
```

A local window displays detected objects and inference FPS. Press **Q** or **Escape** in the video window, or close it, to stop. The camera feed is processed locally and is not saved. On macOS, allow camera access for the terminal or application running Python if prompted. If the default camera does not open, close other camera apps or try `python -m app.camera --camera 1`. Use `--confidence 0.4` to reduce low-confidence detections, or `--size 416` to try faster inference. The displayed FPS measures inference only, not capture and display. Run `python -m app.camera --help` for options.

This is a local camera preview, not a network stream or an Android app. Performance depends on the computer and lighting; it may not reach real-time frame rates on every device.

## Experimental indoor obstacle detection

To try a broader indoor vocabulary that includes doors, run:

```sh
python -m app.camera --indoor-obstacles
```

This uses the medium open-vocabulary YOLOE-26 model (`yoloe-26m-seg.pt`) to draw labeled boxes for a predefined set of indoor objects. The first run downloads model weights and may install the CLIP text encoder dependency and download the MobileCLIP2 text encoder (about 254 MB); allow extra time and disk space. To change the vocabulary, edit `INDOOR_OBJECTS` in `app/camera.py`. `--model` or `DETECTION_MODEL` may override the model, but this mode requires a compatible text-promptable YOLOE or YOLO-World checkpoint. For a smaller, potentially faster model, add `--model yoloe-26s-seg.pt`. YOLOE returns segmentation masks internally; the preview displays boxes only. The list is a test target, not a promise that every object will be detected. Live camera processing happens locally once dependencies and weights are available.

This is an exploratory visual prototype, not a navigation aid: it cannot reliably identify every obstacle, determine whether an object blocks the path, or measure distance. It may miss stairs, drop-offs, glass, small objects, and objects outside the camera view. Do not rely on it as the sole means of safe navigation. Evaluate recorded indoor footage and benchmark the intended Android device before considering mobile use.

## Export the indoor model for Android

Install the optional export dependencies into the project environment with `pip install -r requirements-export.txt`. This may take several minutes and download large packages. Then run:

```sh
python - <<'PY'
from ultralytics import YOLOE
from app.camera import INDOOR_OBJECTS

model = YOLOE('yoloe-26s-seg.pt')
model.set_classes(INDOOR_OBJECTS)
print(model.export(format='litert', imgsz=640))
PY
```

The resulting `yoloe-26s-seg.tflite` has the current class names baked in. Change the vocabulary before export and export again if needed. Exporting creates a model file, not an Android app; device compatibility, camera overlays, and performance require testing on the target phone.

## Scope and limitations

This is a prototype, not a public production service: jobs and state live in one process, are lost on restart, and must use one API worker. Files persist in `storage/` until manually deleted. Jobs run serially with no queue limit, authentication, automatic cleanup, or cancellation. The upload limit defaults to 200 MiB and can be changed with `MAX_UPLOAD_BYTES`; change the data directory with `DETECTION_STORAGE`. Only serve trusted local users and test inputs. Network streaming and mobile apps are not implemented yet. Check [Ultralytics licensing](https://www.ultralytics.com/license) before distributing or commercially deploying the model or library.
