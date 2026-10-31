import cv2
from ultralytics import FastSAM

model = FastSAM("FastSAM-s.pt")
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open camera. Check camera permissions.")

try:
    while True:
        ok, frame = camera.read()
        if not ok:
            break

        result = model(frame, imgsz=640, conf=0.4, verbose=False)[0]
        cv2.imshow("FastSAM - press Q to quit", result.plot())

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()
