"""G7: a resident object detector (run with .runtime/world-model/vision-venv/bin/python, CPU).

stdin: one image path per line; stdout: one JSON line per image
{"detections": [{"coco": name, "conf": p, "box": [x0, y0, x1, y1]}, ...]}. "ready" once the model loaded.
"""
import json
import sys

from ultralytics import YOLO

model = YOLO(sys.argv[1] if len(sys.argv) > 1 else "yolo11x.pt")
print("ready", flush=True)
for line in sys.stdin:
    path = line.strip()
    if not path:
        continue
    try:
        r = model.predict(path, conf=0.3, verbose=False, device="cpu")[0]
        out = [{"coco": r.names[int(c)], "conf": round(float(p), 3), "box": [round(float(v), 1) for v in b]}
               for b, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        print(json.dumps({"detections": out}), flush=True)
    except Exception as error:   # keep serving
        print(json.dumps({"error": str(error)}), flush=True)
