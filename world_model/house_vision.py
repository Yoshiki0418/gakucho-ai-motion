"""G7-0: can the eyes see the house? Head-camera views of every place in eight directions, the
ground truth of what each view shows, and how much of it a detector (YOLO, COCO classes) finds.

    python -m world_model.house_vision --output .runtime/world-model/house2f/vision [--model yolo11x.pt]

Views: every walk place of ``house2f`` at the standing eye height (head origin 1.612 m above the
floor), 15 degrees down, 90 degrees across, 640 x 480 (the depth camera's pose and field of view),
headings every 45 degrees. Rendering: tools/house_render.mjs (/world-model/house in headless Chrome).
Ground truth: an object of a COCO class is in a view when its box projects into the image (>= 400 px
after clipping) and a ray from the eye to its centre or one of 8 inner points is not stopped earlier
by the house's physics boxes. Detection: ultralytics YOLO in .runtime/world-model/vision-venv (CPU).
A ground-truth object is found when a detection of its class overlaps its projected box (IoU >= 0.25)
or contains its projected centre; detections matching nothing listed are reported separately (the
production LDK also has cups, plates and books that the object list does not name).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np

from .house2f import FLOOR_2F, create_house
from .tracker.navigation_eval import scene_elements

ROOT = Path(__file__).resolve().parents[1]
EYE_HEIGHT, PITCH, HFOV, WIDTH, HEIGHT = 1.612, 15., 90., 640, 480
VISION_PYTHON = ROOT / ".runtime/world-model/vision-venv/bin/python"


def camera(eye, heading, pitch=PITCH):
    p = math.radians(pitch)
    forward = np.array([math.cos(p) * math.sin(heading), -math.cos(p) * math.cos(heading), -math.sin(p)])
    right = np.array([-math.cos(heading), -math.sin(heading), 0.])
    return np.asarray(eye, float), right, np.cross(right, forward), forward


def project(points, cam):
    eye, right, up, forward = cam
    focal = WIDTH / 2 / math.tan(math.radians(HFOV) / 2)
    d = np.asarray(points, float) - eye
    z = d @ forward
    with np.errstate(divide="ignore", invalid="ignore"):
        u = WIDTH / 2 + (d @ right) / z * focal
        v = HEIGHT / 2 - (d @ up) / z * focal
    return u, v, z


def corners(o, shrink=1.):
    w, d, h = (np.asarray(o["size"]) / 2) * shrink
    c, s = math.cos(o["yaw"]), math.sin(o["yaw"])
    out = []
    for dx in (-w, w):
        for dy in (-d, d):
            for dz in (-h, h):
                out.append([o["position"][0] + c * dx - s * dy, o["position"][1] + s * dx + c * dy, o["position"][2] + dz])
    return np.asarray(out)


def world_model(scene):
    root = ET.Element("mujoco")
    body = ET.SubElement(root, "worldbody")
    for element in scene_elements(scene):
        body.append(element)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def ground_truth(scene, cam, model, data):
    """Objects of a COCO class this view shows: projected box (clipped) and whether a ray reaches them."""
    eye = cam[0]
    names = {model.geom(g).name: g for g in range(model.ngeom)}
    out = []
    for o in scene["objects"]:
        if not o["coco"]:
            continue
        u, v, z = project(corners(o), cam)
        cu, cv, cz = project([o["position"]], cam)
        if cz[0] < 0.2 or np.linalg.norm(np.asarray(o["position"]) - eye) > 8.:
            continue
        front = z > 0.05
        if not front.any():
            continue
        full = [u[front].min(), v[front].min(), u[front].max(), v[front].max()]
        box = [max(0., full[0]), max(0., full[1]), min(WIDTH, full[2]), min(HEIGHT, full[3])]
        if box[2] - box[0] < 1 or box[3] - box[1] < 1 or (box[2] - box[0]) * (box[3] - box[1]) < 400:
            continue
        # share of the object's projected box inside the frame (an object partly behind the camera counts as cut)
        inside = (box[2] - box[0]) * (box[3] - box[1]) / max((full[2] - full[0]) * (full[3] - full[1]), 1e-6)
        inside = inside if front.all() else min(inside, 0.4)
        own = names.get(o["id"], -2)
        seen = False
        for point in [np.asarray(o["position"]), *corners(o, 0.6)]:
            vec = point - eye
            dist = float(np.linalg.norm(vec))
            geom = np.zeros(1, np.int32)
            hit = mujoco.mj_ray(model, data, eye, vec / dist, None, 1, -1, geom)
            if hit < 0 or hit >= dist - 0.08 or geom[0] == own:
                pu, pv, pz = project([point], cam)
                if pz[0] > 0.05 and 0 <= pu[0] < WIDTH and 0 <= pv[0] < HEIGHT:
                    seen = True
                    break
        if seen:
            out.append({"id": o["id"], "coco": o["coco"], "label": o["label"], "room": o["room"], "box": [round(b, 1) for b in box],
                        "centre": [round(float(cu[0]), 1), round(float(cv[0]), 1)], "distance": round(float(cz[0]), 2),
                        "inside": round(float(inside), 2)})
    return out


DETECT = r'''
import json, sys
from ultralytics import YOLO
images, weights, out = json.loads(sys.argv[1]), sys.argv[2], sys.argv[3]
model = YOLO(weights)
result = {}
for path in images:
    r = model.predict(path, conf=0.25, verbose=False, device="cpu")[0]
    result[path] = [{"coco": r.names[int(c)], "conf": round(float(p), 3), "box": [round(float(v), 1) for v in b]}
                    for b, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
json.dump(result, open(out, "w"))
'''


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x1 - x0) * max(0., y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="yolo11x.pt")
    parser.add_argument("--base", default="http://127.0.0.1:4010")
    parser.add_argument("--headings", type=int, default=8)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    scene = create_house()
    model, data = world_model(scene)
    views = []
    for goal in scene["goals"]:
        if goal["kind"] != "walk":
            continue
        floor_z = FLOOR_2F if goal["position"][2] > 1 else 0.
        for k in range(args.headings):
            heading = 2 * math.pi * k / args.headings
            eye = [goal["position"][0], goal["position"][1], floor_z + EYE_HEIGHT]
            name = f"{goal['id']}-{round(math.degrees(heading)):03d}"
            views.append({"name": name, "place": goal["id"], "room": goal["room"], "x": eye[0], "y": eye[1], "z": eye[2],
                          "heading": heading, "pitch": PITCH, "width": WIDTH, "height": HEIGHT, "hfov": HFOV,
                          "truth": ground_truth(scene, camera(eye, heading), model, data)})
    (out / "views.json").write_text(json.dumps(views, ensure_ascii=False, indent=1))
    missing = [v for v in views if not (images / f"{v['name']}.jpg").exists()]
    if missing:
        (out / "render.json").write_text(json.dumps([{k: v[k] for k in ("name", "x", "y", "z", "heading", "pitch", "width",
                                                                          "height", "hfov")} for v in missing]))
        print(subprocess.run(["node", str(ROOT / "tools/house_render.mjs"), str(out / "render.json"), str(images), args.base],
                             check=True, capture_output=True, text=True).stdout.strip())
    detections_path = out / f"detections-{Path(args.model).stem}.json"
    if not detections_path.exists():
        subprocess.run([str(VISION_PYTHON), "-c", DETECT, json.dumps([str(images / f"{v['name']}.jpg") for v in views]),
                        args.model, str(detections_path)], check=True, cwd=out)
    detections = json.loads(detections_path.read_text())

    found, total, extra = defaultdict(int), defaultdict(int), defaultdict(int)
    whole_found, whole_total = defaultdict(int), defaultdict(int)          # at least half of it in the frame, >= 1500 px
    place_seen, place_found = defaultdict(set), defaultdict(set)           # looking round at a place (8 views)
    per_view = []
    for v in views:
        dets = detections[str(images / f"{v['name']}.jpg")]
        used = set()
        hits = []
        for t in v["truth"]:
            total[t["coco"]] += 1
            match = next((i for i, d in enumerate(dets) if i not in used and d["coco"] == t["coco"] and
                          (iou(d["box"], t["box"]) >= 0.25 or (d["box"][0] <= t["centre"][0] <= d["box"][2] and
                                                                d["box"][1] <= t["centre"][1] <= d["box"][3]))), None)
            whole = bool(t["inside"] >= 0.5 and (t["box"][2] - t["box"][0]) * (t["box"][3] - t["box"][1]) >= 1500)
            whole_total[t["coco"]] += whole
            if whole:
                place_seen[v["place"]].add((t["id"], t["coco"]))
            if match is not None:
                used.add(match)
                found[t["coco"]] += 1
                whole_found[t["coco"]] += whole
                place_found[v["place"]].add((t["id"], t["coco"]))
                hits.append(t["id"])
        for i, d in enumerate(dets):
            if i not in used:
                extra[d["coco"]] += 1
        per_view.append({"name": v["name"], "truth": len(v["truth"]), "found": len(hits), "detections": len(dets)})
    classes = sorted(total, key=lambda c: -total[c])
    looked = defaultdict(lambda: [0, 0])
    for place, seen in place_seen.items():
        for item in seen:
            looked[item[1]][1] += 1
            looked[item[1]][0] += item in place_found[place]
    report = {"model": args.model, "views": len(views),
              "recall": {"found": sum(found.values()), "visible": sum(total.values()),
                         "rate": round(sum(found.values()) / max(sum(total.values()), 1), 3)},
              "recallWhole": {"found": sum(whole_found.values()), "visible": sum(whole_total.values()),
                              "rate": round(sum(whole_found.values()) / max(sum(whole_total.values()), 1), 3),
                              "byClass": {c: [whole_found[c], whole_total[c]] for c in classes if whole_total[c]}},
              "lookAround": {"found": sum(v[0] for v in looked.values()), "visible": sum(v[1] for v in looked.values()),
                             "rate": round(sum(v[0] for v in looked.values()) / max(sum(v[1] for v in looked.values()), 1), 3),
                             "byClass": {c: looked[c] for c in sorted(looked, key=lambda c: -looked[c][1])}},
              "byClass": {c: {"found": found[c], "visible": total[c], "rate": round(found[c] / total[c], 2)} for c in classes},
              "unlistedDetections": dict(sorted(extra.items(), key=lambda kv: -kv[1])), "perView": per_view}
    (out / f"report-{Path(args.model).stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: report[k] for k in ("model", "views", "recall", "recallWhole", "lookAround")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
