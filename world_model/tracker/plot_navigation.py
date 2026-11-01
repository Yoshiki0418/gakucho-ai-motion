"""Top-down plots of navigation episodes: walls, furniture, route, reference
root path and the simulated root path, with the outcome in each title."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from ..navigation_scenes import generate_navigation_scene
from ..scenes import box_corners


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, help="navigation_eval output directory")
    parser.add_argument("--columns", type=int, default=8)
    args = parser.parse_args()
    root = Path(args.result)
    data = json.loads((root / "summary.json").read_text())
    episodes = data["episodes"]
    rows = int(np.ceil(len(episodes) / args.columns))
    fig, axes = plt.subplots(rows, args.columns, figsize=(2.4 * args.columns, 3.2 * rows), squeeze=False)
    for ax, ep in zip(axes.ravel(), episodes):
        scene = generate_navigation_scene(ep["sceneSeed"], mode="obstacle", difficulty=ep["difficulty"])
        for g in scene["geoms"]:
            if g.get("semantic") == "floor":
                continue
            color = "#8a6d4a" if g.get("entityId") == "navigation-furniture" else "#999999"
            ax.add_patch(Polygon(box_corners(g), closed=True, color=color))
        trace = np.load(root / "trajectories" / f"scene-{ep['sceneSeed']}.npz")
        ax.plot(trace["route"][:, 0], trace["route"][:, 1], ":", color="#2b6cb0", lw=1)
        ax.plot(trace["reference"][:, 0], trace["reference"][:, 1], "-", color="#90cdf4", lw=1)
        ax.plot(trace["qpos"][:, 0], trace["qpos"][:, 1], "-", color="#c53030", lw=1.4)
        goal = scene["goals"][0]["position"]
        ax.plot(*goal[:2], "g*", ms=9)
        outcome = ("success" if ep["success"] else "fell" if ep["fellAt"] else
                   f"hit {ep['firstContact']['obstacle']}" if ep["firstContact"] else "no stop")
        ax.set_title(f"{ep['sceneSeed']}\n{outcome}", fontsize=8)
        ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
    for ax in axes.ravel()[len(episodes):]:
        ax.axis("off")
    fig.suptitle("red: simulated root   light blue: reference root   dotted: planned route", fontsize=9)
    fig.tight_layout()
    fig.savefig(root / "overview.png", dpi=110)
    print(root / "overview.png")


if __name__ == "__main__":
    main()
