"""Generate MuJoCo FK calibration poses for the actual-GLB browser check.

These are deliberately specified static calibration poses, not action rollouts
or task success evidence. Real body transforms come from MuJoCo mj_forward.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from world_model.environment import HumanoidRoomEnv
from world_model.humanoid import JOINT_SPECS


def main():
    profile = ROOT / "frontend/public/world-model/gakucho-avatar-profile.json"
    scene = {"geoms": [
        {"id": "floor", "type": "box", "position": [0, 0, -.1], "size": [10, 10, .2]},
        {"id": "calibration-seat", "type": "box", "position": [0, .15, .413916], "size": [.6, .6, .06]},
    ]}
    env = HumanoidRoomEnv(avatar_profile=profile, randomize=False, scene=scene)
    cases = []
    names = [joint[0] for joint in JOINT_SPECS]
    for name, yaw, translation, bent in (
        ("rest", 0, [0, 0, 0], False),
        ("yaw", 1.2, [1.3, -.7, 0], False),
        ("bent", -.85, [-.3, .4, -.3], True),
        ("seated", 0, [0, 0, .443916 + env.seat_pelvis_offset - env.model.qpos0[2]], True),
    ):
        env.data.qpos[:] = env.model.qpos0
        env.data.qvel[:] = 0
        env.data.qpos[:3] += translation
        env.data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        if bent:
            for side in ("left", "right"):
                angles = (("hip_pitch", -np.pi / 2), ("knee", np.pi / 2)) if name == "seated" else (
                    ("hip_pitch", -1.0), ("knee", 1.65), ("ankle_pitch", -.55),
                    ("shoulder_pitch", .2), ("elbow", .6))
                for suffix, angle in angles:
                    env.data.qpos[env._qpos_indices[names.index(f"{side}_{suffix}")]] = angle
            if name != "seated":
                env.data.qpos[env._qpos_indices[names.index("torso_pitch")]] = .15
        mujoco.mj_forward(env.model, env.data)
        cases.append({"name": name, "frame": env.snapshot()})
    output = ROOT / ".runtime/world-model/verification/avatar-retarget-cases.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "purpose": "static-calibration-not-simulated-action",
        "profile": env.avatar_profile, "restBodies": env.physics_rest_bodies,
        "bodyModelSha256": env.body_model_sha256,
        "avatarProfileSha256": env.avatar_profile_sha256, "cases": cases,
    }))
    env.close()
    print(json.dumps({"path": str(output), "caseCount": len(cases)}))


if __name__ == "__main__":
    main()
