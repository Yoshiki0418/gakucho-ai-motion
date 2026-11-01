"""Independent checks against the actual suit asset and coordinate contracts."""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from world_model.avatar_profile import load_avatar_profile


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "frontend/public/world-model/gakucho-avatar-profile.json"
ASSET_PATH = ROOT / "frontend/public/avatars/gakucho_v3_suit.glb"


def _asset():
    raw = ASSET_PATH.read_bytes()
    assert raw[:4] == b"glTF"
    json_length, chunk_type = struct.unpack_from("<II", raw, 12)
    assert chunk_type == 0x4E4F534A
    return raw, json.loads(raw[20:20 + json_length])


def _raw_world_positions(gltf):
    """Compute unmodified glTF node transforms without Three or the exporter."""
    nodes = gltf["nodes"]
    parents = {child: index for index, node in enumerate(nodes)
               for child in node.get("children", [])}
    cache = {}

    def world(index):
        if index in cache:
            return cache[index]
        node = nodes[index]
        if "matrix" in node:
            local = np.asarray(node["matrix"]).reshape(4, 4).T
        else:
            local = np.eye(4)
            local[:3, :3] = Rotation.from_quat(node.get("rotation", [0, 0, 0, 1])).as_matrix()
            local[:3, :3] @= np.diag(node.get("scale", [1, 1, 1]))
            local[:3, 3] = node.get("translation", [0, 0, 0])
        cache[index] = world(parents[index]) @ local if index in parents else local
        return cache[index]

    return {nodes[index].get("name"): world(index)[:3, 3]
            for index in gltf["skins"][0]["joints"]}


def test_measurement_provenance_is_actual_suit_and_current_rig():
    profile = load_avatar_profile(PROFILE_PATH)
    raw, _ = _asset()
    digest = hashlib.sha256(raw).hexdigest()
    assert profile["provenance"]["modelSha256"] == digest
    assert profile["display"]["modelSha256"] == digest
    assert profile["provenance"]["modelBytes"] == len(raw)
    for filename, expected in profile["provenance"]["sourceSha256"].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected
    assert profile["provenance"]["measurementScriptSha256"] == hashlib.sha256(
        (ROOT / profile["provenance"]["measurementScript"]).read_bytes()).hexdigest()
    assert profile["calibration"]["massWasMeasured"] is False
    assert profile["calibration"]["jointStrengthWasMeasured"] is False


def test_limb_lengths_equal_original_glb_lengths_after_display_scaling():
    profile = load_avatar_profile(PROFILE_PATH)
    _, gltf = _asset()
    raw_positions = _raw_world_positions(gltf)
    scale = profile["display"]["scale"]
    names = profile["display"]["joints"]
    for side in ("left", "right"):
        for section, key, start, finish in (
            ("legs", "thighLength", "UpLeg", "LowerLeg"),
            ("legs", "shankLength", "LowerLeg", "Foot"),
            ("arms", "upperArmLength", "UpperArm", "LowerArm"),
            ("arms", "forearmLength", "LowerArm", "Hand"),
        ):
            start_name = names[side + start]["boneName"]
            finish_name = names[side + finish]["boneName"]
            expected = np.linalg.norm(raw_positions[finish_name] - raw_positions[start_name]) * scale
            # The GLB stores rounded, slightly non-unit float quaternions;
            # the existing rest alignment preserves them through hierarchy
            # multiplication. Its largest limb difference is 3.7 micrometres.
            np.testing.assert_allclose(profile["physical"][section][side][key], expected,
                                       rtol=0, atol=5e-6)


def test_physics_positions_are_a_right_handed_conversion_of_display():
    profile = load_avatar_profile(PROFILE_PATH)
    basis = np.asarray([[1., 0, 0], [0, 0, -1.], [0, 1., 0]])
    np.testing.assert_allclose(basis.T @ basis, np.eye(3))
    assert np.linalg.det(basis) == 1
    assert len(profile["display"]["joints"]) >= 20
    for name, joint in profile["display"]["joints"].items():
        np.testing.assert_allclose(profile["physical"]["joints"][name]["position"],
                                   basis @ joint["position"], atol=1e-12)
        for key in ("worldQuaternion", "parentWorldQuaternion"):
            np.testing.assert_allclose(np.linalg.norm(joint[key]), 1, rtol=0, atol=2e-6)
        # Store the existing renderer's aligned local rotations exactly. Deep
        # finger chains contain accumulated non-unit rest values; these are
        # display data and do not define any physical hinge or limb length.
        local = np.asarray(joint["localQuaternion"])
        assert np.isfinite(local).all() and .99 < np.linalg.norm(local) < 1.01
    np.testing.assert_allclose(profile["physical"]["joints"]["hips"]["position"][:2], 0,
                               atol=1e-12)


def test_collision_foot_boxes_reproduce_measured_shoe_bounds_and_floor():
    profile = load_avatar_profile(PROFILE_PATH)
    physical = profile["physical"]
    for side in ("left", "right"):
        foot = physical["foot"][side]
        assert foot["measurement"]["selectedVertices"] >= 50
        ankle = np.asarray(physical["joints"][side + "Foot"]["position"])
        center = ankle + foot["centerOffset"]
        lo = center - foot["halfSize"]
        hi = center + foot["halfSize"]
        bounds = foot["measurement"]["boundsYUp"]
        np.testing.assert_allclose(lo, [bounds["min"][0], -bounds["max"][2], bounds["min"][1]],
                                   atol=1e-12)
        np.testing.assert_allclose(hi, [bounds["max"][0], -bounds["min"][2], bounds["max"][1]],
                                   atol=1e-12)
        assert 0 <= lo[2] < .002
        closure = (physical["rootHeight"] + physical["hipOffsets"][side][2]
                   - sum(physical["legs"][side].values())
                   + foot["centerOffset"][2] - foot["halfSize"][2])
        np.testing.assert_allclose(closure, lo[2], atol=1e-8)


def test_head_contact_reaches_the_actual_rendered_head_top():
    profile = load_avatar_profile(PROFILE_PATH)
    head = profile["physical"]["head"]
    height = profile["physical"]["joints"]["head"]["position"][2]
    measured_top = profile["calibration"]["wholeMeshBoundsYUp"]["max"][1]
    np.testing.assert_allclose(height + head["centerOffset"][2] + head["radius"],
                               measured_top, atol=1e-10)
    assert .08 < head["radius"] < .18
    assert head["measurement"]["selectedVertices"] >= 50
    np.testing.assert_allclose(profile["physical"]["referenceHeight"], 1.8)


def test_pelvis_seat_contact_matches_the_deformed_avatar_in_seated_pose():
    profile = load_avatar_profile(PROFILE_PATH)
    pelvis = profile["physical"]["pelvis"]
    evidence = profile["calibration"]["pelvisSkin"]["seatedCalibration"]
    np.testing.assert_allclose(pelvis["centerOffset"][2] - pelvis["radius"],
                               evidence["rootRelativeBottom"], atol=1e-12)
    np.testing.assert_allclose(pelvis["seatOffset"], -evidence["rootRelativeBottom"],
                               atol=1e-12)
    bounds = evidence["boundsYUp"]
    np.testing.assert_allclose(
        2 * (pelvis["axisHalfLength"] + pelvis["radius"]),
        bounds["max"][0] - bounds["min"][0], atol=1e-12)
    np.testing.assert_allclose(2 * pelvis["radius"],
                               bounds["max"][2] - bounds["min"][2], atol=1e-12)
    assert pelvis["axisHalfLength"] > .01
    assert pelvis["measurement"]["massWasMeasured"] is False
