"""Audit actual flat-floor MuJoCo traces without treating survival as walking.

The input must be recorded simulated geometry/solver statistics, not reference
poses. Foot boxes use native MuJoCo half extents and wxyz world quaternions.
This audit measures sampled clearance/support; it does not certify a policy's
generalization to different rooms or reconstruct contact slip from pictures.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEFAULT_THRESHOLDS = {
    "exclude_initial_seconds": .5,
    "floor_height_meters": 0.,
    "swing_min_clearance_meters": .025,
    "unloaded_force_max_newtons": 20.,
    "swing_min_duration_seconds": .06,
    "min_swing_start_interval_seconds": .4,
    "min_swings_per_foot": 2,
    "min_alternations": 2,
    "min_forward_travel_meters": .8,
    "min_root_height_meters": .65,
    "min_upright": .8,
    "goal_distance_max_meters": .25,
    "require_other_foot_support": True,
}


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def _vector(value, count, field):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f"{field} must contain {count} numbers")
    return [_number(item, field) for item in value]


def _sole_height(geom):
    if geom.get("type") != "box":
        raise ValueError("Gait audit requires actual box foot geometry")
    position = _vector(geom.get("position"), 3, "foot.position")
    size = _vector(geom.get("size"), 3, "foot.size")
    if min(size) <= 0:
        raise ValueError("Foot half extents must be positive")
    quaternion = _vector(geom.get("quaternion"), 4, "foot.quaternion")
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm < 1e-8:
        raise ValueError("Foot quaternion cannot be zero")
    w, x, y, z = [value / norm for value in quaternion]
    # The world-Z row of the quaternion's rotation matrix. The extreme corner
    # projection accounts for foot pitch/roll, rather than centre-Z minus sz.
    row = [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
    return position[2] - sum(abs(axis) * half for axis, half in zip(row, size))


def _configuration(thresholds):
    result = dict(DEFAULT_THRESHOLDS)
    if thresholds is not None:
        if not isinstance(thresholds, dict) or set(thresholds) - set(result):
            raise ValueError("Unknown gait audit threshold")
        result.update(thresholds)
    for name, value in result.items():
        if name == "require_other_foot_support":
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")
        else:
            _number(value, name)
            if name != "floor_height_meters" and value < 0:
                raise ValueError(f"{name} cannot be negative")
    for name in ("min_swings_per_foot", "min_alternations"):
        if result[name] != int(result[name]):
            raise ValueError(f"{name} must be an integer")
    if not 0 <= result["min_upright"] <= 1:
        raise ValueError("min_upright must lie in [0, 1]")
    return result


def audit_gait(trace: dict, thresholds: dict | None = None) -> dict:
    """Return measurements and explicit gates; a failed gate is not an error."""
    config = _configuration(thresholds)
    if not isinstance(trace, dict) or trace.get("sizeConvention") != "mujoco":
        raise ValueError("Trace must declare sizeConvention='mujoco'")
    frames = trace.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError("Trace must contain at least two actual snapshots")
    dt = _number(trace.get("dt"), "trace.dt")
    if dt <= 0:
        raise ValueError("Trace dt must be positive")
    times, positions, heights, upright, falls, distances = [], [], [], [], [], []
    samples = {side: [] for side in ("left", "right")}
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict) or not isinstance(frame.get("stats"), dict):
            raise ValueError("Snapshot must contain actual solver stats")
        stats = frame["stats"]
        time = _number(frame.get("time"), "snapshot.time")
        if times and time <= times[-1]:
            raise ValueError("Snapshot times must increase strictly")
        times.append(time)
        positions.append(_vector(stats.get("root_position"), 3, "stats.root_position"))
        heights.append(_number(stats.get("root_height"), "stats.root_height"))
        upright.append(_number(stats.get("upright"), "stats.upright"))
        if not isinstance(stats.get("fall"), bool):
            raise ValueError("stats.fall must be boolean")
        falls.append(stats["fall"])
        distances.append(_number(stats.get("goal_distance"), "stats.goal_distance"))
        if distances[-1] < 0:
            raise ValueError("Goal distance cannot be negative")
        geoms = frame.get("geoms")
        if not isinstance(geoms, list):
            raise ValueError("Snapshot must contain world geometry")
        for side in samples:
            name = f"humanoid_{side}_foot"
            found = [geom for geom in geoms if geom.get("id") == name]
            if len(found) != 1:
                raise ValueError(f"Snapshot {index} must contain one {name}")
            force = _number(stats.get(f"{side}_foot_force"), f"stats.{side}_foot_force")
            samples[side].append({"clearance": _sole_height(found[0]) - config["floor_height_meters"],
                                  "force": max(force, 0.)})
    cutoff = times[0] + config["exclude_initial_seconds"]
    steady = [index for index, time in enumerate(times) if time >= cutoff - 1e-10]
    enough_samples = len(steady) >= 2
    first_steady = steady[0] if steady else len(times) - 1
    direction = [0., -1.]
    scene = trace.get("scene") or {}
    for goal in scene.get("goals", []):
        if goal.get("kind") == "walk":
            goal_position = _vector(goal.get("position"), 3, "scene.goal.position")
            delta = [goal_position[i] - positions[0][i] for i in range(2)]
            length = math.hypot(*delta)
            if length > 1e-8:
                direction = [value / length for value in delta]
            break
    delta = [positions[-1][i] - positions[first_steady][i] for i in range(3)]
    forward = sum(delta[i] * direction[i] for i in range(2))
    maximum_gap = max(dt * 1.5, 1e-6)
    sampling_complete = all(times[i] - times[i - 1] <= maximum_gap + 1e-9 for i in range(1, len(times)))
    feet = {}
    swing_events = []
    for side, other in (("left", "right"), ("right", "left")):
        events = []
        candidates = []
        raw_events = []
        exclusions = []
        start = None
        previous_candidate_start = None
        cluster_start = None
        cluster_counted = False

        def finish(end):
            nonlocal start, previous_candidate_start, cluster_start, cluster_counted
            if start is None:
                return
            duration = times[end] - times[start]
            event = {"foot": side, "start": times[start], "end": times[end], "duration": duration,
                     "maxClearanceMeters": max(samples[side][i]["clearance"] for i in range(start, end + 1)),
                     "maxVerticalForceNewtons": max(samples[side][i]["force"] for i in range(start, end + 1))}
            candidates.append(event)
            interval = None if previous_candidate_start is None else times[start]-previous_candidate_start
            if interval is None or interval > config["min_swing_start_interval_seconds"] + 1e-9:
                cluster_start = times[start]
                cluster_counted = False
            if duration <= config["swing_min_duration_seconds"] + 1e-9:
                exclusions.append({**event, "reason": "duration_below_minimum"})
            else:
                raw_events.append(event)
                if cluster_counted:
                    exclusions.append({**event, "reason": "same_foot_start_interval_too_short",
                                       "previousCandidateStart": previous_candidate_start,
                                       "candidateClusterStart": cluster_start,
                                       "startIntervalSeconds": interval})
                else:
                    events.append(event)
                    swing_events.append(event)
                    cluster_counted = True
            # Short fragments keep a flicker chain connected, but do not use
            # its one step count. Thus an initial brief toe-off does not hide
            # the first complete swing, and a longer flicker chain cannot
            # manufacture another count by mixing short and long fragments.
            previous_candidate_start = times[start]
            start = None

        for index in range(len(times)):
            valid = (index in steady and samples[side][index]["clearance"] > config["swing_min_clearance_meters"]
                     and samples[side][index]["force"] < config["unloaded_force_max_newtons"])
            if config["require_other_foot_support"]:
                valid = valid and samples[other][index]["force"] >= config["unloaded_force_max_newtons"]
            if index and times[index] - times[index - 1] > maximum_gap:
                finish(index - 1)
            if valid and start is None:
                start = index
            elif not valid:
                finish(index - 1)
        finish(len(times) - 1)
        values = [samples[side][index] for index in steady]
        feet[side] = {
            "swingCount": len(events), "swings": events,
            "rawCandidateCount": len(candidates), "rawCandidates": candidates,
            "rawSwingCount": len(raw_events), "rawSwings": raw_events,
            "excludedSwingCount": len(exclusions), "excludedSwings": exclusions,
            "maxSoleClearanceMeters": max((value["clearance"] for value in values), default=None),
            "loadedFraction": sum(value["force"] >= config["unloaded_force_max_newtons"] for value in values) / len(values) if values else None,
            "unloadedFraction": sum(value["force"] < config["unloaded_force_max_newtons"] for value in values) / len(values) if values else None,
        }
    swing_events.sort(key=lambda event: (event["start"], event["foot"]))
    sequence = [event["foot"] for event in swing_events]
    alternations = sum(left != right for left, right in zip(sequence, sequence[1:]))
    min_height = min((heights[index] for index in steady), default=None)
    min_upright = min((upright[index] for index in steady), default=None)
    gates = {
        "walkTask": trace.get("task") == "walk",
        "postInitializationSamples": enough_samples,
        "samplingComplete": sampling_complete,
        "survived": not any(falls),
        "leftRepeatedSwing": feet["left"]["swingCount"] >= config["min_swings_per_foot"],
        "rightRepeatedSwing": feet["right"]["swingCount"] >= config["min_swings_per_foot"],
        "alternatingSupport": alternations >= config["min_alternations"],
        "forwardTravel": forward > config["min_forward_travel_meters"],
        "rootHeight": min_height is not None and min_height > config["min_root_height_meters"],
        "upright": min_upright is not None and min_upright > config["min_upright"],
        "goalReached": distances[-1] < config["goal_distance_max_meters"],
    }
    return {
        "version": 2, "walkingVerified": all(gates.values()), "gates": gates,
        "failedGates": [name for name, passed in gates.items() if not passed], "thresholds": config,
        "task": trace.get("task"), "controller": trace.get("controller"),
        "durationSeconds": times[-1] - times[0], "steadyStartSeconds": times[first_steady],
        "rootDisplacementAfterInitialization": delta, "forwardTravelMeters": forward,
        "horizontalTravelMeters": math.hypot(delta[0], delta[1]), "directionXY": direction,
        "minRootHeightMeters": min_height, "minUpright": min_upright,
        "finalGoalDistanceMeters": distances[-1], "feet": feet,
        "swingSequence": sequence, "supportAlternations": alternations,
        "eventCounting": "one duration-qualified swing per same-foot candidate cluster; successive starts within the minimum interval remain one cluster; no source phase is used",
        "scope": "sampled flat-floor gait and goal gate; not an unseen-room or stairs evaluation",
        "limitations": ["requires trustworthy actual simulated snapshots and vertical solver foot forces",
                        "does not infer tangential slip from foot centre displacement",
                        "reference-start survival alone does not establish starting to walk"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--thresholds", type=Path, help="JSON threshold overrides, reported with the result")
    args = parser.parse_args()
    try:
        result = audit_gait(json.loads(args.trace.read_text()),
                            json.loads(args.thresholds.read_text()) if args.thresholds else None)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    text = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
