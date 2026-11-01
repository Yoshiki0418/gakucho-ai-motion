"""Read-only foot slip measurements from actual MuJoCo contact states.

The physics rig is Z-up. These measurements cover contact with static world
geometry whose normal can support body weight; they do not treat foot centre
motion, foot rolling, or a reference trajectory as measured contact slip.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import mujoco
import numpy as np


def measure_foot_contact_slip(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    foot_geom_ids: Sequence[int] | None = None,
    *,
    verify_jacobian: bool = False,
    minimum_vertical_force: float = 20.,
) -> dict:
    """Return force-weighted tangential speed at left/right contact points.

    Only positive-load, near-surface contacts with static world geometry and
    abs(normal.Z) > .7 contribute. The filters match the environment's vertical
    foot force sensor. ``loaded`` uses ``minimum_vertical_force`` in Newtons.

    ``weightedSlipSpeedMetersPerSecond`` is the weighted mean of individual
    contact speed magnitudes, not the magnitude of their mean vector. Integrate
    this speed with the actual sampling interval to estimate slip path length.
    Its integral is not a foot's net displacement and needs adequate sampling.

    ``mj_objectVelocity`` supplies geom-centred angular and linear velocity in
    world axes. For a material point at each current contact, velocity is
    v_centre + omega x (contact.pos - geom centre). Setting ``verify_jacobian``
    additionally checks this velocity against an independent point Jacobian
    times qvel. All MuJoCo calls write only allocated output arrays; model and
    data are neither integrated nor edited.
    """
    if not math.isfinite(minimum_vertical_force) or minimum_vertical_force < 0:
        raise ValueError("minimum_vertical_force must be finite and nonnegative")
    if foot_geom_ids is None:
        foot_geom_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"humanoid_{side}_foot")
            for side in ("left", "right")
        ]
    if len(foot_geom_ids) != 2:
        raise ValueError("foot_geom_ids must contain left and right geom IDs")
    ids = [int(value) for value in foot_geom_ids]
    if any(value < 0 or value >= model.ngeom for value in ids) or ids[0] == ids[1]:
        raise ValueError("Foot geom IDs must be distinct existing geoms")

    result = {}
    for side, geom_id in zip(("left", "right"), ids):
        spatial = np.empty(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM,
                                geom_id, spatial, 0)
        centre = data.geom_xpos[geom_id]
        body_id = int(model.geom_bodyid[geom_id])
        contacts = []
        for index in range(data.ncon):
            contact = data.contact[index]
            pair = (int(contact.geom1), int(contact.geom2))
            if geom_id not in pair:
                continue
            other = pair[1] if pair[0] == geom_id else pair[0]
            if model.geom_bodyid[other] != 0:
                continue
            force = np.empty(6)
            mujoco.mj_contactForce(model, data, index, force)
            normal = np.asarray(contact.frame[:3])
            if abs(normal[2]) <= .7 or contact.dist > .002 or force[0] <= 1.:
                continue
            velocity = spatial[3:] + np.cross(spatial[:3], contact.pos-centre)
            tangent = velocity - np.dot(velocity, normal)*normal
            oracle_error = None
            if verify_jacobian:
                jacobian_position = np.empty((3, model.nv))
                jacobian_rotation = np.empty_like(jacobian_position)
                mujoco.mj_jac(model, data, jacobian_position, jacobian_rotation,
                              contact.pos, body_id)
                oracle_error = float(np.linalg.norm(velocity-jacobian_position @ data.qvel))
            contacts.append({
                "position": contact.pos.tolist(),
                "normal": normal.tolist(),
                "verticalForceNewtons": float(force[0]*abs(normal[2])),
                "pointVelocity": velocity.tolist(),
                "tangentialVelocity": tangent.tolist(),
                "tangentSpeedMetersPerSecond": float(np.linalg.norm(tangent)),
                "jacobianOracleErrorMetersPerSecond": oracle_error,
                "otherGeom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other),
            })
        total_force = sum(contact["verticalForceNewtons"] for contact in contacts)
        mean_speed = 0.
        mean_velocity = np.zeros(3)
        if total_force > 0:
            for contact in contacts:
                weight = contact["verticalForceNewtons"]/total_force
                mean_speed += weight*contact["tangentSpeedMetersPerSecond"]
                mean_velocity += weight*np.asarray(contact["tangentialVelocity"])
        result[side] = {
            "verticalForceNewtons": total_force,
            "loaded": total_force >= minimum_vertical_force,
            "weightedSlipSpeedMetersPerSecond": mean_speed,
            "weightedTangentialVelocity": mean_velocity.tolist(),
            "maxContactSlipSpeedMetersPerSecond": max(
                (contact["tangentSpeedMetersPerSecond"] for contact in contacts), default=0.),
            "centreVelocity": spatial[3:].tolist(),
            "angularVelocity": spatial[:3].tolist(),
            "contacts": contacts,
        }
    return result
