"""GPU-parallel motion-tracking environment (MuJoCo Warp through MJX).

The policy receives the current physical state and the next reference frames
and outputs bounded residuals added to the reference joint angles; PD
actuators turn those targets into torques. The floating root is never set
after reset: the body moves only by gravity, contacts and joint torques.

Episodes start at a random frame of a random clip with the reference state
(reference state initialisation). They end when the mean body position error
exceeds ``termination_distance`` (failure), or are truncated at the clip end or
time limit (not a failure; PPO bootstraps through truncation).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.wrapper import Wrapper
from brax.envs.wrappers import training as brax_training

from . import body as B


def default_config():
    return config_dict.create(
        ctrl_dt=B.TIMESTEP * B.CONTROL_SUBSTEPS,
        sim_dt=B.TIMESTEP,
        episode_length=300,
        action_scale=.5,
        termination_distance=.5,
        future_frames=(1, 10),
        solver_iterations=6,
        solver_ls_iterations=10,
        impl="warp",
        naconmax=64 * 8192,
        njmax=256,
        hinge_init_noise=.02,
        reward=config_dict.create(
            position=.5, rotation=.3, velocity=.1, angular_velocity=.1,
            position_k=100., rotation_k=10., velocity_k=.1, angular_velocity_k=.1,
            # Energy/smoothness penalties made the training reward negative
            # under exploration noise (episodes then shortened); kept at 0.
            power=0., action_rate=0.),
        include_seated=False,
        # stage-4 steps: platform, three treads and a landing as repositionable boxes (v11+)
        include_terrain=False,
        split="train",
        # JSON {"weights": {clip path: weight}} from sweep_clips; empty = uniform over clips
        clip_weights="",
    )


# ---------------------------------------------------------------- quaternions (w, x, y, z)
def quat_mul(a, b):
    aw, ax, ay, az = jp.moveaxis(a, -1, 0)
    bw, bx, by, bz = jp.moveaxis(b, -1, 0)
    return jp.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def quat_conj(q):
    return q * jp.asarray([1., -1., -1., -1.])


def quat_rotate(q, v):
    qv = jp.concatenate([jp.zeros(v.shape[:-1] + (1,)), v], axis=-1)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[..., 1:]


def quat_to_6d(q):
    """First two columns of the rotation matrix (continuous representation)."""
    w, x, y, z = jp.moveaxis(q, -1, 0)
    c0 = jp.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], axis=-1)
    c1 = jp.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], axis=-1)
    return jp.concatenate([c0, c1], axis=-1)


def quat_angle(q):
    return 2. * jp.arctan2(jp.linalg.norm(q[..., 1:], axis=-1), jp.abs(q[..., 0]))


def quat_log(q):
    """Rotation vector of a unit quaternion, taking the shorter rotation."""
    q = jp.where(q[..., :1] < 0, -q, q)
    norm = jp.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    angle = 2. * jp.arctan2(norm, q[..., :1])
    return q[..., 1:] * jp.where(norm > 1e-8, angle / jp.maximum(norm, 1e-8), 2.)


def heading_quat_inv(root_quat):
    """Rotation about Z that removes the root heading (pelvis X axis yaw)."""
    w, x, y, z = root_quat
    yaw = jp.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))
    return jp.asarray([jp.cos(-yaw / 2), 0., 0., jp.sin(-yaw / 2)])


# ---------------------------------------------------------------- motion library
def load_clips(library_dir, *, split="train", include_seated=False, include_terrain=False, keys=("qpos",)):
    manifest = json.loads((Path(library_dir) / "manifest.json").read_text())
    clips = [c for c in manifest["clips"] if c["status"] == "ok"
             and (split == "all" or c["split"] == split)
             and (include_seated or not c["needsSeat"])
             and (include_terrain or not c.get("terrain"))]
    seats_path = Path(library_dir) / "seats.json"
    seats = json.loads(seats_path.read_text())["seats"] if seats_path.exists() else {}
    for clip in clips:
        clip["seat"] = seats.get(clip["path"]) if clip["needsSeat"] else None
    data = np.load(Path(library_dir) / "library.npz")
    return manifest, clips, {k: data[k] for k in keys}


SEAT_PARKED = (1000., 1000., -5.)


def seat_xml():
    """A static, repositionable seat box (mocap body) that is parked far away
    for clips without a chair. Its top is set per clip from ``seats.json``."""
    from xml.etree import ElementTree as ET
    from .seats import SEAT_HALF
    seat = ET.Element("body", name="seat", mocap="true", pos=" ".join(map(str, SEAT_PARKED)))
    ET.SubElement(seat, "geom", name="seat", type="box", size=" ".join(map(str, SEAT_HALF)),
                  contype=str(B.WORLD_CONTYPE), conaffinity=str(B.WORLD_CONAFFINITY),
                  rgba="0.55 0.42 0.30 1")
    return seat


# Terrain slots (kind, box half sizes): the shapes of the stage-4 low step and short stairs, and the
# 15 treads of the house staircase (house_stairs.py; its 2F floor uses the landing slot).
# Boxes are 1 m tall and placed by their top; they reach below the floor, as the scene's do.
TERRAIN_BOXES = (("platform", (1.3, .9, .5)), ("tread", (.9, .19, .5)), ("tread", (.9, .19, .5)),
                 ("tread", (.9, .19, .5)), ("landing", (1.6, .8, .5))) + (("house-tread", (.125, .45, .5)),) * 15


def _terrain_parked(i):
    return (1000. + 10. * (i + 1), 1000., -5.)


def terrain_xml():
    """Repositionable terrain boxes (mocap bodies), parked far away for flat clips."""
    from xml.etree import ElementTree as ET
    bodies = []
    for i, (kind, half) in enumerate(TERRAIN_BOXES):
        body = ET.Element("body", name=f"terrain-{i}", mocap="true", pos=" ".join(map(str, _terrain_parked(i))))
        ET.SubElement(body, "geom", name=f"terrain-{i}-{kind}", type="box", size=" ".join(map(str, half)),
                      contype=str(B.WORLD_CONTYPE), conaffinity=str(B.WORLD_CONAFFINITY), friction="1.0 0.005 0.0001",
                      rgba="0.6 0.6 0.55 1")
        bodies.append(body)
    return bodies


def terrain_poses(boxes):
    """Mocap positions (len(TERRAIN_BOXES), 3) for a clip's terrain list (None or [] = all parked)."""
    pos = np.asarray([_terrain_parked(i) for i in range(len(TERRAIN_BOXES))], dtype=np.float64)
    free = list(range(len(TERRAIN_BOXES)))
    for box in boxes or ():
        slot = next(i for i in free if TERRAIN_BOXES[i][0] == box["kind"])
        free.remove(slot)
        pos[slot] = (box["x"], box["y"], box["top"] - TERRAIN_BOXES[slot][1][2])
    return pos


def seat_pose(seat):
    """Mocap position and quaternion (w, x, y, z) of a seat entry or None."""
    if seat is None:
        return np.asarray(SEAT_PARKED), np.asarray([1., 0., 0., 0.])
    from .seats import SEAT_HALF
    yaw = seat["yaw"]
    return (np.asarray([seat["x"], seat["y"], seat["top"] - SEAT_HALF[2]]),
            np.asarray([np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)]))


class TrackingEnv(mjx_env.MjxEnv):
    def __init__(self, library_dir, config=None, *, clips=None):
        config = config or default_config()
        super().__init__(config)
        terrain = bool(config.get("include_terrain"))
        extra = ([seat_xml()] if config.include_seated else []) + (terrain_xml() if terrain else [])
        self._mj_model = B.build_model(extra_world=extra or None)
        self._mj_model.opt.iterations = config.solver_iterations
        self._mj_model.opt.ls_iterations = config.solver_ls_iterations
        self._mjx_model = mjx.put_model(self._mj_model, impl=config.impl)
        manifest, selected, arrays = load_clips(library_dir, split=config.split,
                                                include_seated=config.include_seated, include_terrain=terrain)
        if clips is not None:
            selected = [c for c in selected if c["path"] in set(clips)]
        if not selected:
            raise ValueError("no clips selected")
        self.clips = selected
        self.library_summary = manifest["summary"]
        self._starts = jp.asarray([c["start"] for c in selected], dtype=jp.int32)
        self._lengths = jp.asarray([c["length"] for c in selected], dtype=jp.int32)
        # Clips the policy fails are drawn more often (PHC hard-negative mining).
        self._clip_logits = None
        if config.get("clip_weights"):
            table = json.loads(Path(config.clip_weights).read_text())["weights"]
            weights = np.asarray([table.get(c["path"], 1.) for c in selected], dtype=np.float64)
            self._clip_logits = jp.asarray(np.log(np.maximum(weights, 1e-6)), dtype=jp.float32)
        # Only joint coordinates are stored on the GPU; body poses come from FK.
        self._qpos = jp.asarray(arrays["qpos"])
        self._seated = bool(config.include_seated)
        poses = [seat_pose(c["seat"]) for c in selected]
        self._seat_pos = jp.asarray(np.stack([p for p, _ in poses]), dtype=jp.float32)
        self._seat_quat = jp.asarray(np.stack([q for _, q in poses]), dtype=jp.float32)
        self._terrain = terrain
        if terrain:
            # all mocap bodies per clip: the seat first (when present), then the terrain slots
            pos = [np.concatenate(([p[None]] if self._seated else []) + [terrain_poses(c.get("terrain"))])
                   for c, (p, _) in zip(selected, poses)]
            quat = [np.concatenate(([q[None]] if self._seated else []) + [np.tile([1., 0., 0., 0.], (len(TERRAIN_BOXES), 1))])
                    for _, q in poses]
            self._mocap_pos = jp.asarray(np.stack(pos), dtype=jp.float32)
            self._mocap_quat = jp.asarray(np.stack(quat), dtype=jp.float32)
        self._future = tuple(config.future_frames)
        self._nb = len(B.SEGMENTS)
        self._nu = self._mj_model.nu
        first = self._mj_model.body("pelvis").id
        self._parent = [int(self._mj_model.body_parentid[first + b]) - first for b in range(self._nb)]
        self._body_pos = jp.asarray(self._mj_model.body_pos[first:first + self._nb], dtype=jp.float32)
        axes = []
        for b in range(1, self._nb):
            joints = range(self._mj_model.body_jntadr[first + b],
                           self._mj_model.body_jntadr[first + b] + self._mj_model.body_jntnum[first + b])
            axes.append([self._mj_model.jnt_axis[j] for j in joints])
        self._axes = np.asarray(axes, dtype=np.float32)
        depth = [0] * self._nb
        for b in range(1, self._nb):
            depth[b] = depth[self._parent[b]] + 1
        self._depth = max(depth)
        self._kp = jp.asarray(B.KP)
        self._torque = jp.asarray(B.TORQUE)

    # -- required properties
    @property
    def xml_path(self) -> str:
        return "<generated by world_model.tracker.body>"

    @property
    def action_size(self) -> int:
        return self._nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    # -- reference access
    def _frame(self, clip, frame):
        """Global library row of a clip frame, clamped to the clip end."""
        frame = jp.clip(frame, 0, self._lengths[clip] - 1)
        return self._starts[clip] + frame

    def _fk(self, qpos):
        """World position and quaternion of every body (identity rest frames,
        hinge anchors at body origins), identical to MuJoCo's kinematics.

        Local rotations of all bodies are built at once; world poses are
        then propagated level by level (tree depth iterations), which keeps
        the traced graph small compared with a per-body unrolled chain.
        """
        angles = qpos[7:].reshape(self._nb - 1, 3)
        half = angles[..., None] / 2.
        axis_quats = jp.concatenate([jp.cos(half), jp.sin(half) * self._axes], axis=-1)   # (nb-1, 3, 4)
        local = quat_mul(quat_mul(axis_quats[:, 0], axis_quats[:, 1]), axis_quats[:, 2])
        local = jp.concatenate([qpos[3:7][None], local])                                  # (nb, 4)
        parent = jp.asarray([max(p, 0) for p in self._parent])
        is_root = jp.asarray([p < 0 for p in self._parent])[:, None]
        quat, pos = local, jp.zeros((self._nb, 3)).at[0].set(qpos[:3])
        for _ in range(self._depth):
            quat = jp.where(is_root, local, quat_mul(quat[parent], local))
            pos = jp.where(is_root, qpos[:3], pos[parent] + quat_rotate(quat[parent], self._body_pos))
        return pos, quat

    def _ref_qpos(self, clip, frame):
        return self._qpos[self._frame(clip, frame)]

    def _ref_kin(self, clip, frame):
        return self._fk(self._ref_qpos(clip, frame))

    def _ref_linvel(self, clip, frame):
        """Backward difference of body positions (matches the simulated velocity)."""
        same = (frame > 0).astype(jp.float32)
        return same * (self._ref_kin(clip, frame)[0] - self._ref_kin(clip, frame - 1)[0]) / self.dt

    def _ref_angvel(self, clip, frame):
        """World angular velocity, forward difference with the last frame repeated."""
        last = self._lengths[clip] - 1
        frame = jp.clip(frame, 0, last)
        now = jp.where(frame >= last, frame - 1, frame)
        q0, q1 = self._ref_kin(clip, now)[1], self._ref_kin(clip, now + 1)[1]
        return quat_log(quat_mul(q1, quat_conj(q0))) / self.dt

    WINDOW = (-1, 0, 1, 2)

    def _ref_window(self, clip, frame):
        """FK of frames f-1..f+2 and the far future frames, computed once."""
        offsets = self.WINDOW + tuple(o for o in self._future if o > 2)
        rows = jp.stack([self._ref_qpos(clip, frame + o) for o in offsets])
        pos, quat = jax.vmap(self._fk)(rows)
        last = self._lengths[clip] - 1

        def angvel(k, g):   # k: window index of frame g (k-1, k, k+1 must exist)
            q0 = jp.where(g >= last, quat[k - 1], quat[k])
            q1 = jp.where(g >= last, quat[k], quat[k + 1])
            return quat_log(quat_mul(q1, quat_conj(q0))) / self.dt

        return {"offsets": offsets, "pos": pos, "quat": quat,
                "lin": [(frame > 0) * (pos[1] - pos[0]) / self.dt, (pos[2] - pos[1]) / self.dt],
                "ang": [angvel(1, frame), angvel(2, frame + 1)]}

    def _ref_qvel(self, clip, frame):
        """Generalised velocity like mj_differentiatePos(q[f], q[f+1])."""
        last = self._lengths[clip] - 1
        now = jp.where(jp.clip(frame, 0, last) >= last, last - 1, jp.clip(frame, 0, last))
        q0, q1 = self._ref_qpos(clip, now), self._ref_qpos(clip, now + 1)
        local = quat_log(quat_mul(quat_conj(q0[3:7]), q1[3:7]))
        return jp.concatenate([(q1[:3] - q0[:3]) / self.dt, local / self.dt, (q1[7:] - q0[7:]) / self.dt])

    # -- environment API
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, k_clip, k_frame = jax.random.split(rng, 3)
        if self._clip_logits is None:
            clip = jax.random.randint(k_clip, (), 0, self._starts.shape[0])
        else:
            clip = jax.random.categorical(k_clip, self._clip_logits)
        frame = jax.random.randint(k_frame, (), 0, jp.maximum(self._lengths[clip] - 1, 1))
        return self.reset_at(rng, clip, frame)

    def reset_at(self, rng: jax.Array, clip, frame, init_noise=True) -> mjx_env.State:
        """Start ``clip`` at ``frame`` (used by reset and by the GPU clip sweep)."""
        clip, frame = jp.asarray(clip, jp.int32), jp.asarray(frame, jp.int32)
        rng, k_noise = jax.random.split(rng)
        qpos = self._ref_qpos(clip, frame)
        noise = self._config.hinge_init_noise * jax.random.normal(k_noise, (self._nu,))
        qpos = qpos.at[7:].add(noise if init_noise else 0. * noise)
        qvel = self._ref_qvel(clip, frame)
        seat = {"mocap_pos": self._seat_pos[clip], "mocap_quat": self._seat_quat[clip]} if self._seated else {}
        if self._terrain:
            seat = {"mocap_pos": self._mocap_pos[clip], "mocap_quat": self._mocap_quat[clip]}
        data = mjx_env.make_data(self._mj_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:],
                                 impl=self._mjx_model.impl.value,
                                 naconmax=self._config.naconmax, njmax=self._config.njmax, **seat)
        data = mjx.forward(self._mjx_model, data)
        info = {"rng": rng, "clip": clip, "frame": frame, "last_act": jp.zeros(self._nu),
                "prev_xpos": data.xpos[1:self._nb + 1], "linvel": self._ref_linvel(clip, frame),
                "clip_end": jp.float32(0.)}
        obs = self._observation(data, info)
        metrics = {k: jp.float32(0.) for k in ("r_position", "r_rotation", "r_velocity",
                                              "r_angular_velocity", "position_error",
                                              "rotation_error", "power", "failed")}
        return mjx_env.State(data, obs, jp.float32(0.), jp.float32(0.), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = dict(state.info)
        clip, frame = info["clip"], info["frame"] + 1
        action = jp.clip(action, -1., 1.)
        target = self._ref_qpos(clip, frame)[7:] + self._config.action_scale * action
        data = mjx_env.step(self._mjx_model, state.data, target, self.n_substeps)

        xpos, xquat = data.xpos[1:self._nb + 1], data.xquat[1:self._nb + 1]
        linvel = (xpos - info["prev_xpos"]) / self.dt
        angvel = data.cvel[1:self._nb + 1, :3]
        window = self._ref_window(clip, frame)
        ref_pos, ref_quat = window["pos"][1], window["quat"][1]
        ref_lin, ref_ang = window["lin"][0], window["ang"][0]
        pos_err = jp.linalg.norm(xpos - ref_pos, axis=-1)
        rot_err = quat_angle(quat_mul(quat_conj(xquat), ref_quat))
        w = self._config.reward
        r_pos = jp.exp(-w.position_k * jp.mean(pos_err ** 2))
        r_rot = jp.exp(-w.rotation_k * jp.mean(rot_err ** 2))
        r_vel = jp.exp(-w.velocity_k * jp.mean(jp.sum((linvel - ref_lin) ** 2, -1)))
        r_ang = jp.exp(-w.angular_velocity_k * jp.mean(jp.sum((angvel - ref_ang) ** 2, -1)))
        power = jp.sum(jp.abs(data.actuator_force * data.qvel[6:]))
        rate = jp.mean((action - info["last_act"]) ** 2)
        reward = (w.position * r_pos + w.rotation * r_rot + w.velocity * r_vel
                  + w.angular_velocity * r_ang - w.power * power - w.action_rate * rate)

        failed = (jp.mean(pos_err) > self._config.termination_distance) | jp.any(jp.isnan(data.qpos))
        clip_end = frame >= self._lengths[clip] - 1
        info.update(frame=frame, last_act=action, prev_xpos=xpos, linvel=linvel,
                    clip_end=clip_end.astype(jp.float32))
        obs = self._observation(data, info, window)
        metrics = {"r_position": r_pos, "r_rotation": r_rot, "r_velocity": r_vel,
                   "r_angular_velocity": r_ang, "position_error": jp.mean(pos_err),
                   "rotation_error": jp.mean(rot_err), "power": power,
                   "failed": failed.astype(jp.float32)}
        reward = jp.where(failed, 0., reward)
        return state.replace(data=data, obs=obs, reward=reward,
                             done=failed.astype(jp.float32), metrics=metrics, info=info)

    def _observation(self, data, info, window=None):
        xpos, xquat = data.xpos[1:self._nb + 1], data.xquat[1:self._nb + 1]
        root_pos, root_quat = xpos[0], xquat[0]
        inv = heading_quat_inv(root_quat)
        local = lambda v: quat_rotate(jp.broadcast_to(inv, v.shape[:-1] + (4,)), v)
        local_rot = lambda q: quat_to_6d(quat_mul(jp.broadcast_to(inv, q.shape), q))
        linvel = info["linvel"]   # body-origin velocity over the last control step
        angvel = data.cvel[1:self._nb + 1, :3]
        parts = [root_pos[2:3], local(xpos - root_pos).ravel(), local_rot(xquat).ravel(),
                 local(linvel).ravel(), local(angvel).ravel(),
                 data.qpos[7:], data.qvel[6:], info["last_act"]]
        clip, frame = info["clip"], info["frame"]
        window = self._ref_window(clip, frame) if window is None else window
        if self._future[0] != 1:
            raise ValueError("the first future frame must be the next frame")
        for k, offset in enumerate(self._future):
            index = window["offsets"].index(offset)
            ref_pos, ref_quat = window["pos"][index], window["quat"][index]
            parts += [local(ref_pos - root_pos).ravel(), local_rot(ref_quat).ravel()]
            if k == 0:
                parts += [local(ref_pos - xpos).ravel(),
                          quat_to_6d(quat_mul(quat_conj(xquat), ref_quat)).ravel(),
                          local(window["lin"][1] - linvel).ravel(),
                          local(window["ang"][1] - angvel).ravel(),
                          self._ref_qpos(clip, frame + offset)[7:]]
        return jp.concatenate(parts)


class TrackingEpisodeWrapper(Wrapper):
    """Brax episode bookkeeping where clip ends are truncations, not failures."""

    def __init__(self, env, episode_length):
        super().__init__(env)
        self.episode_length = episode_length

    def reset(self, rng):
        state = self.env.reset(rng)
        shape = rng.shape[:-1]
        state.info["steps"] = jp.zeros(shape)
        state.info["truncation"] = jp.zeros(shape)
        state.info["episode_done"] = jp.zeros(shape)
        metrics = {"sum_reward": jp.zeros(shape), "length": jp.zeros(shape)}
        metrics.update({k: jp.zeros(shape) for k in state.metrics})
        state.info["episode_metrics"] = metrics
        return state

    def step(self, state, action):
        state = self.env.step(state, action)
        steps = state.info["steps"] + 1
        limit = (steps >= self.episode_length) | (state.info["clip_end"] > 0)
        truncation = jp.where(limit, 1. - state.done, 0.)
        done = jp.where(limit, 1., state.done)
        state.info["steps"] = steps
        state.info["truncation"] = truncation
        previous = state.info["episode_done"]
        em = state.info["episode_metrics"]
        em["sum_reward"] = em["sum_reward"] * (1 - previous) + state.reward
        em["length"] = em["length"] * (1 - previous) + 1
        for k, v in state.metrics.items():
            em[k] = em[k] * (1 - previous) + v
        state.info["episode_done"] = done
        return state.replace(done=done)


class TrackingAutoResetWrapper(Wrapper):
    """Reset finished environments to a new random clip and frame.

    Unlike a cached first state, every reset samples a new reference state.
    Episode bookkeeping (truncation flag, episode metrics) of the finished
    transition is preserved so PPO and logging see how the episode ended.
    """

    PRESERVE = ("steps", "truncation", "episode_done", "episode_metrics")

    def reset(self, rng):
        keys = jax.vmap(jax.random.split)(rng)
        state = self.env.reset(keys[..., 1])
        state.info["autoreset_rng"] = keys[..., 0]
        return state

    def step(self, state, action):
        keys = jax.vmap(jax.random.split)(state.info["autoreset_rng"])
        state.info["steps"] = jp.where(state.done > 0, 0., state.info["steps"])
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)
        fresh = self.env.reset(keys[..., 1])
        done = state.done

        def where_done(new, old):
            # Warp keeps some solver buffers shared across the batch; those
            # leaves have no per-environment axis and stay as stepped.
            if old.ndim == 0 or old.shape[0] != done.shape[0]:
                return old
            mask = jp.reshape(done > 0, (done.shape[0],) + (1,) * (old.ndim - 1))
            return jp.where(mask, new, old)

        data = jax.tree.map(where_done, fresh.data, state.data)
        obs = jax.tree.map(where_done, fresh.obs, state.obs)
        info = {k: (v if k in self.PRESERVE else jax.tree.map(where_done, fresh.info[k], v))
                for k, v in state.info.items() if k != "autoreset_rng"}
        info["autoreset_rng"] = keys[..., 0]
        return state.replace(data=data, obs=obs, info=info)


def wrap_for_training(env, episode_length=300, action_repeat=1, randomization_fn=None):
    del action_repeat, randomization_fn
    env = brax_training.VmapWrapper(env)
    env = TrackingEpisodeWrapper(env, episode_length)
    return TrackingAutoResetWrapper(env)
