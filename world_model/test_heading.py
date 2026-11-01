"""Physical controls must work in the body's axes after the room is rotated."""

import copy
import math

import numpy as np
import pytest

from .environment import HumanoidRoomEnv
from .home_scene import create_seated_task_scene
from .imitation import ReferenceHumanoidEnv
from .scenes import generate_scene


def rotate_scene(scene, heading):
    result = copy.deepcopy(scene)
    rotation = np.asarray([[math.cos(heading), -math.sin(heading)],
                           [math.sin(heading), math.cos(heading)]])
    for geom in result["geoms"]:
        geom["position"][:2] = (rotation @ geom["position"][:2]).tolist()
        geom["yaw"] = geom.get("yaw", 0)+heading
    for goal in result["goals"]:
        goal["position"][:2] = (rotation @ goal["position"][:2]).tolist()
    result["spawn"][:2] = (rotation @ result["spawn"][:2]).tolist()
    result["spawnHeading"] = heading
    return result, rotation


@pytest.mark.parametrize("heading", [math.pi/2, -math.pi/2, 1.1])
def test_standing_gravity_control_is_rotation_equivariant(heading):
    scene = generate_scene(1, include_chair=False)
    turned_scene, rotation = rotate_scene(scene, heading)
    base = HumanoidRoomEnv(scene=scene, randomize=True)
    turned = HumanoidRoomEnv(scene=turned_scene, randomize=True)
    first, _ = base.reset(seed=184)
    second, _ = turned.reset(seed=184)
    np.testing.assert_allclose(first, second, atol=1e-7)
    for _ in range(100):
        base.step(np.zeros(20))
        turned.step(np.zeros(20))
    np.testing.assert_allclose(rotation @ base.data.qpos[:2], turned.data.qpos[:2], atol=.001)
    assert abs(base.data.qpos[2]-turned.data.qpos[2]) < .001
    assert base._info()["success"] and turned._info()["success"]
    base.close()
    turned.close()


@pytest.mark.parametrize("heading", [0., math.pi/2, math.pi, -math.pi/2])
def test_physical_seat_support_for_rotated_chair(heading):
    scene, _ = rotate_scene(create_seated_task_scene(.34), heading)
    env = ReferenceHumanoidEnv(task="sit", reference_start=False, scene=scene)
    env.reset(seed=184)
    for _ in range(300):
        _, _, done, _, info = env.step(np.zeros(20))
        assert not done
    assert info["success"]
    assert info["pelvis_support_force"] > 100
    assert info["left_foot_force"] > 50 and info["right_foot_force"] > 50
    env.close()
