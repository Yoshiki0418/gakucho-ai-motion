"""The image actor really uses pixels, and training privileges stay outside it."""
import copy

import gymnasium as gym
import numpy as np
import pytest
import torch

from .visual_policy import DepthStateExtractor
from .visual_teacher import oracle_route, TrainingGeometryTeacher
from .navigation_scenes import generate_navigation_scene, segment_intersects_box


def test_depth_features_depend_on_pixels_with_fixed_state_and_have_gradients():
    torch.set_num_threads(1)
    space = gym.spaces.Dict({"depth": gym.spaces.Box(0., 1., (3, 48, 64), np.float32),
                             "state": gym.spaces.Box(-5., 5., (11,), np.float32)})
    encoder = DepthStateExtractor(space)
    obs = {"depth": torch.zeros(2, 3, 48, 64), "state": torch.zeros(2, 11)}
    before = encoder(obs)
    obs["depth"][1, :, 12:36, 16:48] = 1.
    after = encoder(obs)
    assert after.shape == (2, 160)
    assert torch.equal(before[0], after[0])
    assert not torch.equal(before[1], after[1])
    after.square().mean().backward()
    assert encoder.depth_encoder[0].weight.grad.abs().max() > 0


def test_visual_actor_rejects_privileged_map_input():
    space = gym.spaces.Dict({"depth": gym.spaces.Box(0., 1., (3, 48, 64), np.float32),
                             "state": gym.spaces.Box(-5., 5., (11,), np.float32),
                             "route": gym.spaces.Box(-5., 5., (4,), np.float32)})
    with pytest.raises(ValueError, match="depth and state only"):
        DepthStateExtractor(space)


def test_oracle_routes_are_body_inflated_and_do_not_mutate_scene():
    scene = generate_navigation_scene(90801, mode="obstacle", difficulty=.25)
    original = copy.deepcopy(scene)
    route = oracle_route(scene, margin=.5)
    obstacle = next(g for g in scene["geoms"] if g.get("entityId") == "navigation-furniture")
    assert len(route) > 2
    assert all(not segment_intersects_box(a, b, obstacle, margin=.5, interior=True)
               for a, b in zip(route, route[1:]))
    assert scene == original


def test_privileged_teacher_rejects_heldout_before_accessing_scene():
    class Navigation:
        split = "heldout"
    class Env:
        navigation_env = Navigation()
    with pytest.raises(ValueError, match="held-out"):
        TrainingGeometryTeacher(Env())
