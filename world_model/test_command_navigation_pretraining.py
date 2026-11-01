"""Actor warm-start must preserve the actual critic, variance and RL lineage."""
import math
from types import SimpleNamespace
import unittest

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .command_navigation_pretraining import (fit_actor, goal_bearing_teacher,
                                              inherited_source_provenance,
                                              validated_motor_limit)


class _SpacesEnv(gym.Env):
    observation_space = spaces.Box(-np.inf, np.inf, (112,), dtype=np.float32)
    action_space = spaces.Box(-1., 1., (2,), dtype=np.float32)


class CommandPretrainingTests(unittest.TestCase):
    def test_optional_motor_limit_is_backward_compatible_and_bounded(self):
        self.assertIsNone(validated_motor_limit(SimpleNamespace()))
        self.assertIsNone(validated_motor_limit(SimpleNamespace(motor_hip_yaw_action_limit=None)))
        self.assertEqual(validated_motor_limit(SimpleNamespace(motor_hip_yaw_action_limit=.15)), .15)
        for invalid in (0., -1., 1.01, float("inf"), float("nan"), True):
            with self.assertRaises(ValueError):
                validated_motor_limit(SimpleNamespace(motor_hip_yaw_action_limit=invalid))

    def test_teacher_uses_current_physical_heading_and_final_goal(self):
        heading = .2
        c, s = math.cos(heading), math.sin(heading)
        rotation = np.asarray([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        body = SimpleNamespace(pelvis_body_id=0, goal=np.array([0., -2., 0.]),
            data=SimpleNamespace(qpos=np.zeros(27), xmat=rotation.reshape(1, 9)))
        env = SimpleNamespace(unwrapped=body, max_heading_radians=.45)
        actual_goal = body.goal.copy()
        action = goal_bearing_teacher(env)
        np.testing.assert_allclose(action, [-.2/.45, 1.], atol=1e-7)
        np.testing.assert_array_equal(body.goal, actual_goal)
        body.goal[:2] = [2., 0.]
        np.testing.assert_array_equal(goal_bearing_teacher(env), [1., 1.])

    def test_actor_fit_preserves_critic_variance_and_training_counters(self):
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        policy = PPO("MlpPolicy", _SpacesEnv(), seed=17, device="cpu",
            n_steps=8, batch_size=8, policy_kwargs={"net_arch": {"pi": [16, 16], "vf": [16, 16]}})
        policy.num_timesteps = 32768
        policy.command_physical_control_timesteps = 163376
        with torch.no_grad():
            policy.policy.log_std.fill_(-1.7)
        before = {name: value.clone() for name, value in policy.policy.state_dict().items()}
        angles = np.linspace(-.25, .25, 64, dtype=np.float32)
        observations = np.zeros((64, 112), dtype=np.float32)
        observations[:, 50] = np.sin(angles)
        observations[:, 51] = -np.cos(angles)
        targets = np.stack([angles/.45, np.ones(64)], axis=1)
        result = fit_actor(policy, observations, targets, epochs=60, batch_size=64,
                           learning_rate=.003, seed=17)
        self.assertLess(result["errorsAfter"]["mse"], result["errorsBefore"]["mse"]*.2)
        self.assertEqual(policy.num_timesteps, 32768)
        self.assertEqual(policy.command_physical_control_timesteps, 163376)
        after = policy.policy.state_dict()
        actor = ("mlp_extractor.policy_net.", "action_net.")
        self.assertTrue(any(not torch.equal(before[name], value)
                            for name, value in after.items() if name.startswith(actor)))
        self.assertTrue(all(torch.equal(before[name], value)
                            for name, value in after.items() if not name.startswith(actor)))
        self.assertEqual(len(policy.policy.optimizer.state), 0)
        self.assertEqual(result["rlDecisionTimestepsAdded"], 0)

    def test_bad_dataset_is_rejected_before_fitting(self):
        for observations, actions in ((np.zeros((3, 110)), np.zeros((3, 2))),
                                      (np.zeros((3, 112)), np.zeros((2, 2))),
                                      (np.full((3, 112), np.nan), np.zeros((3, 2)))):
            with self.assertRaises(ValueError):
                fit_actor(None, observations, actions)

    def test_provenance_keeps_actual_motor_counts_and_rejects_false_lineage(self):
        policy = SimpleNamespace(num_timesteps=32768, command_physical_control_timesteps=163376,
            avatar_profile_sha256="profile", body_model_sha256="body",
            command_control_contract={"motorCheckpointSha256": "motor"})
        source = {"checkpointSha256": "source", "motorCheckpoint": "motor.zip",
            "motorCheckpointSha256": "motor", "motorFrozen": True,
            "observationCount": 112, "actionCount": 2, "motorObservationCount": 110,
            "motorActionCount": 20, "motorTrainingTimesteps": 6381568,
            "actualDecisionTimesteps": 32768, "physicalControlTimesteps": 163376,
            "avatarProfileSha256": "profile", "bodyModelSha256": "body",
            "commandControlContract": policy.command_control_contract}
        copied = inherited_source_provenance(source, source_sha="source", policy=policy)
        self.assertEqual(copied["motorTrainingTimesteps"], 6381568)
        self.assertEqual(copied["sourceRlProvenance"], source)
        copied["commandControlContract"]["motorCheckpointSha256"] = "changed"
        self.assertEqual(source["commandControlContract"]["motorCheckpointSha256"], "motor")
        for broken in ({**source, "actualDecisionTimesteps": 999},
                       {**source, "motorFrozen": False},
                       {**source, "checkpointSha256": "other"},
                       {**source, "motorActionCount": None}):
            with self.assertRaises(ValueError):
                inherited_source_provenance(broken, source_sha="source", policy=policy)


if __name__ == "__main__":
    unittest.main()
