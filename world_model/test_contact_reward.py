"""Changing the learning objective must not alter physical state or support."""
import numpy as np

from .locomotion import FootstepHumanoidEnv


def test_contact_objective_is_read_only_with_respect_to_simulation():
    old = FootstepHumanoidEnv(randomize=True, strict_gait_reward=False)
    new = FootstepHumanoidEnv(randomize=True, strict_gait_reward=True)
    old.reset(seed=31571)
    new.reset(seed=31571)
    rng = np.random.default_rng(9002)
    for _ in range(40):
        action = rng.uniform(-.05, .05, 20)
        old_observation, _, old_done, _, _ = old.step(action)
        new_observation, reward, new_done, _, info = new.step(action)
        np.testing.assert_array_equal(old.data.qpos, new.data.qpos)
        np.testing.assert_array_equal(old.data.qvel, new.data.qvel)
        np.testing.assert_array_equal(old.data.ctrl, new.data.ctrl)
        np.testing.assert_array_equal(old_observation, new_observation)
        assert np.isfinite(reward) and old_done == new_done
        if old_done:
            break
    old.close()
    new.close()


def test_initial_double_support_does_not_count_as_a_swing():
    env = FootstepHumanoidEnv(randomize=False, strict_gait_reward=True)
    env.reset(seed=72)
    for _ in range(20):
        _, _, _, _, info = env.step(np.zeros(20))
        assert info["actual_swing_counts"] == [0, 0]
    env.reset(seed=73)
    assert env._swing_counts.tolist() == [0, 0]
    assert env._swing_durations.tolist() == [0., 0.]
    env.close()
