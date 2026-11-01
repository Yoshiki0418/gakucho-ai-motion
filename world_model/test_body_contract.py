"""A trained policy cannot silently cross physical-avatar geometries."""
from types import SimpleNamespace

import pytest
from stable_baselines3 import PPO

from .body_contract import assert_policy_body, avatar_profile_digest, bind_policy_body
from .environment import HumanoidRoomEnv


def test_unrecorded_legacy_policy_only_matches_the_legacy_body():
    policy = SimpleNamespace()
    assert_policy_body(policy, SimpleNamespace(avatar_profile=None))
    with pytest.raises(ValueError, match="body profile"):
        assert_policy_body(policy, SimpleNamespace(avatar_profile={"physical": {"rootHeight": 1.}}))


def test_recorded_profile_and_physics_contract_both_must_match():
    env = SimpleNamespace(avatar_profile={"id": "measured", "physical": {"rootHeight": 1.}},
                          body_model_sha256="body-a")
    policy = SimpleNamespace()
    bind_policy_body(policy, env)
    assert_policy_body(policy, env)
    with pytest.raises(ValueError, match="physical body contract"):
        assert_policy_body(policy, SimpleNamespace(avatar_profile=env.avatar_profile,
                                                  body_model_sha256="body-b"))
    with pytest.raises(ValueError, match="body profile"):
        assert_policy_body(policy, SimpleNamespace(avatar_profile={"id": "other"}))
    assert policy.avatar_profile_sha256 == avatar_profile_digest(env.avatar_profile)


def test_body_contract_survives_actual_sb3_checkpoint_roundtrip(tmp_path):
    env = HumanoidRoomEnv()
    policy = PPO("MlpPolicy", env, n_steps=8, batch_size=8, policy_kwargs={"net_arch": [8]}, device="cpu")
    bind_policy_body(policy, env)
    policy.save(tmp_path/"contract")
    loaded = PPO.load(tmp_path/"contract.zip", device="cpu")
    assert loaded.body_model_sha256 == env.body_model_sha256
    assert loaded.avatar_profile_sha256 is None
    assert_policy_body(loaded, env)
    env.close()
