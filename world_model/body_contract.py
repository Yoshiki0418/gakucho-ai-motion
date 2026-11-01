"""Keep trained policies and physical avatar dimensions explicitly paired."""
from __future__ import annotations

from .avatar_profile import avatar_profile_sha256 as avatar_profile_digest


def bind_policy_body(policy, env):
    policy.avatar_profile_sha256 = avatar_profile_digest(env.avatar_profile)
    policy.body_model_sha256 = env.body_model_sha256


def assert_policy_body(policy, env):
    if policy is None:
        return
    expected = avatar_profile_digest(env.avatar_profile)
    trained = getattr(policy, "avatar_profile_sha256", None)
    if trained != expected:
        raise ValueError("Policy body profile differs from the simulated avatar; fine-tune with an explicit body transfer first")
    actual = getattr(policy, "body_model_sha256", None)
    if expected is not None and actual != env.body_model_sha256:
        raise ValueError("Policy physical body contract differs from the simulated avatar")


def trace_body_metadata(env):
    if env.avatar_profile is None:
        return {}
    return {"avatarProfile": env.avatar_profile,
            "avatarProfileSha256": avatar_profile_digest(env.avatar_profile),
            "bodyModelSha256": env.body_model_sha256,
            "physicsRestBodies": env.physics_rest_bodies,
            "physicalAssumptions": env.physical_assumptions}
