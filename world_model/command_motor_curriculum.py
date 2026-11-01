"""Train-only, balanced sustained local commands for the unchanged joint motor.

This changes the distribution of short commands, not the reward, physics,
observation/action semantics or an inference navigator. Sampling occurs in
the parent's post-physics-reset command hook. Nothing moves the root.
"""
from __future__ import annotations

import math

import numpy as np

from .command_motor import CommandMotorEnv
from .generalization import validate_scene_seed


# A complete shuffled bag has exactly half saturated commands and equal
# coverage of each mirrored pair. Its order is sampled from the actual
# reset body's RNG, after body/mass/friction/initial-state creation.
LOCAL_COMMAND_OFFSETS = (-.45, -.3, -.15, 0., .15, .3, .45)
LOCAL_COMMAND_BAG_COUNTS = (5, 2, 2, 2, 2, 2, 5)


class CommandMotorCurriculumEnv(CommandMotorEnv):
    """Hold one balanced body-relative heading request for each train episode."""

    def __init__(self, *, max_episode_seconds=8., **kwargs):
        if kwargs.get("split", "train") != "train":
            raise ValueError("local-command curriculum is restricted to actual train scenes")
        if kwargs.get("mode", "turn") != "turn":
            raise ValueError("local-command curriculum currently requires empty turn rooms")
        if not math.isfinite(max_episode_seconds) or not 6. <= max_episode_seconds <= 8.:
            raise ValueError("sustained local-command episodes must last 6..8 seconds")
        if kwargs.get("max_heading_radians", .45) != .45:
            raise ValueError("balanced local commands require the existing .45 radian bound")
        if kwargs.get("heading_slew_radians_per_second", .175) != .175:
            raise ValueError("local-command curriculum preserves the existing .175 radian/s slew")
        kwargs.setdefault("mode", "turn")
        kwargs.setdefault("split", "train")
        self._curriculum_bag = []
        self._curriculum_requested_offset = None
        self._curriculum_episode = 0
        self._curriculum_bag_number = 0
        super().__init__(max_episode_seconds=max_episode_seconds, **kwargs)
        self.training_command_generator_contract = {
            "version": 1,
            "role": "training-only balanced sustained local-command curriculum; not learned room navigation",
            "sampleAfterActualPhysicsReset": True,
            "randomSource": "actual reset humanoid np_random; no held-out or prerecorded teacher commands",
            "requestedHeadingOffsetsRadians": list(LOCAL_COMMAND_OFFSETS),
            "shuffledBagCountsPerOffset": list(LOCAL_COMMAND_BAG_COUNTS),
            "episodesPerBalancedBag": sum(LOCAL_COMMAND_BAG_COUNTS),
            "requestedSaturatedEpisodeTargetFractionPerCompleteBag": .5,
            "actualConsumedSaturatedControlFractionGuaranteed": False,
            "mirroredCommandsHaveEqualCountsPerCompleteBag": True,
            "partialBagBalanceGuaranteed": False,
            "holdRequestedOffsetForWholeEpisode": True,
            "maxEpisodeSeconds": float(max_episode_seconds),
            "headingOffsetBoundRadians": self.max_heading_radians,
            "headingSlewRadiansPerSecond": self.heading_slew_radians_per_second,
            "sampledLookaheadRangeMeters": list(self.command_lookahead_range),
            "lookaheadSampling": "unchanged parent sampler after physics reset; held for whole episode",
            "commandRegeneratedAfterEachActual20msControl": True,
            "actualTaskGoalFixed": True,
            "goalDistanceCapped": True,
            "worldGoalUsedToRequestHeading": False,
            "geometryPathsOrWaypointsUsed": False,
            "rewardContractUnchanged": True,
            "unobservedRequestedOffsetUsedInDenseReward": False,
        }

    def _update_next_command(self, *, advance):
        body = self.unwrapped
        # This hook is called by CommandMotorEnv.reset only after its real
        # navigation/physics reset and parent lookahead sampling have finished.
        if not advance:
            if self.navigation_env.split != "train":
                raise RuntimeError("held-out scenes cannot enter the local-command curriculum")
            validate_scene_seed(self.navigation_env._last_scene_seed, "train")
            if not isinstance(body.np_random, np.random.Generator):
                raise RuntimeError("local-command sampling requires the actual reset body's RNG")
            if not self._curriculum_bag:
                bag = np.repeat(LOCAL_COMMAND_OFFSETS, LOCAL_COMMAND_BAG_COUNTS)
                self._curriculum_bag = body.np_random.permutation(bag).tolist()
                self._curriculum_bag_number += 1
            self._curriculum_requested_offset = float(self._curriculum_bag.pop())
            self._curriculum_episode += 1
        if self._curriculum_requested_offset is None:
            raise RuntimeError("reset must sample a local command before actual controls")
        requested = self._curriculum_requested_offset
        if advance:
            change = self.heading_slew_radians_per_second*body.dt
            self._effective_heading_offset += float(np.clip(
                requested-self._effective_heading_offset, -change, change))
        rotation = body.data.xmat[body.pelvis_body_id].reshape(3, 3)
        current_heading = math.atan2(rotation[1, 0], rotation[0, 0])
        command_heading = current_heading+self._effective_heading_offset
        actual_goal_distance = float(np.linalg.norm(body.goal[:2]-body.data.qpos[:2]))
        command_distance = min(self._lookahead, actual_goal_distance)
        self._next_command_goal = np.asarray([
            body.data.qpos[0]+command_distance*math.sin(command_heading),
            body.data.qpos[1]-command_distance*math.cos(command_heading),
            body.goal[2]], dtype=np.float64)
        body.command_goal = self._next_command_goal.copy()
        self._command = {
            "source": "training-only balanced sampled local heading; not actual final-goal bearing",
            "requestedHeadingOffsetRadians": requested,
            "effectiveHeadingOffsetRadians": self._effective_heading_offset,
            "headingRadians": command_heading,
            "requestedLookaheadMeters": self._lookahead,
            "effectiveLookaheadMeters": command_distance,
            "worldCommandGoal": self._next_command_goal.tolist(),
            "actualFinalGoal": body.goal.tolist(),
            "geometryPathTeacherUsed": False,
            "curriculumEpisodeIndex": self._curriculum_episode,
            "curriculumBalancedBagIndex": self._curriculum_bag_number,
            "curriculumCommandsRemainingInBag": len(self._curriculum_bag),
            "requestedOffsetHeldForWholeEpisode": True,
        }
