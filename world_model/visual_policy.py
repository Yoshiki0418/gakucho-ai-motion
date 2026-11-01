"""CUDA-capable depth/state policy; no scene, route or privileged map inputs."""
from __future__ import annotations

import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DepthStateExtractor(BaseFeaturesExtractor):
    """Encode genuine depth frames and an eleven-value proprioceptive goal state."""

    def __init__(self, observation_space, features_dim=160):
        if set(observation_space.spaces) != {"depth", "state"}:
            raise ValueError("vision navigation accepts depth and state only")
        super().__init__(observation_space, features_dim)
        channels, height, width = observation_space["depth"].shape
        if channels not in (2, 3) or min(height, width) < 32:
            raise ValueError("depth policy requires two/three real camera frames >=32 pixels")
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(channels, 16, 5, 2), nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2), nn.SiLU(),
            nn.Conv2d(32, 32, 3, 2), nn.SiLU(), nn.Flatten())
        with torch.no_grad():
            flattened = self.depth_encoder(torch.zeros(1, channels, height, width)).shape[1]
        self.depth_projection = nn.Sequential(nn.Linear(flattened, features_dim-32), nn.SiLU())
        self.state_encoder = nn.Sequential(nn.Linear(observation_space["state"].shape[0], 32), nn.SiLU())

    def forward(self, observations):
        depth = observations["depth"].float()
        return torch.cat((self.depth_projection(self.depth_encoder(depth)),
                          self.state_encoder(observations["state"].float())), dim=1)


def visual_policy_kwargs():
    return {"features_extractor_class": DepthStateExtractor,
            "features_extractor_kwargs": {"features_dim": 160},
            "net_arch": {"pi": [128, 128], "vf": [128, 128]},
            "activation_fn": nn.Tanh, "normalize_images": False}
