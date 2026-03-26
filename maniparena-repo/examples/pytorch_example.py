"""Example: serving a LeRobot/PyTorch model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

# Ensure policy/config classes are registered for PreTrainedConfig.from_pretrained.
from lerobot import policies as _lerobot_policies  # noqa: F401
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

from maniparena.policy import ModelPolicy
from maniparena.utils import convert_model_output_to_action, convert_observation_to_model_input


@dataclass
class LeRobotRuntime:
    policy: Any
    preprocessor: Any
    postprocessor: Any
    image_keys: list[str]


def _map_camera_to_policy_image_keys(policy_image_keys: list[str]) -> dict[str, str]:
    """Map ManipArena camera aliases to policy image keys."""
    key_map: dict[str, str] = {}
    remaining = list(policy_image_keys)

    def pick(matchers: tuple[str, ...]) -> str | None:
        for i, key in enumerate(remaining):
            lowered = key.lower()
            if any(m in lowered for m in matchers):
                return remaining.pop(i)
        if remaining:
            return remaining.pop(0)
        return None

    left_key = pick(("left", "l_wrist", "cam_left"))
    front_key = pick(("front", "face", "head", "top", "agent"))
    right_key = pick(("right", "r_wrist", "cam_right"))

    if left_key:
        key_map["left"] = left_key
    if front_key:
        key_map["front"] = front_key
    if right_key:
        key_map["right"] = right_key

    return key_map


def build_model(checkpoint_path: str, device: str) -> LeRobotRuntime:
    cfg = PreTrainedConfig.from_pretrained(checkpoint_path)
    cfg.device = device
    cfg.pretrained_path = Path(checkpoint_path)

    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(pretrained_name_or_path=checkpoint_path, config=cfg)
    policy.to(device).eval()

    device_override = {"device": device}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=checkpoint_path,
        preprocessor_overrides={"device_processor": device_override},
        postprocessor_overrides={"device_processor": device_override},
    )

    return LeRobotRuntime(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        image_keys=list(policy.config.image_features.keys()),
    )


class TorchPolicy(ModelPolicy):
    def load_model(self, checkpoint_path: str, device: str) -> LeRobotRuntime:
        return build_model(checkpoint_path, device)

    def convert_input(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        model_input = convert_observation_to_model_input(obs, self.control_mode, decode_images=True)

        batch: Dict[str, Any] = {
            OBS_STATE: torch.from_numpy(model_input["state"]).float(),
            "task": model_input.get("instruction", ""),
        }

        camera_to_key = _map_camera_to_policy_image_keys(self.model.image_keys)
        for camera_name, policy_key in camera_to_key.items():
            image = model_input.get(camera_name)
            if image is None:
                continue
            image_tensor = torch.from_numpy(np.asarray(image)).contiguous()
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.permute(2, 0, 1).float() / 255.0
            batch[policy_key if policy_key.startswith(OBS_IMAGES) else f"{OBS_IMAGES}.{policy_key}"] = image_tensor

        return batch

    def run_inference(self, model_input: Dict[str, Any]) -> Any:
        with torch.no_grad():
            processed = self.model.preprocessor(model_input)
            action_chunk = self.model.policy.predict_action_chunk(processed)

            if action_chunk.ndim == 2:
                action_chunk = action_chunk.unsqueeze(1)

            processed_actions = []
            for i in range(action_chunk.shape[1]):
                single_action = action_chunk[:, i, :]
                processed_action = self.model.postprocessor(single_action)
                processed_actions.append(processed_action)

            action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)

        return action_tensor.detach().cpu().numpy()

    def convert_output(self, model_output: Any) -> Dict[str, Any]:
        actions = np.asarray(model_output, dtype=np.float32)

        if actions.shape[0] < self.action_horizon:
            pad = np.repeat(actions[-1:, :], self.action_horizon - actions.shape[0], axis=0)
            actions = np.concatenate([actions, pad], axis=0)
        elif actions.shape[0] > self.action_horizon:
            actions = actions[: self.action_horizon]

        return convert_model_output_to_action(actions, self.control_mode, self.action_horizon)
