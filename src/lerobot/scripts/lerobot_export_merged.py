#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub.constants import CONFIG_NAME, SAFETENSORS_SINGLE_FILE

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

PEFT_FILES = {"adapter_config.json", "adapter_model.safetensors"}


@dataclass
class ExportMergedCheckpointConfig:
    policy: PreTrainedConfig | None = None
    output_dir: Path | None = None
    overwrite: bool = False
    copy_auxiliary_files: bool = True
    keep_peft_files: bool = False

    def __post_init__(self) -> None:
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = Path(policy_path)

        if self.policy is None or self.policy.pretrained_path is None:
            raise ValueError("Please provide a PEFT checkpoint directory with `--policy.path=...`.")

        if self.output_dir is None:
            self.output_dir = self.policy.pretrained_path.parent / f"{self.policy.pretrained_path.name}_merged"

        if self.output_dir.exists() and any(self.output_dir.iterdir()) and not self.overwrite:
            raise FileExistsError(
                f"Output directory {self.output_dir} exists and is not empty. "
                "Use `--overwrite=true` or choose a different `--output_dir`."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def copy_auxiliary_files(source_dir: Path, output_dir: Path, keep_peft_files: bool) -> None:
    for item in source_dir.iterdir():
        if item.is_file():
            if item.name in {CONFIG_NAME, SAFETENSORS_SINGLE_FILE}:
                continue
            if not keep_peft_files and item.name in PEFT_FILES:
                continue
            shutil.copy2(item, output_dir / item.name)
        elif item.is_dir():
            shutil.copytree(item, output_dir / item.name, dirs_exist_ok=True)


def _build_base_policy_candidates(policy_cls: type, base_model_name_or_path: str, cfg_policy: Any) -> list:
    # Candidate 1: use loaded adapter-side config first so architecture/feature overrides are preserved.
    # Candidate 2: fall back to base checkpoint's own config for compatibility with untouched base models.
    return [
        (
            "loaded-config",
            lambda: policy_cls.from_pretrained(pretrained_name_or_path=base_model_name_or_path, config=cfg_policy),
        ),
        # ("base-config", lambda: policy_cls.from_pretrained(pretrained_name_or_path=base_model_name_or_path)),
    ]


@parser.wrap()
def export_merged_checkpoint(cfg: ExportMergedCheckpointConfig) -> None:
    assert cfg.policy is not None
    assert cfg.policy.pretrained_path is not None
    assert cfg.output_dir is not None

    checkpoint_path = cfg.policy.pretrained_path
    checkpoint_str = str(checkpoint_path)

    policy_cls = get_policy_class(cfg.policy.type)
    if not cfg.policy.use_peft:
        logging.warning(
            "Policy config indicates `use_peft=False`. Saving checkpoint as-is without PEFT merge."
        )
        merged_policy = policy_cls.from_pretrained(pretrained_name_or_path=checkpoint_str, config=cfg.policy)
    else:
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(checkpoint_str)
        base_model_name_or_path = peft_config.base_model_name_or_path
        if not base_model_name_or_path:
            raise ValueError(
                "No base model path found in adapter config. "
                "Cannot merge adapter without `base_model_name_or_path`."
            )

        peft_policy = None
        last_error = None

        base_policy = policy_cls.from_pretrained(pretrained_name_or_path=base_model_name_or_path, config=cfg.policy)
        peft_policy = PeftModel.from_pretrained(base_policy, checkpoint_str, config=peft_config)

        if peft_policy is None:
            raise RuntimeError(
                "Could not load adapter on any base-policy configuration candidate. "
                "Check that adapter and base model are from the same training lineage."
            ) from last_error

        logging.info("Merging PEFT adapter into base policy weights.")
        merged_policy = peft_policy.merge_and_unload()

    merged_policy.to("cpu")
    if hasattr(merged_policy, "config") and merged_policy.config is not None:
        merged_policy.config.use_peft = False
        merged_policy.config.pretrained_path = None
        if cfg.policy.device:
            merged_policy.config.device = cfg.policy.device

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    merged_policy.save_pretrained(cfg.output_dir)

    if cfg.copy_auxiliary_files:
        copy_auxiliary_files(
            source_dir=checkpoint_path, output_dir=cfg.output_dir, keep_peft_files=cfg.keep_peft_files
        )

    logging.info(f"Exported merged checkpoint to: {cfg.output_dir}")


def main() -> None:
    init_logging()
    register_third_party_plugins()
    export_merged_checkpoint()


if __name__ == "__main__":
    main()
