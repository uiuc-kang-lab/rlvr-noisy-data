"""
uv run --isolated --extra vllm -m examples.algorithm.custom_policy_loss.main_custom_policy_loss
"""

import ray
import hydra
import torch
from typing import Optional
from omegaconf import DictConfig
from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_train.utils.ppo_utils import PolicyLossRegistry, reduce_loss
from skyrl_gym.envs import register

# Example of custom policy loss: "reinforce"
def compute_reinforce_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: DictConfig,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
):
    """
    SAPO: https://arxiv.org/pdf/2511.20347
    hard-coded tau_pos = 1.0, tau_neg = 1.05
    """    
    tau = torch.where(advantages > 0, 1.0, 1.05).detach()
    ratio = (log_probs - old_log_probs).exp()
    clipped_ratio = torch.sigmoid(tau * (ratio - 1)) * 4.0 / tau
    sapo_objective = clipped_ratio * advantages
    print(clipped_ratio)
    loss = -sapo_objective
    loss = reduce_loss(loss, loss_mask, config.loss_reduction, config.max_seq_len)

    # Return loss and dummy clip_ratio (no clipping in REINFORCE)
    return loss, 0.0


# Register the custom policy loss
PolicyLossRegistry.register("sapo", compute_reinforce_policy_loss)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # Register the multiply environment inside the entrypoint task (no need to modify the skyrl-gym package).
    register(
        id="math_hard",
        entry_point="rl_noise.math_hard.env:MathEnv",
    )
    
    exp = BasePPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)

    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
