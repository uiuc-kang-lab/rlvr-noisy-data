"""
PGFC / PGBC from Cai et al., 2025, "Reinforcement Learning with Verifiable yet Noisy Rewards
under Imperfect Verifiers" (arXiv:2510.00915).

  uv run --isolated --extra vllm -m rl_noise.math.main_math_hard_pgfc

Notation (from the paper)
-------------------------
    R*  true reward, R~  observed (noisy) verifier reward, both in {0, 1}
    rho_0 = P(R~ = 1 | R* = 0)   false positive rate
    rho_1 = P(R~ = 0 | R* = 1)   false negative rate

For the noisy-annotation setup in this repo, a wrongly annotated item marks a truly correct
response as incorrect, so the annotation noise rate p corresponds to the FN rate rho_1.

Theorem 1 - PGBC (backward correction), an unbiased surrogate reward:

    R_hat = (R~ - rho_0) / (1 - rho_0 - rho_1)          E[R_hat] = R*

Proposition 2 / Theorem 2 - PGFC (forward correction), reweighted score-function terms:

    w_1 = rho_1        when R~ = 1
    w_0 = rho_1 - 1    when R~ = 0
    h_t = w_{R~} * G_t                  G_t is the score function grad log pi(a_t | s_t)
    E[dtheta] = c * grad J(theta)       c = 1 - rho_0 - rho_1,  valid while rho_0 + rho_1 < 1

PGFC needs only rho_1, and avoids PGBC's 1/(1 - rho_0 - rho_1) division, so it stays stable at
high noise.

Why the reward-space implementation was inert
---------------------------------------------
The previous implementation put w_{R~} in the *reward*: `rl_noise/math/env.py` with
`noise_rate=p` returns p for a correct response and p - 1 for an incorrect one. Those are exactly
the paper's forward weights, so the numbers were right - but they were then handed to GRPO, which
subtracts the group mean in `compute_grpo_outcome_advantage` (both `grpo_norm_by_std` branches).
Since w_{R~} = R~ + (rho_1 - 1) is affine in R~ with slope 1, the added constant cancels exactly
and the advantages come out bit-identical to uncorrected GRPO.

PGFC's weights are meant to multiply the score function directly (h_t = w_{R~} G_t), with no
group-mean baseline and no std normalization - the paper's core formulation applies neither. The
`pgfc` estimator below does that: it uses w_{R~} itself as the per-token advantage, which is
precisely h_t = w_{R~} G_t once the loss multiplies advantage by grad log pi.

Estimators registered here
--------------------------
    pgfc       Proposition 2 / Theorem 2. advantage = w_{R~}, no centering, no std norm.
               This is the faithful one and the one to use.
    pgfc_grpo  Variant for the paper's remark that a baseline and std normalization are added in
               their practical GRPO implementation: the standard GRPO advantage multiplied by
               w_{R~}. Asymmetric, so unlike the reward-space version it is not cancelled.
    pgbc       Theorem 1. Substitutes R_hat for R~ and then runs the standard GRPO advantage.
               NOTE: R_hat is affine in R~ with positive slope, so group-mean subtraction removes
               the -rho_0 offset and, with grpo_norm_by_std=true, the std division also cancels
               the 1/(1 - rho_0 - rho_1) factor - making this *exactly* plain GRPO. Only with
               grpo_norm_by_std=false does the factor survive, and then it is just a constant
               rescale of every advantage. Provided for completeness/comparison, not as a fix.

Set the rates with `trainer.algorithm.pgfc_fn_rate` (rho_1) and, for pgbc only,
`trainer.algorithm.pgfc_fp_rate` (rho_0). The env must emit plain {0, 1} rewards, i.e.
`environment.skyrl_gym.math_hard.noise_rate=0`, so the correction is applied exactly once;
`main()` below refuses to start otherwise.
"""

import ray
import hydra
import torch
import numpy as np
from typing import Optional
from omegaconf import DictConfig
from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_train.utils.ppo_utils import AdvantageEstimatorRegistry, compute_grpo_outcome_advantage
from skyrl_gym.envs import register


def _get_rate(config: Optional[DictConfig], key: str, upper_exclusive: float = 1.0) -> float:
    if config is None or key not in config:
        raise ValueError(f"`trainer.algorithm.{key}` is required for the PGFC/PGBC estimators")
    rate = float(config[key])
    if not 0.0 <= rate < upper_exclusive:
        raise ValueError(
            f"`trainer.algorithm.{key}` must be in [0, {upper_exclusive}), got {rate}"
        )
    return rate


def _forward_weights(token_level_rewards: torch.Tensor, rho_1: float) -> torch.Tensor:
    """Per-sequence PGFC weights: w_1 = rho_1 for R~ = 1, w_0 = rho_1 - 1 for R~ = 0."""
    observed_reward = token_level_rewards.sum(dim=-1)
    return torch.where(
        observed_reward > 0,
        torch.full_like(observed_reward, rho_1),
        torch.full_like(observed_reward, rho_1 - 1.0),
    )


def compute_pgfc_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    config: Optional[DictConfig] = None,
    **kwargs,
):
    """PGFC (Cai et al., Proposition 2 / Theorem 2): h_t = w_{R~} * G_t.

    The forward weights are used directly as the advantage, with no group-mean baseline and no
    std normalization - applying either would cancel the correction, since w_{R~} is affine in
    R~ with slope 1.
    """
    rho_1 = _get_rate(config, "pgfc_fn_rate")

    with torch.no_grad():
        weights = _forward_weights(token_level_rewards, rho_1)
        advantages = weights.unsqueeze(-1) * response_mask

    return advantages, advantages.clone()


def compute_pgfc_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    config: Optional[DictConfig] = None,
    grpo_norm_by_std: bool = True,
    epsilon: float = 1e-6,
    **kwargs,
):
    """PGFC applied on top of GRPO's normalized advantage (A_grpo * w_{R~}).

    For the paper's remark that a baseline and std normalization are added in their practical
    GRPO implementation. The per-sample multiplication is asymmetric between R~ = 1 and R~ = 0,
    so it survives group-mean subtraction.
    """
    rho_1 = _get_rate(config, "pgfc_fn_rate")

    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        epsilon=epsilon,
        grpo_norm_by_std=grpo_norm_by_std,
    )
    with torch.no_grad():
        weights = _forward_weights(token_level_rewards, rho_1)
        advantages = advantages * weights.unsqueeze(-1)

    return advantages, advantages.clone()


def compute_pgbc_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    config: Optional[DictConfig] = None,
    grpo_norm_by_std: bool = True,
    epsilon: float = 1e-6,
    **kwargs,
):
    """PGBC (Cai et al., Theorem 1): R_hat = (R~ - rho_0) / (1 - rho_0 - rho_1), then GRPO.

    Kept for comparison. R_hat is affine in R~ with positive slope: group-mean subtraction removes
    the -rho_0 offset, and with grpo_norm_by_std=true the std division cancels the
    1/(1 - rho_0 - rho_1) factor as well, so this reduces exactly to plain GRPO.
    """
    rho_1 = _get_rate(config, "pgfc_fn_rate")
    rho_0 = _get_rate(config, "pgfc_fp_rate")
    if rho_0 + rho_1 >= 1.0:
        raise ValueError(
            f"PGBC requires rho_0 + rho_1 < 1 for a positive denominator, got "
            f"pgfc_fp_rate={rho_0} + pgfc_fn_rate={rho_1} = {rho_0 + rho_1}"
        )

    with torch.no_grad():
        corrected_rewards = (token_level_rewards - rho_0 * response_mask) / (1.0 - rho_0 - rho_1)

    return compute_grpo_outcome_advantage(
        token_level_rewards=corrected_rewards,
        response_mask=response_mask,
        index=index,
        epsilon=epsilon,
        grpo_norm_by_std=grpo_norm_by_std,
    )


AdvantageEstimatorRegistry.register("pgfc", compute_pgfc_advantage)
AdvantageEstimatorRegistry.register("pgfc_grpo", compute_pgfc_grpo_advantage)
AdvantageEstimatorRegistry.register("pgbc", compute_pgbc_advantage)

PGFC_ESTIMATORS = ("pgfc", "pgfc_grpo", "pgbc")


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # Register the math environment inside the entrypoint task (no need to modify the skyrl-gym package).
    register(
        id="math_hard",
        entry_point="rl_noise.math.env:MathEnv",
    )

    exp = BasePPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    # The correction lives in advantage space, so the env must emit plain {0, 1} rewards.
    if cfg.trainer.algorithm.advantage_estimator in PGFC_ESTIMATORS:
        env_noise_rate = float(cfg.environment.skyrl_gym.math_hard.noise_rate)
        if env_noise_rate > 0:
            raise ValueError(
                "`environment.skyrl_gym.math_hard.noise_rate` shifts rewards in env.py, which is the "
                "old reward-space PGFC (inert under GRPO). It must be 0 when using "
                f"`advantage_estimator={cfg.trainer.algorithm.advantage_estimator}`, otherwise the "
                f"correction is applied twice. Got {env_noise_rate}. Set the FN rate via "
                "`trainer.algorithm.pgfc_fn_rate` instead."
            )
        # Batch-level advantage whitening re-centers and rescales the forward weights, which
        # destroys the absolute w_1 / w_0 asymmetry PGFC depends on.
        if cfg.trainer.algorithm.advantage_batch_normalize:
            raise ValueError(
                "`trainer.algorithm.advantage_batch_normalize=true` whitens advantages across the "
                "batch, which cancels the PGFC forward weights. Set it to false."
            )

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
