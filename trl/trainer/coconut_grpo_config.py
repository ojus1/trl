from dataclasses import dataclass, field
from .grpo_config import GRPOConfig

@dataclass
class CoconutGRPOConfig(GRPOConfig):
    """
    Extended configuration for CoconutGRPOTrainer to enable GRPO in the latent (continuous thought) space.

    Additional parameters:
      - latent_initial_prob: The initial probability of generating via latent reasoning.
      - latent_final_prob: The final probability (after ramp-up) of generating via latent reasoning.
      - latent_prob_rampup_steps: The number of training steps over which latent reasoning probability increases.
      - max_continuous_tokens: Maximum number of continuous (latent) tokens that can be generated.
      - continuous_tokens_rampup_steps: Number of training steps to ramp up the continuous token count.
    """
    latent_initial_prob: float = field(
        default=0.0,
        metadata={"help": "Initial probability of using latent reasoning at rollout."}
    )
    latent_final_prob: float = field(
        default=1.0,
        metadata={"help": "Final probability of using latent reasoning at rollout."}
    )
    latent_prob_rampup_steps: int = field(
        default=10000,
        metadata={"help": "Number of training steps over which latent reasoning probability increases."}
    )
    max_continuous_tokens: int = field(
        default=10,
        metadata={"help": "Maximum number of continuous (latent) tokens to generate."}
    )
    continuous_tokens_rampup_steps: int = field(
        default=10000,
        metadata={"help": "Number of training steps over which the continuous token count ramps up."}
    ) 