"""
    src/config/model.py
    -------------------
    Manages the configurations for respective generative ai model,
    there is a collective base class of common feature, which is 
    base for the specified configurations which inherents the base
    setup to each specified model.
"""
from dataclasses import dataclass
from typing import Final, Literal, Tuple

VALID_MODELS: Final[Tuple[Literal["vae"], ...]] = ("vae",)



@dataclass(slots=True)
class ModelConfig:
    n_latent        : int   = 10
    min_variance    : float = 1e-4
    dropout_rate    : float = 0.10
    init_kernel_std : float = 10.0
    init_bias_std   : float = 10.0

    def __post_init__(self):
        if self.n_latent <= 0:
            raise ValueError("n_latent must be positive")
        if self.min_variance < 0:
            raise ValueError("min_variance cannot be negative")
        if not (0.0 <= self.dropout_rate <= 1.0):
            raise ValueError("dropout_rate must be between 0.0 and 1.0")


@dataclass(slots=True)
class VaeConfig(ModelConfig):
    encoder_layers: Tuple[int, ...] = (200, 80)
    decoder_layers: Tuple[int, ...] = (80, 200)
    beta: float = 0.50
    beta_annealing_step: int = 100_000
    kl_warmup_steps: int = 80

    def __post_init__(self):
        ModelConfig.__post_init__(self)

        if any(layer <= 0 for layer in self.encoder_layers):
            raise ValueError("All encoder layers must be positive")
        if any(layer <= 0 for layer in self.decoder_layers):
            raise ValueError("All decoder layers must be positive")
        if not (0.0 <= self.beta <= 1.0):
            raise ValueError("beta must be between 0.0 and 1.0")



CONFIGS = {"vae": VaeConfig, }

def get_config(model_type: str) -> ModelConfig:
    """
    Return the configuration object for a given model type.

    Args:
    -----
        model_type: Name of the model (e.g., "vae").

    Raises:
    -------
        ValueError: If model_type is not supported.
    -------
    """
    model_type = model_type.lower()
    try: return CONFIGS[model_type]()
    except KeyError: raise ValueError(f"Unknown model_type '{model_type}', "
                                      f"must be one of {VALID_MODELS}")
