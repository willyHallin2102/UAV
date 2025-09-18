"""
    src/math/random.py
    ------------------
    Collection of random mathematics for various usages, such 
    as noise generation, seed assignment, and reproducibility utilities.
"""

import random
import numpy as np
import tensorflow as tf

from typing import Callable, Optional, Tuple, Union


def set_global_seed(seed: int) -> None:
    """
        Ensure reproducibility across Python, NumPy as well as 
        TensorFlow TF globally. Seed should always be passed 
        to this function to guarantee consistency.
    """
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def generate_noise(
    shape: Union[int, Tuple[int, ...]],
    noise_type: str = "normal",
    backend: str = "tf",
    **kwargs
) -> Union[np.ndarray, tf.Tensor]:
    """
    Generate random noise from a chosen distribution.

    Args:
        shape: int or tuple defining the output shape.
        noise_type: distribution type ("normal", "uniform", "truncated_normal",
                    "lognormal", "gamma", "exponential", "laplace").
        backend: "tf" (TensorFlow) or "np" (NumPy).
        **kwargs: extra distribution parameters (e.g., mean, stddev, minval, maxval).

    Returns:
        Random tensor/array from the specified distribution.
    """
    if isinstance(shape, int): shape = (shape,)

    noise_type = noise_type.lower()
    if backend == "tf":
        dist_map: dict[str, Callable] = {
            "normal"        : lambda: tf.random.normal(shape, **kwargs),
            "uniform"       : lambda: tf.random.uniform(shape, **kwargs),
            "trunc_normal"  : lambda: tf.random.truncated_normal(shape, **kwargs),
            "lognormal"     : lambda: tf.random.log_normal(shape, **kwargs),
            "gamma"         : lambda: tf.random.gamma(shape, **kwargs),
            "exponential"   : lambda: tf.random.exponential(shape, **kwargs),
            "laplace"       : lambda: tf.random.normal(shape, **kwargs) * tf.sqrt(0.5),
        }
    elif backend == "np":
        dist_map: dict[str, Callable] = {
            "normal"    : lambda: np.random.normal(size=shape, **kwargs),
            "uniform"   : lambda: np.random.uniform(size=shape, **kwargs),
            "lognormal" : lambda: np.random.lognormal(size=shape, **kwargs),
            "gamma"     : lambda: np.random.gamma(kwargs.get("shape", 1.0),
                                                  kwargs.get("scale", 1.0),
                                                  size=shape),
            "exponential": lambda: np.random.exponential(size=shape, **kwargs),
            "laplace"   : lambda: np.random.laplace(size=shape, **kwargs),
        }
    else: raise ValueError(f"Unknown backend: {backend}")

    if noise_type not in dist_map:
        raise ValueError(f"Unknown noise_type: {noise_type}")
    
    return dist_map[noise_type]()


def random_choice(values, size: int, replace: bool = True) -> np.ndarray:
    """Sample values with or without replacement using NumPy."""
    return np.random.choice(values, size=size, replace=replace)


def shuffle_tensor(tensor: Union[tf.Tensor, np.ndarray]) -> Union[tf.Tensor, np.ndarray]:
    """Shuffle a tensor/array along its first dimension."""
    if isinstance(tensor, tf.Tensor): return tf.random.shuffle(tensor)
    return np.random.permutation(tensor)


def bernoulli_mask(shape: Tuple[int, ...], p: float = 0.5, backend: str = "tf"):
    """Generate a binary mask from a Bernoulli distribution."""
    if backend == "tf":
        return tf.cast(tf.random.uniform(shape) < p, tf.float32)
    return (np.random.rand(*shape) < p).astype(np.float32)