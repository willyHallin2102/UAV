"""
    src/math/random.py
    ------------------
    collection of random mathematics for various usages, such 
    as noise generations, seed assignment for reproducibility.
"""
import random

import numpy as np
import tensorflow as tf


def set_global_seed(seed: int) -> None:
    """ Ensure Reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
