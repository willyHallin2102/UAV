"""
    src/models/utilities/helpers.py
    -------------------------------
    Some methods used in general for models
    stored here temporary...
"""
from __future__ import annotations

import warnings
from typing import Dict, Literal, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
tfkl = tfk.layers


# -------------------------------------------------------------------
#   Sorting Layers Modules...
# -------------------------------------------------------------------

class SortLayer(tfkl.Layer):
    """
        Sort the first n_sort features for each sample while keeping 
        the rest unchanged.
    """
    def __init__(self, n_sort: int, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(n_sort, int) or n_sort <= 0:
            raise ValueError("n_sort must be a positive integer")
        self.n_sort = n_sort

    def call(self, 
        inputs: tf.Tensor, 
        direction: Literal["ASCENDING", "DESCENDING"] = "DESCENDING"
    ) -> tf.Tensor:
        """
            Call
        """
        inputs = tf.convert_to_tensor(inputs)
        tf.debugging.assert_rank_at_least(
            inputs, 2, 
            message="SortLayer expects rank >= 2 (batch, features)"
        )
        head = tf.sort(inputs[:, :self.n_sort], direction=direction)
        tail = inputs[:, self.n_sort:]
        return tf.concat([head, tail], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"n_sort": self.n_sort})
        return config


class SplitSortLayer(tfkl.Layer):
    """Sort only the first n_sort columns, leave the rest unchanged."""
    def __init__(self, n_sort: int, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(n_sort, int) or n_sort <= 0:
            raise ValueError("n_sort must be a positive integer")
        self.n_sort = n_sort

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.convert_to_tensor(x)
        head = tf.sort(x[:, :self.n_sort], direction="DESCENDING")
        tail = x[:, self.n_sort:]
        return tf.concat([head, tail], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"n_sort": self.n_sort})
        return config




def extract_inputs(inputs: Union[
    Tuple[Union[np.ndarray, tf.Tensor], Union[np.ndarray, tf.Tensor]],
    Dict[str, Union[np.ndarray, tf.Tensor]], tf.Tensor,
    Sequence[Union[np.ndarray, tf.Tensor]]
]) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    """
    if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
        x, cond = inputs
        if isinstance(x, (list, tuple)):
            x = tf.concat([
                tf.convert_to_tensor(part, dtype=tf.float32) for part in x
            ], axis=-1)
        else: x = tf.convert_to_tensor(x, dtype=tf.float32)
        return x, tf.convert_to_tensor(cond, dtype=tf.float32)
    
    elif isinstance(inputs, dict):
        if "x" not in inputs or "cond" not in inputs:
            raise ValueError("Dictionary is requred to contain `x` and `cond`")
        x = tf.convert_to_tensor(inputs["x"], dtype=tf.float32)
        cond = tf.convert_to_tensor(inputs["cond"], dtype=tf.float32)
        return x, cond
    

    # Assign Zeros to Cond --- Temp (Dire need of correction) -- assign zero (X)
    # elif isinstance(inputs, (np.ndarray, tf.Tensor)):
    #     x = tf.convert_to_tensor(inputs, dtype=tf.float32)
    #     cond = tf.zeros((tf.shape(x)[0], 1), dtype=tf.float32)
    #     return x, cond

    # elif isinstance(inputs, Sequence):
    #     parts = [tf.convert_to_tensor(part, dtype=tf.float32) for part in inputs]
    #     x = tf.concat(parts, axis=-1)
    #     cond = tf.zeros((tf.shape(x)[0], 1), dtype=tf.float32)
    #     return x, cond
    
    raise ValueError(f"Unsupported input type: '{type(inputs)}'")



def set_initialization(
    model: tfk.Model,
    names: Sequence[str],
    noise: str = "gaussian",
    kernel_init: float = 1.0,
    bias_init: float = 1.0,
) -> None:
    """Reinitialize Dense layer weights with custom noise."""
    for name in names:
        try: layer = model.get_layer(name)
        except Exception:
            warnings.warn(f"Layer {name} not found; skipping initialization")
            continue

        if not isinstance(layer, tfkl.Dense) or not layer.built:
            warnings.warn(f"Layer {name} is not a built Dense layer; " 
                          f"skipping initialization")
            continue

        input_dim, output_dim = layer.kernel.shape
        if noise.lower() == "gaussian":
            kernel = tf.random.normal(shape=(int(input_dim), int(output_dim)),
                                      mean=0.0,
                                      stddev=kernel_init / np.sqrt(float(input_dim)),
                                      dtype=tf.float32,)
            bias = tf.random.normal(shape=(int(output_dim),),
                                    mean=0.0, stddev=bias_init,
                                    dtype=tf.float32,)

        elif noise.lower() == "uniform":
            limit = kernel_init / np.sqrt(float(input_dim))
            kernel = tf.random.uniform((int(input_dim), int(output_dim)), 
                                        -limit, limit, dtype=tf.float32)
            bias = tf.random.uniform((int(output_dim),), 
                                     -bias_init, bias_init, 
                                     dtype=tf.float32)
        else: raise ValueError(f"Unsupported noise type: {noise}")

        try: layer.set_weights([kernel.numpy(), bias.numpy()])
        except Exception as e:
            warnings.warn(f"Could not set weights for layer {name}: {e}")
