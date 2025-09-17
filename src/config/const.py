"""
    src/config/const.py
    -------------------
    Constants recurrent within the project are collected here for simplistic 
    alterations and accessibility while enhance the maintainability by
    collecting all constants at one single place.
"""
from typing import Final

# ----------===== Physical Constants =====---------- #

LIGHT_SPEED         : Final[float]  = 2.99792458e8   # m/s
THERMAL_NOISE       : Final[float]  = -174.0         # dBm


# ----------===== File Constants =====---------- #

PREPROCESSOR_FN     : Final[str]    = "preprocessor.pkl"
WEIGHTS_FN          : Final[str]    = "model.weights.h5"
CONFIG_FN           : Final[str]    = "model_config.json"