"""
    src/config/data.py
    ------------------
    Configuration of the environment, features that is collected 
    in section. Simplifies setting up the environment for tweakings 
    and overhaul, generally altering the scenario.
"""
import datetime

from dataclasses import dataclass, field
from typing import Final, Tuple


# ----------------------------------------------------------------------------------
#   Angle Indexing, Common access to AoA/AoD
# ----------------------------------------------------------------------------------

class AngleIndex:
    AOA_PHI   = 0
    AOA_THETA = 1
    AOD_PHI   = 2
    AOD_THETA = 3

    N_ANGLES : Final[int] = 4


# ----------------------------------------------------------------------------------
#   Link State, Indexing the Communication Link Status
# ----------------------------------------------------------------------------------

class LinkState:
    NO_LINK = 0
    LOS     = 1
    NLOS    = 2

    N_STATES : Final[int] = 3


# ----------------------------------------------------------------------------------
#   Data Config: Collection of Environmental Parameters
# ----------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DataConfig:
    """ Meta Data for the environmental parameters """
    frequency       : float = 28e9
    data_created    : str = field(default_factory=lambda:
                                  datetime.datetime.now().isoformat())
    description     : str = "dataset"
    rx_types        : Tuple[str, ...] = ("RX0", "RX1")
    max_pathloss    : float = 200.0
    tx_power_dbm    : float = 16.0
    n_max_paths     : int = 20
    n_unit_links    : Tuple[int, ...] = (50, 25, 10)
    add_zero_los_frac   : float = 0.10
    n_dimensions    : int = 3
    dropout_rate    : float = 0.05
