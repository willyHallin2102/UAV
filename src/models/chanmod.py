import numpy as np

from pathlib import Path
from src.config.data import DataConfig
from src.models.link import LinkStatePredictor
# from src.models.path import PathModel

from typing import Union



class ChannelModel:
    """
    """
    def __init__(self,
        config      : DataConfig = DataConfig(),
        model_type  : str = "vae",
        directory   : Union[str, Path]  = 'beijing'
    ):
        """ Initialize the Channel Model instance """
        # Create the directory root for the model 
        self.directory = Path(__file__).parent / "archives" / directory
        self.directory.mkdir(parents=True, exist_ok=True)

        # Initialize the LinkStatePredictor instance for link predictions
        self.link = LinkStatePredictor(
            directory=self.directory / "link",
            rx_types=config.rx_types,
            n_unit_links=config.n_unit_links,
            add_zero_los_frac=config.add_zero_los_frac,
            dropout_rate=config.dropout_rate
        )