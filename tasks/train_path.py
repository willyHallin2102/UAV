"""
    tasks/train_path.py
    --------------------
    Script to train the Path model, conducted from termminal calls,
    to run this script:
        1.  cd ../path-to-UAV
        2.  source venv/bin/active  # Activates virtual environment 
        3.  cd tasks    # Where script should be located from UAV
        4.  python train_path.py --city city --epochs 50 --batch 512 ..
    stores the training (all includes) in models/archives/"city" 
"""
import os
import sys

path = os.path.abspath("..")
if not path in sys.path:
    sys.path.append(path)

import argparse

import numpy as np

from pathlib import Path
from data.loader import DataLoader, shuffle_and_split
from src.models.chanmod import ChannelModel
from logs.logger import setup_logging, get_logger
from src.math.random import set_global_seed


def parse_args():
    """
        Contains all bash-commands the script manages
    """
    parser = argparse.ArgumentParser(
        description="Train Path-Model in the ChannelModel on specific data"
    )

    # Arguments
    # ------------------------
    parser.add_argument("--city", type=str, default="beijing",
                        help="Comma-separated list of cities or 'all'")
    
    parser.add_argument("--ratio", type=float, default=0.10,
                        help="Validation -split, remaining is in training")
    
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (full forward backward"
                             "propagation)")
    
    parser.add_argument("--batch", type=int, default=512,
                        help="Batch size (nr samples per epoch)")
    
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for optimizer in backpropagation")
    
    parser.add_argument("--seed", type=int, default=42, 
                        help="Seed for reproducibility")

    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Setting logging verbosity")
    
    parser.add_argument("--model_type", type=str, default="vae",
                        choices=["vae"],
                        help="Which generative model the path-model does assign")
    
    return parser.parse_args()




args = parse_args()
setup_logging(logger_directory=Path(args.city)/args.model_type,
              logger_level=args.log_level)
logger = get_logger(__name__)

set_global_seed(seed=args.seed)


# -----------------------------------------------------------------------
#   City Assignment
# ----------------------------------------------------------------------

if args.city.lower() == "all":
    files = ["uav_beijing/train.csv",
             "uav_boston/train.csv",
             "uav_london/train.csv",
             "uav_moscow/train.csv",
             "uav_tokyo/train.csv"]
else:
    city_list = [c.strip().lower() for c in args.city.split(",")]
    supported = {"beijing", "boston", "london", "moscow", "tokyo"}
    invalid = set(city_list) - supported
    if invalid:
        logger.error(f"Unsupported city/cities: {invalid}")
        sys.exit(1)
    files = [f"uav_{city}/train.csv" for city in city_list]

logger.info(f"Training cities: {args.city}")
logger.info(f"Files to load: {files}")

# ------------ Load the City

loader = DataLoader(debug_level=args.log_level)
data = loader.load(files)
dtr, dts = shuffle_and_split(data=data, val_ratio=args.ratio)
logger.info(f"Loaded {len(dtr)} training samples, {len(dts)} validation samples")

model = ChannelModel(directory=args.city, model_type=args.model_type)
model.path.build()
logger.info(f"Starting training for {args.epochs} epochs, batch size {args.batch}")
history = model.path.fit(dtr=dtr, dts=dts,
                         epochs=args.epochs, batch_size=args.batch,
                         learning_rate=args.learning_rate,)

# Save training history
history_file = os.path.join(args.city, f"{args.city}_history.json")
try:
    import json
    with open(history_file, "w") as fp:
        json.dump(history.history, fp)
    logger.info(f"Saved training history to {history_file}")
except Exception as e:
    logger.warning(f"Could not save training history: {e}")

model.path.save()
logger.info(f"Model saved to {args.city}")




# """
# """
# import os
# import argparse
# import sys

# path = os.path.abspath("..")
# if not path in sys.path:
#     sys.path.append(path)

# import numpy as np
# from data.loader import DataLoader, LogLevel, shuffle_and_split



# parser = argparse.ArgumentParser(
#     "Train path model in the Channel Model on specific data"
# )

# parser.add_argument("--directory", type=str, required=True, default="beijing", 
#                     help="Directory within models/archives/--directory")
# parser.add_argument("--cities", type=str, required=True, default="beijing",
#                     help="Comma-separated list of cities e.g., beijing,london")
# parser.add_argument("--ratio", type=float, default=0.20,
#                     help="Ratio of the validation of data loaded")
# parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
# parser.add_argument("--batch_size", type=int, default=100, help="Number of elements per epochs")
# parser.add_argument("--name", type=str, default="vae", 
#                     help="name of the path modeling model")
# parser.add_argument("--learning_rate", type=float, default=1e-4,
#                     help="Learning Rate for the generative AI model")
# parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffle in the file loaded")

# args = parser.parse_args()


# # --------------------------------------------------------
# #   Retrieve the corresponding files to cities
# # --------------------------------------------------------

# if args.cities == 'all':
#     city_list = ['uav_beijing/train.csv', 'uav_boston/train.csv',
#                  'uav_london/train.csv',  'uav_moscow/train.csv',
#                  'uav_tokyo/train.csv']
# else:
#     city_list  = [city.strip().lower() for city in args.cities.split(",")]
#     supported_cities    = {"beijing", "boston", "london",
#                            "moscow", "tokyo"}
#     invalid = set(city_list) - supported_cities
#     if invalid:
#         raise ValueError(f"Unsupported city/cities: { invalid }")
#     files   = [f"uav_{ city }/train.csv" for city in city_list]


# # ---------------------------------------------
# #   Load the files into dictionaries
# # ---------------------------------------------

# loader = DataLoader(debugging_level=LogLevel.ERROR)
# data = loader.load(files)
# # n_workers to add
# dtr, dts = shuffle_and_split(data=data, val_ratio=args.ratio)


# # ---------------------------------------------------
# #   Train the Path Model on the Loaded Data
# # ---------------------------------------------------

# from src.models.chanmod import ChannelModel
# model =  ChannelModel(directory=args.directory, path_name=args.name)
# model.path_model.build()
# model.path_model.fit(dtr=dtr, dts=dts, epochs=args.epochs, 
#                      batch_size=args.batch_size, learning_rate=args.learning_rate)
# model.path_model.save()
