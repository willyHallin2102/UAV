"""
"""
import os
import sys

path = os.path.abspath("..")
if not path in sys.path:
    sys.path.append(path)

import argparse
import logging
import random
import numpy as np
import tensorflow as tf

from pathlib import Path
from data.loader import DataLoader, shuffle_and_split
from src.models.chanmod import ChannelModel
from logs.logger import setup_logging, get_logger


def set_global_seed(seed: int):
    """
        Ensure reproducibility
    """
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Link Model in the ChannelModel on specific data"
    )
    parser.add_argument(
        "--city", type=str, default="beijing",
        help="Comma-separated list of cities or 'all'"
    )
    parser.add_argument(
        "--ratio", type=float, default=0.20,
        help="Validation split ratio"
    )
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=512,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for the optimizer")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging verbosity"
    )
    return parser.parse_args()


args = parse_args()
setup_logging(logger_directory=Path(args.city)/"links")
logger = get_logger(__name__)

set_global_seed(args.seed)

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


# ---------------------------------------------
#   Load data
# ---------------------------------------------

loader = DataLoader(debug_level=logging.INFO)
data = loader.load(files)
dtr, dts = shuffle_and_split(data=data, val_ratio=args.ratio)
logger.info(f"Loaded {len(dtr)} training samples, {len(dts)} validation samples")


# ---------------------------------------------------
#   Build and train the model
# ---------------------------------------------------

model = ChannelModel(directory=args.city)
model.link.build()

logger.info(f"Starting training for {args.epochs} epochs, batch size {args.batch}")
history = model.link.fit(dtr=dtr, dts=dts,
                               epochs=args.epochs,
                               batch_size=args.batch,
                               learning_rate=args.learning_rate,)
# Save training history
history_file = os.path.join(args.city, f"{args.city}_history.json")
try:
    import json
    with open(history_file, "w") as f:
        json.dump(history.history, f)
    logger.info(f"Saved training history to {history_file}")
except Exception as e:
    logger.warning(f"Could not save training history: {e}")

# Save the model
model.link.save()
logger.info(f"Model saved to {args.city}")
