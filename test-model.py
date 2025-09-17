import logging
from data.loader import DataLoader, shuffle_and_split
from src.models.chanmod import ChannelModel

loader = DataLoader(debug_level=logging.INFO)
data = loader.load("uav_london/train.csv")
dtr, dts = shuffle_and_split(data=data, val_ratio=0.05)

channel = ChannelModel()
# channel.link.build()
# channel.link.fit(dtr=dtr, dts=dts, epochs=100)
# channel.link.save()
channel.path.build()
channel.path.fit(dtr=dtr, dts=dts, epochs=10, batch_size=512)