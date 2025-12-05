# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com
from gazefollower.misc import DefaultConfig

from gazefollower import GazeFollower

config = DefaultConfig()
config.cali_mode = 9  # 5-, 9-, 13-point calibration

gf = GazeFollower(config=config)
