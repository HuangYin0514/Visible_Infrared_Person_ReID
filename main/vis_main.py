import argparse
import os
import warnings

import torch
import util
from data import Data_Loder
from model import ReIDNet
from visualization import visualization

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="main/cfg/test.yml", help="path to config file")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args


def run(config):
    ######################################################################
    # Logger
    logger = util.Logger(path_dir=os.path.join(config.SAVE.OUTPUT_PATH, "logs/"), name="logger.log")
    logger(config)

    ######################################################################
    # Device
    DEVICE = torch.device(config.TASK.DEVICE)

    ######################################################################
    # Data
    data_loder = Data_Loder(config)

    ######################################################################
    # Model
    net = ReIDNet(config, data_loder.N_class).to(DEVICE)

    ########################################################
    # 可视化
    ########################################################
    visualization(config, net, data_loder)


if __name__ == "__main__":
    args = get_args()
    config = util.load_config(args.config_file, args.opts)
    util.set_seed_torch(config.TASK.SEED)
    run(config)
