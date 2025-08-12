import argparse
import os
import warnings

import numpy as np
import torch
import util
from data_loader import Data_Loder
from logger import Logger

import wandb

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
    util.make_dirs(os.path.join(config.SAVE.OUTPUT_PATH, "logs/"))
    logger = Logger(file_path=os.path.join(config.SAVE.OUTPUT_PATH, "logs/", "logger.log"))
    logger(config)

    ######################################################################
    # Data
    data_loder = Data_Loder(config)

    ######################################################################
    # Model

    pass


if __name__ == "__main__":
    args = get_args()
    config = util.load_config(args.config_file, args.opts)
    util.set_seed_torch(config.TASK.SEED)

    # 初始化wandb
    wandb.init(
        entity="yinhuang-team-projects",
        project="VI_",
        name=config.TASK.NAME,
        notes=config.TASK.NOTES,
        tags=config.TASK.TAGS,
        config=config,
    )
    run(config)
    wandb.finish()
