import argparse
import ast
import os
import random
import warnings

import numpy as np
import torch
from config import ConfigNode, load_config
from core import Base, test, train, visualization
from data_loader.loader import Loader
from tools import Logger, make_dirs, os_walk, time_now

import wandb

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="main/cfg/test.yml", help="path to config file")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args


def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(config):
    print("#" * 10)
    print("Main running...")

    # 加载数据
    print("Loading data...")
    loaders = Loader(config)

    # 加载模型
    print("Loading model...")
    model = Base(config)

    # 初始化最佳指标
    best_mAP = 0
    best_rank1 = 0
    best_epoch = 0

    # 初始化日志记录器
    logger_path = os.path.join(config.SAVE.OUTPUT_PATH, "logs/", "logger.log")
    logger = Logger(logger_path)
    logger("\n" * 3)
    logger(config)

    if config.TASK.MODE == "train":
        ########################################################
        # 训练
        ########################################################
        for current_epoch in range(0, config.SOLVER.TOTAL_TRAIN_EPOCH):
            model.model_lr_scheduler.step(current_epoch)
            results_meter = train(model, loaders, config)
            logger("Time: {}; Epoch: {}; {}".format(time_now(), current_epoch, results_meter.get_str()))
            wandb.log({"Lr": model.model_optimizer.param_groups[0]["lr"], **results_meter.get_dict()})

            # 每隔一定的epoch进行评估
            if current_epoch + 1 >= 1 and (current_epoch + 1) % config.SOLVER.EVAL_EPOCH == 0:
                mAP, CMC = test(config, model, loaders)
                is_best_rank = CMC[0] >= best_rank1
                if is_best_rank:
                    best_epoch = current_epoch
                    best_rank1 = CMC[0]
                    best_mAP = mAP
                    wandb.log({"best_epoch": best_epoch, "best_rank1": best_rank1, "best_mAP": best_mAP})
                model.save_model(current_epoch, is_best_rank)
                logger("Time: {}; Test on Dataset: {}, \nmAP: {} \nRank: {}".format(time_now(), config.DATASET.TEST_DATASET, mAP, CMC))
                wandb.log({"test_epoch": current_epoch, "mAP": mAP, "Rank1": CMC[0], "Rank5": CMC[4], "Rank10": CMC[9], "Rank20": CMC[19]})

        # 训练结束后打印最佳指标
        if best_rank1:
            logger("=" * 50)
            logger("Best model is: epoch: {}, rank1: {}, mAP: {}".format(best_epoch, best_rank1, best_mAP))
            logger("=" * 50)

    elif config.TASK.MODE == "test":
        ########################################################
        # 测试
        ########################################################
        model.resume_model(config.TEST.RESUME_TEST_MODEL)
        mAP, CMC = test(config, model, loaders)
        logger("Time: {}; Test on Dataset: {}, \nmAP: {} \n Rank: {}".format(time_now(), config.test_dataset, mAP, CMC))

    elif config.TASK.MODE == "visualization":
        ########################################################
        # 可视化
        ########################################################
        loaders._visualization_load()
        model.resume_model(config.TEST.RESUME_TEST_MODEL)
        visualization(config, model, loaders)


if __name__ == "__main__":
    args = get_args()
    config = load_config(args.config_file, args.opts)
    seed_torch(config.TASK.SEED)

    # 初始化wandb
    wandb.init(
        entity="yinhuang-team-projects",
        project="multi-view",
        name=config.TASK.NAME,
        notes=config.TASK.NOTES,
        tags=config.TASK.TAGS,
        config=config,
    )
    main(config)
    wandb.finish()
