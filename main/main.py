import argparse
import os
import warnings

import numpy as np
import torch
import torch.utils.data as data
import util
from criterion import Criterion
from data_loader import Data_Loder
from eval_metrics import eval_sysu
from identity_sampler import IdentitySampler
from logger import Logger
from model import ReIDNet
from optimizer import Optimizer
from scheduler import Scheduler

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
    # Device
    DEVICE = torch.device(config.TASK.DEVICE)

    ######################################################################
    # Data
    data_loder = Data_Loder(config)

    ######################################################################
    # Model
    net = ReIDNet(config, data_loder.N_class).to(DEVICE)

    ######################################################################
    # Criterion
    criterion = Criterion(config)

    ######################################################################
    # Optimizer
    optimizer = Optimizer(config, net).optimizer

    ######################################################################
    # Scheduler
    scheduler = Scheduler()

    ######################################################################
    # Scheduler
    print("==> Start Training...")
    for epoch in range(0, config.OPTIMIZER.TOTAL_TRAIN_EPOCH):
        #########
        # data
        #########
        print("==> Preparing Data Loader...")
        # identity sampler
        sampler = IdentitySampler(
            data_loder.trainset.color_label,
            data_loder.trainset.thermal_label,
            data_loder.color_pos,
            data_loder.thermal_pos,
            config.DATALOADER.NUM_INSTANCES,
            config.DATALOADER.BATCHSIZE,
            epoch,
        )
        data_loder.trainset.cIndex = sampler.index1  # color index
        data_loder.trainset.tIndex = sampler.index2  # thermal index
        # print(epoch)
        # print(data_loder.trainset.cIndex)
        # print(data_loder.trainset.tIndex)

        # dataloder
        loader_batch = config.DATALOADER.BATCHSIZE * config.DATALOADER.NUM_INSTANCES
        trainloader = data.DataLoader(
            data_loder.trainset,
            batch_size=loader_batch,
            sampler=sampler,
            num_workers=4,
            drop_last=True,
        )

        #########
        # train
        #########
        current_lr = scheduler.adjust_learning_rate(config, optimizer, epoch)
        meter = util.MultiItemAverageMeter()
        for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):
            net.train()
            if config.MODEL.MODULE == "Lucky":
                total_loss = 0

                labels = torch.cat([label1, label2], 0).to(DEVICE)
                input1, input2 = input1.to(DEVICE), input2.to(DEVICE)
                input = torch.cat([input1, input2], 0)

                backbone_feature_map = net(input)

                # Backbone
                backbone_feature = net.backbone_pooling(backbone_feature_map).squeeze()
                backbone_bn_features, backbone_cls_score = net.backbone_classifier(backbone_feature)
                global_pid_loss = criterion.id(backbone_cls_score, labels)
                total_loss += global_pid_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                meter.update({"global_pid_loss": global_pid_loss.item()})
        logger("Time: {}; Epoch: {}; {}".format(util.time_now(), epoch, meter.get_str()))
        wandb.log({"Lr": optimizer.param_groups[0]["lr"], **meter.get_dict()})

        #########
        # Test
        #########
        if epoch % config.SOLVER.EVAL_EPOCH == 0:
            net.eval()

            query_feat = np.zeros((data_loder.N_query, 2048))
            gall_feat = np.zeros((data_loder.N_gallery, 2048))
            loaders = [data_loder.query_loader, data_loder.gallery_loader]
            print(util.time_now(), "Start extracting features...")
            with torch.no_grad():
                for loader_id, loader in enumerate(loaders):
                    ptr = 0
                    for input, label in loader:
                        batch_num = input.size(0)

                        bn_features = net(input)
                        flip_images = torch.flip(input, [3])
                        flip_bn_features = net(flip_images)
                        bn_features = bn_features + flip_bn_features

                        if loader_id == 0:
                            query_feat[ptr : ptr + batch_num, :] = bn_features.detach().cpu().numpy()
                        elif loader_id == 1:
                            gall_feat[ptr : ptr + batch_num, :] = bn_features.detach().cpu().numpy()

            # compute the similarity
            distmat = np.matmul(query_feat, np.transpose(gall_feat))
            if config.DATASET.TRAIN_DATASET == "sysu_mm01":
                cmc, mAP, mINP = eval_sysu(
                    -distmat,
                    data_loder.query_label,
                    data_loder.gallery_label,
                    data_loder.query_cam,
                    data_loder.gallery_cam,
                )
                logger("Time: {}; Test on Dataset: {}, \nmAP: {} \nRank: {}".format(util.time_now(), config.DATASET.TRAIN_DATASET, mAP, cmc))


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
