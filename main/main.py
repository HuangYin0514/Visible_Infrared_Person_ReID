import argparse
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import util
from criterion import Criterion
from data import Data_Loder, IdentitySampler
from eval_metrics import eval_regdb, eval_sysu
from model import ReIDNet
from optimizer import Optimizer
from scheduler import Scheduler
from tqdm import tqdm

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

    ######################################################################
    # Criterion
    criterion = Criterion(config)

    ######################################################################
    # Optimizer
    optimizer = Optimizer(config, net).optimizer

    ######################################################################
    # Scheduler
    scheduler = Scheduler(config, optimizer)

    ######################################################################
    # Scheduler
    print("==> Start Training...")
    # 初始化最佳指标
    best_epoch, best_mAP, best_rank1 = 0, 0, 0
    for epoch in range(0, config.OPTIMIZER.TOTAL_TRAIN_EPOCH):
        #########
        # data
        #########
        # print("==> Preparing Data Loader...")
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
            num_workers=config.DATALOADER.NUM_WORKERS,
            drop_last=True,
        )

        #########
        # train
        #########
        scheduler.lr_scheduler.step(epoch)
        meter = util.MultiItemAverageMeter()
        for batch_idx, (vis_imgs, inf_imgs, vis_labels, inf_labels) in enumerate(tqdm(trainloader)):

            net.train()
            if config.MODEL.MODULE == "Lucky":
                total_loss = 0
                B = vis_imgs.size(0) * 2

                vis_labels, inf_labels = vis_labels.to(DEVICE), inf_labels.to(DEVICE)
                labels = torch.cat([vis_labels, inf_labels], 0)
                vis_imgs, inf_imgs = vis_imgs.to(DEVICE), inf_imgs.to(DEVICE)

                backbone_feat_map = net(vis_imgs, inf_imgs, modal="all")

                # ------------- Partialization -----------------------
                num_stripes = 6
                stripe_h = int(18 / num_stripes)
                local_feat_list = []
                for i in range(num_stripes):
                    # gm pool
                    local_feat = backbone_feat_map[:, :, i * stripe_h : (i + 1) * stripe_h, :]
                    local_feat = local_feat.view(B, 2048, -1)
                    p = 10.0  # regDB: 10.0    SYSU: 3.0
                    local_feat = (torch.mean(local_feat**p, dim=-1) + 1e-12) ** (1 / p)
                    local_feat = net.local_conv_list[i](local_feat.view(B, 2048, 1, 1))
                    local_feat = local_feat.view(B, -1)
                    local_feat_list.append(local_feat)

                # ----------- Global ------------
                global_feat = torch.cat(local_feat_list, dim=1)
                global_ctl_loss = criterion.ctl(global_feat, labels)[0]
                global_loss = global_ctl_loss

                # ----------- Local ------------
                local_loss = 0
                for i in range(num_stripes):
                    local_feat_i = local_feat_list[i]
                    local_bn_feat, local_cls_score = net.local_classifier_list[i](local_feat_i)
                    local_pid_loss = criterion.id(local_cls_score, labels)
                    local_ctl_loss = criterion.ctl(local_feat_i, labels)[0]
                    local_loss += local_pid_loss + local_ctl_loss * 2.0

                total_loss += local_loss + global_loss
                meter.update(
                    {
                        "local_loss": local_loss.item(),
                        "global_loss": global_loss.item(),
                    }
                )
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        logger("Time: {}; Epoch: {}; {}".format(util.time_now(), epoch, meter.get_str()))
        wandb.log({"Lr": optimizer.param_groups[0]["lr"], **meter.get_dict()})

        #########
        # Test
        #########
        if epoch % config.TEST.EVAL_EPOCH == 0:
            net.eval()
            feat_dim = 512 * 6  # 2048
            query_feat = np.zeros((data_loder.N_query, feat_dim))
            gall_feat = np.zeros((data_loder.N_gallery, feat_dim))
            loaders = [data_loder.query_loader, data_loder.gallery_loader]
            print(util.time_now(), "Start extracting features...")
            with torch.no_grad():
                for loader_id, loader in enumerate(loaders):
                    if config.DATASET.TRAIN_DATASET == "sysu_mm01":
                        modal_map = {0: "inf", 1: "vis"}
                        modal = modal_map.get(loader_id)
                    elif config.DATASET.TRAIN_DATASET == "reg_db":
                        modal_map = {1: "inf", 0: "vis"}
                        modal = modal_map.get(loader_id)
                    ptr = 0
                    for imgs, labels in loader:
                        batch_num = imgs.size(0)
                        imgs = imgs.to(DEVICE)

                        bn_feat = net(imgs, imgs, modal)
                        flip_imgs = torch.flip(imgs, [3])
                        flip_bn_feat = net(flip_imgs, flip_imgs, modal)
                        bn_feat = bn_feat + flip_bn_feat

                        if loader_id == 0:
                            query_feat[ptr : ptr + batch_num, :] = bn_feat.detach().cpu().numpy()
                        elif loader_id == 1:
                            gall_feat[ptr : ptr + batch_num, :] = bn_feat.detach().cpu().numpy()

                        ptr = ptr + batch_num

            # compute the similarity
            distmat = np.matmul(query_feat, np.transpose(gall_feat))

            cmc, mAP = None, None
            if config.DATASET.TRAIN_DATASET == "sysu_mm01":
                cmc, mAP, mINP = eval_sysu(
                    -distmat,
                    data_loder.query_label,
                    data_loder.gallery_label,
                    data_loder.query_cam,
                    data_loder.gallery_cam,
                )
            elif config.DATASET.TRAIN_DATASET == "reg_db":
                cmc, mAP, mINP = eval_regdb(
                    -distmat,
                    data_loder.query_label,
                    data_loder.gallery_label,
                )

            is_best_rank_flag = cmc[0] >= best_rank1
            if is_best_rank_flag:
                best_epoch = epoch
                best_rank1 = cmc[0]
                best_mAP = mAP
                wandb.log({"best_epoch": best_epoch, "best_rank1": best_rank1, "best_mAP": best_mAP})
                # if epoch>50:
                #     util.save_model(
                #         model=net,
                #         epoch=epoch,
                #         path_dir=os.path.join(config.SAVE.OUTPUT_PATH, "models/"),
                #     )

            logger("Time: {}; Test on Dataset: {}, \nmAP: {} \nRank: {}".format(util.time_now(), config.DATASET.TRAIN_DATASET, mAP, cmc))
            wandb.log({"test_epoch": epoch, "mAP": mAP, "Rank1": cmc[0]})

    logger("=" * 50)
    logger("Best model is: epoch: {}, rank1: {}, mAP: {}".format(best_epoch, best_rank1, best_mAP))
    logger("=" * 50)


if __name__ == "__main__":
    args = get_args()
    config = util.load_config(args.config_file, args.opts)
    util.set_seed_torch(config.TASK.SEED)

    # 初始化wandb
    wandb.init(
        entity="yinhuang-team-projects",
        project=config.TASK.PROJECT,
        name=config.TASK.NAME,
        notes=config.TASK.NOTES,
        tags=config.TASK.TAGS,
        config=config,
    )
    run(config)
    wandb.finish()
