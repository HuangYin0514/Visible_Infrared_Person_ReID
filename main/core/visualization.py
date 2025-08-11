import os
import random
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from method import evaluation
from sklearn.manifold import TSNE
from tools import CatMeter, time_now
from torch.nn import functional as F


def visualization(config, base, loader):
    visualization_heatmap(config, base, loader)
    visualization_rank(config, base, loader)
    visualization_tsne(config, base, loader)


def visualization_heatmap(config, base, loader):
    print(time_now(), "CAM start")
    base.set_eval()
    cam_loader = loader.visualization_train_loader  # visualization_train_loader/visualization_query_loader/visualization_gallery_loader
    heatmap_core = Heatmap_Core(config)
    with torch.no_grad():
        for index, data in enumerate(cam_loader):
            if index % 100 == 0:
                print(time_now(), "CAM: {}/{}".format(index, len(cam_loader)))
            images, pids, cids, path = data
            images = images.to(base.device)
            heatmap_core.__call__(images, base.model, base.model.module.global_classifier, pids)
            # break
    print(time_now(), "CAM done.")


def visualization_rank(config, base, loader):
    print(time_now(), "Visualization_ranked_results start")
    base.set_eval()
    loaders = [loader.visualization_query_loader, loader.visualization_gallery_loader]
    # ------------------------------------------------
    query_features_meter, query_pids_meter, query_cids_meter = CatMeter(), CatMeter(), CatMeter()
    gallery_features_meter, gallery_pids_meter, gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()
    with torch.no_grad():
        for loader_id, loader_i in enumerate(loaders):
            for data in loader_i:
                images, pids, cids, path = data
                bn_features = base.model(images)
                flip_images = torch.flip(images, [3])
                flip_bn_features = base.model(flip_images)
                bn_features = bn_features + flip_bn_features
                if loader_id == 0:
                    query_features_meter.update(bn_features.data)
                    query_pids_meter.update(pids)
                    query_cids_meter.update(cids)
                elif loader_id == 1:
                    gallery_features_meter.update(bn_features.data)
                    gallery_pids_meter.update(pids)
                    gallery_cids_meter.update(cids)

    query_features = query_features_meter.get_val_numpy()
    gallery_features = gallery_features_meter.get_val_numpy()

    mAP, CMC = evaluation.ReIDEvaluator(dist="cosine", mode=config.TEST.TEST_MODE).evaluate(
        query_features,
        query_pids_meter.get_val_numpy(),
        query_cids_meter.get_val_numpy(),
        gallery_features,
        gallery_pids_meter.get_val_numpy(),
        gallery_cids_meter.get_val_numpy(),
    )

    print("mAP: {:.2%}\t , CMC:{:.2%}".format(mAP, CMC[0]))

    # ------------------------------------------------
    # t_dir = os.path.join(config.output_path, "tmp")
    # if not os.path.exists(t_dir):
    #     os.makedirs(t_dir)
    #     print("Successfully make dirs: {}".format(dir))

    # torch.save(query_features, os.path.join(config.output_path, "tmp", "query_features" + ".pt"))
    # torch.save(gallery_features, os.path.join(config.output_path, "tmp", "gallery_features" + ".pt"))

    # query_features = torch.load(os.path.join(config.output_path, "tmp", "query_features" + ".pt"))
    # gallery_features = torch.load(os.path.join(config.output_path, "tmp", "gallery_features" + ".pt"))

    # ------------------------------------------------
    def cos_sim(x, y):
        def normalize(x):
            norm = np.tile(np.sqrt(np.sum(np.square(x), axis=1, keepdims=True)), [1, x.shape[1]])
            return x / norm

        x = normalize(x)
        y = normalize(y)
        return np.matmul(x, y.transpose([1, 0]))

    dist = cos_sim(query_features, gallery_features)
    rank_core = Rank_Core(config)
    rank_core.__call__(dist, [loaders[0].dataset, loaders[1].dataset])
    print(time_now(), "Visualization_ranked_results done.")
    return


def visualization_tsne(config, base, loader):
    print(time_now(), "t-SNE start")
    base.set_eval()
    loaders = [loader.visualization_query_loader, loader.visualization_gallery_loader]
    # ------------------------------------------------
    query_features_meter, query_pids_meter, query_cids_meter = CatMeter(), CatMeter(), CatMeter()
    gallery_features_meter, gallery_pids_meter, gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()
    with torch.no_grad():
        for loader_id, loader_i in enumerate(loaders):
            for data in loader_i:
                images, pids, cids, path = data
                bn_features = base.model(images)
                flip_images = torch.flip(images, [3])
                flip_bn_features = base.model(flip_images)
                bn_features = bn_features + flip_bn_features
                if loader_id == 0:
                    query_features_meter.update(bn_features.data)
                    query_pids_meter.update(pids)
                    query_cids_meter.update(cids)
                elif loader_id == 1:
                    gallery_features_meter.update(bn_features.data)
                    gallery_pids_meter.update(pids)
                    gallery_cids_meter.update(cids)

    query_features = query_features_meter.get_val_numpy()
    gallery_features = gallery_features_meter.get_val_numpy()
    query_pids_features = query_pids_meter.get_val_numpy()
    gallery_pids_features = gallery_pids_meter.get_val_numpy()

    mAP, CMC = evaluation.ReIDEvaluator(dist="cosine", mode=config.TEST.TEST_MODE).evaluate(
        query_features,
        query_pids_meter.get_val_numpy(),
        query_cids_meter.get_val_numpy(),
        gallery_features,
        gallery_pids_meter.get_val_numpy(),
        gallery_cids_meter.get_val_numpy(),
    )

    print("mAP: {:.2%}\t , CMC:{:.2%}".format(mAP, CMC[0]))
    # # ------------------------------------------------
    # t_dir = os.path.join(config.output_path, "tmp")
    # if not os.path.exists(t_dir):
    #     os.makedirs(t_dir)
    #     print("Successfully make dirs: {}".format(dir))

    # torch.save(query_features, os.path.join(config.output_path, "tmp", "query_features" + ".pt"))
    # torch.save(gallery_features, os.path.join(config.output_path, "tmp", "gallery_features" + ".pt"))
    # torch.save(query_pids_features, os.path.join(config.output_path, "tmp", "query_pids_features" + ".pt"))
    # torch.save(gallery_pids_features, os.path.join(config.output_path, "tmp", "gallery_pids_features" + ".pt"))

    # query_features = torch.load(os.path.join(config.output_path, "tmp", "query_features" + ".pt"))
    # gallery_features = torch.load(os.path.join(config.output_path, "tmp", "gallery_features" + ".pt"))
    # query_pids_features = torch.load(os.path.join(config.output_path, "tmp", "query_pids_features" + ".pt"))
    # gallery_pids_features = torch.load(os.path.join(config.output_path, "tmp", "gallery_pids_features" + ".pt"))
    # ------------------------------------------------

    t_dir = os.path.join(config.SAVE.OUTPUT_PATH, "tSNE")
    if not os.path.exists(t_dir):
        os.makedirs(t_dir)
        print("Successfully make dirs: {}".format(dir))

    print("query_features shape:", query_features.shape)
    print("gallery_features shape:", gallery_features.shape)
    print("query_pids_features shape:", query_pids_features.shape)
    print("gallery_pids_features shape:", gallery_pids_features.shape)

    # ------------------------------------------------
    # query 可视化结果
    # tsne = TSNE(n_components=2, random_state=42)
    # X_tsne = tsne.fit_transform(query_features)
    # plt.figure(figsize=(8, 5))
    # for idx, pid in enumerate(np.unique(query_pids_features)):
    #     if pid not in [577, 479, 4272, 834]:
    #         continue
    #     plt.scatter(X_tsne[query_pids_features == pid, 0], X_tsne[query_pids_features == pid, 1], label=f"Class {pid}")
    # plt.title("t-SNE Visualization of Simulated Data with Numpy")
    # plt.xlabel("Component 1")
    # plt.ylabel("Component 2")
    # plt.legend()
    # plt.savefig(os.path.join(t_dir, "query_tsne.png"))  # 保存图像
    # plt.show()
    # print(time_now(), "t-SNE done.")

    # ------------------------------------------------
    # # gallery 可视化结果
    # tsne = TSNE(n_components=2, random_state=42)
    # X_tsne = tsne.fit_transform(gallery_features)
    # plt.figure(figsize=(8, 5))
    # for idx, pid in enumerate(np.unique(gallery_pids_features)):
    #     if pid not in [577, 479, 4272, 834]:
    #         continue
    #     plt.scatter(X_tsne[gallery_pids_features == pid, 0], X_tsne[gallery_pids_features == pid, 1], label=f"Class {pid}")
    # plt.title("t-SNE Visualization of Simulated Data with Numpy")
    # plt.xlabel("Component 1")
    # plt.ylabel("Component 2")
    # plt.legend()
    # plt.savefig(os.path.join(t_dir, "gallery_tsne.png"))  # 保存图像
    # plt.show()
    # print(time_now(), "t-SNE done.")

    # ------------------------------------------------
    # all 可视化结果(查询模式)
    all_features = np.concatenate((query_features, gallery_features), axis=0)
    all_pids_features = np.concatenate((query_pids_features, gallery_pids_features), axis=0)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(len(all_pids_features) - 1, 30))
    X_tsne = tsne.fit_transform(all_features)
    plt.figure(figsize=(10, 10), dpi=300)

    # plot_list = [19, 21, 31, 33, 83]
    plot_list = list(set(all_pids_features))
    plot_list = random.sample(plot_list, min(5, len(plot_list)))

    for idx, pid in enumerate(np.unique(all_pids_features)):
        if pid not in plot_list:
            continue
        plt.scatter(X_tsne[all_pids_features == pid, 0], X_tsne[all_pids_features == pid, 1], label=f"Class {pid}")

    # Query Marked
    query_features_len = query_features.shape[0]
    mask = np.isin(query_pids_features, plot_list)
    plt.scatter(X_tsne[:query_features_len][mask, 0], X_tsne[:query_features_len][mask, 1], s=100, c="red", marker="x", alpha=0.8, label="Query Marked")

    plt.title("t-SNE Visualization of Simulated Data with Numpy")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.savefig(os.path.join(t_dir, "all_tsne.png"))
    # plt.show()
    print(time_now(), "t-SNE done.")


##########################################################
# Core
##########################################################


class Heatmap_Core:
    def __init__(self, config):
        super(Heatmap_Core, self).__init__()
        self.config = config

        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        self.GRID_SPACING = 10

        self.actmap_dir = os.path.join(config.SAVE.OUTPUT_PATH, "actmap/")
        if not os.path.exists(self.actmap_dir):
            os.makedirs(self.actmap_dir)
            print("Successfully make dirs: {}".format(dir))
        else:
            shutil.rmtree(self.actmap_dir)
            os.makedirs(self.actmap_dir)

    def actmap_fn(self, images, model, classifier, pids):
        _, _, height, width = images.shape
        features_map = model.module.heatmap(images)
        bs, c, h, w = features_map.shape

        # CAM
        classifier_params = [param for name, param in classifier.named_parameters()]
        heatmaps = torch.zeros((bs, h, w))
        for i in range(bs):
            heatmap_i = torch.matmul(classifier_params[-1][pids[i]].unsqueeze(0), features_map[i].unsqueeze(0).reshape(c, h * w)).detach()
            if heatmap_i.max() != 0:
                heatmap_i = (heatmap_i - heatmap_i.min()) / (heatmap_i.max() - heatmap_i.min())
            heatmap_i = heatmap_i.reshape(h, w)
            heatmaps[i] = heatmap_i

        # Channel
        # heatmaps = torch.abs(features_map)
        # # max_channel_indices = torch.argmax(heatmaps, dim=1, keepdim=True)[0]
        # # print(max_channel_indices, max_channel_indices.shape)
        # # heatmaps = torch.max(heatmaps[:, 476 : 476 + 1, :, :], dim=1, keepdim=True)[0]
        # heatmaps = torch.max(heatmaps, dim=1, keepdim=True)[0]
        # heatmaps = heatmaps.squeeze()

        heatmaps = heatmaps.view(bs, h * w)
        heatmaps = F.normalize(heatmaps, p=2, dim=1)
        heatmaps = heatmaps.view(bs, h, w)

        for j in range(bs):

            # Image
            img = images[j, ...]
            for t, m, s in zip(img, self.IMAGENET_MEAN, self.IMAGENET_STD):
                t.mul_(s).add_(m).clamp_(0, 1)
            img_np = np.uint8(np.floor(img.cpu().detach().numpy() * 255))
            img_np = img_np.transpose((1, 2, 0))  # (c, h, w) -> (h, w, c)

            # Activation map
            am = heatmaps[j, ...].cpu().detach().numpy()
            # am = outputs[j, 2:-2:, 2:-2].numpy()
            am = cv2.resize(am, (width, height))
            am = 255 * (am - np.min(am)) / (np.max(am) - np.min(am) + 1e-12)
            am = np.uint8(np.floor(am))
            am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

            # 重叠图像
            overlapped = img_np * 0.5 + am * 0.5
            overlapped[overlapped > 255] = 255
            overlapped = overlapped.astype(np.uint8)

            # from left to right: original image, activation map, overlapped image
            grid_img = 255 * np.ones((height, 3 * width + 2 * self.GRID_SPACING, 3), dtype=np.uint8)
            grid_img[:, :width, :] = img_np[:, :, ::-1]
            grid_img[:, width + self.GRID_SPACING : 2 * width + self.GRID_SPACING, :] = am
            grid_img[:, 2 * width + 2 * self.GRID_SPACING :, :] = overlapped

            random_number = random.randint(100000, 999999)
            cv2.imwrite(os.path.join(self.actmap_dir, str(pids[j].item()) + "_" + str(random_number) + ".jpg"), grid_img)

    def __call__(self, images, model, classifier, pids):
        # model.eval()
        # classifier.eval()
        self.actmap_fn(images, model, classifier, pids)
        # model.train()
        # classifier.train()


class Rank_Core:
    def __init__(self, config):
        self.config = config

        self.GRID_SPACING = 10
        self.QUERY_EXTRA_SPACING = 90
        self.BW = 5  # border width
        self.GREEN = (0, 255, 0)
        self.RED = (0, 0, 255)

        self.width = 128
        self.height = 256
        self.topk = 10
        self.data_type = "image"

        self.ranked_dir = os.path.join(config.SAVE.OUTPUT_PATH, "rank/")
        if not os.path.exists(self.ranked_dir):
            os.makedirs(self.ranked_dir)
            print("Successfully make dirs: {}".format(dir))
        else:
            shutil.rmtree(self.ranked_dir)
            os.makedirs(self.ranked_dir)

    def visualize_ranked_results(self, distmat, dataset, data_type, width=128, height=256, save_dir="", topk=10):
        print("Visualizing top-{} ranks ...".format(topk))
        num_q, num_g = distmat.shape
        print("# query: {}\t # gallery: {}".format(num_q, num_g))
        print("Visualizing top-{} ranks ...".format(topk))

        query, gallery = dataset
        indices = np.argsort(distmat)[:, ::-1]

        for q_idx in range(num_q):
            _, qpid, qcamid, qimg_path = query[q_idx]
            qimg_path_name = qimg_path[0] if isinstance(qimg_path, (tuple, list)) else qimg_path

            if data_type == "image":
                qimg = cv2.imread(qimg_path)
                qimg = cv2.resize(qimg, (width, height))
                qimg = cv2.copyMakeBorder(qimg, self.BW, self.BW, self.BW, self.BW, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                # resize twice to ensure that the border width is consistent across images
                qimg = cv2.resize(qimg, (width, height))
                num_cols = topk + 1
                grid_img = 255 * np.ones((height, num_cols * width + topk * self.GRID_SPACING + self.QUERY_EXTRA_SPACING, 3), dtype=np.uint8)
                grid_img[:, :width, :] = qimg

            rank_idx = 1
            for g_idx in indices[q_idx, :]:
                _, gpid, gcamid, gimg_path = gallery[g_idx]
                invalid = (qpid == gpid) & (qcamid == gcamid)
                if not invalid:
                    matched = gpid == qpid

                    # if matched and rank_idx == 1:  # 过滤, rank-1 错误的情况
                    #     continue

                    if data_type == "image":
                        border_color = self.GREEN if matched else self.RED
                        gimg = cv2.imread(gimg_path)
                        gimg = cv2.resize(gimg, (width, height))
                        gimg = cv2.copyMakeBorder(gimg, self.BW, self.BW, self.BW, self.BW, cv2.BORDER_CONSTANT, value=border_color)
                        gimg = cv2.resize(gimg, (width, height))
                        start = rank_idx * width + rank_idx * self.GRID_SPACING + self.QUERY_EXTRA_SPACING
                        end = (rank_idx + 1) * width + rank_idx * self.GRID_SPACING + self.QUERY_EXTRA_SPACING
                        grid_img[:, start:end, :] = gimg
                    rank_idx += 1

                    if rank_idx > topk:
                        break

            if data_type == "image":
                # if qpid != 19:  # 查询特定的行人图像
                #     continue
                imname = os.path.basename(os.path.splitext(qimg_path_name)[0])
                cv2.imwrite(os.path.join(save_dir, imname + ".jpg"), grid_img)

            if (q_idx + 1) % 100 == 0:
                print("- done {}/{}".format(q_idx + 1, num_q))

    def __call__(self, distmat, dataset):
        # model.eval()
        # classifier.eval()
        self.visualize_ranked_results(
            distmat,
            dataset,
            data_type=self.data_type,
            width=self.width,
            height=self.height,
            save_dir=self.ranked_dir,
            topk=self.topk,
        )
        # model.train()
        # classifier.train()
