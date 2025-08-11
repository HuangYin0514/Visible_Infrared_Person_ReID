import torch
from method import evaluation
from tools import CatMeter, time_now


def test(config, base, loader):
    # 设置模型为评估模式
    base.set_eval()

    # 初始化度量器，用于存储查询和图库的特征、身份ID和摄像头ID
    query_features_meter, query_pids_meter, query_cids_meter = CatMeter(), CatMeter(), CatMeter()
    gallery_features_meter, gallery_pids_meter, gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()

    # 获取查询和图库的加载器
    loaders = [loader.query_loader, loader.gallery_loader]

    # 提取查询和图库的特征
    print(time_now(), "Start extracting features...")
    with torch.no_grad():
        for loader_id, loader in enumerate(loaders):
            for data in loader:
                images, pids, cids = data
                images = images.to(base.device)

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
    print(time_now(), "Feature extraction completed.")

    # 获取查询和图库的特征值
    query_features = query_features_meter.get_val_numpy()
    gallery_features = gallery_features_meter.get_val_numpy()

    # 评估模型，计算mAP和CMC
    mAP, CMC = evaluation.ReIDEvaluator(dist="cosine", mode=config.TEST.TEST_MODE).evaluate(query_features, query_pids_meter.get_val_numpy(), query_cids_meter.get_val_numpy(), gallery_features, gallery_pids_meter.get_val_numpy(), gallery_cids_meter.get_val_numpy())

    return mAP, CMC[0:20]
