import os

import torch
import torch.nn as nn
import torch.optim as optim
from method import Model, scheduler
from tools import Logger, make_dirs, os_walk, time_now


class Base:
    def __init__(self, config):
        self.config = config

        self.pid_num = config.DATASET.PID_NUM
        self.device = torch.device(config.TASK.DEVICE)

        self.max_save_model_num = config.SAVE.MAX_SAVE_MODEL_NUM
        self.output_path = config.SAVE.OUTPUT_PATH
        self.save_model_path = os.path.join(self.output_path, "models/")
        self.save_logs_path = os.path.join(self.output_path, "logs/")
        make_dirs(self.output_path)
        make_dirs(self.save_model_path)
        make_dirs(self.save_logs_path)

        self.learning_rate = config.SOLVER.LEARNING_RATE
        self.weight_decay = config.SOLVER.WEIGHT_DECAY
        self.milestones = config.SOLVER.MILESTONES

        self._init_model()
        self._init_optimizer()

    def _init_model(self):
        self.model = Model(self.config)
        self.model = nn.DataParallel(self.model).to(self.device)

    def _init_optimizer(self):
        model_params_group = [{"params": self.model.parameters(), "lr": self.learning_rate, "weight_decay": self.weight_decay}]
        self.model_optimizer = optim.Adam(model_params_group)
        self.model_lr_scheduler = scheduler.WarmupMultiStepLR(self.model_optimizer, self.milestones, gamma=0.1, warmup_factor=0.01, warmup_iters=10)

    def save_model(self, save_epoch, is_best):
        if is_best:
            model_file_path = os.path.join(self.save_model_path, "model_{}.pth".format(save_epoch))
            torch.save(self.model.state_dict(), model_file_path)

        # 移除多余文件
        if self.max_save_model_num > 0:
            root, _, files = os_walk(self.save_model_path)
            for file in files:
                if ".pth" not in file:
                    files.remove(file)
            if len(files) > 1 * self.max_save_model_num:
                file_iters = sorted([int(file.replace(".pth", "").split("_")[1]) for file in files], reverse=False)
                model_file_path = os.path.join(root, "model_{}.pth".format(file_iters[0]))
                os.remove(model_file_path)

    def resume_model(self, resume_epoch):
        model_path = os.path.join(self.save_model_path, "model_{}.pth".format(resume_epoch))
        self.model.load_state_dict(torch.load(model_path), strict=False)
        print("Successfully resume model from {}".format(model_path))

    def set_train(self):
        self.model = self.model.train()
        self.training = True

    def set_eval(self):
        self.model = self.model.eval()
        self.training = False
