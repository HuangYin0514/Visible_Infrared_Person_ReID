import os
import pdb

import numpy as np
from PIL import Image

DATA_PATH = "/Users/drhy/Documents/projects/Visible_Infrared_Person_ReID/_dataset_processing/SYSU_MM01_concise"
# DATA_PATH = "kaggle"
FIX_IMAGE_WIDTH = 192
FIX_IMAGE_HEIGHT = 384
RGB_CAMERAS = ["cam1", "cam2", "cam4", "cam5"]
IR_CAMERAS = ["cam3", "cam6"]

#################################################################
# Load id info from train_id.txt and val_id.txt
file_path_train = os.path.join(DATA_PATH, "exp/train_id.txt")
file_path_val = os.path.join(DATA_PATH, "exp/val_id.txt")
with open(file_path_train, "r") as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(",")]
    id_train = ["%04d" % x for x in ids]

with open(file_path_val, "r") as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(",")]
    id_val = ["%04d" % x for x in ids]

# combine train and val split
id_train.extend(id_val)

#################################################################
# Load Constructing an image path
files_rgb = []
files_ir = []
for id in sorted(id_train):
    for cam in RGB_CAMERAS:
        img_dir = os.path.join(DATA_PATH, cam, id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir + "/" + i for i in os.listdir(img_dir)])
            files_rgb.extend(new_files)

    for cam in IR_CAMERAS:
        img_dir = os.path.join(DATA_PATH, cam, id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir + "/" + i for i in os.listdir(img_dir)])
            files_ir.extend(new_files)

#################################################################
# relabel
pid_container = set()
for img_path in files_ir:
    pid = int(img_path[-13:-9])
    pid_container.add(pid)
pid2label = {pid: label for label, pid in enumerate(pid_container)}


def read_imgs(train_image):
    train_img = []
    train_label = []
    for img_path in train_image:
        # img
        img = Image.open(img_path)
        img = img.resize((FIX_IMAGE_WIDTH, FIX_IMAGE_HEIGHT), Image.LANCZOS)
        pix_array = np.array(img)

        train_img.append(pix_array)

        # label
        pid = int(img_path[-13:-9])
        pid = pid2label[pid]
        train_label.append(pid)
    return np.array(train_img), np.array(train_label)


#################################################################
# Save images
# rgb images
train_img, train_label = read_imgs(files_rgb)
np.save(DATA_PATH + "/train_rgb_resized_img.npy", train_img)
np.save(DATA_PATH + "/train_rgb_resized_label.npy", train_label)

# ir images
train_img, train_label = read_imgs(files_ir)
np.save(DATA_PATH + "/train_ir_resized_img.npy", train_img)
np.save(DATA_PATH + "/train_ir_resized_label.npy", train_label)
