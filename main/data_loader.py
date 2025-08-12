import time

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from data_manager import process_gallery_sysu, process_query_sysu
from PIL import Image
from random_erasing import RandomErasing


class Data_Loder:
    def __init__(self, config):
        self.name = "Data_Loder"
        self.load_data(config)

    def load_data(self, config):
        print("==> Loading data..")
        end = time.time()

        ###################################################################################################
        # Data loading code
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Pad(10),
                transforms.RandomCrop(config.DATALOADER.IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0]),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop(config.DATALOADER.IMAGE_SIZE),
                transforms.ToTensor(),
                normalize,
            ]
        )

        ###################################################################################################
        # Load data
        if config.DATASET.TRAIN_DATASET == "sysu_mm01":
            # training set
            trainset = Dataset4Sysu_mm01(data_dir=config.DATASET.TRAIN_DATASET_PATH, transform=transform_train)
            # generate the idx of each person identity
            color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

            # testing set
            query_img, query_label, query_cam = process_query_sysu(config.DATASET.TRAIN_DATASET_PATH, mode=config.DATASET.MODE)
            gallery_img, gallery_label, gallery_cam = process_gallery_sysu(config.DATASET.TRAIN_DATASET_PATH, mode=config.DATASET.MODE, trial=0)

        queryset = TestDataset(query_img, query_label, transform=transform_test, img_size=config.DATALOADER.IMAGE_SIZE)
        gallset = TestDataset(gallery_img, gallery_label, transform=transform_test, img_size=config.DATALOADER.IMAGE_SIZE)

        query_loader = data.DataLoader(queryset, batch_size=config.DATALOADER.TEST_BATCH, shuffle=False, num_workers=4)
        gallery_loader = data.DataLoader(gallset, batch_size=config.DATALOADER.TEST_BATCH, shuffle=False, num_workers=4)

        self.trainset = trainset
        self.color_pos = color_pos
        self.thermal_pos = thermal_pos
        self.query_loader = query_loader
        self.gallery_loader = gallery_loader

        ###################################################################################################
        # Print dataset statistics
        N_class = len(np.unique(trainset.train_color_label))
        N_query = len(query_label)
        N_gallery = len(gallery_label)

        print("Dataset {} statistics:".format(config.DATASET.TRAIN_DATASET))
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  visible  | {:5d} | {:8d}".format(N_class, len(trainset.train_color_label)))
        print("  thermal  | {:5d} | {:8d}".format(N_class, len(trainset.train_thermal_label)))
        print("  ------------------------------")
        print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), N_query))
        print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gallery_label)), N_gallery))
        print("  ------------------------------")
        print("Data Loading Time:\t {:.3f}".format(time.time() - end))
        ###################################################################################################


class Dataset4Sysu_mm01(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex=None):

        # Load training images (path) and labels
        train_color_image = np.load(data_dir + "train_rgb_resized_img.npy")
        self.train_color_label = np.load(data_dir + "train_rgb_resized_label.npy")

        train_thermal_image = np.load(data_dir + "train_ir_resized_img.npy")
        self.train_thermal_label = np.load(data_dir + "train_ir_resized_label.npy")

        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)


class TestDataset(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size=(192, 384)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)


def GenIdx(train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k, v in enumerate(train_color_label) if v == unique_label_color[i]]
        color_pos.append(tmp_pos)

    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k, v in enumerate(train_thermal_label) if v == unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)
    return color_pos, thermal_pos
