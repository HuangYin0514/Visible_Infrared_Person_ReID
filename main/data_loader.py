import random
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
                RandomGrayscale(),
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
            color_pos, thermal_pos = GenIdx(trainset.color_label, trainset.thermal_label)

            # testing set
            query_img, query_label, query_cam = process_query_sysu(config.DATASET.TRAIN_DATASET_PATH, mode=config.DATASET.MODE)
            gallery_img, gallery_label, gallery_cam = process_gallery_sysu(config.DATASET.TRAIN_DATASET_PATH, mode=config.DATASET.MODE, trial=0)

        queryset = TestDataset(query_img, query_label, transform=transform_test, img_size=config.DATALOADER.IMAGE_SIZE)
        gallset = TestDataset(gallery_img, gallery_label, transform=transform_test, img_size=config.DATALOADER.IMAGE_SIZE)

        query_loader = data.DataLoader(queryset, batch_size=config.DATALOADER.TEST_BATCH, shuffle=False, num_workers=config.DATALOADER.NUM_WORKERS)
        gallery_loader = data.DataLoader(gallset, batch_size=config.DATALOADER.TEST_BATCH, shuffle=False, num_workers=config.DATALOADER.NUM_WORKERS)

        ###################################################################################################
        # Print dataset statistics
        N_class = len(np.unique(trainset.color_label))
        N_query = len(query_label)
        N_gallery = len(gallery_label)

        print("Dataset {} statistics:".format(config.DATASET.TRAIN_DATASET))
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  visible  | {:5d} | {:8d}".format(N_class, len(trainset.color_label)))
        print("  thermal  | {:5d} | {:8d}".format(N_class, len(trainset.thermal_label)))
        print("  ------------------------------")
        print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), N_query))
        print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gallery_label)), N_gallery))
        print("  ------------------------------")
        print("Data Loading Time:\t {:.3f}".format(time.time() - end))

        ###################################################################################################
        # Set values
        self.trainset = trainset
        self.color_pos = color_pos
        self.thermal_pos = thermal_pos

        self.query_loader = query_loader
        self.query_label = query_label
        self.query_cam = query_cam
        self.gallery_loader = gallery_loader
        self.gallery_label = gallery_label
        self.gallery_cam = gallery_cam

        self.N_class = N_class
        self.N_query = N_query
        self.N_gallery = N_gallery


class Dataset4Sysu_mm01(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex=None):

        # Load training images (path) and labels
        color_image = np.load(data_dir + "train_rgb_resized_img_288_144.npy")
        self.color_label = np.load(data_dir + "train_rgb_resized_label_288_144.npy")

        thermal_image = np.load(data_dir + "train_ir_resized_img_288_144.npy")
        self.thermal_label = np.load(data_dir + "train_ir_resized_label_288_144.npy")

        # BGR to RGB
        self.color_image = color_image
        self.thermal_image = thermal_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_vis = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Pad(10),
                # transforms.RandomCrop((288, 144)),
                transforms.RandomCrop((256, 192)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                RandomErasing(probability=0.5, mode="pixel", max_count=1, device="cpu"),
                ChannelAdapGray(probability=1.0),
            ]
        )
        self.transform_inf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Pad(10),
                # transforms.RandomCrop((288, 144)),
                transforms.RandomCrop((256, 192)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                RandomErasing(probability=0.5, mode="pixel", max_count=1, device="cpu"),
                AdapGray(probability=0.5),
            ]
        )

    def __getitem__(self, index):

        img1, target1 = self.color_image[self.cIndex[index]], self.color_label[self.cIndex[index]]
        img2, target2 = self.thermal_image[self.tIndex[index]], self.thermal_label[self.tIndex[index]]

        img1 = self.transform_vis(img1)
        img2 = self.transform_inf(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)


class TestDataset(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size=(144, 288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[1], img_size[0]), Image.LANCZOS)
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


###################################################################################################################
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


class RandomGrayscale:
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        op = random.randint(0, 10)

        if op == 0:
            x[1] = x[2] = x[0]
        elif op == 1:
            x[0] = x[2] = x[1]
        elif op == 2:
            x[0] = x[1] = x[2]

        return x


class ChannelAdapGray(object):
    """Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):

        # if random.uniform(0, 1) > self.probability:
        # return img

        idx = random.randint(0, 3)

        if idx == 0:
            # random select R Channel
            img[1, :, :] = img[0, :, :]
            img[2, :, :] = img[0, :, :]
        elif idx == 1:
            # random select B Channel
            img[0, :, :] = img[1, :, :]
            img[2, :, :] = img[1, :, :]
        elif idx == 2:
            # random select G Channel
            img[0, :, :] = img[2, :, :]
            img[1, :, :] = img[2, :, :]
        else:
            if random.uniform(0, 1) > self.probability:
                # return img
                img = img
            else:
                tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]  # float trabsform to gray image
                img[0, :, :] = tmp_img
                img[1, :, :] = tmp_img
                img[2, :, :] = tmp_img
        return img


class ChannelExchange(object):
    """Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, gray=2):
        self.gray = gray

    def __call__(self, img):

        idx = random.randint(0, self.gray)

        if idx == 0:
            # random select R Channel
            img[1, :, :] = img[0, :, :]
            img[2, :, :] = img[0, :, :]
        elif idx == 1:
            # random select B Channel
            img[0, :, :] = img[1, :, :]
            img[2, :, :] = img[1, :, :]
        elif idx == 2:
            # random select G Channel
            img[0, :, :] = img[2, :, :]
            img[1, :, :] = img[2, :, :]
        else:
            tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
            img[0, :, :] = tmp_img
            img[1, :, :] = tmp_img
            img[2, :, :] = tmp_img
        return img


class GrayExchange(object):
    def __init__(self, gray=2):
        self.gray = gray

    def __call__(self, img):

        tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
        img[0, :, :] = tmp_img
        img[1, :, :] = tmp_img
        img[2, :, :] = tmp_img
        return img


class AdapGray(object):
    """Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):

        # if random.uniform(0, 1) > self.probability:
        # return img

        if random.uniform(0, 1) > self.probability:
            # return img
            img = img
        else:
            tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
            img[0, :, :] = tmp_img
            img[1, :, :] = tmp_img
            img[2, :, :] = tmp_img
        return img


class reAdapGray(object):
    """Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):

        # if random.uniform(0, 1) > self.probability:
        # return img

        if random.uniform(0, 1) > self.probability:
            # return img
            img = img
        else:
            img[0, :, :] = 0.2989 * img[0, :, :]
            img[1, :, :] = 0.5870 * img[1, :, :]
            img[2, :, :] = 0.1140 * img[2, :, :]
        return img
