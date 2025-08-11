import torchvision.transforms as transforms
from data_loader.dataset import (
    Dataset,
    Samples4Duke,
    Samples4Market,
    Samples4MSMT17,
    Samples4OccludedDuke,
    Samples4OccludedReID,
    Samples4PartialDuke,
    TestSamples4Duke,
    TestSamples4Market,
    TestSamples4MSMT17,
    TestSamples4OccludedDuke,
    TestSamples4OccludedReid,
    TestSamples4PartialDuke,
    TestSamples4PartialiLIDS,
    TestSamples4PartialReID,
    VisualizationDataset,
)
from data_loader.preprocessing import RandomErasing
from data_loader.sampler import TripletSampler
from torch.utils.data import DataLoader


class Loader:

    def __init__(self, config):
        transform_train = [
            transforms.Resize(config.DATALOADER.IMAGE_SIZE, interpolation=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop(config.DATALOADER.IMAGE_SIZE),
        ]
        if config.DATALOADER.USE_COLORJITOR:
            transform_train.append(transforms.ColorJitter(brightness=0.25, contrast=0.15, saturation=0.25, hue=0))
        transform_train.extend([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        if config.DATALOADER.USE_REA:
            transform_train.append(RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]))
        self.transform_train = transforms.Compose(transform_train)

        self.transform_test = transforms.Compose(
            [
                transforms.Resize(config.DATALOADER.IMAGE_SIZE, interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.train_dataset = config.DATASET.TRAIN_DATASET
        self.test_dataset = config.DATASET.TEST_DATASET
        self.train_dataset_path = config.DATASET.TRAIN_DATASET_PATH
        self.test_dataset_path = config.DATASET.TEST_DATASET_PATH

        self.batchsize = config.DATALOADER.BATCHSIZE
        self.num_instances = config.DATALOADER.NUM_INSTANCES

        if config.TASK.MODE == "train":
            self._load()
        elif config.TASK.MODE == "visualization":
            self._visualization_load()

    def _load(self):
        samples = self._get_samples(self.train_dataset).samples
        self.train_loader = self._get_train_loader(samples, self.transform_train, self.batchsize)
        query_samples, gallery_samples = self._get_test_samples(self.test_dataset)
        self.query_loader = self._get_test_loader(query_samples, self.transform_test, 128)
        self.gallery_loader = self._get_test_loader(gallery_samples, self.transform_test, 128)

    def _get_train_loader(self, samples, transform, batchsize):
        dataset = Dataset(samples, transform=transform)
        loader = DataLoader(dataset, batch_size=batchsize, sampler=TripletSampler(dataset.samples, batchsize, self.num_instances), num_workers=8)
        return loader

    def _get_test_loader(self, samples, transform, batch_size):
        dataset = Dataset(samples, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=False)
        return loader

    def _visualization_load(self):
        samples = self._get_samples(self.train_dataset).samples
        self.visualization_train_loader = self._get_visualization_loader(samples, self.transform_test, self.batchsize)
        query_samples, gallery_samples = self._get_test_samples(self.test_dataset)
        self.visualization_query_loader = self._get_visualization_loader(query_samples, self.transform_test, 128)
        self.visualization_gallery_loader = self._get_visualization_loader(gallery_samples, self.transform_test, 128)

    def _get_visualization_loader(self, samples, transform, batch_size):
        dataset = VisualizationDataset(samples, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=False)
        return loader

    def _get_samples(self, dataset):
        if dataset == "occluded_duke":
            samples = Samples4OccludedDuke(self.train_dataset_path)
        elif dataset == "occluded_reid":
            samples = Samples4OccludedReID(self.train_dataset_path)
        elif dataset == "partial_duke":
            samples = Samples4PartialDuke(self.train_dataset_path)
        elif dataset == "market":
            samples = Samples4Market(self.train_dataset_path)
        elif dataset == "duke":
            samples = Samples4Duke(self.train_dataset_path)
        elif dataset == "msmt":
            samples = Samples4MSMT17(self.train_dataset_path)
        return samples

    def _get_test_samples(self, dataset):
        if dataset == "occluded_duke":
            query_samples = TestSamples4OccludedDuke(self.test_dataset_path).query_samples
            gallery_samples = TestSamples4OccludedDuke(self.test_dataset_path).gallery_samples
        elif dataset == "occluded_reid":
            query_samples = TestSamples4OccludedReid(self.test_dataset_path).query_samples
            gallery_samples = TestSamples4OccludedReid(self.test_dataset_path).gallery_samples
        elif dataset == "partial_duke":
            query_samples = TestSamples4PartialDuke(self.test_dataset_path).query_samples
            gallery_samples = TestSamples4PartialDuke(self.test_dataset_path).gallery_samples
        elif dataset == "partial_reid":
            query_samples = TestSamples4PartialReID(self.test_dataset_path).query_samples
            gallery_samples = TestSamples4PartialReID(self.test_dataset_path).gallery_samples
        elif dataset == "partial_ilids":
            query_samples = TestSamples4PartialiLIDS(self.test_dataset_path).query_samples
            gallery_samples = TestSamples4PartialiLIDS(self.test_dataset_path).gallery_samples
        elif dataset == "market":
            query_samples = TestSamples4Market(self.test_dataset_path).query_samples
            gallery_samples = TestSamples4Market(self.test_dataset_path).gallery_samples
        elif dataset == "duke":
            query_samples = TestSamples4Duke(self.test_dataset_path).query_samples
            gallery_samples = TestSamples4Duke(self.test_dataset_path).gallery_samples
        elif dataset == "msmt":
            query_samples = TestSamples4MSMT17(self.test_dataset_path).query_samples
            gallery_samples = TestSamples4MSMT17(self.test_dataset_path).gallery_samples

        return query_samples, gallery_samples
