from base import BaseDataLoader
from .cifar_data_loaders import ImbalanceCIFAR10DataLoader, CIFAR100DataLoader, ImbalanceCIFAR100DataLoader
from .imagenet_lt_data_loaders import ImageNetLTDataLoader
from .inaturalist_data_loaders import iNaturalistDataLoader
from torchvision import transforms, datasets
import torch
import numpy as np

class ImbalanceCIFAR10DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1, training=True, imb_factor=0.1, resize_size=None):
        self.data_dir = data_dir
        
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        
        # Add resize if specified
        if resize_size:
            transform_list.insert(0, transforms.Resize(resize_size))
            
        self.transform = transforms.Compose(transform_list)

        # Get CIFAR10 dataset
        self.train_dataset = datasets.CIFAR10(self.data_dir, train=True, download=True, transform=self.transform)
        self.val_dataset = datasets.CIFAR10(self.data_dir, train=False, download=True, transform=self.transform)

        # Get number of classes
        self.n_classes = len(self.train_dataset.classes)

        # Create imbalanced dataset
        if training:
            img_num_list = self.get_img_num_per_cls(self.n_classes, imb_factor)
            self.gen_imbalanced_data(img_num_list)
            self.cls_num_list = img_num_list
        
        # Initialize base class
        super().__init__(self.train_dataset, batch_size, shuffle, validation_split=0.0, num_workers=num_workers)

    def get_img_num_per_cls(self, num_classes, imb_factor):
        """Get number of images per class"""
        img_max = len(self.train_dataset.data) / num_classes
        img_num_per_cls = []
        for cls_idx in range(num_classes):
            num = img_max * (imb_factor**(cls_idx / (num_classes - 1.0)))
            img_num_per_cls.append(int(num))
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        """Generate imbalanced data"""
        new_data = []
        new_targets = []
        targets_np = np.array(self.train_dataset.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        for the_class, the_img_num in zip(classes, img_num_per_cls):
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.train_dataset.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)

        new_data = np.vstack(new_data)
        self.train_dataset.data = new_data
        self.train_dataset.targets = new_targets