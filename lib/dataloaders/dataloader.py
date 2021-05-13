import os
import sys
sys.path.append('/home/zongdaoming/cv/multi-organ/multi-organ-ijcai')
import shutil
import argparse
import logging
import time
import random
import numpy as np
import logging
import SimpleITK as sitk
import medpy
import nibabel as nib

import types
import torch
from torch.utils.data import Dataset
import itertools
from torch.utils.data.sampler import Sampler
from torchvision.utils import make_grid
from torchvision import transforms
from glob import glob
import torch.nn.functional as F
import random
import yaml


def setup_config(default_path="config.yaml"):
    # Setup Config files
    if os.path.exists(default_path):
        with open(default_path, "r") as f:
            config = yaml.load(f)
            return config
    try:
        if not os.path.exists(default_path):
            print("Path is not exists, check!")
            return None
    except Exception:
        print("Error: File not find")


def read_nii_gz(filename):
    itk_img = sitk.ReadImage(filename)
    img_or_label_arr = sitk.GetArrayFromImage(itk_img)
    # print(img_array.shape)
    # output: (frame_num, width, height) (574, 512, 512)
    print(f"Image or label shape is {img_or_label_arr.shape}")
    return img_or_label_arr


def load_nii_gz(filename):
    img_or_label_arr = nib.load(filename)
    # Image array shape is (512, 512, 574)
    # print(f"Image or label shape is {img_or_label_arr.shape}")
    return img_or_label_arr


def loading_nii(filename):
    # image_data is a numpy ndarray with the image data and
    # image_header is a header object holding the associated metadata.
    from medpy.io import load
    image_data, image_header = load(filename)
    # print(f"Image or label shape is: {image_data.shape}")
    return image_data


class LiverDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        # make sure label match image
        self._base_dir = base_dir
        self.transform = transform
        # assert os.path.exists(base_dir), f"{base_dir} is not exists!"
        assert isinstance(base_dir, dict), f"{base_dir} error!"
        self.sample_list = []
        # self._base_dir —> dict {
        # 'image_tr': '/home/zongdaoming/cv/multi-organ/LiTS/Task03_Liver/imagesTr',
        # 'label_tr': '/home/zongdaoming/cv/multi-organ/LiTS/Task03_Liver/labelsTr',
        # 'image_ts': '/home/zongdaoming/cv/multi-organ/LiTS/Task03_Liver/imagesTs'
        # }
        if split == 'train':
            self.imagepth = self._base_dir["image_tr"]
            self.labelpth = self._base_dir['label_tr']
            for _, _, filenames in os.walk(self._base_dir['image_tr']):
                for filename in filenames:
                    self.sample_list.append((os.path.join(self.imagepth, filename),
                                             os.path.join(self.labelpth, filename)))
        elif split == 'test':
            with open(self._base_dir+'test.list', 'r') as f:
                # self.sample_list = f.readlines().split('\n')[0]
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]
        if num is not None:
            print(f"Total {len(self.sample_list)} samples.")

    def __getitem__(self, idx):
        image_pth = self.sample_list[idx][0]
        label_pth = self.sample_list[idx][1]
        image = loading_nii(image_pth)
        label = loading_nii(label_pth)
        # make sure only two num_classes, namely background+single organ
        label[np.where(label>1)]=0
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.sample_list)

class PancreasDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir=None, split='train', num = None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        # assert os.path.isdir(base_dir), f"{base_dir} Error!"
        self.sample_list = []
        if split == "train":
            self.imagepth = self._base_dir['image_tr']
            self.labelpth = self._base_dir['label_tr']
            for _, _, filenames in os.walk(self._base_dir['image_tr']):
                for filename in filenames:
                    self.sample_list.append((os.path.join(self.imagepth, filename),
                    os.path.join(self.labelpth, filename)))
        elif split ==  'test':
            with open(self._base_dir+"/test.list", 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n','') for item in self.sample_list]
        if num is not None:
            # print(f"Total {len(self.sample_list)} samples !")
            print("Total {} samples !".format(len(self.sample_list)))

    def __getitem__(self, idx):
        image_pth = self.sample_list[idx][0]
        label_pth = self.sample_list[idx][1]
        image = loading_nii(image_pth)
        label = loading_nii(label_pth)
        sample = {'image':image, 'label':label}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.sample_list)

class SpleenDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        # make sure label match image
        self._base_dir = base_dir
        self.transform = transform
        # assert os.path.exists(base_dir), f"{base_dir} is not exists!"
        assert isinstance(base_dir, dict), f"{base_dir} error!"
        self.sample_list = []
        # self._base_dir —> dict {
        # 'image_tr': '/home/zongdaoming/cv/multi-organ/LiTS/Task03_Liver/imagesTr',
        # 'label_tr': '/home/zongdaoming/cv/multi-organ/LiTS/Task03_Liver/labelsTr',
        # 'image_ts': '/home/zongdaoming/cv/multi-organ/LiTS/Task03_Liver/imagesTs'
        # }
        if split == 'train':
            self.imagepth = self._base_dir["image_tr"]
            self.labelpth = self._base_dir['label_tr']
            for _, _, filenames in os.walk(self._base_dir['image_tr']):
                for filename in filenames:
                    self.sample_list.append((os.path.join(self.imagepth, filename),
                                             os.path.join(self.labelpth, filename)))
        elif split == 'test':
            with open(self._base_dir+'test.list', 'r') as f:
                # self.sample_list = f.readlines().split('\n')[0]
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]
        if num is not None:
            print(f"Total {len(self.sample_list)} samples.")

    def __getitem__(self, idx):
        image_pth = self.sample_list[idx][0]
        label_pth = self.sample_list[idx][1]
        image = loading_nii(image_pth)
        label = loading_nii(label_pth)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.sample_list)




class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(
            image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros(
            (self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label, 'onehot_label': onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(
            1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


if __name__ == "__main__":
    config = setup_config(
        default_path="/home/zongdaoming/cv/multi-organ/multi-organ-ijcai/config.yaml")
    
    liver_data_dir = config["liver_data_pth"]
    # patch_str_list = config["train_settings"]['patch_size'].split(',')
    # patch_size = tuple(map(int,patch_str_list))
    patch_size = tuple(config["train_settings"]['liver']['patch_size'])
    db_train = LiverDataset(base_dir=liver_data_dir,
                            split='train',
                            transform=transforms.Compose([
                                RandomRotFlip(),
                                RandomCrop(patch_size),
                                ToTensor(),
                            ])
                            )

    pancreas_data_dir = config['pancreas_data_pth']
    patch_size = tuple(config["train_settings"]['pancreas']['patch_size'])
    db_train = PancreasDataset(base_dir=pancreas_data_dir,
                            split='train',
                            transform=transforms.Compose([
                                RandomRotFlip(),
                                RandomCrop(patch_size),
                                ToTensor(),
                            ])
                            )

    pass
