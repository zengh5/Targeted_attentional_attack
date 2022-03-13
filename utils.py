# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import pathlib
from torch.utils.data.dataset import Subset

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

# TODO: need to refactor path to keep a single copy of file

ROOT_PATH = os.path.expanduser("~/.advertorch")
DATA_PATH = os.path.join(ROOT_PATH, "data")

path_of_this_module = os.path.dirname(sys.modules[__name__].__file__)
TRAINED_MODEL_PATH = os.path.join(path_of_this_module, "trained_models")


def mkdir(directory):
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


def get_madry_et_al_cifar10_train_transform():
    return transforms.Compose([
        transforms.Pad(4, padding_mode="reflect"),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])



def get_train_val_loaders(
        dataset, datapath=DATA_PATH,
        train_size=None, val_size=5000,
        train_batch_size=100, val_batch_size=1000,
        kwargs=None, train_transform=None, val_transform=None,
        train_shuffle=True, val_shuffle=False):
    """Support MNIST and CIFAR10"""
    if kwargs is None:
        kwargs = {}
    if train_transform is None:
        train_transform = transforms.ToTensor()
    if val_transform is None:
        val_transform = transforms.ToTensor()

    datapath = os.path.join(datapath, dataset)

    trainset = datasets.__dict__[dataset](
        datapath, train=True, download=True, transform=train_transform)

    if train_size is not None:
        assert train_size + val_size <= len(trainset)

    if val_size > 0:
        indices = list(range(len(trainset)))
        trainset = Subset(trainset, indices[val_size:])

        valset = datasets.__dict__[dataset](
            datapath, train=True, download=True, transform=val_transform)
        valset = Subset(valset, indices[:val_size])
        val_loader = torch.utils.data.DataLoader(
            valset, batch_size=val_batch_size, shuffle=val_shuffle, **kwargs)

    else:
        val_loader = None

    if train_size is not None:
        trainset = Subset(trainset, list(range(train_size)))

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, shuffle=train_shuffle, **kwargs)

    return train_loader, val_loader


def get_test_loader(
        dataset, datapath=DATA_PATH, test_size=None, batch_size=1000,
        transform=None, kwargs=None, shuffle=False):
    """Support MNIST and CIFAR10"""
    if kwargs is None:
        kwargs = {}
    if transform is None:
        transform = transforms.ToTensor()

    datapath = os.path.join(datapath, dataset)

    testset = datasets.__dict__[dataset](
        datapath, train=False, download=True, transform=transform)

    if test_size is not None:
        testset = Subset(testset, list(range(test_size)))

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    return test_loader


def bchw2bhwc(x):
    if isinstance(x, np.ndarray):
        pass
    else:
        raise

    if x.ndim == 3:
        return np.moveaxis(x, 0, 2)
    if x.ndim == 4:
        return np.moveaxis(x, 1, 3)


def bhwc2bchw(x):
    if isinstance(x, np.ndarray):
        pass
    else:
        raise

    if x.ndim == 3:
        return np.moveaxis(x, 2, 0)
    if x.ndim == 4:
        return np.moveaxis(x, 3, 1)


def _imshow(img):
    import matplotlib.pyplot as plt
    img = bchw2bhwc(img.detach().cpu().numpy())
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    plt.imshow(img, vmin=0, vmax=1)
    plt.axis("off")


class ImageNetClassNameLookup(object):

    def _load_list(self):
        import json
        with open(self.json_path) as f:
            class_idx = json.load(f)
        self.label2classname = [
            class_idx[str(k)][1] for k in range(len(class_idx))]

    def __init__(self):
        self.json_url = ("https://s3.amazonaws.com/deep-learning-models/"
                         "image-models/imagenet_class_index.json")
        self.json_path = os.path.join(DATA_PATH, "imagenet_class_index.json")
        if os.path.exists(self.json_path):
            self._load_list()
        else:
            import urllib
            urllib.request.urlretrieve(self.json_url, self.json_path)
            self._load_list()


    def __call__(self, label):
        return self.label2classname[label]


def get_panda_image():
    img_path = os.path.join(DATA_PATH, "panda.jpg")
    img_url = "https://farm1.static.flickr.com/230/524562325_fb0a11d1e1.jpg"

    def _load_panda_image():
        from skimage.io import imread
        return imread(img_path) / 255.

    if os.path.exists(img_path):
        return _load_panda_image()
    else:
        import urllib
        urllib.request.urlretrieve(img_url, img_path)
        return _load_panda_image()


def tensor2npimg(tensor):
    return bchw2bhwc(tensor[0].cpu().numpy())


def _show_images(img, advimg, enhance=31):
    np_img = tensor2npimg(img)
    np_advimg = tensor2npimg(advimg)
    np_perturb = tensor2npimg(advimg - img)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(np_img)
    plt.axis("off")
    plt.title("original image")
    plt.subplot(1, 3, 2)

    plt.imshow(np_perturb * enhance + 0.5)
    plt.axis("off")
    plt.title("the perturbation")

    plt.subplot(1, 3, 3)
    plt.imshow(np_advimg)
    plt.axis("off")
    plt.title("perturbed image")
    plt.show()


mkdir(ROOT_PATH)
mkdir(DATA_PATH)
