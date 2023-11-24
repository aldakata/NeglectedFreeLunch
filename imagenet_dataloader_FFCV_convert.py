import numpy as np
import cv2
import json
import random

import torch
import torchvision
import torchvision.transforms as transforms
import timm
from timm.data import create_transform
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
import matplotlib
import sys

import torchvision.transforms.functional as F


from imagenet_dataloader import ImageNetwithLUAB, ImageNetwithLUAB_dataloader, RRCFlipReturnParams
import os


from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField, FloatField, NDArrayField


root_train = "/common/datasets/ImageNet_ILSVRC2012/train"
xml_path = "/home/catalantatjer/researchproject/NeglectedFreeLunch/imagenet_ab_v1_0/train_xml"

input_size = 224
batch_size = 8
num_workers = 10

_, transform_2nd, transform_final = create_transform(
    input_size=input_size,
    is_training=True,
    auto_augment=None,
    color_jitter=0,
    re_prob=0,
    interpolation="bicubic",
    separate=True,
)

# ImageNetwithLUAB inherits from torchvision.datasets.folder.ImageFolder
# which is a DataLoader and NOT a torch Dataset.
# Will this work? Would it work with a dataset_train.dataset?
# https://pytorch.org/vision/main/generated/torchvision.datasets.DatasetFolder.html#torchvision.datasets.DatasetFolder

dataset_train = ImageNetwithLUAB(
    root=root_train,
    xml_root=xml_path,
    num_classes=1000,
    transform=transforms.Compose([transform_2nd, transform_final]),
    pre_transform=RRCFlipReturnParams(
        size=input_size, scale=(0.08, 1), interpolation="bicubic"
    ),
)
max_resolution = 400
compress_probability = 0.1
jpeg_quality = 90

BETON_PATH = f"../imagenet_AB_train_{max_resolution}_{compress_probability}_{jpeg_quality}.ffcv"
write_path = BETON_PATH

print('Dataset read, setting up writter')
# Pass a type for each data field
writer = DatasetWriter(
    write_path,
    {
        # Tune options to optimize dataset size, throughput at train-time
        "image": RGBImageField(write_mode='proportion',
                               max_resolution=max_resolution,
                               compress_probability=compress_probability,
                               jpeg_quality=jpeg_quality),
        "label": IntField(),
        "weight": FloatField(),
        "fg_point_w": FloatField(),
        "fg_point_h": FloatField(),
        "loc_info_w": FloatField(),
        "loc_info_h": FloatField(),
        "estimateTime": IntField(),
    },
)
# sample, (target, weight, fg_point, loc_info)

print('Green light, time to write the dataset')
# Write dataset
writer.from_indexed_dataset(dataset_train)
# END

# from ffcv.loader import Loader, OrderOption
# from ffcv.fields.decoders import NDArrayDecoder, FloatDecoder, JSON

# # Deciding ORDERING.
# # Random is most expensive, but most random.
# # Quasi_random is a trade off in between.
# # Sequential is most efficient but not random.

# from ffcv.loader import OrderOption

# # Truly random shuffling (shuffle=True in PyTorch)
# ORDERING = OrderOption.RANDOM
# # Unshuffled (i.e., served in the order the dataset was written)
# ORDERING = OrderOption.SEQUENTIAL
# # Memory-efficient but not truly random loading
# # Speeds up loading over RANDOM when the whole dataset does not fit in RAM!
# ORDERING = OrderOption.QUASI_RANDOM

# # Deciding PIPEPLINE.
# # A key-value dictionary where the key matches the one used in writing the dataset,
# # and the value is a sequence of operations to perform on top. JIT-able
# # transformations will be JITted for fastest experience, and such. Our example could be
# PIPELINES = {
#     "covariate": [
#         NDArrayDecoder(),
#         ToTensor(),
#         transforms.Compose([transform_2nd, transform_final]),
#     ],  # How to do the pre_transform random_resize_and_interpolation?
#     "label": [FloatDecoder(), ToTensor()],
#     "byprodut_annotations": [],  # JSON encoder to do
# }

# loader = Loader(
#     BETON_PATH,
#     batch_size=BATCH_SIZE,
#     num_workers=NUM_WORKERS,
#     order=ORDERING,
#     pipelines=PIPELINES,
# )
