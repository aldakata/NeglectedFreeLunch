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
from imagenet_dataloader import ImageNetwithLUAB_datalaoder, ImageNetwithLUAB_dataloader

import argparse


def build_args():
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument(
        "--img_path",
        type=str,
        help="Path to the ImageNet .jpg folder.",
        default="/common/datasets/ImageNet_ILSVRC2012/train",
    )
    main_parser.add_argument(
        "--ab_path",
        type=str,
        default="/home/stud132/researchproject/NeglectedFreeLunch/train_xml",
        help="Path to the Annotation Byproducts folder.",
    )
    main_parser.add_argument(
        "--clsidx_path",
        type=str,
        default="/home/stud132/researchproject/NeglectedFreeLunch/imagenet1000_clsidx_to_labels.txt",
        help="Path to the Annotation Byproducts clsidx to labels .txt.",
    )

    return main_parser.parse_args()


def train(net, loader):


if __name__ == "__main__":
    args = build_args()

    print("Start")
    root_train = args.img_path
    xml_path = args.ab_path

    input_size = 224
    batch_size = 8
    num_workers = 4

    _, transform_2nd, transform_final = create_transform(
        input_size=input_size,
        is_training=True,
        auto_augment=None,
        color_jitter=0,
        re_prob=0,
        interpolation="bicubic",
        separate=True,
    )
    print("Created transform")

    loader = ImageNetwithLUAB_dataloader(
        root=root_train,
        xml_root=xml_path,
        num_classes=1000,
        input_size=input_size,
        batch_size=batch_size,
        num_workers=num_workers,
        loss_weight=1,
    ).run()

def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model
