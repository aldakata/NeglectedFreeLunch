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

import os


def load_points(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    points = []
    for obj in root.findall("metadata"):
        if obj.find("selected").text == "True":
            selected_point = obj.find("selectedRecord")
            points.append(
                [
                    float(selected_point.find("x").text),
                    float(selected_point.find("y").text),
                ]
            )

    return np.array(points)


def get_imagenet_selected_point_info(image_path, xml_root):
    """
    image_path: /img_root/Data/CLS-LOC/train/n15075141/n15075141_9993.JPEG
    box_xml: /xml_root/Annotations/CLS-LOC/train/n15075141/n15075141_9993.xml

    Returns:
      boxes: Numpy array or None
      sizes: [width, height, depth] or None
      None is returned if no box sup exists.
    """
    fragments = image_path.split("/")
    file_name_no_extension = fragments[-1].split('.')[0]
    image_class = file_name_no_extension.split('_')[0]
    source_xml = os.path.join(xml_root, image_class, f'{file_name_no_extension}.xml')
    if not os.path.isfile(source_xml):
        return None

    return load_points(source_xml)

def check_in_point(loc_info, gt_points):
    if len(gt_points) != 0:
        img_h, img_w = loc_info["img_h"], loc_info["img_w"]
        is_flip = loc_info["f"]
        crop_x0 = loc_info["j"] / img_w
        crop_y0 = loc_info["i"] / img_h
        crop_x1 = crop_x0 + loc_info["w"] / img_w
        crop_y1 = crop_y0 + loc_info["h"] / img_h

        avg_point = sum(gt_points) / len(gt_points)
        if is_flip:
            crop_x0, crop_x1 = 1 - crop_x1, 1 - crop_x0
            avg_point[0] = 1 - avg_point[0]

        return (crop_x0 <= avg_point[0] and crop_x1 >= avg_point[0]) and (
            crop_y0 <= avg_point[1] and crop_y1 >= avg_point[1]
        ), np.array(
            [
                (avg_point[0] - crop_x0) * img_w / loc_info["w"],
                (avg_point[1] - crop_y0) * img_h / loc_info["h"],
            ]
        )

    return False, None

def compute_cls(
    original_label,
    LUAB_points,
    loc_info,
    class_gt=False,
    num_classes=1000,
    loss_weight=1,
):
    is_fg = False
    weight = np.array(1, dtype=np.float32)
    if LUAB_points is not None:
        is_fg, fg_point = check_in_point(loc_info=loc_info, gt_points=LUAB_points)

    if not is_fg or LUAB_points is None:
        weight = np.array(0, dtype=np.float32)
        fg_point = np.array([-1, -1], dtype=np.float32)

    return original_label, weight, fg_point


class RRCFlipReturnParams(timm.data.transforms.RandomResizedCropAndInterpolation):
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        processed_img = F.resized_crop(img, i, j, h, w, self.size, interpolation)

        f = torch.rand(1) < 0.5
        if f:
            processed_img = F.hflip(processed_img)
        return processed_img, {
            "i": i,
            "j": j,
            "h": h,
            "w": w,
            "f": f,
            "img_h": img.size[1],
            "img_w": img.size[0],
        }


class ImageNetwithLUAB(torchvision.datasets.folder.ImageFolder):
    def __init__(
        self,
        root,
        xml_root,
        transform=None,
        pre_transform=None,
        num_classes=1000,
        loss_weight=1,
        seed=0,
    ):
        super(ImageNetwithLUAB, self).__init__(root, transform=transform)
        self.xml_root = xml_root
        self.pre_transform = pre_transform
        self.num_classes = num_classes
        self.loss_weight = loss_weight

    def get_point_ingredients(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        image_path = self.imgs[index][0].strip()
        points = get_imagenet_selected_point_info(image_path, self.xml_root)
        sample, loc_info = self.pre_transform(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, points, loc_info

    def __getitem__(self, index):
        sample, target, points, loc_info = self.get_point_ingredients(index)
        target, weight, fg_point = compute_cls(
            original_label=target,
            LUAB_points=points,
            loc_info=loc_info,
            num_classes=self.num_classes,
            loss_weight=self.loss_weight,
        )
        return (np.asarray(sample.permute(1, 2, 0),dtype=np.uint8()), 
            target,
            weight,
            fg_point[0],
            fg_point[1],            
            int(loc_info["w"]),
            int(loc_info["h"]))
        


class ImageNetwithLUAB_dataloader:
    def __init__(
        self,
        root_train,
        xml_path,
        input_size,
        batch_size,
        num_workers,
        num_classes,
        loss_weight,
    ):
        self.root_train = root_train
        self.xml_path = xml_path
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.num_workers = num_workers
        self.input_size = input_size
        self.batch_size = batch_size

        _, self.transform_2nd, self.transform_final = create_transform(
            input_size=self.input_size,
            is_training=True,
            auto_augment=None,
            color_jitter=0,
            re_prob=0,
            interpolation="bicubic",
            separate=True,
        )

    def run(self):
        dataset_train = ImageNetwithLUAB(
            root=self.root_train,
            xml_root=self.xml_path,
            num_classes=1000,
            transform=transforms.Compose([self.transform_2nd, self.transform_final]),
            pre_transform=RRCFlipReturnParams(
                size=self.input_size, scale=(0.08, 1), interpolation="bicubic"
            ),
            loss_weight=self.loss_weight,
        )
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return train_loader
