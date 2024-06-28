import numpy as np
import random

import torch
import torchvision
import torchvision.transforms as transforms
import timm
from timm.data import create_transform
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as F
import os
from scipy.interpolate import make_interp_spline

REGULAR_TS = np.array([   
        0,  100,  200,  300,  400,  500,  600,  700,  800, 900, 
        1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,1800, 1900, 
        2000, 2100, 2200, 2300, 2400, 2500, 2600,2700, 2800, 2900, 
        # 3000., 3100., 3200., 3300., 3400., 3500.,3600., 3700., 3800., 3900.
        ])

def regularize_mouse_record(mouse_record):
        default_mouse_record = np.array([np.ones_like(REGULAR_TS)*-1, np.ones_like(REGULAR_TS)*-1]).T
        t0 = mouse_record[0,0]
        mouse_record[:,0] = mouse_record[:,0] - t0
        tf = mouse_record[-1,0]
        idxf = min(int(tf//100) + 1, len(REGULAR_TS)-1)
        # if idxf < 2:
        #     return default_mouse_record
        try:
            bspl = make_interp_spline(mouse_record[:,0], mouse_record[:, 1:].T, k=3,axis=1)
            default_mouse_record[:idxf] = bspl(REGULAR_TS[:idxf]).T
            return default_mouse_record

        except Exception as e: 
            # print(f"Error: {e}")#, mouse_record: ts : {mouse_record[:,0]-t0}, x-y: {mouse_record[:,1:]}")
            # continue
            return default_mouse_record

def load_points_interpolated(xml_file):
    selected_record_default = -1 * np.ones((1,2))
    estimateTime_default = -1
    mouse_record_default = np.array([np.ones_like(REGULAR_TS)*-1, np.ones_like(REGULAR_TS)*-1]).T

    if not os.path.isfile(xml_file):
        return selected_record_default, estimateTime_default, mouse_record_default

    tree = ET.parse(xml_file)
    root = tree.getroot()
    selected_record = []
    mouse_record = []
    estimateTime = estimateTime_default
    for obj in root.findall("metadata"):
        if obj.find("selected").text == "True":
            selected_point = obj.find("selectedRecord")
            if len(selected_record) == 0:
                selected_record.append(
                    [
                        float(selected_point.find("x").text),
                        float(selected_point.find("y").text),
                    ]
                )
            estimateTime = float(obj.find("estimateTime").text)

            for mouse_point in  obj.findall("mouseTracking"):
                time = mouse_point.find("time")
                x = mouse_point.find("x")
                y = mouse_point.find("y")
                if (x is not None) and (y is not None) and (time is not None):
                    payload = [float(time.text),float(x.text), float(y.text)]
                    mouse_record.append(payload)


    if len(selected_record) == 0:
        selected_record = selected_record_default

    if len(mouse_record) > 3:
        mouse_record = regularize_mouse_record(np.array(mouse_record))
    else:
        mouse_record = mouse_record_default

    if estimateTime == 0:
        estimateTime = estimateTime_default

    return np.array(selected_record), estimateTime, mouse_record


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
    
    return load_points_interpolated(source_xml)

def check_in_point(loc_info, gt_points):
    """
        Return 
            bool: whether the point selected point is cropped out because of the image transformation
            np.array: the transformed point

    """
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
    weight = 1.
    if np.all(LUAB_points) > 0 :
        is_fg, fg_point = check_in_point(loc_info=loc_info, gt_points=LUAB_points)

    if not is_fg or np.all(LUAB_points) < 0:
        weight = 0.
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

class ImageNetwithLUAB_extended(torchvision.datasets.folder.ImageFolder):
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
        super(ImageNetwithLUAB_extended, self).__init__(root, transform=transform)
        self.xml_root = xml_root
        self.pre_transform = pre_transform
        self.num_classes = num_classes
        self.loss_weight = loss_weight

    def get_point_ingredients(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        image_path = self.imgs[index][0].strip()
        selected_record, estimateTime, mouse_record = get_imagenet_selected_point_info(image_path, self.xml_root)
        sample, loc_info = self.pre_transform(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, selected_record, estimateTime, mouse_record, loc_info

    def __getitem__(self, index):
        sample, target, selected_record, estimateTime, mouse_record, loc_info = self.get_point_ingredients(index)
        target, weight, fg_point = compute_cls(
            original_label=target,
            LUAB_points=selected_record,
            loc_info=loc_info,
            num_classes=self.num_classes,
            loss_weight=self.loss_weight,
        )

        return sample, (target, weight, fg_point, np.array([loc_info['w'], loc_info['h']], dtype=np.float32), mouse_record)


class ImageNetwithLUAB_extended_dataloader:
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
        dataset_train = ImageNetwithLUAB_extended(
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
            pin_memory=False,
            drop_last=False,
            shuffle=False
        )
        return train_loader
