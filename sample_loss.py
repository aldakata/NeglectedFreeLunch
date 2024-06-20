import torch as ch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.distributed as dist
ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)

from torchvision import models
import torchmetrics
import numpy as np
from tqdm import tqdm

import os
import time
import json
from uuid import uuid4
from typing import List
from pathlib import Path
from argparse import ArgumentParser

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

class BlurPoolConv2d(ch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = ch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer('blur_filter', filt)

    def forward(self, x):
        blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                           groups=self.conv.in_channels, bias=None)
        return self.conv.forward(blurred)

def create_model_and_scaler(gpu, arch, pretrained, distributed, use_blurpool):
    scaler = GradScaler()
    model = getattr(models, arch)(pretrained=pretrained)
    def apply_blurpool(mod: ch.nn.Module):
        for (name, child) in mod.named_children():
            if isinstance(child, ch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16):
                setattr(mod, name, BlurPoolConv2d(child))
            else: apply_blurpool(child)
    if use_blurpool: apply_blurpool(model)

    model = model.to(memory_format=ch.channels_last)
    model = model.to(gpu)

    return model, scaler

def create_train_loader(gpu, train_dataset, num_workers, batch_size, in_memory):
    res = 192
    distributed = False
    this_device = f'cuda:{gpu}'
    train_path = Path(train_dataset)
    assert train_path.is_file()
    decoder = RandomResizedCropRGBImageDecoder((res, res))
    image_pipeline: List[Operation] = [
        decoder,
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(ch.device(this_device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
    ]

    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(ch.device(this_device), 
        non_blocking=True)
    ]

    loader = Loader(train_dataset,
                    batch_size=batch_size,
                    # indices = [i for i in range(2500) ],
                    num_workers=num_workers,
                    order=OrderOption.SEQUENTIAL,
                    os_cache=in_memory,
                    drop_last=False,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline
                    },
                    distributed=distributed)
    return loader

def loop(model, loader, log):
    print('Init loop')
    model.eval()
    losses = []
    celoss = ch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none')
    iterator = tqdm(loader)
    with ch.no_grad():
        with autocast():
            for ix, (images, target) in enumerate(iterator):
                output = model(images)
                output += model(ch.flip(images, dims=[3]))
                loss = celoss(output, target)
                losses.append(loss.detach().cpu().numpy().astype(np.float64) )
    np.save(log, np.hstack(losses))
    print(f'Losses saved at {log}')

if __name__ == "__main__":
    print('Imports finished')
    num_workers=8
    batch_size=1024
    in_memory=0


    train_dataset_path='/mnt/qb/datasets/ffcv_imagenet_data/train_500_0.50_90.ffcv'
    GPU = 0
    model,scaler = create_model_and_scaler(GPU,'resnet50',False,False,True) 
    print('Model created')
    loader = create_train_loader(GPU, train_dataset_path, num_workers, batch_size, in_memory)
    print('Dataloader successful')
    # PATH='../ffcv-train/log/c42c9eca-afe8-49f9-9d61-1447c175df7c/final_weights.pt'
    LOG = '/mnt/qb/work/oh/owl156/NeglectedFreeLunch/data/resnet50_losses_{}.npy' 
    ROOT_PATH='/mnt/qb/work/oh/owl156/ffcv-train/log/e8463e7a-ae56-4dea-b6d8-95933227199f/{}.pt'
    PATHS = [
        ROOT_PATH.format(0),
        ROOT_PATH.format(5),
        ROOT_PATH.format(10),
        ROOT_PATH.format(15),
        ROOT_PATH.format("final_weights"),
    ]
    LOGS = [
        LOG.format(0),
        LOG.format(5),
        LOG.format(10),
        LOG.format(15),
        LOG.format("final_weights"),
    ]
    for ckpt, log in zip(PATHS, LOGS):
        model.load_state_dict(ch.load(ckpt))
        print('Model loaded')
        loop(model, loader, log)
