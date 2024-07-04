from training_utils import create_siamese_dataset
from ffcv.writer import DatasetWriter
import numpy as np
import argparse
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField, FloatField, TorchTensorField
import torch

def build_args():
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument(
        "--sh_path",
        type=str,
        help="Path to the data file.",
        default="/scratch_local/owl156-462029/NeglectedFreeLunch/data/resnet50_losses_0.npy",
    )
    main_parser.add_argument(
        "--margin",
        type=float,
        help="sh margin",
        default=1.,
    )
    main_parser.add_argument(
        "--destiny_path",
        type=str,
        help="Path to the data file.",
        default="data/test.beton",
    )
    return main_parser.parse_args()

args = build_args()
tensor_dataset = create_siamese_dataset(args.sh_path, args.margin)

write_path = args.destiny_path
# Pass a type for each data field
writer = DatasetWriter(write_path, {
    # Tune options to optimize dataset size, throughput at train-time
    'mr0': TorchTensorField(
        dtype=torch.float64,
        shape=(60, 1),
    ),
    'mr1': TorchTensorField(
        dtype=torch.float64,
        shape=(60, 1),
    ),
    'target': IntField(),
    't0': FloatField(),
    't1': FloatField(),
    'w0': FloatField(),
    'w1': FloatField(),
})

# Write dataset
writer.from_indexed_dataset(tensor_dataset)
print(f"Dataset written to { write_path}")