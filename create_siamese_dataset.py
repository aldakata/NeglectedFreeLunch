from training_utils import create_siamese_data
import numpy as np
import argparse

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
        default="data/siamese_data_cleaned.npy",
    )
    return main_parser.parse_args()

args = build_args()
arr = create_siamese_data(args.sh_path, args.margin)
np.save(args.destiny_path, arr)