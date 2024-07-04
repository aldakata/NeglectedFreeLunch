#!/bin/bash

#create the dataset

run () {
    python create_siamese_dataset.py --sh_path ${1} --margin ${2} --destiny_path ${3}
    python train_siamese.py --data_path ${3} --epochs ${4}
}

#train with the dataset

SAMPLE_HARDNESS_PATH = "/scratch_local/owl156-462029/NeglectedFreeLunch/data/resnet50_losses_0.npy"
MARGIN = 1.0
DESTINY="/scratch_local/owl156-462029/NeglectedFreeLunch/data/siamese_dataset_resnet50_0.npy"

run $SAMPLE_HARDNESS_PATH $MARGIN $DESTINY