#!/bin/bash

#create the dataset

run () {
    python create_siamese_dataset.py --sh_path ${1} --margin ${2} --destiny_path ${3}
    python train_siamese.py --data_path ${3} --epochs ${4}
}

#Train with the dataset

SAMPLE_HARDNESS_PATH = "/scratch_local/owl156-466225/NeglectedFreeLunch/data/resnet50_losses_0.npy"
MARGIN = 1.0
DESTINY="/scratch_local/owl156-466225/NeglectedFreeLunch/data/siamese_dataset_resnet50_0_1.npy"
EPOCHS=10000

run $SAMPLE_HARDNESS_PATH $MARGIN $DESTINY $EPOCHS