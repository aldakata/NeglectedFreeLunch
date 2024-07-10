import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from siamese_network import SiameseNetworkConv
from torch import nn
from training_utils import train_one_epoch, validate, collate_fn, create_siamese_dataset
from time import gmtime, strftime
import argparse
import os
import time
import shutil
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice
from ffcv.fields.decoders import IntDecoder, NDArrayDecoder, FloatDecoder
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, FloatField, TorchTensorField


def build_args():
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the data file.",
        default="data/resnet50_losses_0.npy",
    )

    main_parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
    )

    main_parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
    )

    main_parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )
    main_parser.add_argument(
        "--debug",
        type=bool,
        default=False,
    )
    main_parser.add_argument(
        "--log",
        type=str,
        default=f'checkpoints/{strftime("%Y-%m-%d %H:%M:%S", gmtime())}',
    )
    
    main_parser.add_argument(
        "--beton",
        type=str,
        default=f'checkpoints/{strftime("%Y-%m-%d %H:%M:%S", gmtime())}/dataset.beton',
    )
    
    main_parser.add_argument(
        "--margin",
        type=float,
        default=1.
    )
    return main_parser.parse_args()

if __name__ == "__main__":
    args = build_args()
    data_path = args.data_path
    data = np.load(data_path)
    print(
        f"Loading data from {data_path}, training for {args.epochs} epochs.\nLogging at {args.log}"
    )
    os.mkdir(f"{args.log}")
    log_file = open(f"{args.log}/log.txt", "w")
    with open(f"{args.log}/args.txt", "w") as f:
        f.write(str(args))

    tensor_dataset = create_siamese_dataset(args.data_path, args.margin)
    write_path = args.beton
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

    if args.debug:
        SUBSELECT_TO_DEBUG = 100
    else:
        SUBSELECT_TO_DEBUG = len(data)
    indices = np.arange(len(tensor_dataset))[:SUBSELECT_TO_DEBUG]
    np.random.shuffle(indices)
    train_indices = indices[: int(0.8 * len(indices))]
    val_indices = indices[int(0.8 * len(indices)) : int(0.85 * len(indices))]
    test_indices = indices[int(0.85 * len(indices)) :]

    # Data decoding and augmentation
    mr_pipeline = [ NDArrayDecoder(), ToTensor(),ToDevice(0)]
    target_pipeline = [IntDecoder(), ToTensor(), ToDevice(0)]
    float_pipeline = [FloatDecoder(), ToTensor(), ToDevice(0)]

    # Pipeline for each data field
    pipelines = {
        'mr0': mr_pipeline,
        'mr1': mr_pipeline,
        'target': target_pipeline,
        'sh0': float_pipeline,
        'sh1': float_pipeline,
        't0': float_pipeline,
        't1': float_pipeline,
        'w0': float_pipeline,
        'w1': float_pipeline, 
    }

    # Replaces PyTorch data loader (`torch.utils.data.Dataloader`)
    dataloader_train = Loader(write_path, 
                    batch_size=args.batch_size, 
                    num_workers=1,
                    order=OrderOption.RANDOM, 
                    pipelines=pipelines,
                    indices =train_indices                
                    )
    dataloader_val = Loader(args.beton, 
                    batch_size=args.batch_size, 
                    num_workers=1,
                    order=OrderOption.RANDOM, 
                    pipelines=pipelines,
                    indices =val_indices                
                    )
    dataloader_test = Loader(args.beton, 
                    batch_size=args.batch_size, 
                    num_workers=1,
                    order=OrderOption.RANDOM, 
                    pipelines=pipelines,
                    indices =test_indices                
                    )
    print(
        f"Size of the datasets: train: {len(dataset_train)}, val: {len(dataset_val)}, test: {len(dataset_test)}"
    )
    log_file.write(
        f"Size of the datasets: train: {len(dataset_train)}, val: {len(dataset_val)}, test: {len(dataset_test)}\n"
    )
    losses = []
    train_accuracies = []
    val_accuracies = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    net = SiameseNetworkConv()
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
    # number of params
    print(
        f"Number of params: {sum(p.numel() for p in net.parameters() if p.requires_grad)}"
    )
    log_file.write(
        f"Number of params: {sum(p.numel() for p in net.parameters() if p.requires_grad)}\n"
    )
    # initial_train_accuracy = validate(net, dataloader_train, device)
    # print(f"Initial train accuracy: {initial_train_accuracy}")
    # log_file.write(f"Initial train accuracy: {initial_train_accuracy}\n")
    for epoch in range(args.epochs):
        start = time.time()
        loss, train_accuracy = train_one_epoch(
            net, dataloader_train, optimizer, criterion, device
        )
        end = time.time()
        val_accuracy = validate(net, dataloader_val, device)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        losses.append(loss)
        if epoch % 50 == 0:
            print(
                f"Epoch {epoch},  Time: {end - start} loss: {loss}, train accuracy: {train_accuracy}, val accuracy: {val_accuracy}"
            )
            log_file.write(
                f"Epoch {epoch}, loss: {loss}, train accuracy: {train_accuracy}, val accuracy: {val_accuracy}\n"
            )
    test_accuracy = validate(net, dataloader_test, device)
    torch.save(net.state_dict(), f"{args.log}/model.pth")
    print(f"Test accuracy: {test_accuracy}")
    log_file.write(f"Test accuracy: {test_accuracy}\n")
    plt.figure(0)
    plt.plot(losses, label="loss")
    plt.title("Loss")
    plt.savefig(f"{args.log}/loss.png")
    plt.figure(1)
    plt.plot(val_accuracies, label="val accuracy")
    plt.plot(train_accuracies, label="train accuracy")
    plt.plot(np.ones_like(val_accuracies)*test_accuracy, label="test accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.savefig(f"{args.log}/accuracy.png")
    log_file.close()

    work = "/mnt/qb/work/oh/owl156/NeglectedFreeLunch"
    shutil.copytree(f"{args.log}", f"{work}/{args.log}")