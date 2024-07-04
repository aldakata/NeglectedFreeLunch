import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from siamese_network import SiameseNetworkConv
from torch import nn
from training_utils import train_one_epoch, validate, collate_fn
from time import gmtime, strftime
import argparse
import os


def build_args():
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the data file.",
        default="data/siamese_data_cleaned.npy",
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
    return main_parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    data_path = args.data_path
    data = np.load(data_path)
    print(
        f"Loading data from {data_path}, training for {args.epochs} epochs.\nLogging at {args.log}"
    )
    log_file = open(f"{args.log}/log.txt", "w")
    os.mkdir(f"{args.log}")
    with open(f"{args.log}/args.txt", "w") as f:
        f.write(str(args))
    data = torch.from_numpy(data)
    dataset = TensorDataset(data)
    if args.debug:
        SUBSELECT_TO_DEBUG = 100
    else:
        SUBSELECT_TO_DEBUG = len(data)
    indices = np.arange(len(data))[:SUBSELECT_TO_DEBUG]
    np.random.shuffle(indices)
    train_indices = indices[: int(0.8 * len(indices))]
    val_indices = indices[int(0.8 * len(indices)) : int(0.85 * len(indices))]
    test_indices = indices[int(0.85 * len(indices)) :]
    dataset_train = Subset(dataset, train_indices)
    dataset_val = Subset(dataset, val_indices)
    dataset_test = Subset(dataset, test_indices)
    dataloader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    dataloader_val = DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    dataloader_test = DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
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
    initial_train_accuracy = validate(net, dataloader_train, device)
    print(f"Initial train accuracy: {initial_train_accuracy}")
    log_file.write(f"Initial train accuracy: {initial_train_accuracy}\n")
    for epoch in range(args.epochs):
        loss, train_accuracy = train_one_epoch(
            net, dataloader_train, optimizer, criterion, device
        )
        val_accuracy = validate(net, dataloader_val, device)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        losses.append(loss)
        if epoch % 50 == 0:
            print(
                f"Epoch {epoch}, loss: {loss}, train accuracy: {train_accuracy}, val accuracy: {val_accuracy}"
            )
            log_file.write(
                f"Epoch {epoch}, loss: {loss}, train accuracy: {train_accuracy}, val accuracy: {val_accuracy}\n"
            )
    plt.plot(losses, label="loss")
    plt.title("Loss")
    plt.savefig(f"{args.log}/loss.png")
    plt.plot(val_accuracies, label="val accuracy")
    plt.plot(train_accuracies, label="train accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.savefig(f"{args.log}/accuracy.png")
