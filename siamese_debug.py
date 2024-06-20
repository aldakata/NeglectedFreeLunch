from siamese_dataloader import Siamese_dataset, Siamese_dataloader
from torch import nn as nn
from torch import device, cuda, no_grad, manual_seed, cat
import torch.optim as optim
from torch.utils.data import Subset
from numpy import load, logical_not, logical_and, arange
from tqdm import tqdm

def train_one_epoch(train_loader, net, criterion, optimizer, epoch):
    running_loss = 0.0
    for _, (inputs_0, inputs_1, labels) in enumerate(train_loader, 0):
        inputs_0, inputs_1, labels = inputs_0.to(device), inputs_1.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs_0, inputs_1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss

class My_little_siamese_network(nn.Module):
    # It should predict 0, 1.
    # 0 if 1st sample has higher loss
    # 1 if 2nd sample has higher loss
    def __init__(self):
        super().__init__()
        # define a head for the network
        # input is 1 dim and output is 1 dim
        self.net = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )

        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        x = self.net(x)
        y = self.net(y)

        return self.cls_head(x * y)

if __name__ == "__main__":
    print('Imports finished')
    SEED = 0
    manual_seed(SEED)

    # Training utils
    EPOCHS = 5
    device = device('cuda:0' if cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data utils
    root="/mnt/qb/work/oh/owl156/train_xml"
    loss_path="/mnt/qb/work/oh/owl156/NeglectedFreeLunch/data/resnet50_losses_final_weights.npy"
    subset = logical_not(load("/mnt/qb/work/oh/owl156/NeglectedFreeLunch/data/missing_file_names_mask.npy"))
    estimate_times=load('/mnt/qb/work/oh/owl156/NeglectedFreeLunch/data/sample_estimate_times.npy')
    emask = estimate_times < 1000
    and_subset = logical_and(subset, emask)
    
    # print(estimate_times.shape)
    # print(estimate_times[subset].shape)
    # print(subset.shape)
    # print(subset[subset].shape)
    # print(and_subset.shape)

    # Start
    dataset = Siamese_dataset(
        loss_path=loss_path,
        estimate_times_path='/mnt/qb/work/oh/owl156/NeglectedFreeLunch/data/sample_estimate_times.npy',
        mask = and_subset,
        seed=SEED
    )
    # print(arange(len(dataset))[and_subset].max())
    # print(arange(len(dataset))[and_subset].shape)
    indices = list(arange(len(dataset)))
    train_indices = indices[:int(len(indices)*0.8)]
    test_indices = indices[int(len(indices)*0.8):]
    dataset_train = Subset(
        dataset, 
        train_indices
    )
    dataloader_train = Siamese_dataloader(
        dataset=dataset_train,
        batch_size=len(dataset_train),
        num_workers=4
    ).run()

    dataset_test = Subset(
        dataset, 
        test_indices
    )
    dataloader_test = Siamese_dataloader(
        dataset=dataset_test,
        batch_size=len(dataset_test),
        num_workers=4
    ).run()

    net = My_little_siamese_network()
    print(f"The number of parameters is {sum(p.numel() for p in net.parameters())}")
    # number of parameters of the networ
    net.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.8)

    print("TRAININNGGGG!")
    history = []

    with no_grad():
        loss = 0
        accuracy = 0
        predictions = []
        all_labels = []
        for _, (inputs_0, inputs_1, labels) in enumerate(dataloader_test, 0):
            inputs_0, inputs_1, labels = inputs_0.to(device), inputs_1.to(device), labels.to(device)
            outputs = net(inputs_0, inputs_1)
            predictions.append(outputs.round())
            all_labels.append(labels)
            loss += criterion(outputs, labels).item()

        predictions = cat(predictions)
        all_labels = cat(all_labels)
        accuracy = (predictions == all_labels).sum().item() / len(predictions)
    print(f"Initial test accuracy is {accuracy} and loss is {loss}")

    for _, epoch in enumerate(tqdm(range(EPOCHS))):
        e_loss = train_one_epoch(dataloader_train, net, criterion, optimizer, epoch)
        print(f"Epoch: {epoch}/{EPOCHS} | Loss : {e_loss}")
        history.append(e_loss)

    dataloader_test = Siamese_dataloader(
        dataset=dataset_test,
        batch_size=len(dataset_test),
        num_workers=4
    ).run()
    with no_grad():
        loss = 0
        accuracy = 0
        predictions = []
        all_labels = []
        for _, (inputs_0, inputs_1, labels) in enumerate(dataloader_test, 0):
            inputs_0, inputs_1, labels = inputs_0.to(device), inputs_1.to(device), labels.to(device)
            outputs = net(inputs_0, inputs_1)
            predictions.append(outputs.round())
            all_labels.append(labels)
            loss += criterion(outputs, labels).item()

        predictions = cat(predictions)
        all_labels = cat(all_labels)
        accuracy = (predictions == all_labels).sum().item() / len(predictions)
    # save('/mnt/qb/work/oh/owl156/NeglectedFreeLunch/data/history.npy', history)
    print(f"Finished Training, final accuracy is {accuracy} and loss is {loss}")
