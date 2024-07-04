import torch
import numpy as np


def train_one_epoch(net, dataloader, optimizer, criterion, device):
    predictions = []
    targets = []
    for i, (mr0, mr1, target, _, _, t0, t1, w0, w1) in enumerate(dataloader):
        mr0, mr1, target, t0, t1, w0, w1 = mr0.to(device), mr1.to(device), target.to(device), t0.to(device), t1.to(device), w0.to(device), w1.to(device)
        optimizer.zero_grad()
        logits = net(mr0, mr1, t0, t1, w0, w1)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        predictions.append(logits.detach().cpu().numpy()>0.5)
        targets.append(target.detach().cpu().numpy())

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    accuracy = predictions == targets
    return loss.item(), sum(accuracy)/len(accuracy)

def validate(net, dataloader, device):
    predictions = []
    targets = []
    with torch.no_grad():
        for i, (mr0, mr1, target, _, _, t0, t1, w0, w1) in enumerate(dataloader):
            mr0, mr1, target, t0, t1, w0, w1 = mr0.to(device), mr1.to(device), target.to(device), t0.to(device), t1.to(device), w0.to(device), w1.to(device)
            logits = net(mr0, mr1, t0, t1, w0, w1)
            predictions.append(logits.detach().cpu().numpy()>0.5)
            targets.append(target.detach().cpu().numpy())
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    accuracy = predictions == targets
    return sum(accuracy)/len(accuracy)

def collate_fn(batch):
    mr0=torch.zeros((len(batch), 60))
    mr1=torch.zeros((len(batch), 60))
    target=torch.zeros((len(batch), 1))
    sh0=torch.zeros((len(batch), 1))
    sh1=torch.zeros((len(batch), 1))
    t0=torch.zeros((len(batch), 1))
    t1=torch.zeros((len(batch), 1))
    w0=torch.zeros((len(batch), 1))
    w1=torch.zeros((len(batch), 1))
    
    for i, b in enumerate(batch):
        mr0[i, :]=batch[i][0][:60]
        mr1[i, :]=batch[i][0][60:120]
        target[i, :]=batch[i][0][120]
        sh0[i, :]=batch[i][0][121]
        sh1[i, :]=batch[i][0][122]
        t0[i, :]=batch[i][0][123]
        t1[i, :]=batch[i][0][124]
        w0[i, :]=batch[i][0][125]
        w1[i, :]=batch[i][0][126]
    mr0 = mr0.view((len(batch), 2, 30))
    mr1 = mr1.view((len(batch), 2, 30))
    return mr0, mr1, target, sh0, sh1, t0, t1, w0, w1