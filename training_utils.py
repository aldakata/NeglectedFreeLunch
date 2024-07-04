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


def create_siamese_data(sample_hardness_path, margin):
    subset = np.logical_not(np.load("data/missing_file_names_mask.npy"))
    sample_hardness = np.load(sample_hardness_path)[subset]
    mouse_records  = np.load('data/mouse_record_interpolated.npy')
    assert len(sample_hardness) == len(mouse_records)
    
    estimate_times=np.load('data/sample_estimate_times.npy')[subset]
    worker_ids=np.load('data/sample_worker_ids.npy')[subset]
    clean_subset = estimate_times < 3000
    clean_subset = np.logical_and(clean_subset, estimate_times > 0)

    sample_hardness = sample_hardness[clean_subset]
    mouse_records = mouse_records[clean_subset]
    estimate_times = estimate_times[clean_subset]
    worker_ids = worker_ids[clean_subset]
    
    not_zero = []
    for i, el in enumerate(mouse_records):
        if not np.allclose(el, np.zeros_like(el)):
            not_zero.append(i)
    mouse_records = mouse_records[not_zero]
    sample_hardness = sample_hardness[not_zero]
    estimate_times  = estimate_times[not_zero]
    worker_ids  = worker_ids[not_zero]
    worker_ids_int = np.zeros_like(worker_ids, dtype = np.float32)
    for i, w in enumerate(np.unique(worker_ids)):
        worker_ids_int[np.where(worker_ids == w)] = i
        
    NUM_SAMPLES = 2000000
    idx0 = np.random.choice(len(sample_hardness), NUM_SAMPLES)
    idx1 = np.random.choice(len(sample_hardness), NUM_SAMPLES)
    invalid = idx0 == idx1
    idx0 = idx0[~invalid]
    idx1 = idx1[~invalid]
    lower0 = sample_hardness[idx0] < sample_hardness[idx1]
    over_margin_0 = sample_hardness[idx0] + margin < sample_hardness[idx1]
    discard0 = np.logical_and(lower0, ~over_margin_0)
    lower1 = sample_hardness[idx1] < sample_hardness[idx0]
    over_margin_1 = sample_hardness[idx1] + margin < sample_hardness[idx0]
    discard1 = np.logical_and(lower1, ~over_margin_1)
    to_discard = np.logical_or(discard0, discard1)

    idx0 = idx0[~to_discard]
    idx1 = idx1[~to_discard]
    data = np.zeros((len(idx0), mouse_records.shape[1]*mouse_records.shape[2]+mouse_records.shape[1]*mouse_records.shape[2]+7))
    print(data.shape)
    target = (sample_hardness[idx0]<sample_hardness[idx1])*1.

    data[:, :30]=mouse_records[idx0][:, :, 0] # x-axis
    data[:, 30:60]=mouse_records[idx0][:, :, 1] # y-axis
    data[:, 60:90]=mouse_records[idx1][:, :, 0] # x-axis
    data[:, 90:120]=mouse_records[idx1][:, :, 1] # x-axis
    data[:, 120]=target
    data[:, 121]=sample_hardness[idx0] # hardness
    data[:, 122]=sample_hardness[idx1] # hardness
    data[:, 123]=estimate_times[idx0] # estimate time
    data[:, 124]=estimate_times[idx1] # estimate time
    data[:, 125]=worker_ids_int[idx0] # worker id
    data[:, 126]=worker_ids_int[idx1] # worker id
    return data