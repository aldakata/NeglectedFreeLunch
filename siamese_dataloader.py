# import numpy as np
from torch.utils.data import DataLoader, Dataset 
import torch
import numpy as np

class Siamese_dataset_et(Dataset):
    def __init__(
        self,
        loss_path,
        estimate_times_path,
        mask,
        seed=0,
    ):
        super(Siamese_dataset, self).__init__()
        self.losses = torch.tensor(np.load(loss_path)[mask], dtype=torch.float32)
        self.estimateTimes = torch.tensor(np.load(estimate_times_path)[mask], dtype=torch.float32)
        self.indices = list(range(len(self.losses)))
        generator = np.random.default_rng(seed=seed)
        generator.shuffle(self.indices)

        self.offset = len(self.losses)//2

    def __getitem__(self, index):
        idx_0 = self.indices[index]
        idx_1 = self.indices[index + self.offset]
        estimateTime_0 = torch.tensor([self.estimateTimes[idx_0]])
        target_0 = torch.tensor([self.losses[idx_0]])
        estimateTime_1 = torch.tensor([self.estimateTimes[idx_1]])
        target_1 = torch.tensor([self.losses[idx_1]])
        target = torch.tensor([1.0]) if target_0 < target_1 else torch.tensor([0.0])
        return (
            estimateTime_0,
            estimateTime_1,
            target
        )
    def __len__(self) -> int:
        return len(self.losses)//2


class Siamese_dataset(Dataset):
    def __init__(
        self,
        losses,
        mouse_records,
        seed=0,
        margin=1,
    ):
        super(Siamese_dataset, self).__init__()
        self.losses = torch.from_numpy(losses)
        self.mouse_records = torch.from_numpy(mouse_records)
        self.indices = list(range(len(self.losses)))
        generator = np.random.default_rng(seed=seed)
        generator.shuffle(self.indices)

        self.offset = len(self.losses)//2
        self.margin = margin

    def __getitem__(self, index):
        idx_0 = self.indices[index]
        if index >= self.offset:
            idx_1 = self.indices[index - self.offset]
        else:
            idx_1 = self.indices[index + self.offset]

        # print(f"indices {index} {idx_0} {idx_1}")

        mr_0 = self.mouse_records[idx_0]
        target_0 = self.losses[idx_0]
        weight_0 = 0. if torch.allclose(torch.zeros_like(mr_0), mr_0) else 1.

        mr_1 = self.mouse_records[idx_1]
        target_1 = self.losses[idx_1]
        weight_1 = 0. if torch.allclose(torch.zeros_like(mr_1), mr_1) else 1.
        
        delta = target_0 - target_1
        # Case mr_1 has higher loss
        if delta <= -self.margin:
            target = 1
            weight = 1.
        # Case mr_0 has higher loss
        elif delta >= self.margin:
            target = 0
            weight = 1.
        # Case loss is similar
        else:
            target = 0
            weight = 0.

        return (
            mr_0,
            mr_1,
            target,
            weight
        )

    def __len__(self) -> int:
        assert len(self.losses) == len(self.mouse_records)
        return len(self.losses)//2

class Siamese_dataloader:
    def __init__(
        self,
        dataset,
        batch_size,
        num_workers,
        shuffle = False,
        mouse_record_shape = (30,2),
    ):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset = dataset
        self.shuffle = shuffle
        self.mouse_record_shape = mouse_record_shape
        assert len(mouse_record_shape) == 2
        
    def _collate_fn(self, batch):
        """
            As per: https://github.com/pytorch/pytorch/issues/123439
            should be np arrays
        """
        batch_size = len(batch)
        mouse_record_1 =  torch.zeros((batch_size, self.mouse_record_shape[0], self.mouse_record_shape[1]), dtype=torch.float32)
        mouse_record_0 =  torch.zeros((batch_size, self.mouse_record_shape[0], self.mouse_record_shape[1]), dtype=torch.float32)
        labels = torch.zeros((batch_size,1))
        weights = torch.zeros((batch_size,1))
        for i,x in enumerate(batch):
            mouse_record_0[i] = x[0]
            mouse_record_1[i] = x[1]
            labels[i] = x[2]
            weights[i] = x[3]

        return (
            mouse_record_0,
            mouse_record_1,
            labels,
            weights
        )

    def run(self):
        train_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._collate_fn,
            shuffle=self.shuffle
        )
        return train_loader
