from torchvision.datasets.folder import DatasetFolder
import numpy as np
from dataloader_utils import is_turd, load_points_interpolated, load_points
from torch.utils.data import DataLoader

class Mouse_records_dataset(DatasetFolder):
    def __init__(
        self,
        root,
        is_valid_file=is_turd,
    ):
        super(Mouse_records_dataset, self).__init__(root, loader=load_points_interpolated, is_valid_file=is_valid_file)

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        selected_record, estimateTime, mouse_record = sample
        # estimateTime = torch.tensor([estimateTime])#, dtype=torch.float32)
        # target = torch.tensor([target])
        # target = self.losses[index]
        # print(f"__get_item__ index: {index}", hovered_record, mouse_record)
        # As per https://github.com/pytorch/pytorch/issues/123439 we should return numpy arrays.

        return (
            selected_record,
            estimateTime, 
            mouse_record,
            target,
        )


class Mouse_records_dataloader:
    def __init__(
        self,
        dataset,
        batch_size,
        num_workers,
        shuffle = False
    ):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset = dataset
        self.shuffle = shuffle
        
    def _collate_fn(self, batch):
        """
            As per: https://github.com/pytorch/pytorch/issues/123439
            should be np arrays
        """
        batch_size = len(batch)
        selected_record = np.zeros((batch_size,1,2))
        estimateTime =  np.zeros((batch_size,1))
        mouse_record =  np.zeros((batch_size,30,2), dtype=np.float32)
        labels = np.zeros((batch_size,1))
        for i,x in enumerate(batch):
            selected_record[i] = x[0]
            estimateTime[i] = x[1]
            mouse_record[i] = x[2]
            labels[i] = x[3]
        return (
            selected_record,
            estimateTime,
            mouse_record,
            labels,
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



###########
### RAW ###
###########


class Mouse_records_raw_dataset(DatasetFolder):
    # TODO: PAD WITH THE FOREGROUND POINT
    def __init__(
        self,
        root,
        is_valid_file=is_turd,
        seed=0,
    ):
        super(Mouse_records_raw_dataset, self).__init__(root, loader=load_points, is_valid_file=is_valid_file)

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        selected_record, estimateTime, mouse_record = sample
        # estimateTime = torch.tensor([estimateTime])#, dtype=torch.float32)
        # target = torch.tensor([target])
        # target = self.losses[index]
        # print(f"__get_item__ index: {index}", hovered_record, mouse_record)
        # As per https://github.com/pytorch/pytorch/issues/123439 we should return numpy arrays.
        if estimateTime > 3000 or mouse_record.shape[0] > 30 or mouse_record.shape[-1] != 3:
            mouse_record = np.zeros((30,3))
        mouse_record[:,0] = mouse_record[:,0] - mouse_record[0,0]
        return (
            selected_record,
            estimateTime, 
            mouse_record,
            target,
        )


class Mouse_records_raw_dataloader:
    def __init__(
        self,
        dataset,
        batch_size,
        num_workers,
        shuffle = False
    ):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset = dataset
        self.shuffle = shuffle

    def _collate_fn(self, batch):
        """
            As per: https://github.com/pytorch/pytorch/issues/123439
            should be np arrays
        """
        batch_size = len(batch)
        selected_record = np.zeros((batch_size,1,2))
        estimateTime =  np.zeros((batch_size,1))
        mouse_record =  np.zeros((batch_size,30,3), dtype=np.float32)
        labels = np.zeros((batch_size,1))
        for i,x in enumerate(batch):
            selected_record[i] = x[0]
            estimateTime[i] = x[1]
            mouse_record[i, :x[2].shape[0]] = x[2]
            labels[i] = x[3]
        return (
            selected_record,
            estimateTime,
            mouse_record,
            labels,
        )

    def run(self):
        train_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._collate_fn,
        )
        return train_loader


