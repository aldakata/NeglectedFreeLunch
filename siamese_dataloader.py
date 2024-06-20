# import numpy as np
from numpy import ones, load, array, logical_not
from torch.utils.data import DataLoader, Dataset 
from torchvision.datasets.folder import DatasetFolder
import xml.etree.ElementTree as ET
import os
import torch
import numpy as np
from scipy.interpolate import make_interp_spline

REGULAR_TS = np.array([   
        0.,  100.,  200.,  300.,  400.,  500.,  600.,  700.,  800., 900., 
        1000., 1100., 1200., 1300., 1400., 1500., 1600., 1700.,1800., 1900., 
        2000., 2100., 2200., 2300., 2400., 2500., 2600.,2700., 2800., 2900., 
        # 3000., 3100., 3200., 3300., 3400., 3500.,3600., 3700., 3800., 3900.
        ])

def regularize_mouse_record(mouse_record):
        default_mouse_record = np.array([REGULAR_TS, np.ones_like(REGULAR_TS)*-1, np.ones_like(REGULAR_TS)*-1]).T
        t0 = mouse_record[0,0]
        mouse_record[:,0] = mouse_record[:,0] - t0
        tf = mouse_record[-1,0]
        idxf = min(int(tf//100) + 1, len(REGULAR_TS)-1)
        if idxf < 2:
                return default_mouse_record
        bspl = make_interp_spline(mouse_record[:,0], mouse_record[:, 1:].T, k=3,axis=1)
        default_mouse_record[:idxf, 1:] = bspl(REGULAR_TS[:idxf]).T
        return default_mouse_record


def load_points_interpolated(xml_file):
    selected_record = -1 * np.ones((1,2))
    estimateTime = -1
    mouse_record =  -1 * np.ones((2,3))

    
    if not os.path.isfile(xml_file):
        return selected_record, estimateTime, mouse_record

    tree = ET.parse(xml_file)
    root = tree.getroot()
    selected_record = []
    mouse_record = []
    for obj in root.findall("metadata"):
        if obj.find("selected").text == "True":
            selected_point = obj.find("selectedRecord")
            selected_record.append(
                [
                    float(selected_point.find("x").text),
                    float(selected_point.find("y").text),
                ]
            )
            estimateTime = float(obj.find("estimateTime").text)

            for mouse_point in  obj.findall("mouseTracking"):
                time = mouse_point.find("time")
                x = mouse_point.find("x")
                y = mouse_point.find("y")
                if (x is not None) and (y is not None) and (time is not None):
                    payload = [float(time.text),float(x.text), float(y.text)]
                    mouse_record.append(payload)


    if len(selected_record) == 0:
        selected_record = -1 * np.ones((1,2))
    if len(mouse_record) != 0:
        mouse_record = regularize_mouse_record(np.array(mouse_record))
    if estimateTime == 0:
        estimateTime = -1

    return np.array(selected_record), estimateTime, mouse_record


def load_points(xml_file):
    log = ""
    selected_record = -1 * ones((1,2))
    selected_record_time = -1
    selected = False
    estimateTime = -1
    selectedCount = 0
    worker_id = ''
    assignment_id = ''
    hovered_record = [[-1,-1], [-1,-1]]
    mouse_record =  [[-1,-1, -1], [-1,-1, -1]]
    if not os.path.isfile(xml_file):
        return selected_record, selected_record_time, selected, estimateTime, worker_id, assignment_id, hovered_record, mouse_record, selectedCount

    tree = ET.parse(xml_file)
    root = tree.getroot()
    selected_record = []
    hovered_record = []
    mouse_record = []
    for obj in root.findall("metadata"):
        selected = bool(obj.find("selected").text)
        worker_id = obj.find("worker_id").text
        assignment_id = obj.find("assignment_id").text

        if obj.find("selected").text == "True":
            selected_point = obj.find("selectedRecord")
            selected_record.append(
                [
                    float(selected_point.find("x").text),
                    float(selected_point.find("y").text),
                ]
            )
            selected_record_time = float(selected_point.find("time").text)
            estimateTime = float(obj.find("estimateTime").text)
            selectedCount = int(obj.find("selectedCount").text)

            for hovered_point in  obj.findall("hoveredRecord"):
                action = hovered_point.find("action")
                point = hovered_point.find("time")
                if (action is not None) and (point is not None):
                    if action.text == 'enter':
                        payload = [0.,float(point.text)]
                    if action.text == 'leave':
                        payload = [1.,float(point.text)]
                    hovered_record.append(payload)

            for mouse_point in  obj.findall("mouseTracking"):
                time = mouse_point.find("time")
                x = mouse_point.find("x")
                y = mouse_point.find("y")
                if (x is not None) and (y is not None) and (time is not None):
                    payload = [float(time.text),float(x.text), float(y.text)]
                    mouse_record.append(payload)
                    log = log + f"\t{mouse_point.find('time').text}"
    # print(f"Inside the XML LOAD_POINTS: {log}")
    return array(selected_record), selected_record_time, selected, estimateTime, worker_id, assignment_id, hovered_record, mouse_record, selectedCount

def is_turd(path):
    ret = True
    ret &= path.endswith('.xml')
    ret &= "._" not in path
    return ret

class Siamese_dataset_folder(DatasetFolder):
    def __init__(
        self,
        root,
        is_valid_file=is_turd,
        loader=load_points_interpolated,
        transform=None,
        seed=0,
    ):
        super(Siamese_dataset_folder, self).__init__(root, loader=loader, transform=transform, is_valid_file=is_valid_file)
        self._subset = logical_not(load("/mnt/qb/work/oh/owl156/NeglectedFreeLunch/data/missing_file_names_mask.npy"))
        # self.losses = torch.tensor(load(loss_path)[self._subset], dtype=torch.float32)
        self.loader = loader

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        selected_record, estimateTime,mouse_record = sample
        estimateTime = torch.tensor([estimateTime])#, dtype=torch.float32)
        # target = torch.tensor([self.losses[index]])
        # print(f"__get_item__ index: {index}", hovered_record, mouse_record)

        return (
            selected_record,
            estimateTime, 
            mouse_record,
            target,
            )

class Siamese_dataset(Dataset):
    def __init__(
        self,
        loss_path,
        estimate_times_path,
        mask,
        seed=0,
    ):
        super(Siamese_dataset, self).__init__()
        self.losses = torch.tensor(load(loss_path)[mask], dtype=torch.float32)
        self.estimateTimes = torch.tensor(load(estimate_times_path)[mask], dtype=torch.float32)
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


class Siamese_dataloader:
    def __init__(
        self,
        dataset,
        batch_size,
        num_workers,
    ):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset = dataset

    def run(self):
        train_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return train_loader
