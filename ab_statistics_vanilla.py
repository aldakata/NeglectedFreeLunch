import numpy as np
from tqdm import tqdm
import pickle

import torchvision.transforms as transforms
from timm.data import create_transform

from imagenet_dataloader import ImageNetwithLUAB, RRCFlipReturnParams

root_train = "/mnt/qb/datasets/ImageNet2012/train/"
xml_path = "/mnt/qb/work/oh/owl156/train_xml/"

input_size = 224

_, transform_2nd, transform_final = create_transform(
    input_size=input_size,
    is_training=True,
    auto_augment=None,
    color_jitter=0,
    re_prob=0,
    interpolation="bicubic",
    separate=True,
)

dataset_train = ImageNetwithLUAB(
    root=root_train,
    xml_root=xml_path,
    num_classes=1000,
    transform=transforms.Compose([transform_2nd, transform_final]),
    pre_transform=RRCFlipReturnParams(
        size=input_size, scale=(0.08, 1), interpolation="bicubic"
    ),
)

iterator = tqdm(dataset_train)
not_selected = []
worker_sum = {}
worker_min = {}
worker_max = {}
worker_counts = {}
selected_count_freq = {}

print('Loop')
for i, (image, label, weight, loc_info, selected_record, selected_record_time, selected, estimateTime, worker_id, assignment_id, selected_count) in enumerate(iterator):
    workerid=worker_id
    if workerid not in worker_sum.keys():
        worker_sum[workerid] = estimateTime 
        worker_min[workerid] = estimateTime
        worker_max[workerid] = estimateTime 
        worker_counts[workerid] = 1 
    else:
        worker_sum[workerid] += estimateTime
        if worker_min[workerid] > estimateTime:
            worker_min[workerid] = estimateTime
        if worker_min[workerid] < estimateTime:
            worker_max[workerid] = estimateTime
        worker_counts[workerid] += 1
    if selected_count not in selected_count_freq.keys():    
        selected_count_freq[selected_count] = 1
    else:
        selected_count_freq[selected_count] +=1
    if not i%10000:
        print(f"item {i} covariate {selected}, label {estimateTime}, workerid {workerid}, assignmentid {assignment_id}, Number of workers so Far{len(worker_min)}, selected_count_freq {selected_count_freq}")
    if not selected == 1:
        not_selected.append(i)

print(selected_count_freq)
with open('data/selected_counts.txt', 'wb') as file:
    pickle.dump(selected_count_freq, file, protocol=pickle.HIGHEST_PROTOCOL)
