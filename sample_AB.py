import torch as ch

import numpy as np
from tqdm import tqdm

import torchvision.transforms as transforms


from timm.data import create_transform
from imagenet_dataloader import ImageNetwithLUAB, RRCFlipReturnParams


def loop(loader, log):
    print('Init loop')
    # iterator = tqdm(loader)
    iterator = loader
    selected_counts = []
    estimate_times = []
    worker_ids = []
    selecteds = []

    mouse_record_times = []
    hovered_record_times = []

    targets = []

    hovered_records = []
    mouse_records = []

    assingment_ids = []

    with ch.no_grad():
        for i, (_, target, _, _, _, _, selected, estimate_time, worker_id, assignment_id, selected_count, hovered_record, mouse_record) in enumerate(iterator):
            # selected_counts.append(selected_count)
            # estimate_times.append(estimate_time)
            # worker_ids.append(worker_id)
            # selecteds.append(selected)
            # if len(mouse_record[0])>0:
            #     mr_time = mouse_record[0][-1][0]-mouse_record[0][0][0]
            # else:
            #     mr_time = 0
            # if len(hovered_record[0])>0:
            #     hv_time =hovered_record[0][-1][-1]-hovered_record[0][0][-1]
            # else:
            #     hv_time = 0
            # mouse_record_times.append(mr_time)
            # hovered_record_times.append(hv_time)
            # assingment_ids.append(assignment_id)
            # targets.append(target)
            hovered_records.append(hovered_record)
            mouse_records.append(mouse_record)

    # np.save(log.format('selected_counts'), np.hstack(selected_counts))
    # np.save(log.format('estimate_times'), np.hstack(estimate_times))
    # np.save(log.format('worker_ids'), np.hstack(worker_ids))
    # np.save(log.format('mouse_record_times'), np.hstack(mouse_record_times))
    # np.save(log.format('hovered_record_times'), np.hstack(hovered_record_times))
    # np.save(log.format('assingment_ids'), np.hstack(assingment_ids))
    # np.save(log.format('labels'), np.hstack(targets))
    np.save(log.format('hovered_records'), np.hstack(hovered_records))
    np.save(log.format('mouse_records'), np.hstack(mouse_records))
    print(f'Saved at {log}')

if __name__ == "__main__":
    print('Imports finished')
    num_workers=8
    batch_size=1
    in_memory=0

    LOG = '/mnt/qb/work/oh/owl156/NeglectedFreeLunch/data/sample_{}.npy' 

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
        time_series_info=True
    )
    print('Loader loaded')
    loader = ch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

    loop(loader, LOG)