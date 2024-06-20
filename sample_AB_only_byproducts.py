import torch as ch

import numpy as np
from tqdm import tqdm

from siamese_dataloader import Siamese_dataset_folder, Siamese_dataloader

def loop(loader, log):
    print('Init loop')
    iterator = tqdm(loader)
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
        for i, (hovered_record, mouse_record) in enumerate(iterator):
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
    ch.save(hovered_records, log.format('hovered_records'))
    ch.save(mouse_records, log.format('mouse_records'))
    # np.save(log.format('hovered_records'), np.hstack(hovered_records))
    # np.save(log.format('mouse_records'), np.hstack(mouse_records))
    print(f'Saved at {log}')

if __name__ == "__main__":
    print('Imports finished')
    num_workers=8
    batch_size=1
    in_memory=1

    LOG = '/mnt/qb/work/oh/owl156/NeglectedFreeLunch/data/sample_{}.npy' 
    root_train = "/mnt/qb/datasets/ImageNet2012/train/"
    xml_path = "/mnt/qb/work/oh/owl156/train_xml/"

    dataset = Siamese_dataset_folder(
        root=xml_path,
    )

    print('Loader loaded')
    loader = Siamese_dataloader(
        dataset=dataset,
        batch_size=1,
        num_workers=num_workers,
    ).run()

    loop(loader, LOG)