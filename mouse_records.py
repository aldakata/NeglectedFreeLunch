import torch
from siamese_dataloader import Siamese_dataset_folder, Siamese_dataloader
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    xml_path = "/mnt/qb/work/oh/owl156/train_xml/"
    dataset = Siamese_dataset_folder(
        root=xml_path,
    )
    print('Loader loaded')
    loader = Siamese_dataloader(
        dataset=dataset,
        batch_size=1,
        num_workers=32,
    ).run()
    mouse_record_history = []
    loader = tqdm(loader)
    for i, (et, hr, mr, target) in enumerate(loader):
        mouse_record_history.append(mr)
    print('Mouse record history loaded', mouse_record_history)
    np.save('data/mouse_record_interpolated.npy', mouse_record_history)