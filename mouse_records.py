import torch
from siamese_dataloader import Siamese_dataset_folder, Siamese_dataloader
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    xml_path = "/scratch_local/owl156-429256/train_xml/"
    dataset = Siamese_dataset_folder(
        root=xml_path,
    )
    print('Loader loaded')
    loader = Siamese_dataloader(
        dataset=dataset,
        batch_size=4096,
        num_workers=8,
    ).run()
    mouse_record_history = []
    loader = tqdm(loader)
    for i, (_, _, mr, _) in enumerate(loader):
        mouse_record_history.append(mr)

    np.save('data/mouse_record_interpolated.npy', np.concatenate(mouse_record_history, axis=0))