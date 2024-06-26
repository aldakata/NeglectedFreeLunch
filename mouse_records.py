import torch
from siamese_dataloader import Mouse_records_dataset, Mouse_records_dataloader
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    xml_path = "/scratch_local/owl156-438025/train_xml/"
    dataset = Mouse_records_dataset(
        root=xml_path,
    )
    print('Loader loaded')
    loader = Mouse_records_dataloader(
        dataset=dataset,
        batch_size=4096,
        num_workers=8,
    ).run()
    mouse_record_history = []
    loader = tqdm(loader)
    for i, (_, _, mr, _) in enumerate(loader):
        mouse_record_history.append(mr)

    np.save('data/mouse_record_interpolated.npy', np.concatenate(mouse_record_history, axis=0))