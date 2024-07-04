from mouse_record_dataloader import Mouse_records_dataset, Mouse_records_dataloader
from tqdm import tqdm
import numpy as np
import os
if __name__ == "__main__":
    root = os.getcwd().replace('/NeglectedFreeLunch', '')
    xml_path = f"{root}/train_xml/"
    print(f"Loading data from {xml_path}...")
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