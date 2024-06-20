from siamese_dataloader import Siamese_dataset, Siamese_dataloader
from numpy import hstack, save
        
if __name__ == "__main__":
    print('Imports finished')

    root="/mnt/qb/work/oh/owl156/train_xml"
    loss_path="/mnt/qb/work/oh/owl156/NeglectedFreeLunch/data/losses.npy"
    dataset = Siamese_dataset(
        root=root,
        loss_path=loss_path,
    )
    dataloader = Siamese_dataloader(
        dataset=dataset,
        batch_size=1,
        num_workers=1
    ).run()
    print("SIAMESE NOW")
    neglected_free_lunch = []
    for i, _ in enumerate(dataloader):
        neglected_free_lunch.append(dataset.samples[i][0].split("/")[-1])

    neglected_free_lunch = hstack(neglected_free_lunch)
    save('/mnt/qb/work/oh/owl156/NeglectedFreeLunch/data/neglected_free_lunch_file_names.npy', neglected_free_lunch)
    print("FINISHED")