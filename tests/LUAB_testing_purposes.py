/home/aldakata/Projects/Tuebingen/23-WS/ResearchProject/NeglectedFreeLunch/study_ab.ipynbimport torch
from ResNet import resnet18

from imagenet_dataloader import ImageNetwithLUAB_dataloader

import argparse

def build_args():
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument(
        "--img_path",
        type=str,
        help="Path to the ImageNet .jpg folder.",
        default="/common/datasets/ImageNet_ILSVRC2012/",
    )
    main_parser.add_argument(
        "--ab_path",
        type=str,
        default="/home/catalantatjer/researchproject/NeglectedFreeLunch/train_xml",
        help="Path to the Annotation Byproducts folder.",
    )
    main_parser.add_argument(
        "--clsidx_path",
        type=str,
        default="/home/stud132/researchproject/NeglectedFreeLunch/imagenet1000_clsidx_to_labels.txt",
        help="Path to the Annotation Byproducts clsidx to labels .txt.",
    )
    main_parser.add_argument(
        "--nr_epochs",
        type=int,
        default=100,
    )
    main_parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )    
    main_parser.add_argument(
        "--lambda_",
        type=float,
        default=0.5,
    )
    main_parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
    )
    main_parser.add_argument(
        "--num_class",
        type=int,
        default=1000,
    )
    return main_parser.parse_args()



if __name__ == "__main__":
    args = build_args()

    root_train = f'{args.img_path}train'
    xml_path = args.ab_path
    nr_epochs = args.nr_epochs
    batch_size = args.batch_size
    input_size = 224
    num_workers = 4
    num_class=args.num_class

    loader = ImageNetwithLUAB_dataloader(
        root_train=root_train,
        xml_path=xml_path,
        num_classes=num_class,
        input_size=input_size,
        batch_size=batch_size,
        num_workers=num_workers,
        loss_weight=1,
    ).run()

    (inputs, ab_information) = next(iter(loader))
    print('Dataloader sanity check, this should be (batch_size, (2))')
    print(f'Shapes :{ab_information[1].shape, ab_information[2].shape, ab_information[3].shape}')
    print(f'Content :{ab_information[1], ab_information[2], ab_information[3]}')