import torch
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
        default="/home/stud132/researchproject/NeglectedFreeLunch/train_xml",
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
    print('Dataloader sanity check, this should be (batch_size, (2))', ab_information[-1].shape)
    model = resnet18()
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    L_image = torch.nn.CrossEntropyLoss()
    L_ab = torch.nn.SmoothL1Loss()
    # Are the batches shuffled here?
    print('---Training---')
    for epoch in range(nr_epochs):
        total_loss = 0
        for batch_idx, (inputs, ab_information) in enumerate(loader):
            targets, weight, fg_point, loc_info = ab_information
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            visual_outputs, ab_outputs = model(inputs)
            loss = L_image(visual_outputs, targets)
            if loc_info is not None:
                loc_info = loc_info.cuda()
                loss += args.lambda_ * L_ab(ab_outputs, loc_info)
            
            loss.backward()
            optimizer.step()
            total_loss+=loss
        print(f"[Training] - E{epoch} - Loss {total_loss}")
