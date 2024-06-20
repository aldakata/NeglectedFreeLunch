import torch
from ResNet import resnet18, resnet50

from imagenet_dataloader import ImageNetwithLUAB_dataloader, ONLY_IMAGES, IMAGES_AND_POINT_CLICKED, IMAGES_AND_POINT_CLICKED_AND_MOUSE_RECORD

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
        default="/home/stud132/researchproject/NeglectedFreeLunch/imagenet_ab_v1_0/train_xml",
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
    main_parser.add_argument(
        "--compile",
        action='store_true',
        default=False,
    )
    main_parser.add_argument(
        "--mode",
        type=int,
        default=ONLY_IMAGES,
        help="0: only images, 1: images and point clicked, 2: images and point clicked and mouse record."
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
        mode = args.mode
    ).run()

    (inputs, ab_information) = next(iter(loader))
    print('Dataloader sanity check, this should be (batch_size, (2))', ab_information[-1].shape)
    model = resnet50().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")

    L_image = torch.nn.CrossEntropyLoss()
    L_ab = torch.nn.SmoothL1Loss()
    print('---Training---')
    for epoch in range(nr_epochs):
        total_loss = 0
        for batch_idx, (inputs, target) in enumerate(loader):
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
