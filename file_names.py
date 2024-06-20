from torchvision.datasets import ImageNet
from numpy import hstack, save
        
if __name__ == "__main__":
    print('Imports finished')

    imagenet_root = '/mnt/qb/datasets/ImageNet2012'
    imagenet = ImageNet(imagenet_root, split='train')
    imagenet_file_names = []

    for i in range(len(imagenet)):
        imagenet_file_names.append(imagenet.samples[i][0].split("/")[-1])
    imagenet_file_names = hstack(imagenet_file_names)
    save('/mnt/qb/work/oh/owl156/NeglectedFreeLunch/data/imagenet_file_names.npy', imagenet_file_names)