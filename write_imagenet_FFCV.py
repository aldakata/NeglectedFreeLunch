import torchvision.transforms as transforms
from timm.data import create_transform

from imagenet_dataloader import ImageNetwithLUAB, RRCFlipReturnParams

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField, FloatField, NDArrayField, BytesField
import numpy as np


class StringField(NDArrayField):
    def __init__(self, max_len: int, pad_char='\0'):
        self.max_len = max_len
        self.pad_char = pad_char
        super().__init__(np.dtype('uint8'), (max_len,))
    
    def encode(self, destination, field, malloc):
        padded_field = (field + self.pad_char * self.max_len)[:self.max_len]
        field = np.frombuffer(padded_field.encode('ascii'), dtype='uint8')
        return super().encode(destination, field, malloc)

MAX_STRING_SIZE = 30

root_train = "/mnt/qb/datasets/ImageNet2012/train/"
xml_path = "/mnt/qb/work/oh/owl156/train_xml/"

input_size = 224

_, transform_2nd, transform_final = create_transform(
    input_size=input_size,
    is_training=True,
    auto_augment=None,
    color_jitter=0,
    re_prob=0,
    interpolation="bicubic",
    separate=True,
)

dataset_train = ImageNetwithLUAB(
    root=root_train,
    xml_root=xml_path,
    num_classes=1000,
    transform=transforms.Compose([transform_2nd, transform_final]),
    pre_transform=RRCFlipReturnParams(
        size=input_size, scale=(0.08, 1), interpolation="bicubic"
    ),
)
max_resolution = 500
compress_probability = 0.5
jpeg_quality = 90

BETON_PATH = f"/mnt/qb/work/oh/owl156/imagenet_AB_train_{max_resolution}_{compress_probability}_{jpeg_quality}.ffcv"
write_path = BETON_PATH

print('Dataset read, setting up writter')
# Pass a type for each data field
writer = DatasetWriter(
    write_path,
    {
        # Tune options to optimize dataset size, throughput at train-time
        "image": RGBImageField(write_mode='proportion',
                               max_resolution=max_resolution,
                               compress_probability=compress_probability,
                               jpeg_quality=jpeg_quality),
        "label": IntField(),
        "weight": IntField(),
        "loc_info": NDArrayField(shape=(2,), dtype=np.dtype('float32')),
        "selected_record": NDArrayField(shape=(2,), dtype=np.dtype('float32')),
        "selected_record_time": IntField(),
        "selected": IntField(),
        "estimateTime": IntField(),
        "worker_id": StringField(15),
        "assignment_id": StringField(35),
        # "selected_count": IntField(), # Not implemented yet
    },
)

print('Green light, time to write the dataset')
# Write dataset
writer.from_indexed_dataset(dataset_train)
# END