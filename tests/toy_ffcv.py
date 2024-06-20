import numpy as np
from tqdm import tqdm
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder, FloatDecoder, BytesDecoder
from ffcv.transforms import ToTensor
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField, FloatField, BytesField
from uuid import uuid4


class StringDecoder(NDArrayDecoder):
    pass
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


class LinearRegressionDataset:
    def __init__(self, N, d):
        self.X = np.random.randn(N, d)
        self.Y = np.random.randn(N)

    def __getitem__(self, idx):
        return (self.X[idx].astype('float32'), self.Y[idx], 'jajaxdaixotbfunca', )

    def __len__(self):
        return len(self.X)

if __name__ =="__main__":
    path = 'toy.ffcv'
    WRITE = False
    if WRITE:

        N, d = (128, 2)
        dataset = LinearRegressionDataset(N, d)
        write_path = path

        writer = DatasetWriter(write_path, {
            'covariate': NDArrayField(shape=(d,), dtype=np.dtype('float32')),
            'label': FloatField(),
            'worker_id': StringField(MAX_STRING_SIZE)
        }, num_workers=8)
        writer.from_indexed_dataset(dataset)
        print('Written')

    print('Finished importing')

    PIPELINES = {
        'covariate': [NDArrayDecoder(), ToTensor()],
        'label': [FloatDecoder(), ToTensor()],
        'worker_id': [StringDecoder()]
        }

    loader = Loader(path,
            batch_size=5,
            num_workers=2,
            order=OrderOption.SEQUENTIAL,
            pipelines=PIPELINES,
            custom_fields={
                        'worker_id': StringField
                    },
            indices = [0,1,2,3,4,5,6,7,8,9]
                    )
    iterator = tqdm(loader)
    print(next(enumerate(iterator)))
    for i, (c,l, workerid) in enumerate(iterator):
        payment_daddy=workerid.tobytes().decode('ascii').replace('\0', '')
        print(f"covariate {c}, label {l}, workerid {payment_daddy}")

        