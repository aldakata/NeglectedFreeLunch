import numpy as np
from tqdm import tqdm
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder, IntDecoder
from ffcv.fields import NDArrayField
import csv


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


if __name__=='__main__':
    path = "../imagenet_AB_train_500_0.5_90.ffcv"
    print('Imports finished')
    PIPELINES = {
            'selected': [IntDecoder()],
            'estimateTime': [IntDecoder()],
            'worker_id': [StringDecoder()],
            'assignment_id': [StringDecoder()],
            }

    loader = Loader(path,
            batch_size=1,
            num_workers = 2,
            order=OrderOption.SEQUENTIAL,
            pipelines=PIPELINES,
            custom_fields={
                        'worker_id': StringField,
                        'assignment_id': StringField
                    })
    iterator = tqdm(loader)
    not_selected = []
    worker_sum = {}
    worker_min = {}
    worker_max = {}
    worker_counts = {}

    print('Loop')
    for i, (_, _, _, _, _, _, selected, estimateTime, workerid, assignmentid) in enumerate(iterator):
        workerid=workerid.tobytes().decode('ascii').replace('\0', '')
        if workerid not in worker_sum.keys():
            worker_sum[workerid] = estimateTime 
            worker_min[workerid] = estimateTime
            worker_max[workerid] = estimateTime 
            worker_counts[workerid] = 1 
        else:
            worker_sum[workerid] += estimateTime
            if worker_min[workerid] > estimateTime:
                worker_min[workerid] = estimateTime
            if worker_min[workerid] < estimateTime:
                worker_max[workerid] = estimateTime 
            worker_counts[workerid] += 1

        if not i%10000:
            print(f"item {i} covariate {selected}, label {estimateTime}, workerid {workerid}, assignmentid {assignmentid}, Number of workers so Far{len(worker_min)}")
        if not selected == 1:
            not_selected.append(i)

    # Save the indices of the elements to ignore
    np.save('not_selected.npy', np.asarray(not_selected))

    with open('writer_statistics.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['writer_id', 'estimated_time_avg', 'estimated_time_max', 'estimated_time_min'])

        for key in worker_counts.keys():
            writer_id= worker_sum[key]   
            estimated_time_avg= worker_min[key]  
            estimated_time_max= worker_max[key]  
            estimated_time_min= worker_counts[key]  
            writer.writerow([f'{writer_id}', f'{estimated_time_avg}', f'{estimated_time_max}', f'{estimated_time_min}'])


            
            