## Batching Classs

import numpy as np


class Audiobooks_Data_Reader():
    def __init__(self, dataset, batch_size = None):
        npz = np.load("Audiobook_data_{0}.npz".format(dataset))
        
        self.inputs, self.targets = npz['inputs'].astype(np.float32), npz['targets'].astype(np.int32)
        
        if batch_size is None:
            self.batch_size = self.inputs.shape[0]
        else:
            self.batch_size = batch_size
        self.curr_batch = 0
        self.batch_count = self.inputs.shape[0] // self.batch_size
        

    def __next__(self):
        if self.curr_batch >= self.batch_count:
            self.curr_batch = 0
            raise StopIteration
            
        batch_slice = slice(self.curr_batch * self.batch_size, (self.curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self.curr_batch += 1
        
     
        classes_num = 2
        targets_one_hot = np.zeros((targets_batch.shape[0], classes_num))
        targets_one_hot[range(targets_batch.shape[0]), targets_batch] = 1
        
       
        return inputs_batch, targets_batch
    
    
    
    def __iter__(self):
        return self
    
    def __len__(self):
        return self.batch_count
    
        