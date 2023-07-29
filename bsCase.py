import numpy as np

# Create a class that will do the batching for the algorithms
# This code is extremely reusabl. You should just adjust/change Audiobooks_data everywhere in th code

class Audiobooks_Data_Reader():
    # Dataset is a mandatory argument, while batch_size is optional
    # If you dont input batch_size, it will automatically take the value: None
    def __init__(self, dataset, batch_size = None):
        # Th dataset that loads is one of "train", "validation", "test".
        # e.g. if i call this class with x('train', 5), it will load "Audiobook_data_train.npz" with a batch size of 5.
        npz = np.load("Audiobook_data_{0}.npz".format(dataset))
        
        #Two variables that take the values of the inputs and the targets. Inputs are floats, targets are integers
        self.inputs, self.targets = npz['inputs'].astype(np.float32), npz['targets'].astype(np.int32)
        
        # Counts the batch number, given the size you feet it later
        # If the batch_size is None, we are either validating or testing, so we want to take the data in a single batch
        if batch_size is None:
            self.batch_size = self.inputs.shape[0]
        else:
            self.batch_size = batch_size
        self.curr_batch = 0
        self.batch_count = self.inputs.shape[0] // self.batch_size
        
    # A method which loads the next batch
    def __next__(self):
        if self.curr_batch >= self.batch_count:
            self.curr_batch = 0
            raise StopIteration
            
        # You should slice the dataset in batches and then the "next" function loads them one after the other
        batch_slice = slice(self.curr_batch * self.batch_size, (self.curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self.curr_batch += 1
        
        # One-hot encode the targets. In this example its a bit superfuous since we have a 0/1 colums
        # as a target already but we're giving you the code regardless, as it will be useful for any
        # classification task with more than one target colums
        classes_num = 2
        targets_one_hot = np.zeros((targets_batch.shape[0], classes_num))
        targets_one_hot[range(targets_batch.shape[0]), targets_batch] = 1
        
        # This function will return the inputs batch and the one-hot encoded targets
        return inputs_batch, targets_batch
    
    
    # A method needed for iterating over the batches, as we will put them in a loop    
    # This tells Python that the class we're defining is iterable, i.e that we can use it like:
    # for input, output in data:
            #do things
    # An iterator in Python is a class with a method __next__ that defines exactly how to iterate throuth its objects
    def __iter__(self):
        return self
    
    def __len__(self):
        return self.batch_count
        