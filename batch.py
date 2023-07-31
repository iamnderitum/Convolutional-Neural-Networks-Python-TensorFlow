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
    


import tensorflow as tf
import numpy as np
import time

input_size = 10
output_size = 2
hidden_layer_size = 50

# Create the model using the Keras API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer_size, activation='relu', input_shape=(input_size,)),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(output_size)
])

# Define the optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Compile the model
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

batch_size = 100
max_epochs = 15

prev_validation_loss = 9999999

print('Validating and Train Inputs and validate Inputs\n')
print("-------------------------------------------------")
# Assuming you have your data loaded into 'train_inputs' and 'train_targets'.
# Assuming 'Audiobooks_Data_Reader' class is defined and loads the data batches.
train_data = Audiobooks_Data_Reader('train', batch_size)
validation_data = Audiobooks_Data_Reader('validation')

start_time = time.time()
for epoch_counter in range(max_epochs):
    curr_epoch_loss = 0.

    for input_batch, target_batch in train_data:
        batch_loss = model.train_on_batch(input_batch, target_batch)
        curr_epoch_loss += batch_loss[0]

    curr_epoch_loss /= len(train_data)

    validation_loss = 0
    validation_accuracy = 0

    for input_batch, target_batch in validation_data:
        batch_loss, batch_accuracy = model.evaluate(input_batch, target_batch, verbose=0)
        validation_loss += batch_loss
        validation_accuracy += batch_accuracy

    validation_loss /= len(validation_data)
    validation_accuracy /= len(validation_data)

    print('Epoch ' + str(epoch_counter + 1) +
          ". Training Loss: " + "{0:.3f}".format(curr_epoch_loss) +
          ". Validation Loss: " + "{0:.3f}".format(validation_loss) +
          ". Validation Accuracy: " + "{0:.2f}".format(validation_accuracy * 100.) + "%"
         )

    if validation_loss > prev_validation_loss:
        break

    prev_validation_loss = validation_loss
end_time = time.time()
total_time = end_time - start_time
print("End of Training. Time taken:{0:.3f}s ".format(total_time) )

print("Testing The Data ... ")

test_data = Audiobooks_Data_Reader('test')

# Assuming you have already compiled the model and used 'accuracy' as one of the metrics
test_accuracy = model.evaluate(test_data, verbose=0)

test_accuracy_percent = test_accuracy[1] * 100.  # Accuracy is at index 1 in the evaluation result
print(test_accuracy)
print("Test Accuracy: " + '{0:.2f}'.format(test_accuracy_percent) + '%')



