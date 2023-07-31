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
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Compile the model
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

batch_size = 100
max_epochs = 15


# Assuming you have your data loaded into 'train_inputs' and 'train_targets'.
# Assuming 'Audiobook_Data_Reader' class is defined and loads the data batches.
train_data = Audiobooks_Data_Reader('train', batch_size)
validation_data = Audiobooks_Data_Reader('validation')

for epoch in range(max_epochs):
    for batch_inputs, batch_targets in train_data:
        model.train_on_batch(batch_inputs, batch_targets)

    # Calculate validation loss and accuracy after each epoch (if validation data available)
    val_loss, val_accuracy = 0.0, 0.0
    for val_batch_inputs, val_batch_targets in validation_data:
        loss, accuracy = model.evaluate(val_batch_inputs, val_batch_targets, verbose=0)
        val_loss += loss
        val_accuracy += accuracy

    val_loss /= len(validation_data)
    val_accuracy /= len(validation_data)

    print(f"Epoch {epoch + 1}/{max_epochs}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

    # Add any early stopping or checkpoint saving logic based on validation performance

# To make predictions, you can use:
# predictions = model.predict(test_inputs)
