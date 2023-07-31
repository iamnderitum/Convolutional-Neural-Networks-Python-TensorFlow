
from .batch import Audiobooks_Data_Reader

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

