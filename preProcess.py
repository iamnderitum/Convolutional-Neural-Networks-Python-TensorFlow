import numpy as np
from sklearn import preprocessing

raw_csv_data = np.loadtxt("audiobooks_data.csv.csv", delimiter = ",")

unscaled_inputs_all = raw_csv_data[:, 1:-1]
targets_all = raw_csv_data[:, -1]
print('targets', targets_all)

num_one_targets = int(np.sum(targets_all)) 
zero_target_counter = 0
indices_to_remove = []

print('Starting to Balance The Data Set.\n')
for i in range(targets_all.shape[0]): 
    #all_record = len(targets_all)
    #print(all_record)
    if targets_all[i] == 0:
       
        zero_target_counter += 1
       
        if zero_target_counter > num_one_targets:
            indices_to_remove.append(i)
print("Found {0} Balanced records. Inputs and Targets".format(num_one_targets))
print('Removing {0} Unnecessary Data\n\n'.format(zero_target_counter))
unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis=0)
targets_equal_prior = np.delete(targets_all, indices_to_remove, axis=0)
print('-------------------------------------------------------------------------\n\n')


scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)

print("Generated PreProcessed Data Of type ===>", type(scaled_inputs))
print("\nSample Data for first 2 to array: \n",)
print(scaled_inputs[:2])
print("------------------------------------------------------------------------------\n\n")
#np.arange([start], stop) is a method that returns a evenly spaced values within a given interval.
print("Shuffling Indices to return an Evenly Spaced Values withing a given Interval.\n\n")
shuffled_indices = np.arange(scaled_inputs.shape[0])

#np.random.shuffle(X) is a method that shuffles the numbers in a given sequence
np.random.shuffle(shuffled_indices)

shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_prior[shuffled_indices]
print('Shuffled Data Set:')
print('Shuffled Targets ===>> ', shuffled_targets)
print('Shuffled Inputs: ')
print(shuffled_inputs)
WSL_AUTH_TOKEN_KEY = "ghp_1WRishVEMEa91o9PdVykkpcWRy4OnY2dLxA"

