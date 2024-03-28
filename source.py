import csv
import os
from keras.models import Sequential
from keras.layers import Dense
import random

# For fitness function, you can use the following (or similar) code of 
#a basic neural network. Do not worry
# about its training parameters for now.
def fitness_function(data_chunk, inputD, chromosone): #inputD is the no. of 1s in the dataRow's chromosone.

    #create neural network model (model is a type of artificial neural network called a multilayer perceptron (MLP))
    n_outputs = 1  # change to 1 for binary classification
    model = Sequential()
    model.add(Dense(120, input_dim=inputD, activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))  # change to 'sigmoid' for binary classification
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # change to 'binary_crossentropy' for binary classification

    model.fit(data_chunk, chromosone, epochs=10, batch_size=32) 
    #epochs = of times the learning algorithm will work through the entire training dataset,
    #batch_size = number of samples to work through before updating the internal model parameters.

    fitness_value = model.predict()

    return fitness_value

def generate_chromosome(data_row_length):
    # Generate a random chromosome from a data row
    chromosome = []
    inputD = 0
    for i in range(data_row_length):
        #randomly select 0 or 1 and add it to the chromosome
        num = random.randint(0, 1)
        chromosome.append(num)
        if num == 1:
            inputD += 1

    return chromosome, inputD

def read_csv_files(directory):
    data = {} #dictionary
    for filename in os.listdir(directory):
        if filename.endswith(".csv"): 
            with open(os.path.join(directory, filename), 'r') as f:
                reader = csv.reader(f)
                next(reader, None)  #skip the header row
                file_data = []
                chunk = []

                for i, row in enumerate(reader):
                    chunk.append(row)
                    if (i + 1) % 10 == 0:  # if row number is divisible by 10
                        file_data.append(chunk)
                        chromo, inputD = generate_chromosome(len(chunk[0])) #generate a chromosome for each chunk
                        fitness_val = fitness_function(chunk, inputD, chromo)
                        chunk = []
                
                # handle the last chunk which may have less than 10 rows
                if chunk:
                    file_data.append(chunk)
                    chromo, inputD = generate_chromosome(len(chunk[0])) #generate a chromosome for the last chunk
                    fitness_val = fitness_function(chunk, inputD)
                
                data[filename] = file_data #adding dictionary entry
    return data

# Use relative path to the 'data' directory
data = read_csv_files('./data')

print(data)


