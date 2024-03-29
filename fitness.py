import random
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

def create_population(size, num_features):
    population = [];
    for i in range(size):
        chromosome = [];
        for j in range(num_features):
            chromosome.append(random.randint(0, 1));
        population.append(chromosome);
    return population;

def get_labels_df(data_df):
    labels = data_df.iloc[:, -1].to_frame();
    return labels;

def get_feature_df(data_df, chromosome):
    chromosome.append(0)  # Append 0 to the end of chromosome for mask to work
    mask = [bool(val) for val in chromosome]
    feature_cols = data_df.loc[:, mask]
    chromosome.pop()  # Remove the appended 0
    return feature_cols;

def fitness(data_chunk, number_of_features, labels_chunk):
    # 80% training and 20% test, X is data_features, y is labels
    X_train, X_test, y_train, y_test = train_test_split(data_chunk, labels_chunk, test_size=0.2, random_state=42)  
    
    # Random under-sampling
    rus = RandomUnderSampler(random_state=42)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
    
    # get the removed data
    removed_data = X_train[~X_train.index.isin(X_train_res.index)]
    removed_labels = y_train[~y_train.index.isin(y_train_res.index)]
    
    # append to the test data
    X_test = pd.concat([X_test, removed_data])
    y_test = pd.concat([y_test, removed_labels])    
    
    # binary classification
    n_outputs = 1
    model = Sequential()
    model.add(Dense(120, input_dim=number_of_features, activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train_res, y_train_res, epochs=10, batch_size=32)
    fitness_value = model.evaluate(X_test, y_test)
    return fitness_value[1];

def calc_fitness(data_df, chromosome, chromosome_index, population_size):
    data_features = get_feature_df(data_df, chromosome);
    num_of_features = len(data_features.columns);
    labels = get_labels_df(data_df);
    
    print("Number of filtered features: ", num_of_features);
    
    data_features = data_features.reset_index(drop=True)
    labels = labels.reset_index(drop=True)
    
    fraction = 1 / population_size
    # Concatenate data_features and labels along the column axis
    data_with_labels = pd.concat([data_features, labels], axis=1)

    randomized_data_with_labels = data_with_labels.sample(frac=fraction, random_state=42)
    randomized_indices = randomized_data_with_labels.index

    # Split shuffled_data_with_labels back into data and labels
    data_chunk = randomized_data_with_labels.iloc[:, :-1]
    labels_chunk = randomized_data_with_labels.iloc[:, -1]

    # remove these data and labels from the original data and labels
    data_df = data_df[~data_df.index.isin(randomized_indices)]

    print(data_chunk.shape)
    print(labels_chunk.shape)
    print(len(set(labels_chunk)))
    
    return data_df, fitness(data_chunk, num_of_features, labels_chunk);