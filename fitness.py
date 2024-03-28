import random
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

def create_population(size, num_features):
    population = [];
    for i in range(size):
        chromosome = [];
        for j in range(num_features):
            chromosome.append(random.randint(0, 1));
        population.append(chromosome);
    return population;

def get_labels(data_df):
    labels = data_df.iloc[:, -1].values;
    return labels;

def get_feature_cols(data_df, chromosome):
    chromosome.append(0)  # Append 0 to the end of chromosome for mask to work
    mask = [bool(val) for val in chromosome]
    feature_cols = data_df.loc[:, mask]
    chromosome.pop()  # Remove the appended 0
    return feature_cols;

def fitness(data_features, number_of_features, labels):
    X_train, X_test, y_train, y_test = train_test_split(data_features, labels, test_size=0.2, random_state=42)  # 80% training and 20% test, X is data_features, y is labels
    
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
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    fitness_value = model.evaluate(X_test, y_test)
    return fitness_value[1];    

def calc_fitness(data_df, chromosome):
    data_features = get_feature_cols(data_df, chromosome);
    #print 1 row of data_features
    print(data_features.iloc[0]);
    
    num_of_features = len(data_features.columns);
    print("Number of features: ", num_of_features);
    
    labels = get_labels(data_df);
    

    print(num_of_features);
    return fitness(data_features, num_of_features, labels);