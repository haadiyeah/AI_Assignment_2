from processing import get_data
from fitness import get_feature_df
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd

def accuracy(y_test, predictions):
    # comparing the predictions with the actual labels
    count = 0
    for i in range(len(predictions)):
        predictions[i] = 1 if predictions[i] >= 0.5 else 0
        # print("Predicted: ", predictions[i], "Actual: ", y_test.iloc[i].values[0])
        if predictions[i] == y_test.iloc[i].values[0]:
            count += 1
    return count/len(predictions)

def train_model(data, chromosome):
    # Get the features
    X = get_feature_df(data, chromosome).reset_index(drop=True)
    y = data.iloc[:, [-1]].reset_index(drop=True)
    
    number_of_features = len(X.columns);

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random under-sampling
    rus = RandomUnderSampler(random_state=42)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

    # get the removed data
    removed_data = X_train[~X_train.index.isin(X_train_res.index)]
    removed_labels = y_train[~y_train.index.isin(y_train_res.index)]

    # append to the test data
    X_test = pd.concat([X_test, removed_data])
    y_test = pd.concat([y_test, removed_labels])  

    #create optimizer
    opt = Adam(learning_rate=0.0001)    # decreasing the learning rate for less loss

    # binary classification
    n_outputs = 1
    model = Sequential()
    model.add(Dense(120, input_dim=number_of_features, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(X_train_res, y_train_res, epochs=5, batch_size=64)

    # making predictions
    predictions = model.predict(X_test)
    fitness_value = model.evaluate(X_test, y_test)

    accuracy_value = accuracy(y_test, predictions)

    print("Accuracy: ", accuracy_value)
    print("Fitness: ", fitness_value[1])

    return accuracy_value, fitness_value[1]