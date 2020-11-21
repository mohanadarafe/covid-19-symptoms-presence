from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def load_data(filename):
    '''
    The following function loads the data set into a numpy array.
    '''
    df=pd.read_csv(filename, sep=',', skiprows=1, header=None)
    return df.values

def preprocess_data(filename):
    '''
    The following function will preprocess the data by limiting our dataset
    to only contain symptoms.
    '''
    data = load_data(filename)
    
    # We removed unwanted columns that are not symptoms of COVID-19
    sanitized_data = np.delete(data, slice(13,20), 1)

    X = sanitized_data[:, :-1]
    y = sanitized_data[:, -1]
    return X, y

def split_data(X, y, training_split):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = training_split, random_state = 0)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X, y = preprocess_data("data/covid-dataset.csv")
    X_train, X_test, y_train, y_test = split_data(X, y, 0.65)
