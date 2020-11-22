import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
import os

def split_data(X, y, training_split):
    '''
    The following function splits the training and testing data sets
    according to a split [0 - 1] passed.
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = training_split, random_state = 42)
    return X_train, X_test, y_train, y_test

def assert_correct_directory() -> bool:
    '''
    Asserts that script is being executed from the correct directory
    '''
    return 'environment.yml' in os.listdir()

def get_data_directory() -> str:
    '''
    Returns the path to the data file.
    '''
    while 'environment.yml' not in os.listdir():
        os.chdir('..')
    os.chdir('data')
    return os.getcwd() + "/covid-dataset.csv"

def get_feature_names() -> list:
    '''
    Returns an array of the feature names.
    '''
    return ['Breathing Problem','Fever','Dry Cough','Sore throat','Running Nose','Asthma','Chronic Lung Disease','Headache','Heart Disease','Diabetes','Hyper Tension','Fatigue ','Gastrointestinal']