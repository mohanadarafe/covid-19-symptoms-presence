import numpy as np
import os, shutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import plot_confusion_matrix

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

def _create_directories(dirName: str):
    '''
    Creates directories for results report
    '''
    while 'environment.yml' not in os.listdir():
        os.chdir('..')

    os.chdir("results")

    if os.path.isdir(dirName):
        shutil.rmtree(dirName)

    os.makedirs(dirName)
    
    while 'environment.yml' not in os.listdir():
        os.chdir('..')

def display_metrics(report: dict):
    '''
    Display the metrics of a model on the console.
    ''' 
    NO_COVID = report['No']
    YES_COVID = report['Yes']

    print(f'\t\tprecision\trecall\t\tf1')
    print(f'\tNo\t{round(NO_COVID["precision"], 3)}\t\t{round(NO_COVID["recall"], 3)}\t\t{round(NO_COVID["f1-score"], 3)}')
    print(f'\tYes\t{round(YES_COVID["precision"], 3)}\t\t{round(YES_COVID["recall"], 3)}\t\t{round(YES_COVID["f1-score"], 3)}')

def _save_metrics(report: dict, dirName: str):
    '''
    Saves small report text file containing information on model metrics.
    '''
    NO_COVID = report['No']
    YES_COVID = report['Yes']

    with open(f"results/{dirName}/no_covid_metrics.txt", mode='w') as n:
        n.write(f'Precision: {round(NO_COVID["precision"], 3)}\n' +
                f'Recall: {round(NO_COVID["recall"], 3)}\n' +
                f'F1-Score: {round(NO_COVID["f1-score"], 3)}\n'
                )

    with open(f"results/{dirName}/yes_covid_metrics.txt", mode='w') as y:
        y.write(f'Precision: {round(YES_COVID["precision"], 3)}\n' +
                f'Recall: {round(YES_COVID["recall"], 3)}\n' +
                f'F1-Score: {round(YES_COVID["f1-score"], 3)}\n'
                )


def _save_confusion_matrix(dirName: str, modelName: str, model, X, y):
    '''
    Saves confusion matrix picture.
    '''
    matrix = plot_confusion_matrix(model, X, y, display_labels=["No", "Yes"])
    plt.title(f"Confusion Matrix for {modelName}")
    plt.savefig(f"results/{dirName}/confusion_matrix")

def generate_report(dirName: str, modelName: str, model, X, y, report: dict):
    '''
    Generates report in results directory for every model trained.
    '''
    _create_directories(dirName)
    assert os.path.isdir(f"results/{dirName}"), "Something went wrong!"

    _save_confusion_matrix(dirName, modelName, model, X, y)
    _save_metrics(report, dirName)
