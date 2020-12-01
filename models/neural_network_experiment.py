import numpy as np
import matplotlib.pyplot as plt 
import models.utils as utils, models.preprocess as preprocess
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import PredefinedSplit
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV

def neural_network_exp():
    print("="*60)
    print("Running experiement on Neural Networks...")
    TRAIN_SET = utils.get_data_directory()
    TEST_SET = utils.get_data_directory(fileName="/experiment-dataset.csv")

    X, y = preprocess.preprocess_data(TRAIN_SET)
    X = np.delete(X, slice(4,13), 1)
    X_train, X_val, y_train, y_val = utils.split_data(X, y, 0.9)
    X_test, y_test = preprocess.preprocess_experiment(TEST_SET)

    X_grid = np.concatenate((X_train, X_val))
    y_grid = np.concatenate((y_train, y_val))
    separation_boundary = [-1 for _ in y_train] + [0 for _ in y_val]
    ps = PredefinedSplit(separation_boundary)

    param_grid = {
        'activation': ['logistic', 'identity', 'tanh', 'relu'],
        'hidden_layer_sizes': [(80), (20, 10, 20, 10, 20)], 
        'solver': ['adam', 'sgd'],
    }

    clf = GridSearchCV(MLPClassifier(random_state=0), param_grid, cv=ps)

    model = clf.fit(X_grid, y_grid)
    report_dict = classification_report(y_test, model.predict(X_test), output_dict = True, target_names=["No", "Yes"])
    utils.display_metrics(report_dict)
    print(f'\nScore of Neural Network: {round(model.score(X_test, y_test), 3)}')
 
    imps = permutation_importance(model, X_test, y_test)
    top_feature_importances = list(sorted(enumerate(imps.importances_mean), key = lambda x: x[1], reverse = True))
    utils.log_results(top_feature_importances)
    utils.generate_report("Experiment NN", "Experimental Neural Network", model, X_test, y_test, report_dict)