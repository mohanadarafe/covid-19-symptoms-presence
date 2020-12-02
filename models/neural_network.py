import numpy as np
import matplotlib.pyplot as plt 
import models.preprocess as preprocess, models.utils as utils
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import PredefinedSplit

def neural_network(sampling = False, isNotebook = False):
    print("="*60)
    print("Neural network...")
    DATA_FILE = utils.get_data_directory()

    # The argument of the function will determine weather we use oversampling or not
    if(sampling):
        process_method = preprocess.oversample(DATA_FILE)
    else:
        process_method = preprocess.preprocess_data(DATA_FILE)

    X, y = process_method
    X_train, X_test, y_train, y_test = utils.split_data(X, y, 0.6)
    X_val, X_test, y_val, y_test = utils.split_data(X_test, y_test, 0.5)

    X_grid = np.concatenate((X_train, X_val))
    y_grid = np.concatenate((y_train, y_val))
    separation_boundary = [-1 for _ in y_train] + [0 for _ in y_val]
    ps = PredefinedSplit(separation_boundary)

    param_grid = {
        'activation': ['logistic', 'identity', 'tanh', 'relu'],
        'hidden_layer_sizes': [(100), (10, 20, 10, 20, 10, 20, 10)], 
        'solver': ['adam', 'sgd'],
    }

    clf = GridSearchCV(MLPClassifier(random_state=0), param_grid, cv=ps)

    model = clf.fit(X_grid, y_grid)
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    test_acc = model.score(X_test, y_test)
    print(f'training score: {round(train_acc, 3)}')
    print(f'validation score: {round(val_acc, 3)}')
    print(f'testing score: {round(test_acc, 3)}')
    report_dict = classification_report(y_test, model.predict(X_test), output_dict = True, target_names=["No", "Yes"])

    imps = permutation_importance(model, X_test, y_test)
    top_feature_importances = list(sorted(enumerate(imps.importances_mean), key = lambda x: x[1], reverse = True))

    if (isNotebook):
        return top_feature_importances, model
    else:
        utils.display_metrics(report_dict)

    utils.log_results(top_feature_importances)
    utils.generate_report("Neural Network", "MLP", model, X_test, y_test, report_dict)