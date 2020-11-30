import numpy as np
import matplotlib.pyplot as plt 
import models.preprocess as preprocess, models.utils as utils
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import RandomizedSearchCV

def support_vector_machine(sampling = False):
    print("="*60)
    print("Running support vector machine...")
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
        'C': [1.0, 10.0, 100.0, 1000.0],
        'gamma': [0.01, 0.10, 1.00, 10.00],
        'kernel': ['linear']
    }

    clf = RandomizedSearchCV(SVC(), param_grid, cv=ps)

    model = clf.fit(X_grid, y_grid)
    report_dict = classification_report(y_test, model.predict(X_test), output_dict = True, target_names=["No", "Yes"])
    utils.display_metrics(report_dict)

    weights = model.best_estimator_.coef_
    top_weights = list(sorted(enumerate(weights[0]), key = lambda x: x[1], reverse = True))

    utils.log_results(top_weights)
    utils.generate_report("SVM", "SVM", model, X_test, y_test, report_dict)