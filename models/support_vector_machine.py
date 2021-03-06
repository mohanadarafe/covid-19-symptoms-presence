import numpy as np
import matplotlib.pyplot as plt 
import models.preprocess as preprocess, models.utils as utils
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV

def support_vector_machine(sampling = False, isNotebook = False):
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
        'kernel': ['rbf', 'poly']
    }

    clf = GridSearchCV(SVC(random_state=0, probability=True), param_grid, cv=ps)

    model = clf.fit(X_grid, y_grid)
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    test_acc = model.score(X_test, y_test)
    print(f'training score: {round(train_acc, 3)}')
    print(f'validation score: {round(val_acc, 3)}')
    print(f'testing score: {round(test_acc, 3)}')
    report_dict = classification_report(y_test, model.predict(X_test), output_dict = True, target_names=["No", "Yes"])

    weights = permutation_importance(model, X_test, y_test)
    top_weights = list(sorted(enumerate(weights.importances_mean), key = lambda x: x[1], reverse = True))

    if isNotebook:
        return top_weights, model
    else:
        utils.display_metrics(report_dict)

    utils.log_results(top_weights)
    utils.generate_report("SVM", "SVM", model, X_test, y_test, report_dict)