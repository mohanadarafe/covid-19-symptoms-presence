import numpy as np
import matplotlib.pyplot as plt 
import models.utils as utils, models.preprocess as preprocess
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import PredefinedSplit
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV

def svm_exp():
    print("="*60)
    print("Running experiement on SVM...")
    TRAIN_SET = utils.get_data_directory()
    TEST_SET = utils.get_data_directory(fileName="/experiment-dataset.csv")

    X, y = preprocess.oversample(TRAIN_SET)
    X = np.delete(X, slice(4,13), 1)
    X_train, X_val, y_train, y_val = utils.split_data(X, y, 0.8)
    X_test, y_test = preprocess.preprocess_experiment(TEST_SET)

    X_grid = np.concatenate((X_train, X_val))
    y_grid = np.concatenate((y_train, y_val))
    separation_boundary = [-1 for _ in y_train] + [0 for _ in y_val]
    ps = PredefinedSplit(separation_boundary)

    param_grid = {
        'C': [1.0, 10.0, 100.0, 1000.0],
        'gamma': [0.01, 0.10, 1.00, 10.00],
        'kernel': ['rbf', 'poly']
    }

    print(X_train.shape)
    clf = GridSearchCV(SVC(random_state=0), param_grid, cv=ps)

    model = clf.fit(X_grid, y_grid)
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    test_acc = model.score(X_test, y_test)
    print(f'training score: {round(train_acc, 3)}')
    print(f'validation score: {round(val_acc, 3)}')
    print(f'testing score: {round(test_acc, 3)}')
    report_dict = classification_report(y_test, model.predict(X_test), output_dict = True, target_names=["No", "Yes"])
    utils.display_metrics(report_dict)
 
    imps = permutation_importance(model, X_test, y_test)
    top_feature_importances = list(sorted(enumerate(imps.importances_mean), key = lambda x: x[1], reverse = True))
    utils.log_results(top_feature_importances)
    utils.generate_report("Experiment SVM", "Experimental SVM", model, X_test, y_test, report_dict)
