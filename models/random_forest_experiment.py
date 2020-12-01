import numpy as np
import matplotlib.pyplot as plt 
import models.utils as utils, models.preprocess as preprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV

def random_forest():
    print("="*60)
    print("Running experiement on Random Forest...")
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
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 4, 5, 10, 13],
        'min_samples_leaf': [1, 2, 5, 8, 13]
    }

    clf = GridSearchCV(RandomForestClassifier(random_state=0), param_grid, cv=ps)

    model = clf.fit(X_grid, y_grid)
    print(model.score(X_test, y_test))
    report_dict = classification_report(y_test, model.predict(X_test), output_dict = True, target_names=["No", "Yes"])
    utils.display_metrics(report_dict)
 
    feature_importances = model.best_estimator_.feature_importances_
    top_feature_importances = list(sorted(enumerate(feature_importances), key = lambda x: x[1], reverse = True))
    utils.log_results(top_feature_importances)
    utils.generate_report("Experiment Results", "Experimental Random Forest", model, X_test, y_test, report_dict)