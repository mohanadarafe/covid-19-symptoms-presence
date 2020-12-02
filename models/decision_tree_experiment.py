import numpy as np
import matplotlib.pyplot as plt 
import models.utils as utils, models.preprocess as preprocess
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV

def decision_tree_exp():
    print("="*60)
    print("Running experiement on Decision Tree...")
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
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 4, 5, 10, 13],
        'min_samples_leaf': [1, 2, 5, 8, 13]
    }

    clf = GridSearchCV(DecisionTreeClassifier(random_state=0), param_grid, cv=ps)

    model = clf.fit(X_grid, y_grid)
    report_dict = classification_report(y_test, model.predict(X_test), output_dict = True, target_names=["No", "Yes"])
    utils.display_metrics(report_dict)
    print(f'\nScore of Decision Tree: {round(model.score(X_test, y_test), 3)}')
 
    feature_importances = model.best_estimator_.feature_importances_
    top_feature_importances = list(sorted(enumerate(feature_importances), key = lambda x: x[1], reverse = True))
    utils.log_results(top_feature_importances)
    utils.generate_report("Experiment DT", "Experimental Decision Tree", model, X_test, y_test, report_dict)