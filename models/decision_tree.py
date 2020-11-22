import matplotlib.pyplot as plt 
import models.preprocess as preprocess, models.utils as utils
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

def decision_tree():
    print("Running Decision Tree...")
    DATA_FILE = utils.get_data_directory()

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'min_samples_split': [2, 4, 5, 10, 13]
    }

    clf = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid)

    X, y = preprocess.preprocess_data(DATA_FILE)
    X_train, X_test, y_train, y_test = utils.split_data(X, y, 0.8)

    model = clf.fit(X_train, y_train)
    report_dict = classification_report(y_test, model.predict(X_test), output_dict = True, target_names=["No", "Yes"])
    report = classification_report(y_test, model.predict(X_test), target_names=["No", "Yes"])
    print(report)
    print(f'Best params are {model.best_params_}')

    utils.generate_report("DecisionTree", "Decision Tree", model, X_test, y_test, report_dict)

if __name__ == "__main__":
    decision_tree()