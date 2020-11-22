import matplotlib.pyplot as plt 
import models.preprocess as preprocess, models.utils as utils
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

def decision_tree():
    print("="*60)
    print("\nRunning Decision Tree...")
    DATA_FILE = utils.get_data_directory()

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'min_samples_split': [2, 4, 5, 10, 13]
    }

    clf = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid)

    X, y = preprocess.preprocess_data(DATA_FILE)
    X_train, X_test, y_train, y_test = utils.split_data(X, y, 0.7)

    model = clf.fit(X_train, y_train)
    report_dict = classification_report(y_test, model.predict(X_test), output_dict = True, target_names=["No", "Yes"])
    report = classification_report(y_test, model.predict(X_test), target_names=["No", "Yes"])
    print(report)
    
    feature_importances = model.best_estimator_.feature_importances_
    features = utils.get_feature_names()
    top_feature_importances = list(sorted(enumerate(feature_importances), key = lambda x: x[1], reverse = True))

    u'•' == u'\u2022'
    print(f'The top three features are: ')
    print(f'\t\u2022 {features[top_feature_importances[0][0]]} with a mean importance of {round(top_feature_importances[0][1], 4)}')
    print(f'\t\u2022 {features[top_feature_importances[1][0]]} with a mean importance of {round(top_feature_importances[1][1], 4)}')
    print(f'\t\u2022 {features[top_feature_importances[2][0]]} with a mean importance of {round(top_feature_importances[2][1], 4)}') 

    utils.generate_report("DecisionTree", "Decision Tree", model, X_test, y_test, report_dict)

if __name__ == "__main__":
    decision_tree()