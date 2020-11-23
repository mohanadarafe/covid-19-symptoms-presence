import matplotlib.pyplot as plt 
import models.preprocess as preprocess, models.utils as utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
from sklearn.metrics import plot_confusion_matrix


clf = RandomForestClassifier()

def random_forest(sampling = False):
    print("="*60)
    print("Running Random Forest...")
    DATA_FILE = utils.get_data_directory()

    # The argument of the function will determine weather we use oversampling or not
    if(sampling):
        process_method = preprocess.oversample(DATA_FILE)
    else:
        process_method = preprocess.preprocess_data(DATA_FILE)

    X, y = process_method
    X_train, X_test, y_train, y_test = utils.split_data(X, y, 0.8)

    model = clf.fit(X_train, y_train)
    report_dict = classification_report(y_test, model.predict(X_test), output_dict = True, target_names=["No", "Yes"])
    utils.display_metrics(report_dict)

    feature_importances = model.feature_importances_
    features = utils.get_feature_names()
    top_feature_importances = list(sorted(enumerate(feature_importances), key = lambda x: x[1], reverse = True))

    u'•' == u'\u2022'
    print(f'\nThe top three features are: ')
    print(f'\t\u2022 {features[top_feature_importances[0][0]]} with a mean importance of {round(top_feature_importances[0][1], 4)}')
    print(f'\t\u2022 {features[top_feature_importances[1][0]]} with a mean importance of {round(top_feature_importances[1][1], 4)}')
    print(f'\t\u2022 {features[top_feature_importances[2][0]]} with a mean importance of {round(top_feature_importances[2][1], 4)}') 

    utils.generate_report("Random Forest", "Random Forest", model, X_test, y_test, report_dict)