import matplotlib.pyplot as plt 
import models.preprocess as preprocess, models.utils as utils
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
from sklearn.metrics import plot_confusion_matrix
import inspect


clf = GaussianNB()

def naive_bayes(sampling = False, isNotebook = False):
    print("Running Gaussian Naive Bayes...")
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

    '''
    Since GNB does not have a native way of getting feature importances, we use permutation importance.
    Permutation importance works by shuffling features. If shuffling a symptom made the model perform
    worse, then it suggests that this symptom is important. Therefore, it is assigned a postiive value.
    '''
    imps = permutation_importance(model, X_test, y_test)
    features = utils.get_feature_names()
    feat_imp = list(sorted(enumerate(imps.importances_mean), key = lambda x: x[1], reverse = True)) 

    if isNotebook:
        return feat_imp

    u'•' == u'\u2022'
    print(f'\nThe top three features are: ')
    print(f'\t\u2022 {features[feat_imp[0][0]]} with a mean importance of {round(feat_imp[0][1], 4)}')
    print(f'\t\u2022 {features[feat_imp[1][0]]} with a mean importance of {round(feat_imp[1][1], 4)}')
    print(f'\t\u2022 {features[feat_imp[2][0]]} with a mean importance of {round(feat_imp[2][1], 4)}')
    utils.generate_report("GNB", "Naive Bayes", model, X_test, y_test, report_dict)