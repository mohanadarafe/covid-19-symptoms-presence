import matplotlib.pyplot as plt 
import models.preprocess as preprocess, models.utils as utils
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
from sklearn.metrics import plot_confusion_matrix


clf = svm.SVC(kernel="linear")

def support_vector_machine(sampling):
    print("Running support vector machine...")
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
    report = classification_report(y_test, model.predict(X_test), target_names=["No", "Yes"])
    print(report)

    weights = model.coef_
    print(weights)
    features = utils.get_feature_names()
    top_weights = list(sorted(enumerate(weights[0]), key = lambda x: x[1], reverse = True))

    u'•' == u'\u2022'
    print(f'The 3 features with the largest weights assigned to them are: ')
    print(f'\t\u2022 {features[top_weights[0][0]]} with a mean importance of {round(top_weights[0][1], 4)}')
    print(f'\t\u2022 {features[top_weights[1][0]]} with a mean importance of {round(top_weights[1][1], 4)}')
    print(f'\t\u2022 {features[top_weights[2][0]]} with a mean importance of {round(top_weights[2][1], 4)}') 

    utils.generate_report("SVM", "SVM", model, X_test, y_test, report_dict)