import matplotlib.pyplot as plt 
import models.preprocess as preprocess, models.utils as utils
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
from sklearn.metrics import plot_confusion_matrix


clf = LinearRegression()

def LinearRegression(sampling):
    print("Running Linear Regression...")
    DATA_FILE = utils.get_data_directory()

    # The argument of the function will determine weather we use oversampling or not
    if(sampling):
        process_method = preprocess.oversample(DATA_FILE)
    else:
        process_method = preprocess.preprocess_data(DATA_FILE)

    X, y = process_method
    X_train, X_test, y_train, y_test = utils.split_data(X, y, 0.8)

    model = clf.fit(X_train, y_train)
    # Linear regression is not classification therefore you cant use classification_report()
    # report_dict = classification_report(y_test, model.predict(X_test), output_dict = True, target_names=["No", "Yes"])
    # report = classification_report(y_test, model.predict(X_test), target_names=["No", "Yes"])
    # print(report)