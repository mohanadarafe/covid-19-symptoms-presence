import preprocess, utils
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

clf = GaussianNB()

def naive_bayes():
    print("Running Gaussian Naive Bayes...")
    DATA_FILE = utils.get_data_directory()

    X, y = preprocess.preprocess_data(DATA_FILE)
    X_train, X_test, y_train, y_test = utils.split_data(X, y, 0.8)

    model = clf.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test), target_names=["No", "Yes"]))

if __name__ == "__main__":
    naive_bayes()