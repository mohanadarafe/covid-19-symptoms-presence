import os, shutil
import models.utils as utils
import models.naive_bayes as gnb
import models.decision_tree as dt
import models.random_forest as rf
import models.support_vector_machine as svm
import models.neural_network as nn

if __name__ == "__main__":
    assert utils.assert_correct_directory(), f"Make sure you execute from the project root directory!"
    if os.path.isdir("results"):
        shutil.rmtree("results")
    os.makedirs("results")

    # Change argument to True to use oversampling
    gnb.naive_bayes()
    dt.decision_tree(True)
    rf.random_forest()
    svm.support_vector_machine()
    nn.neural_network()