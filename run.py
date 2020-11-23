import os, shutil
import models.utils as utils, models.naive_bayes as gnb, models.decision_tree as dt

if __name__ == "__main__":
    assert utils.assert_correct_directory(), f"Make sure you execute from the project root directory!"
    if os.path.isdir("results"):
        shutil.rmtree("results")
    os.makedirs("results")

    gnb.naive_bayes(False)
    dt.decision_tree(True)