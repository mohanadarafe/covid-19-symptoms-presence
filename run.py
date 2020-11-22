import os
import models.utils as utils, models.naive_bayes as gnb

if __name__ == "__main__":
    assert utils.assert_correct_directory(), f"Make sure you execute from the project root directory!"
    gnb.naive_bayes()