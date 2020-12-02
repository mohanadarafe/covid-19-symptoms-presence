import os, shutil, time
import models.utils as utils
import models.svm_experiment as svm

if __name__ == "__main__":
    assert utils.assert_correct_directory(), f"Make sure you execute from the project root directory!"

    start = time.time()
    svm.svm_exp()
    end = time.time()
    print(f'\nDone training the models in {round(end-start, 3)} seconds')