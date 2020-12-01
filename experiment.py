import os, shutil, time
import models.utils as utils
import models.random_forest_experiment as rf

if __name__ == "__main__":
    assert utils.assert_correct_directory(), f"Make sure you execute from the project root directory!"

    start = time.time()
    rf.random_forest()
    end = time.time()
    print(f'Done training the models in {round(end-start, 3)} seconds')