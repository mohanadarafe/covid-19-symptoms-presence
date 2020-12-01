import os, shutil, time
import models.utils as utils
import models.random_forest_experiment as rf
import models.neural_network_experiment as nn

if __name__ == "__main__":
    assert utils.assert_correct_directory(), f"Make sure you execute from the project root directory!"

    start = time.time()
    rf.random_forest_exp()
    nn.neural_network_exp()
    end = time.time()
    print(f'\nDone training the models in {round(end-start, 3)} seconds')