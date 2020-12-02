import os, shutil, time
import models.utils as utils
import models.decision_tree_experiment as dt
import models.neural_network_experiment as nn

if __name__ == "__main__":
    assert utils.assert_correct_directory(), f"Make sure you execute from the project root directory!"

    start = time.time()
    dt.decision_tree_exp()
    nn.neural_network_exp()
    end = time.time()
    print(f'\nDone training the models in {round(end-start, 3)} seconds')