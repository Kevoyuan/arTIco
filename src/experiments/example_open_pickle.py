import pickle
import sys
from pathlib import Path

import numpy as np

SRC_DIR = str(Path(__file__).absolute().parents[2])
if SRC_DIR not in set(sys.path):
    sys.path.append(SRC_DIR)
from src.build.ExamplePipe import ExamplePipe


def example():
    """Example how to deal with pickle file
    """
    global SRC_DIR

    # read pickle
    with open(Path(SRC_DIR) / "experiments" / "2023-10-18-09-48-00_Example" / "pipeline_dev_fit.pkl", "rb") as f:
        obj: ExamplePipe = pickle.load(f)  # type real used pipe enables convenient suggestions in VSCode

    # example prediction
    gen = np.random.default_rng(42)
    feature = gen.random((10, 3, 1))
    feature_2d = gen.random((10, 8))
    print(obj.predict(x=feature, x_2d=feature_2d))


if __name__ == "__main__":
    example()
