import json
import logging
from pathlib import Path, PosixPath, WindowsPath
import pandas as pd

import numpy as np


class GeneralEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, PosixPath) or isinstance(obj, WindowsPath):
            return str(obj).replace("\\\\", "\\")
        elif isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
            return obj.to_dict()
        else:
            return json.JSONEncoder.default(self, obj)


def _path_ending(f_path: Path):
    return f_path if f_path.suffix == ".json" else f_path.parent / f"{f_path.stem}.json"


def dump_with_hash(obj, f_path: Path, log: logging.Logger):
    json_str = json.dumps(obj=obj)
    f_path = f_path.parent / f"{f_path.stem}_{hash(json_str)}.json"
    dump(obj=obj, f_path=f_path, log=log)


def dump(obj, f_path: Path, log: logging.Logger) -> Path:
    f_path = _path_ending(f_path=f_path)
    log.debug(f"Write {f_path}")

    with open(file=f_path, mode="w", encoding="utf-8") as f:
        json.dump(obj=obj, indent=2, fp=f, cls=GeneralEncoder)

    return f_path


def load(f_path: Path, log: logging.Logger):
    f_path = _path_ending(f_path=f_path)
    log.debug(f"Load {f_path}")

    with open(file=f_path, mode="r") as f:
        obj = json.load(fp=f)

    return obj
