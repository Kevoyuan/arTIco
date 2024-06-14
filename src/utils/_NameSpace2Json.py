import sys
from logging import Logger
from pathlib import Path
from typing import Union

src_dir = str(Path(__file__).absolute().parent)
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils._custom_log as custom_log
import src.utils._json_util as json_util


class NameSpace2Json:
    def __init__(self, prefix: str = "_", log: Union[Logger, None] = None) -> None:
        # init logging
        if log is None:
            self.log = custom_log.init_logger(log_lvl=10)
        else:
            self.log = log

        self.prefix = prefix
        self.attributes = {}

    def get_attr(self, class_obj: object):
        attributes_all = class_obj.__dict__
        for attr in attributes_all.keys():
            if len(attr) > 2 and attr[0] == self.prefix and attr[1] != self.prefix:
                self.attributes[attr] = attributes_all[attr]
        self.log.debug(f"Found {len(self.attributes)} attributes")

    def to_json(self, f_path: Path) -> Path:
        return json_util.dump(obj=self.attributes, f_path=f_path, log=self.log)
