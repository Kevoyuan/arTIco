import datetime
import importlib.metadata as metadata
import subprocess
import sys
from logging import Logger
from pathlib import Path
from typing import Union

SRC_DIR = str(Path(__file__).absolute().parents[2])
if SRC_DIR not in set(sys.path):
    sys.path.append(SRC_DIR)
import src.utils._custom_log as custom_log
import src.utils._json_util as json_util
from src._StandardNames import StandardNames


class Parameters:
    def __init__(self, log: Union[Logger, None] = None) -> None:
        """Parameter.json utility

        Args:
            log (Union[Logger, None], optional): logging. Defaults to None.
        """
        self.__log: Logger = custom_log.init_logger(log_lvl=10) if log is None else log
        self.str = StandardNames()

    def create(self, exp_dir: Path, data_dir: Path, pipeline_paras: dict):
        """Create parameter.json with standardized entries and  pipeline specifics

        Args:
            exp_dir (Path): directory of experiment, used for output
            data_dir (Path): directory of standardized data files
            pipeline_paras (dict): pipeline specific parameters compatible with .set_params method
        """
        global SRC_DIR
        self.__log.debug("Store experiment's parameter")

        parameters = {
            self.str.creation: str(datetime.datetime.now()),
            self.str.data: {
                self.str.input: {
                    self.str.dir: data_dir,
                    self.str.feature: self.str.fname_feature,
                    self.str.feature_2d: self.str.fname_feature_2d,
                    self.str.target: self.str.fname_target,
                    self.str.info: self.str.fname_data_info,
                    self.str.info_2d: self.str.fname_data_info_2d,
                },
                self.str.output: exp_dir,
            },
            self.str.pipeline: pipeline_paras,
            self.str.python: {
                # "Python": sys.version,
                # "Git_Parent": {
                #    "Hash": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SRC_DIR).decode("ascii").strip(),
                #    "Branch": subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=SRC_DIR).decode("ascii").strip(),
                # },
                "Packages": {pa.name: metadata.version(pa.name) for pa in metadata.distributions()},
            },
        }

        json_util.dump(obj=parameters, f_path=exp_dir / self.str.fname_para, log=self.__log)

    def read(self, exp_dir: Path) -> dict:
        """Read and check parameters.json

        Args:
            exp_dir (Path): directory with parameters.json

        Returns:
            dict: parameters
        """
        f_path = exp_dir / self.str.fname_para
        paras = json_util.load(f_path=exp_dir / self.str.fname_para, log=self.__log)

        # check
        if self.str.data in paras and self.str.pipeline in paras:
            if self.str.input in paras[self.str.data] and self.str.output in paras[self.str.data]:
                if (
                    self.str.dir in paras[self.str.data][self.str.input]
                    and self.str.feature in paras[self.str.data][self.str.input]
                    and self.str.target in paras[self.str.data][self.str.input]
                    and self.str.info in paras[self.str.data][self.str.input]
                ):
                    self.__log.debug("Parameter file consistent - CONTINUE")
                else:
                    self.__log.critical(
                        "Parameters in %s not consistent, please see example - EXIT",
                        f_path,
                    )
                    sys.exit()
            else:
                self.__log.critical("Parameters in %s not consistent, please see example - EXIT", f_path)
                sys.exit()
        else:
            self.__log.critical("Parameters in %s not consistent, please see example - EXIT", f_path)
            sys.exit()

        return paras
