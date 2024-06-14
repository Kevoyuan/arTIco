import datetime
import sys
from logging import Logger
from pathlib import Path
from typing import List, Union
import numpy as np
import pandas as pd

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils._custom_log as custom_log
import src.utils._json_util as json_util
from src.utils._Csv import Csv
import src.utils._hash_file as hash_file
from src._StandardNames import StandardNames


class LoadFromRaw:
    def __init__(
        self,
        log: Union[Logger, None] = None,
    ) -> None:
        # log
        self.__log: Logger = custom_log.init_logger(log_lvl=10) if log is None else log
        self.str = StandardNames()

        # fixed paths
        self.data_dir = self.__check_directory(self.str.dir_raw_data)
        self.out_dir = self.__check_directory(self.str.dir_processed_data)
        self.info_path = self.out_dir / self.str.fname_data_info
        self.trans_feature_path = self.out_dir / self.str.fname_feature
        self.trans_target_path = self.out_dir / self.str.fname_target

        # adaptable
        self.feature_path = self.__check_file("signals_cae.csv")
        self.target_path = self.__check_file("ratings_experts.csv")
        self.str_use_rating = "rating_total"
        self.target_labels = ("Good", "Acceptable", "Marginal", "Poor")

        # sortings
        self.channels_names: List[str] = []
        self.sample_ids: List[str] = []
        self.time_stamps: List[int] = []

        # init data
        n_samples, n_features, n_channels = 4, 2, 1
        self.feature: np.array = np.zeros((n_samples, n_features, n_channels))
        self.target: np.array = np.zeros((n_samples, 1))

    def __check_directory(self, d_path: Path) -> Path:
        # data directory
        if d_path.is_dir():
            self.__log.info("Data taken from %s", d_path)
        else:
            self.__log.critical("Directory %s does not exist - EXIT", d_path)
            sys.exit()

        return d_path

    def __check_file(self, d_fnames: str) -> Path:
        data_fname = self.data_dir / d_fnames
        if data_fname.is_file():
            self.__log.debug("Use data file %s", data_fname)
        else:
            self.__log.critical("File %s does not exist - EXIT", data_fname)
            sys.exit()

        return data_fname

    def target2vector(self):
        raw: pd.DataFrame = Csv(csv_path=self.target_path).read()
        self.sample_ids = sorted(raw.index)
        target = raw.loc[self.sample_ids, self.str_use_rating].to_numpy()
        self.target = target.reshape(-1, 1)

    def features2tensor(self):
        raw: pd.DataFrame = Csv(csv_path=self.feature_path).read(index_cols=None)

        self.channels_names = sorted(raw[self.str.channels].unique())
        self.time_stamps = sorted(raw[self.str.tsps].unique())

        data = []
        for sid in self.sample_ids:
            # filter
            raw_filt = raw[raw[self.str.rid].eq(sid)]

            # reshape
            raw_filt = raw_filt.pivot(
                index=self.str.channels, columns=self.str.tsps, values=self.str.signal
            )

            # store
            data.append(raw_filt.loc[self.channels_names, self.time_stamps].to_numpy())

        feature = np.array(data)
        self.feature = feature

    def store(self):
        self.__log.info("Store data %s", self.info_path)

        # init book
        book = {self.str.creation: str(datetime.datetime.now())}

        # store transformed data
        np.save(self.trans_feature_path, self.feature)
        np.save(self.trans_target_path, self.target)

        # hash in and out files
        fnames = {
            self.str.input: {
                self.str.feature: self.feature_path,
                self.str.target: self.target_path,
            },
            self.str.output: {
                self.str.feature: self.trans_feature_path,
                self.str.target: self.trans_target_path,
            },
        }
        for in_out in fnames.keys():
            book[in_out] = {}
            for feta in fnames[in_out].keys():
                fpath = fnames[in_out][feta]
                hashed = hash_file.hash_file(fpath=fpath, log=self.__log)
                book[in_out][feta] = {self.str.path: fpath, self.str.hash: hashed}

        # add ids
        book[self.str.id] = {
            self.str.labels: self.target_labels,
            self.str.samples: self.sample_ids,
            self.str.tsps: [f"{x:.2f}" for x in self.time_stamps],
            self.str.channels: self.channels_names,
        }

        # store book
        json_util.dump(obj=book, f_path=self.info_path, log=self.__log)


def run():
    log = custom_log.init_logger()
    loader = LoadFromRaw(log=log)
    loader.target2vector()
    log.debug("Target shape %s", loader.target.shape)
    loader.features2tensor()
    log.debug("Feature shape %s", loader.feature.shape)

    loader.store()

    log.info("DONE")


if __name__ == "__main__":
    run()
