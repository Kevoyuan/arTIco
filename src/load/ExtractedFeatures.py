import datetime
import sys
from logging import Logger
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils._custom_log as custom_log
import src.utils._hash_file as hash_file
import src.utils._json_util as json_util
from src._StandardNames import StandardNames
from src.utils._Csv import Csv

LOG: Logger = custom_log.init_logger(log_lvl=10)


class ExtractFeatures:
    def __init__(self, log_lvl: int = 10) -> None:
        """Create tabular features

        Args:
            log_lvl (int, optional): logging level. Defaults to 10.
        """
        LOG.level = log_lvl
        self.str = StandardNames()

        # fixed paths
        self.data_dir = self.__check_directory(self.str.dir_raw_data)
        self.out_dir = self.__check_directory(self.str.dir_processed_data)
        self.info_path = self.out_dir / self.str.fname_data_info_2d
        self.trans_feature_path = self.out_dir / self.str.fname_feature_2d
        self.trans_target_path = self.out_dir / self.str.fname_target

        # adaptable
        self.feature_path = self.__check_file("feature.npy", b_dir=self.out_dir)
        self.feature_iso_path = self.__check_file("ratings_iso18571.csv", b_dir=self.data_dir)
        self.target_path = self.__check_file("ratings_experts.csv", b_dir=self.data_dir)
        self.info_multichannel_path = self.__check_file(self.str.fname_data_info, b_dir=self.out_dir)
        self.info_data = json_util.load(f_path=self.info_multichannel_path, log=LOG)
        self.str_use_rating = "rating_total"
        self.target_labels = tuple(self.info_data[self.str.id][self.str.labels])

        # sortings
        self.channels_names: List[str] = self.info_data[self.str.id][self.str.channels]
        self.sample_ids: List[str] = self.info_data[self.str.id][self.str.samples]
        self.time_stamps: List[float] = [float(t) for t in self.info_data[self.str.id][self.str.tsps]]

        # init data
        n_samples, n_features, n_channels = len(self.sample_ids), 2, len(self.channels_names)
        self.feature_3d: np.array = np.zeros((n_samples, n_features, n_channels))
        self.feature: pd.DataFrame = pd.DataFrame(index=self.sample_ids)
        self.iso_data: pd.DataFrame = pd.DataFrame()
        self.__all_channels: Dict[str, pd.DataFrame] = []

    def extract(self) -> None:
        LOG.info("Extract features")

        LOG.info("Load data")
        self._load_data()
        self._separate_channels()

        LOG.info("Extract defined features from time series")
        self._extract_from_3d()
        self._add_iso()

        LOG.info("Store tabular features")
        self._store()

    def __check_directory(self, d_path: Path) -> Path:
        """Checks if directory exists, exit if False

        Args:
            d_path (Path): directory path

        Returns:
            Path: directory path
        """
        # data directory
        if d_path.is_dir():
            LOG.info("Data taken from %s", d_path)
        else:
            LOG.critical("Directory %s does not exist - EXIT", d_path)
            sys.exit()

        return d_path

    def __check_file(self, d_fname: str, b_dir: Path) -> Path:
        """Checks if file exists, exit if False

        Args:
            d_fname (str): file name
            b_dir (Path): directory path

        Returns:
            Path: file path
        """
        data_fname = b_dir / d_fname
        if data_fname.is_file():
            LOG.debug("Use data file %s", data_fname)
        else:
            LOG.critical("File %s does not exist - EXIT", data_fname)
            sys.exit()

        return data_fname

    def _load_data(self) -> None:
        """Load data from hard drive"""
        self.feature_3d: np.ndarray = np.load(self.feature_path, allow_pickle=True)
        LOG.info("Got channel data shape %s from %s", self.feature_3d.shape, self.feature_path)

        self.iso_data = Csv(csv_path=self.feature_iso_path, log=LOG).read()
        LOG.info("Got iso data shape %s from %s", self.iso_data.shape, self.feature_iso_path)

    def _separate_channels(self) -> None:
        """Separate channels from feature3D and store it in a dict with channel name as key"""
        self.__all_channels = {
            ch: pd.DataFrame(data=self.feature_3d[:, i, :], columns=self.time_stamps) for i, ch in enumerate(self.channels_names)
        }

        LOG.debug(
            "Loaded %s channels with shapes %s from %s",
            len(self.__all_channels.keys()),
            [d.shape for d in self.__all_channels.values()],
            self.feature_3d.shape,
        )

    def __get_peak(self, channel: str, start_time: Union[int, float] = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Finds positive peaks in data

        Args:
            channel (str): channel name
            start_time (Union[int, float], optional): ignore peaks that happen before this time. Defaults to 0.

        Returns:
            Tuple[np.ndarray, np.ndarray]: time @ peak with shape (n_samples,), peak value with shape (n_samples,)
        """
        # init
        data = self.__all_channels[channel]
        height_peak = np.zeros(data.shape[0])
        time_peak = np.zeros(data.shape[0])

        # get peak
        for i in range(data.shape[0]):
            sample = data.iloc[i, :].to_numpy().reshape(data.shape[1])
            peaks, _ = find_peaks(sample)
            t = 0
            while data.columns[peaks[t]] < start_time:
                t += 1
                if t == len(peaks):
                    break
            try:
                height_peak[i] = data.iloc[i, peaks[t]]
            except ValueError:
                height_peak[i] = 0
            try:
                time_peak[i] = data.columns[peaks[t]]
            except ValueError:
                time_peak[i] = 0

        LOG.debug(
            "Extracted peak for channel %s with shape %s and time @ peak with shape %s",
            channel,
            height_peak.shape,
            time_peak.shape,
        )

        return time_peak, height_peak

    def __get_negative_peak(self, channel: str) -> Tuple[np.ndarray, np.ndarray]:
        """Finds negative peaks in data

        Args:
            channel (str): channel name

        Returns:
            Tuple[np.ndarray, np.ndarray]: time @ peak with shape (n_samples,), peak value with shape (n_samples,)
        """

        data = self.__all_channels[channel]
        flipped_data = -data
        height_peak = np.zeros(data.shape[0])
        time_peak = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            flipped = flipped_data.iloc[i, :].to_numpy().reshape(data.shape[1])
            peaks, _ = find_peaks(flipped)
            try:
                height_peak[i] = data.iloc[i, peaks[0]]
            except ValueError:
                height_peak[i] = 0
            try:
                time_peak[i] = data.columns[peaks[0]]
            except ValueError:
                time_peak[i] = 0

        LOG.debug(
            "Extracted negative peak for channel %s with shape %s and time @ peak with shape %s",
            channel,
            height_peak.shape,
            time_peak.shape,
        )
        return time_peak, height_peak

    def __get_init_velocity(self, channel: str) -> np.ndarray:
        """Extract initial velocity of channel

        Args:
            channel (str): channel name

        Returns:
            np.ndarray: initial velocity with shape (n_samples,)
        """
        init_vel = self.__all_channels[channel].iloc[:, 0].to_numpy()
        LOG.debug("Extracted initial velocity of channel %s with shape %s", channel, init_vel.shape)
        return init_vel

    def __get_max(self, channel: str) -> Tuple[np.ndarray, np.ndarray]:
        """Find maxima in data

        Args:
            channel (str): channel name

        Returns:
            Tuple[np.ndarray, np.ndarray]: time @ max with shape (n_samples,), max value with shape (n_samples,)
        """

        data = self.__all_channels[channel]
        t_at_max, max_val = data.idxmax(axis=1), data.max(axis=1)
        LOG.debug(
            "Extracted maxima for channel %s with shape %s and time @ maxima with shape %s",
            channel,
            max_val.shape,
            t_at_max.shape,
        )
        return t_at_max.to_numpy(), max_val.to_numpy()

    def _extract_from_3d(self) -> None:
        """Extract features from channel data and add it to features"""
        LOG.info("Extract features from channel data and add it to features shape %s", self.feature.shape)

        # for shorter cases tuple unpacking á a[["q", "w"]]=[1,2], [3,2] with a = pd.DataFrame(index=[1,2]) works
        # for longer index, list is treated as row leading to 'ValueError: Columns must be same length as key'

        # add acceleration sensors
        for i in ["04", "12"]:
            self.feature[f"t{i}_tPeak"], self.feature[f"t{i}_hPeak"] = self.__get_peak(channel=f"01THSP{i}00WSAC", start_time=5)

        # add impactor
        self.feature["imp_vInit"] = self.__get_init_velocity(channel="01TIMPLE00WSVE")
        self.feature["imp_tPeak"], self.feature["imp_hPeak"] = self.__get_negative_peak(channel="01TIMPLE00WSAC")

        # add rib deflection
        for i in [1, 2, 3]:
            self.feature[f"r0{i}_tPeak"], self.feature[f"r0{i}_hPeak"] = self.__get_max(channel=f"01TRRILE0{i}WSDS")

        LOG.info("Extracted %s features for %s samples", self.feature.shape[1], self.feature.shape[0])

    def _add_iso(self) -> None:
        """Add ISO 18751 features to feature2D"""
        # select ratings
        rating_names = sorted(self.iso_data["Rating"].unique())
        rating_names.remove("ISO 18571 Label")
        rating_names.remove("ISO 18571 Rank")
        total = "Total"

        i = 0
        for channel in sorted(self.iso_data["Channel"].unique()):
            for rating in rating_names:
                # skip not existing combinations
                if (channel == total or rating == total) and rating != channel:
                    continue

                # filter
                filt = self.iso_data[self.iso_data["Channel"].eq(channel) & self.iso_data["Rating"].eq(rating)].copy()
                filt.set_index("SimID", inplace=True)

                # store
                f_name = f"{channel}_{'ISO_18571_Rating' if rating==total else rating}".replace(" ", "_")
                LOG.debug("Add iso feature %s with shape %s", f_name, filt.shape)
                self.feature[f_name] = filt.loc[self.feature.index, "Value"]

                # for log
                i += 1

        LOG.info("Added %s iso features for %s samples, now %s features", i, self.feature.shape[0], self.feature.shape[1])

    def _store(self) -> None:
        """Store feature2D"""
        # init book
        book = {self.str.creation: str(datetime.datetime.now())}

        # store transformed data
        cols = sorted(self.feature.columns)
        to_store = self.feature.loc[self.sample_ids, cols].to_numpy()
        LOG.info("Store %s array with shape %s", self.trans_feature_path, to_store.shape)
        np.save(self.trans_feature_path, to_store)

        # hash in and out files
        fnames = {
            self.str.input: {
                self.str.feature: self.feature_path,
                self.str.iso: self.feature_iso_path,
            },
            self.str.output: {
                self.str.feature: self.trans_feature_path,
            },
        }
        for in_out in fnames.keys():
            book[in_out] = {}
            for feta in fnames[in_out].keys():
                fpath = fnames[in_out][feta]
                hashed = hash_file.hash_file(fpath=fpath, log=LOG)
                book[in_out][feta] = {self.str.path: fpath, self.str.hash: hashed}

        # add ids
        book[self.str.id] = {
            self.str.samples: self.sample_ids,
            self.str.feature: cols,
        }

        # store book
        json_util.dump(obj=book, f_path=self.info_path, log=LOG)


if __name__ == "__main__":
    ExtractFeatures(log_lvl=10).extract()
    LOG.info("DONE")
