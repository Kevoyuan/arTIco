import sys
from logging import Logger
from pathlib import Path
from typing import List, Literal, Union

import numpy as np
import pandas as pd
from pandas._typing import DtypeArg

src_dir = str(Path(__file__).absolute().parent)
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils._custom_log as custom_log


class Csv:
    def __init__(
        self,
        csv_path: Path,
        compress: bool = False,
        log: Union[Logger, None] = None,
        csv_suffix: Literal[".csv"] = ".csv",
        compr_suffix: Literal["zip", "gzip", "bz2", "zstd", "xz", "tar"] = ".zip",
    ) -> None:
        """Csv file handling, wrapper for pandas.read_csv and df.to_csv()

        Args:
            csv_path (Path): path to csv file
            compress (bool, optional): csv is zip compressed. Defaults to False.
            log (Union[Logger, None], optional): logger. Defaults to None.
            csv_suffix (Literal[&quot;.csv&quot;], optional): csv file extension. Defaults to ".csv".
            compr_suffix (Literal[&quot;zip&quot;, &quot;gzip&quot;, &quot;bz2&quot;, &quot;zstd&quot;, &quot;xz&quot;, &quot;tar&quot;], optional):
                    Archive file extension, must be compatible with current pandas version. Defaults to ".zip".
        """

        # init logging
        if log is None:
            self.log = custom_log.init_logger(log_lvl=10)
        else:
            self.log = log

        # set compression by suffix (pandas will infer)
        compress = True if csv_path.suffix == compr_suffix else compress
        while csv_path.suffix in {csv_suffix, compr_suffix}:
            csv_path = csv_path.with_suffix("")
        self.log.debug("CSV file without suffix is %s, Compress %s", csv_path, compress)
        self.csv_path = (
            csv_path.with_suffix(f"{csv_suffix}{compr_suffix}")
            if compress
            else csv_path.with_suffix(csv_suffix)
        )
        self.log.debug("CSV file is %s", self.csv_path)

    def write(
        self,
        db: pd.DataFrame,
        float_format="%.4f",
    ) -> Path:
        """Wrapper for pandas.DataFrame().to_csv

        Args:
            db (pd.DataFrame): data frame
            float_format (str, optional): float formatting. Defaults to "%.4f".

        Returns:
            Path: path to csv file
        """
        self.log.debug(f"Write {self.csv_path}")
        db.to_csv(self.csv_path, float_format=float_format)

        return self.csv_path

    def read(
        self,
        dtype: Union[DtypeArg, None] = None,
        idx_round_prec: Union[int, None] = None,
        index_cols: Union[int, List[int]] = 0,
    ) -> pd.DataFrame:
        """Wrapper for pandas.read_csv()

        Args:
            dtype (Union[DtypeArg , None], optional): specific datatype to broadcast to. Defaults to None.
            idx_round_prec (Union[int , None], optional):: precision of returned data frame. Defaults to None.
            index_cols (Union[int, List[int]], optional): index of column(s) in CSV file to use as index in DataFrame. Defaults to 0.

        Returns:
            pd.DataFrame: new data frame
        """

        self.log.debug(f"Read {self.csv_path}")

        # get delimiter
        delimiters: List[str] = [",", "\t"]
        for delimiter in delimiters:
            csv: pd.DataFrame = pd.read_csv(
                self.csv_path, index_col=0, dtype=dtype, nrows=2, sep=delimiter
            )
            if csv.shape[1] > 0:
                break

        # read csv
        csv = pd.read_csv(
            self.csv_path, index_col=index_cols, dtype=dtype, sep=delimiter
        )

        # reduce float precision
        if idx_round_prec is not None:
            idx_name = csv.index.name
            csv.index = np.round(csv.index, idx_round_prec)
            csv.index.name = idx_name

        return csv
