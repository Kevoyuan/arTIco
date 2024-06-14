import hashlib
from logging import Logger
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Union
from zipfile import ZipFile

import _hashlib

BLOCK_SIZE = 65536  # The size of each read from the file


def hash_file(fpath: Union[Path, List[Path]], log: Logger) -> str:
    """Generate hash hex str of a file

    Args:
        fpath (Union[Path, List[Path]]): path to file or list of file paths
        log (Logger): logger

    Returns:
        str: hash digest
    """
    file_hash = hashlib.sha256()

    if isinstance(fpath, list):
        for f in fpath:
            file_hash = __update_hash(fpath=f, file_hash=file_hash, log=log)
    else:
        file_hash = __update_hash(fpath=fpath, file_hash=file_hash, log=log)

    return file_hash.hexdigest()


def __update_hash(fpath: Path, file_hash: _hashlib.HASH, log: Logger) -> _hashlib.HASH:
    """Adds hash of single file
    Adapted from https://nitratine.net/blog/post/how-to-hash-files-in-python/

    Args:
        fpath (Path): path to file which should be hashed
        file_hash (_hashlib.HASH): initialized hash object
        log (Logger): logger

    Returns:
        _hashlib.HASH: updated hash object
    """
    if fpath.is_file():
        log.debug("Hash file %s", fpath)
        if fpath.suffix == ".zip":
            with TemporaryDirectory() as tmp:
                with ZipFile(fpath, "r") as zip_ref:
                    zip_ref.extractall(tmp)
                    for sub_f in zip_ref.namelist():
                        file_hash = __update_hash_inner(fpath=Path(tmp) / sub_f, file_hash=file_hash, log=log)
        else:
            file_hash = __update_hash_inner(fpath=fpath, file_hash=file_hash, log=log)
    else:
        log.warning("File %s does not exist", fpath)

    return file_hash


def __update_hash_inner(fpath: Path, file_hash: _hashlib.HASH, log: Logger) -> _hashlib.HASH:
    log.debug("Add file content from '%s' to hash", fpath)
    with open(fpath, "rb") as f:
        fb = f.read(BLOCK_SIZE)
        while len(fb) > 0:
            file_hash.update(fb)
            fb = f.read(BLOCK_SIZE)
    return file_hash

