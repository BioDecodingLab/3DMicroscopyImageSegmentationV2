from pathlib import Path
from shutil import rmtree

from loguru import logger


def overwrite_and_create_directory(dir_path: Path) -> None:
    try:
        dir_path.mkdir(parents=True)
    except FileExistsError:
        logger.info(f"Existing directory {dir_path}. Deleting it.")
        rmtree(dir_path)
        dir_path.mkdir(parents=True)


def create_directory(dir_path: Path) -> None:
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        logger.info(f"Found directory {dir_path}")
