from pathlib import Path
from shutil import rmtree

from loguru import logger


def overwrite_and_create_directory(dir_path: Path) -> None:
    if dir_path.exists():
        logger.info(f"Existing directory {dir_path}. Deleting it.")
        rmtree(dir_path)
    dir_path.mkdir(parents=True)


def create_directory(dir_path: Path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
