import pathlib

PROJECT_DIR = pathlib.Path(__file__).parents[2]
DATA_ROOT = PROJECT_DIR / "data"
CONFIG_ROOT = PROJECT_DIR / "config"
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR_ABS = str((LOG_DIR).absolute())
