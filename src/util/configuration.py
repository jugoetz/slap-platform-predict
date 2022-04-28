import yaml

from src.util.definitions import CONFIG_ROOT


def get_config(file):
    with open(file, "r") as f:
        conf = yaml.safe_load(f)
    return conf


CONFIG = get_config(CONFIG_ROOT / "config.yaml")
