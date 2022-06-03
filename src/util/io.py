import yaml


def index_from_file(path):
    """Load a list of newline-separated integer indices from a file"""
    with open(path, "r") as file:
        indices = [int(l.strip("\n")) for l in file.readlines()]
    return indices


def get_hparam_bounds(path):
    """Read hyperparameter bounds from a yaml file"""
    with open(path, "r") as f:
        bounds = yaml.safe_load(f)
    return bounds
