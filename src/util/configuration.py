import yaml


def get_config(file):
    with open(file, "r") as f:
        conf = yaml.safe_load(f)
    return conf
