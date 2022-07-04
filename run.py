import argparse

from src.run_experiment import run_experiment
from src.util.configuration import get_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparam_optimization", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    CONFIG = get_config(args.config)

    run_experiment(config=CONFIG, hparam_optimization=args.hparam_optimization)
