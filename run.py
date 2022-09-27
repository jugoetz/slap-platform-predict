import argparse

from src.run_experiment import run_experiment
from src.util.configuration import get_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hparam_optimization",
        action="store_true",
        help="Run hyperparameter optimization",
    )
    parser.add_argument(
        "--hparam_config_path",
        type=str,
        help="Path to hyperparameter search config file",
    )
    parser.add_argument(
        "--hparam_n_iter",
        type=int,
        help="Number of trials for hyperparameter optimization",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    CONFIG = get_config(args.config)

    run_experiment(
        config=CONFIG,
        hparam_optimization=args.hparam_optimization,
        hparam_config_path=args.hparam_config_path,
        hparam_n_iter=args.hparam_n_iter,
    )
