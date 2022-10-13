import argparse
import pathlib

from src.run_experiment import run_training, run_prediction
from src.util.configuration import get_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # parent parser for training and prediction
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--config", type=pathlib.Path, help="Path to config file", required=True
    )
    parent_parser.add_argument(
        "--data_path",
        type=pathlib.Path,
        required=True,
        help="Path to the data. For training, this should contain labels. For prediction, labels will be ignored.",
    )
    parent_parser.add_argument(
        "--smiles_column",
        type=str,
        default="SMILES",
        help="Name of the column containing SMILES strings to use as input.",
    )

    # train parser
    train_parser = subparsers.add_parser("train", parents=[parent_parser])
    train_parser.set_defaults(func=run_training)

    train_parser.add_argument(
        "--label_column",
        type=str,
        default="targets",
        help="Name of the column containing ground truth labels.",
    )
    train_parser.add_argument(
        "--hparam_optimization",
        action="store_true",
        help="Run hyperparameter optimization",
    )
    train_parser.add_argument(
        "--hparam_config_path",
        type=str,
        help="Path to hyperparameter search config file",
    )
    train_parser.add_argument(
        "--hparam_n_iter",
        type=int,
        help="Number of trials for hyperparameter optimization",
    )
    train_parser.add_argument(
        "--cv",
        type=int,
        help="Number of CV folds. If > 1, a random CV split is done. Ignored if --split_indices is provided.",
        default=0,
    )
    train_parser.add_argument(
        "--train_size",
        type=float,
        help="Fraction of data to use for training. If cv > 1 or '--split_indices' given , this is ignored.",
        default=0.9,
    )
    train_parser.add_argument(
        "--split_indices",
        type=pathlib.Path,
        help="""Path to directory containing data set splits.
        Expects a directory of csv-files following the naming convention fold{i}_{train,val,test}[_optionalsuffix].csv.
        For each fold, one train and one val set need to be given. An optional arbitrary number of test sets can be
        given. Multiple test sets are distinguished by the optional suffix.
        Fold indices start at 0 and increase monotonically.
        """,
    )
    train_parser.add_argument(
        "--run_test",
        action="store_true",
        help="Run test set evaluation after training. Only takes an effect if `--split_indices` is given and test set "
        "is present in the split directory for this to have an effect.",
    )

    # predict parser
    predict_parser = subparsers.add_parser("predict", parents=[parent_parser])
    predict_parser.add_argument(
        "--model_path",
        required=True,
        type=pathlib.Path,
        help="Path to the model checkpoint to be used for inference.",
    )

    predict_parser.set_defaults(func=run_prediction)

    args = parser.parse_args()

    hparams = get_config(args.config)

    args.func(args, hparams)
