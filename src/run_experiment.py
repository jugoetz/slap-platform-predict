from torch.utils.data import DataLoader

from src.data.dataloader import SLAPDataset, collate_fn
from src.util.definitions import LOG_DIR
from src.util.io import walk_split_directory
from src.cross_validation import cross_validate_sklearn, cross_validate
from src.predict import predict
from src.model.classifier import load_trained_model
from src.hyperopt import optimize_hyperparameters_bayes

# TODO what do we do with test data?
def run_training(args, hparams):
    """
    Handles training and hyperparameter optimization.
    """
    # load data
    data = SLAPDataset(
        name=args.data_path.name,
        raw_dir=args.data_path.parent,
        reaction=hparams["encoder"]["reaction"],
        smiles_columns=(args.smiles_column,),
        label_column=args.label_column,
        graph_type=hparams["encoder"]["graph_type"],
        global_features=hparams["decoder"]["global_features"],
        global_features_file=hparams["decoder"]["global_features_file"],
        featurizers=hparams["encoder"]["featurizers"],
    )

    # update config with data processing specifics
    hparams["atom_feature_size"] = data.atom_feature_size
    hparams["bond_feature_size"] = data.bond_feature_size
    hparams["global_feature_size"] = data.global_feature_size

    # define split index files
    if args.split_indices:
        split_files = walk_split_directory(args.split_indices)
        strategy = "predefined"
    elif args.cv > 1:
        strategy = "KFold"
        split_files = None
    elif args.train_size:
        strategy = "random"
        split_files = None
    else:
        raise ValueError(
            "One of `--split_indices`, `--cv`, or `--train_size` must be given."
        )

    # run either cv or hyperparameter optimization wrapping cv
    if args.hparam_optimization:
        # run bayesian hparam optimization
        best_params, values, experiment = optimize_hyperparameters_bayes(
            data=data,
            hparams=hparams,
            hparam_config_path=args.hparam_config_path,
            cv_parameters={
                "strategy": strategy,
                "split_files": split_files,
                "n_folds": args.cv,
                "train_size": args.train_size,
            },
            n_iter=args.hparam_n_iter,
        )
        print(best_params, values)

    else:
        # run cross-validation with configured hparams
        if hparams["name"] in ["D-MPNN", "GCN", "FFN"]:
            aggregate_metrics, fold_metrics = cross_validate(
                data,
                hparams,
                strategy=strategy,
                n_folds=args.cv,
                train_size=args.train_size,
                split_files=split_files,
                return_fold_metrics=True,
            )
        elif hparams["name"] in ["LogisticRegression", "XGB"]:
            aggregate_metrics, fold_metrics = cross_validate_sklearn(
                data,
                hparams,
                split_files=split_files,
                save_models=False,
                return_fold_metrics=True,
            )
        else:
            raise ValueError(f"Unknown model type {hparams['name']}")
        print(aggregate_metrics)
        print(fold_metrics)

    return


def run_prediction(args, hparams):
    """
    Handles prediction from a trained model.
    """

    # load data
    data = SLAPDataset(
        name=args.data_path.name,
        raw_dir=args.data_path.parent,
        reaction=hparams["encoder"]["reaction"],
        smiles_columns=(args.smiles_column,),
        label_column=None,
        graph_type=hparams["encoder"]["graph_type"],
        global_features=hparams["decoder"]["global_features"],
        featurizers=hparams["encoder"]["featurizers"],
    )

    # instantiate DataLoader
    dl = DataLoader(data, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # load trained model
    model = load_trained_model(hparams["name"], args.model_path)

    # predict
    predictions = predict(model, dl, hparams)

    # save predictions to text file
    pred_file = (
        LOG_DIR / "predictions" / args.model_path.parent.name / "predictions.txt"
    )
    if not pred_file.parent.exists():
        pred_file.parent.mkdir(parents=True)
    with open(pred_file, "w") as f:
        f.write("Prediction\n")
        for i in predictions.tolist():
            f.write(str(i) + "\n")

    print(predictions)
