from src.data.dataloader import SLAPDataset
from src.util.definitions import DATA_ROOT
from src.cross_validation import cross_validate_predefined, cross_validate_sklearn
from src.hyperopt import optimize_hyperparameters_bayes


def run_experiment(config, hparam_optimization):

    # load data
    data = SLAPDataset(name=config["data_name"],
                       raw_dir=DATA_ROOT,
                       reaction=config["reaction"],
                       smiles_columns=("SMILES", ),
                       label_column="targets",
                       graph_type=config["graph_type"],
                       rdkit_features=config["rdkit_features"],
                       featurizers=config["featurizers"],
                       )

    # update config with data processing specifics
    config["atom_feature_size"] = data.atom_feature_size
    config["bond_feature_size"] = data.bond_feature_size
    config["global_feature_size"] = data.global_feature_size

    # define split index files
    split_files = [{"train": DATA_ROOT / "LCMS_split_763records" / f"fold{i}_train.csv",
                    "val": DATA_ROOT / "LCMS_split_763records" / f"fold{i}_val.csv",
                    "test_0D": DATA_ROOT / "LCMS_split_763records" / f"fold{i}_test_0D.csv",
                    "test_1D": DATA_ROOT / "LCMS_split_763records" / f"fold{i}_test_1D.csv",
                    "test_2D": DATA_ROOT / "LCMS_split_763records" / f"fold{i}_test_2D.csv"}
                   for i in range(5)]

    if hparam_optimization:
        # run bayesian hparam optimization
        best_params, values, experiment = optimize_hyperparameters_bayes(config, data, split_files)
        print(best_params, values)
    else:
        # run cross-validation with configured hparams
        if config["decoder"]["type"] == "FFN":
            aggregate_metrics, fold_metrics = cross_validate_predefined(config, data, split_files=split_files, save_models=False, return_fold_metrics=True)
        else:
            aggregate_metrics, fold_metrics = cross_validate_sklearn(config, data, split_files=split_files, save_models=False, return_fold_metrics=True)
        print(aggregate_metrics)
        print(fold_metrics)
    return
