from src.data.dataloader import SLAPDataset
from src.util.configuration import get_config
from src.util.definitions import DATA_ROOT, CONFIG_ROOT
from src.cross_validation import cross_validate, cross_validate_predefined
from src.hyperopt import optimize_hyperparameters_bayes


def main(config, hparam_optimization):

    # load data
    data = SLAPDataset(name=config["data_name"],
                       raw_dir=DATA_ROOT,
                       reaction=config["reaction"],
                       smiles_columns=("SMILES", ),
                       label_column="targets",
                       molecular_graph=config["molecular_graph"],
                       rdkit_features=config["rdkit_features"],
                       featurizers=config["featurizers"],
                       )

    # update config with data processing specifics
    config["atom_feature_size"] = data.atom_feature_size
    config["bond_feature_size"] = data.bond_feature_size

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
        aggregate_metrics, fold_metrics = cross_validate_predefined(config, data, split_files=split_files, save_models=False, return_fold_metrics=True)
        print(aggregate_metrics)
        print(fold_metrics)
    return


if __name__ == "__main__":
    CONFIG = get_config(CONFIG_ROOT / "config.yaml")
    hparam_optimization = False  # invoke hparam sweep if True, else invoke single-point training
    main(CONFIG, hparam_optimization)

