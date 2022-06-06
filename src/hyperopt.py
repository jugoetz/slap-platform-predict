from copy import deepcopy

from ax.service.managed_loop import optimize

from src.cross_validation import cross_validate_predefined
from src.util.definitions import CONFIG_ROOT
from src.util.io import get_hparam_bounds


def optimize_hyperparameters_bayes(hparams, data, split_files):
    def objective_function(parameterization):
        """
        Wrapper around cross_validate() for Bayesian optimization with Ax.

        Ax requires that this function takes only one argument, parameterization.
        Consequently, variables `hparams` and `data` need to be defined in outside the function.

        Args:
            parameterization (dict): Parameters to be probed

        Returns:
            tuple: Score and STD for the probed parameters
        """
        hparams_local = deepcopy(hparams)
        # TODO should these apply to both enc and dec or only to one?
        hparams_local["encoder"]["hidden_size"] = parameterization["hidden_size"]
        hparams_local["encoder"]["depth"] = parameterization["mpnn_depth"]
        hparams_local["decoder"]["depth"] = parameterization["ffn_depth"]
        hparams_local["encoder"]["dropout_ratio"] = parameterization["dropout"]
        metrics = cross_validate_predefined(hparams=hparams_local,
                                            data=data,
                                            split_files=split_files,
                                            save_models=False,
                                            return_fold_metrics=False
                                            )
        # note that metrics returned from cross_validate_predefined are tensors, which ax cannot handle
        # thus we convert to float
        return {k: v.item() for k, v in metrics.items()}

    bounds = get_hparam_bounds(CONFIG_ROOT / "hparam_bounds.yaml")

    best_parameters, values, experiment, model = optimize(parameters=bounds,
                                                          evaluation_function=objective_function,
                                                          objective_name="val/loss_mean",
                                                          total_trials=20
                                                          )

    # print(experiment.trials.values())
    # print([trial.objective_mean for trial in experiment.trials.values()])
    # TODO logging. All the info is contained in experiment
    return best_parameters, values, experiment

