from ax.service.managed_loop import optimize
from src.cross_validation import cross_validate
from copy import deepcopy


def optimize_hyperparameters(hparams, data):
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
        metrics = cross_validate(hparams=hparams_local, data=data)
        return metrics["val/AUROC_mean"], metrics["val/AUROC_std"]

    # TODO move this to external file
    bounds = [{"name": "hidden_size", "type": "range", "bounds": [100, 2500], "value_type": "int"},
              {"name": "mpnn_depth", "type": "range", "bounds": [3, 6], "value_type": "int"},
              {"name": "ffn_depth", "type": "range", "bounds": [1, 3], "value_type": "int"},
              {"name": "dropout", "type": "range", "bounds": [0.0, 0.5], "value_type": "float"},
              ]

    best_parameters, values, experiment, model = optimize(parameters=bounds,
                                                          evaluation_function=objective_function,
                                                          objective_name="val/AUROC_cv",
                                                          total_trials=1
                                                          )

    # print(experiment.trials.values())
    # print([trial.objective_mean for trial in experiment.trials.values()])
    # # assemble a dict with
    # TODO logging. All the info is contained in experiment
    return best_parameters, values, experiment

