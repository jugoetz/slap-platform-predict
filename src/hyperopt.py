from copy import deepcopy

from ax.service.managed_loop import optimize

from src.cross_validation import cross_validate_predefined, cross_validate_sklearn
from src.util.io import get_hparam_bounds


def optimize_hyperparameters_bayes(
    hparams, data, split_files, hparam_config_path, n_iter=50
):
    def objective_function(parameterization):
        """
        Wrapper around cross_validate_predefined() for Bayesian optimization with Ax.

        Ax requires that this function takes only one argument, parameterization.
        Consequently, variables `hparams` and `data` need to be defined in outside the function.

        Args:
            parameterization (dict): Parameters to be probed

        Returns:
            tuple: Score and STD for the probed parameters
        """
        hparams_local = deepcopy(hparams)

        hparams_local["encoder"]["hidden_size"] = parameterization.get(
            "encoder_hidden_size", hparams["encoder"]["hidden_size"]
        )
        hparams_local["encoder"]["depth"] = parameterization.get(
            "encoder_depth", hparams["encoder"]["depth"]
        )
        hparams_local["decoder"]["hidden_size"] = parameterization.get(
            "decoder_hidden_size", hparams["decoder"]["hidden_size"]
        )
        hparams_local["decoder"]["depth"] = parameterization.get(
            "decoder_depth", hparams["decoder"]["depth"]
        )
        hparams_local["encoder"]["dropout_ratio"] = parameterization.get(
            "dropout", hparams["encoder"]["dropout_ratio"]
        )
        hparams_local["decoder"]["dropout_ratio"] = parameterization.get(
            "dropout", hparams["decoder"]["dropout_ratio"]
        )
        hparams_local["optimizer"]["lr"] = parameterization.get(
            "learning_rate", hparams["optimizer"]["lr"]
        )
        hparams_local["optimizer"]["lr_scheduler"]["lr_min"] = (
            hparams_local["optimizer"]["lr"] / 10
        )  # we want min_lr to be 1/10th of max lr
        hparams_local["encoder"]["aggregation"] = parameterization.get(
            "aggregation", hparams["encoder"]["aggregation"]
        )
        metrics = cross_validate_predefined(
            hparams=hparams_local,
            data=data,
            split_files=split_files,
            save_models=False,
            return_fold_metrics=False,
        )
        # note that metrics returned from cross_validate_predefined are tensors, which ax cannot handle
        # thus we convert to float
        return {k: v.item() for k, v in metrics.items()}

    def objective_function_sklearn(parameterization):
        """
        Wrapper around cross_validate_sklearn() for Bayesian optimization with Ax.

        Ax requires that this function takes only one argument, parameterization.
        Consequently, variables `hparams` and `data` need to be defined in outside the function.

        Args:
            parameterization (dict): Parameters to be probed

        Returns:
            tuple: Score and STD for the probed parameters
        """
        hparams_local = deepcopy(hparams)
        if hparams_local["decoder"]["type"] == "LogisticRegression":
            hparams_local["decoder"]["LogisticRegression"]["C"] = parameterization["C"]
        elif hparams_local["decoder"]["type"] == "XGB":
            hparams_local["decoder"]["XGB"]["reg_lambda"] = parameterization[
                "reg_lambda"
            ]
            hparams_local["decoder"]["XGB"]["reg_alpha"] = parameterization["reg_alpha"]
            hparams_local["decoder"]["XGB"]["gamma"] = parameterization["gamma"]
            hparams_local["decoder"]["XGB"]["learning_rate"] = parameterization[
                "learning_rate"
            ]
        else:
            raise ValueError("Unknown decoder type")

        metrics = cross_validate_sklearn(
            hparams=hparams_local,
            data=data,
            split_files=split_files,
            save_models=False,
            return_fold_metrics=False,
        )
        # note that metrics returned from cross_validate_predefined are tensors, which ax cannot handle
        # thus we convert to float
        return metrics

    bounds = get_hparam_bounds(hparam_config_path)

    if hparams["decoder"]["type"] == "FFN":
        obj_func = objective_function
    else:
        obj_func = objective_function_sklearn

    best_parameters, values, experiment, model = optimize(
        parameters=bounds,
        evaluation_function=obj_func,
        objective_name="val/loss_mean",
        total_trials=n_iter,
        minimize=True,
    )
    return best_parameters, values, experiment
