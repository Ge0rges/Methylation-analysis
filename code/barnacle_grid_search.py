#!/usr/bin/env python
# AUTHOR: Stephen Blaskowski
# CREATE DATE: 24 September 2023

# Script to determine optimal hyperparameters (number of components and
# sparsity coefficient) for the sparse tensor decomposition model of
# ProMo metabolite data, using parameter grid search.

from utilities.utils import generate_cross_validation_sets
from concurrent.futures import ProcessPoolExecutor
import datetime
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import ParameterGrid
from tensorly import check_random_state
from barnacle import SparseCP
from tlab.cp_tensor import store_cp_tensor
from tlviz.model_evaluation import relative_sse, core_consistency
from tlviz.factor_tools import factor_match_score, cosine_similarity, degeneracy_score
import xarray as xr


def fit_save_model(model, data, path, rep, fit_params):
    """Helper function that takes an instantiated model and data as input,
    fits the model to the data, and returns the fit model. Optionally, the model
    and its settings can be saved to an input file path.

    Parameters
    ----------
    model : barnacle.SparseCP
        Instantiated and parameterized SparseCP model.
    data : numpy.ndarray
        Input data tensor.
    path : pathlib.Path
        Path directory where output will be saved. If path=None, no data will be saved.
        If a legitimate filepath is provided, the fit model, in addition to parameters
        will be saved at the provided filepath.
    fit_params : dict
        Keyword arguments to be passed to the SparseCP.fit_transform() method.
        Pass empty dictionary if no kwargs are to be passed.

    Returns
    -------
    model : barnacle.SparseCP
        Fit model.
    """
    if path is not None:
        # make path directories if they don't exist yet
        if not path.exists():
            path.mkdir(parents=True)
        # save parameters
        if model._best_cp_index is not None:
            raise AssertionError('The `model` passed has already been fit')
        else:
            with open(path / 'model_parameters.txt', 'w') as f:
                f.write(json.dumps(model.__dict__, indent=4))
    _ = model.fit_transform(data, return_losses=False, **fit_params)
    # save best fit model
    # if path is not None:
    #     print(path / f"fitted_model_{rep}.h5")
    #     store_cp_tensor(model.decomposition_, path / f"fitted_model_{rep}.h5")
    # return model
    return model


# function to count number of nonzero components in a cp tensor
def nonzero_components(cp, return_trimmed_cp=False):
    accumulator = np.ones_like(cp.weights)
    for f in cp.factors:
        accumulator *= f.sum(axis=0)
    if return_trimmed_cp:
        raise NotImplementedError
    else:
        return (accumulator != 0.0).sum()


# Function to fit models to each replicate dataset
def fit_models_to_replicates(replicates_labels, replicates_gen_param, param_grid, boot_id, rns, abundance_cols: list, model_out):
    model_seed = rns.randint(2 ** 32)
    all_models = {}
    fitting_results = {}
    replicate_data = {}

    for i, rep in enumerate(replicates_labels):
        print(f"Fitting replicate {rep} models")
        models = [SparseCP(**params, random_state=model_seed) for params in param_grid]

        # Make Xarray from df
        df = generate_cross_validation_sets(*replicates_gen_param, boot_id).to_pandas()
        tensor = xr.DataArray.from_series(
            df[abundance_cols].reset_index().drop_duplicates().set_index(abundance_cols[:-1])[abundance_cols[-1]]
        )
        tensor = tensor.fillna(0)

        # Assemble job parameters and run jobs
        job_params = (models, [tensor.data]*len(models), [model_out]*len(models), [rep]*len(models), [{'threads': 1, 'verbose': 1}]*len(models))
        executor = ProcessPoolExecutor(max_workers=20)
        fit_models = executor.map(fit_save_model, *job_params)
        executor.shutdown()

        # Save models and data
        all_models[rep] = list(fit_models)
        fitting_results[rep] = []
        replicate_data[rep] = tensor

        # Process fitted models and record metrics
        for model in all_models[rep]:
            metrics = {
                'datetime': datetime.datetime.now(),
                'bootstrap_id': boot_id,
                'replicate': rep,
                'rank': model.rank,
                'lambda': model.lambdas[0],
                'best_init': model._best_cp_index,
                'loss': model.loss_[-1],
                'convergence_iterations': len(model.loss_),
                'sse': relative_sse(model.decomposition_, tensor),
                'degeneracy': degeneracy_score(model.decomposition_),
                'core_consistency': core_consistency(model.decomposition_, tensor),
                'monotonicity': np.all(np.diff(model.loss_) < 0),
                'candidate_monotonicity': [np.all(np.diff(l) < 0) for l in model.candidate_losses_],
                #'candidate_fms': [factor_match_score(model.decomposition_, c, consider_weights=False, allow_smaller_rank=True) for c in model.candidates_],
                'candidate_sse': [relative_sse(c, tensor) for c in model.candidates_]
            }
            fitting_results[rep].append(metrics)


    return all_models, fitting_results, replicate_data


# Perform cross-validation calculations
def cross_validate(boot_id, replicate_labels, all_models, all_tensors, param_grid) -> list:
    cv_results = []
    for params in param_grid:

        # Keep only models that have the right params
        cps = {}
        for key, value_list in all_models.items():
            # Filter the list based on the 'rank' property
            for item in value_list:
                if item.rank == params["rank"] and item.lambdas == params["lambdas"]:
                    cps[key] = item.decomposition_

        for modeled_rep in replicate_labels:
            for comparison_rep in replicate_labels:

                fms_cv, css_cv, scss_cv = calculate_cross_validation_metrics(cps, modeled_rep, comparison_rep)

                cv_results.append({
                    'bootstrap_id': boot_id,
                    'rank': params['rank'],
                    'lambda': params['lambdas'][0],
                    'modeled_replicate': modeled_rep,
                    'comparison_replicate': comparison_rep,
                    'replicate_pair': f'{modeled_rep}, {comparison_rep}',
                    'n_components': nonzero_components(cps[modeled_rep]),
                    'mean_gene_sparsity': (cps[modeled_rep].factors[0] != 0.0).sum(axis=0).mean(),
                    'relative_sse': relative_sse(cps[modeled_rep], all_tensors[comparison_rep].data),
                    'fms_cv': fms_cv,
                    'css_cv_factor0': css_cv,
                    'scss_cv_factor0': scss_cv
                    })


    return cv_results


# Calculate cross-validation metrics
def calculate_cross_validation_metrics(cps, modeled_rep, comparison_rep):
    if modeled_rep < comparison_rep:
        fms_cv = np.nan #factor_match_score(cps[modeled_rep], cps[comparison_rep], consider_weights=False, allow_smaller_rank=True)
        css_cv = cosine_similarity(cps[modeled_rep].factors[0], cps[comparison_rep].factors[0])
        scss_cv = cosine_similarity((cps[modeled_rep].factors[0] != 0), (cps[comparison_rep].factors[0] != 0))
    else:
        fms_cv = css_cv = scss_cv = np.nan
    return fms_cv, css_cv, scss_cv


def barnacle_grid_search(cross_df_gen_params, replicate_labels, abundance_cols, output_dir):
    # Set random state
    rns = check_random_state(9481)

    # Output directory and experiment parameters
    output_dir = Path(output_dir)
    n_bootstraps = 84

    # Define model grid search param
    model_params = {
        'rank': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'lambdas': [[i, 0.0, 0.0] for i in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]],
        # 'nonneg_modes': [[1, 2]],
        'tol': [1e-5],
        'n_iter_max': [2000],
        'n_initializations': [5]
    }

    # Sort by rank to make parallelization more efficient
    param_grid = sorted(list(ParameterGrid(model_params)), key=lambda d: d['rank'])
    
    results = {}

    # begin experiment
    for boot_id in range(n_bootstraps):
        # Make the output directory with parents
        model_out = output_dir / f"models_{boot_id}"
        model_out.mkdir(parents=True, exist_ok=True)

        # Fit models to replicate_labels and cross validate
        models, fitting_results, replicate_data = fit_models_to_replicates(replicate_labels, cross_df_gen_params, param_grid, boot_id, rns, abundance_cols, model_out)
        cv_result = cross_validate(boot_id, replicate_labels, models, replicate_data, param_grid)

        # Save all this to files
        results[boot_id] = (fitting_results, cv_results)
        
    return results
