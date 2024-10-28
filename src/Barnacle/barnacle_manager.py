import polars as pl
from itertools import product
from random import randint
import xarray as xr
import json
from sklearn.model_selection import ParameterGrid
from src.Objects import GeneCollection, Genome
from src.utilities.utils import barcode_replicate_map, readable_methylation_name
from tensorly import check_random_state
from barnacle import SparseCP
from tlab.cp_tensor import store_cp_tensor
from tlviz.model_evaluation import relative_sse, core_consistency
from tlviz.factor_tools import factor_match_score, cosine_similarity, degeneracy_score
from pathlib import Path
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl


class BarnacleManager(object):

    def __init__(self, genome: Genome):
        self.genome: Genome = genome
        self.rns = check_random_state(9481)


    def run_barnacle(self, df: pl.LazyFrame, lambdas: list[float], ranks: list[int], n_bootstraps: int = 27, max_cpus: int = 30) -> dict:
        # Call barnacle grid search on it
        cross_df_gen_params = [df, ["position", "gene_callers_id"], "treatment", "sample"] if "gene_callers_id" in df.collect_schema().names() else [df, ["position"], "treatment", "sample"]

        replicate_labels = ["A", "B", "C"]
        output_dir = Path(f'../data/models/{self.genome.name}/')

        # Define model grid search param
        model_params = {
            'rank': ranks,
            'lambdas': [[i, 0.0, 0.0] for i in lambdas],
            'tol': [1e-5],
            'n_iter_max': [2000],
            'n_initializations': [5]
        }

        # Sort by rank to make parallelization more efficient
        param_grid = sorted(list(ParameterGrid(model_params)), key=lambda d: d['rank'])

        results = {}

        # begin experiment
        for boot_id in range(n_bootstraps):
            print(f"Running bootstrap {boot_id}/{n_bootstraps}")
            # Make the output directory with parents
            model_out = output_dir / f"models_{boot_id}"
            model_out.mkdir(parents=True, exist_ok=False)

            # Fit models to replicate_labels and cross validate
            models, fitting_results, replicate_data = fit_models_to_replicates(replicate_labels, cross_df_gen_params,
                                                                               param_grid, boot_id, self.rns, model_out,
                                                                               max_cpus)
            cv_result = cross_validate(boot_id, replicate_labels, models, replicate_data, param_grid)

            # Save all this to files
            results[boot_id] = (fitting_results, cv_result)

        return results


    def visualize_barnacle_grid_search(self, results, rank_detail = None):
        plot_cross_validation_result(results, rank_detail)


    def get_genome_barnacle_format_by_position(self) -> pl.LazyFrame:
        methyl_data = self.genome.load_all_methylation_data()

        # Filter samples
        methyl_data = methyl_data.with_columns(pl.col("sample").replace_strict(barcode_replicate_map, default=pl.first()).alias("treatment"))
        methyl_data = methyl_data.filter(pl.col("treatment").is_in(["top", "middle", "bottom"]))

        # Add absolute position
        methyl_data = self.genome.add_genome_relative_position(methyl_data).drop("position").rename({"genome_position": "position"})

        # Pivot the dataframe
        methyl_data = methyl_data.unpivot(index=["position", "treatment"],
                                          on=list(readable_methylation_name.keys()),
                                          value_name="value",
                                          variable_name="methylation_type")

        # Filter out NaNs
        methyl_data = methyl_data.filter(pl.col("value").is_not_nan())

        return methyl_data.select("position", "treatment", "methylation_type", "value")


    def get_genome_barnacle_format_by_gene(self) -> pl.LazyFrame:
        gene_collection = GeneCollection(self.genome.gene_ids, self.genome)
        methyl_data = gene_collection.methylation_data

        # Filter samples
        methyl_data = methyl_data.with_columns(
            pl.col("sample").replace_strict(barcode_replicate_map, default=pl.first()).alias("treatment"))
        methyl_data = methyl_data.filter(pl.col("treatment").is_in(["top", "middle", "bottom"]))

        # Pivot the dataframe
        methyl_data = methyl_data.unpivot(index=["position", "treatment", "gene_callers_id"],
                                          on=list(readable_methylation_name.keys()),
                                          value_name="value",
                                          variable_name="methylation_type")

        # Filter out NaNs
        methyl_data = methyl_data.filter(pl.col("value").is_not_nan())

        return methyl_data.select("position", "treatment", "methylation_type", "gene_callers_id", "value")


def generate_cross_validation_sets(df: pl.DataFrame, unique_cols: list[str], treatmeant_col: str, sample_col: str,
                                    boot_id: int) -> pl.DataFrame:
    # Get all possible combinations of replicate_labels and treatments
    all_permutations = list(product(
        *[df.filter(pl.col(treatmeant_col).eq(group)).get_column(sample_col).unique().to_list() for group in
          df.get_column(treatmeant_col).unique().to_list()]))

    if boot_id >= len(all_permutations):
        print(f"Max bootstraps is {len(all_permutations)}")
        boot_id = randint(0, len(all_permutations) - 1)

    # Keep only names (positions) that are in all samples
    labels_in_all_groups = df.group_by(*unique_cols).agg(pl.col(sample_col).n_unique().alias("unique_groups")).filter(
        pl.col("unique_groups") == df.get_column(sample_col).n_unique())

    df = df.filter(pl.col(unique_col).is_in(labels_in_all_groups.get_column(unique_col).to_list()) for unique_col in unique_cols)

    # Get the combination of samples for this bootstrap
    combination = all_permutations[boot_id]
    df = df.filter(pl.col(sample_col).is_in(combination))
    return df


def xarray_from_df(df: pl.DataFrame):
    cols = df.collect_schema().names()
    index_cols = cols[:-1]
    value_col = "value"
    assert cols[-1] == value_col

    df = df.to_pandas()
    tensor = xr.DataArray.from_series(
        df[index_cols].reset_index().drop_duplicates().set_index(index_cols)[value_col]
    )
    tensor = tensor.fillna(0)

    return tensor


# function to count number of nonzero components in a cp tensor
def nonzero_components(cp):
    accumulator = np.ones_like(cp.weights)
    for f in cp.factors:
        accumulator *= f.sum(axis=0)
        return (accumulator != 0.0).sum()


# Calculate cross-validation metrics
def calculate_cross_validation_metrics(cps, modeled_rep, comparison_rep):
    if modeled_rep < comparison_rep:
        fms_cv = factor_match_score(cps[modeled_rep], cps[comparison_rep], consider_weights=False, allow_smaller_rank=True)
        css_cv = cosine_similarity(cps[modeled_rep].factors[0], cps[comparison_rep].factors[0])
        scss_cv = cosine_similarity((cps[modeled_rep].factors[0] != 0), (cps[comparison_rep].factors[0] != 0))
    else:
        fms_cv = css_cv = scss_cv = np.nan
    return fms_cv, css_cv, scss_cv


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
    if path is not None:
        store_cp_tensor(model.decomposition_, path / f"fitted_model_{rep}_r{model.rank}_l{model.lambdas}.h5")
    return model


# Function to fit models to each replicate dataset
def fit_models_to_replicates(replicates_labels, replicates_gen_param, param_grid, boot_id, rns, model_out, max_cpus):
    model_seed = rns.randint(2 ** 32)
    all_models = {}
    fitting_results = {}
    replicate_data = {}

    # Make Xarray from df
    df = generate_cross_validation_sets(*replicates_gen_param, boot_id).to_pandas()
    tensor = xarray_from_df(df)
    tensor.to_netcdf(model_out / f"dataset_bootstrap_{boot_id}.nc")


    for i, rep in enumerate(replicates_labels):
        print(f"Fitting replicate {rep} models")
        models = [SparseCP(**params, random_state=model_seed) for params in param_grid]

        # Assemble job parameters and run jobs
        job_params = (models, [tensor.data]*len(models), [model_out]*len(models), [rep]*len(models), [{'threads': 1, 'verbose': 3}]*len(models))
        executor = ProcessPoolExecutor(max_workers=max_cpus)
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
                'core_consistency': core_consistency(model.decomposition_, tensor.data),  # Gio added the .data here
                'monotonicity': np.all(np.diff(model.loss_) < 0),
                'candidate_monotonicity': [np.all(np.diff(l) < 0) for l in model.candidate_losses_],
                'candidate_fms': [factor_match_score(model.decomposition_, c, consider_weights=False, allow_smaller_rank=True) for c in model.candidates_],
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


def plot_cross_validation_result(result, rank_detail = 3):
    cv_results = []
    for vals in result.values():
        cv_results.extend(vals[1])

    results_df = pd.DataFrame(cv_results)

    # add features
    results_df['comparison'] = (results_df['modeled_replicate'] == results_df['comparison_replicate']).map({
        True: 'fitting', False: 'cross-validation'
    })
    results_df = results_df.sort_values(
        ['bootstrap_id', 'rank', 'lambda', 'replicate_pair']).reset_index(drop=True)
    results_df['sparsity coefficient'] = results_df['lambda'].astype(str)

    plot_df = results_df[results_df['comparison'] == 'cross-validation']

    # plot figure
    fig, axis = plt.subplots(figsize=(10, 4), layout="constrained")

    sns.lineplot(
        x='rank',
        y='relative_sse',
        hue='sparsity coefficient',
        #         style='comparison',
        errorbar='se',
        err_style='bars',
        data=plot_df,
        ax=axis,
        alpha=0.5
        #     label=lamb,
    )

    plt.title('model fit vs. parameterization')
    plt.xlabel('R')
    plt.ylabel('CV SSE')
    plt.legend(title='λ', loc='center left', bbox_to_anchor=[1, 0.5])
    plt.show()

    # define data
    if rank_detail:
        plot_df = results_df[results_df['comparison'] == 'cross-validation']
        plot_df = plot_df[plot_df['rank'] == rank_detail]

        # plot SSE
        mpl.rcParams['axes.spines.left'] = True
        mpl.rcParams['axes.spines.right'] = False
        color = sns.color_palette()[0]
        fig, axis = plt.subplots(figsize=(6, 6), layout="constrained")
        sns.lineplot(
            x='lambda',
            y='relative_sse',
            color=color,
            #     style='rank',
            errorbar='se',
            err_style='bars',
            data=plot_df,
            ax=axis,
            legend=False
        )
        axis.set(ylim=[-0.05, 1.02], xlabel='λ', ylabel='CV SSE')
        axis.yaxis.label.set_color(color)
        axis.tick_params(axis='y', colors=color)
        axis.spines['left'].set_color(color)

        # plot FMS
        color = sns.color_palette()[1]
        mpl.rcParams['axes.spines.right'] = True
        mpl.rcParams['axes.spines.left'] = False
        axis2 = plt.twinx(axis)
        sns.lineplot(
            x='lambda',
            y='fms_cv',
            color=color,
            #     style='rank',
            errorbar='se',
            err_style='bars',
            data=plot_df,
            ax=axis2,
            legend=False
        )
        axis2.set(ylim=[-.05, 1.05], xlabel='λ', ylabel='CV FMS')
        axis2.yaxis.label.set_color(color)
        axis2.tick_params(axis='y', colors=color)
        axis2.spines['right'].set_color(color)

        # put defaults back where you found them
        mpl.rcParams['axes.spines.right'] = True
        mpl.rcParams['axes.spines.left'] = True

        # fix x axis stuff
        plt.xscale('log')
        plt.title('model scores vs. sparsity (R={})'.format(rank_detail))
        plt.show()


def print_suggested_FMS(result, rank):
    cv_results = []
    for vals in result.values():
        cv_results.extend(vals[1])

    results_df = pd.DataFrame(cv_results)

    # add features
    results_df['comparison'] = (results_df['modeled_replicate'] == results_df['comparison_replicate']).map({
        True: 'fitting', False: 'cross-validation'
    })
    results_df = results_df.sort_values(
        ['bootstrap_id', 'rank', 'lambda', 'replicate_pair']).reset_index(drop=True)
    results_df['sparsity coefficient'] = results_df['lambda'].astype(str)

    # Max FMS
    summary_df = results_df[results_df['comparison'] == 'cross-validation']
    summary_df = summary_df[summary_df['rank'] == rank]
    summary_df = summary_df.groupby(['rank', 'lambda'])[['mean_gene_sparsity', 'relative_sse', 'fms_cv']].agg(
        mean_gene_sparsity=('mean_gene_sparsity', 'mean'),
        relative_sse=('relative_sse', 'mean'),
        fms_cv=('fms_cv', 'mean'),
        fms_sem=('fms_cv', 'sem'),
        bootstraps=('fms_cv', 'count')
    ).reset_index()

    best_FMS = summary_df.loc[summary_df.fms_cv.idxmax(), :]
    print('max CV FMS: \n\n{}\n'.format(best_FMS))
    se_fms = best_FMS['fms_cv'] - best_FMS['fms_sem']
    print('max CV FMS - 1SE: {}\n'.format(se_fms))

    # show all models with at least the minimum FMS, sorted from sparsest to least sparse
    print(summary_df[summary_df.fms_cv.ge(se_fms)].sort_values('lambda', ascending=False))
