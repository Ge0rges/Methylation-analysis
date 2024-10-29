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
import multiprocessing
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from pickle import dump


class BarnacleDataManager:
    """Handles data preparation and formatting for Barnacle analysis."""

    def __init__(self, genome: Genome):
        self.genome = genome
        self.position_df = None
        self.gene_df = None


    def get_position_based_data(self, boot_id: int = None) -> xr.DataArray:
        if self.position_df is None:
            methyl_data = self.genome.load_all_methylation_data()

            # Filter samples
            methyl_data = methyl_data.with_columns(pl.col("sample").replace_strict(barcode_replicate_map, return_dtype=pl.String).alias("treatment"))
            methyl_data = methyl_data.filter(pl.col("treatment").is_in(["top", "middle", "bottom"]))

            # Add absolute position
            methyl_data = self.genome.add_genome_relative_position(methyl_data).drop("position").rename({"genome_position": "position"})

            # Pivot the dataframe
            methyl_data = methyl_data.unpivot(index=["position", "treatment", "sample"],
                                              on=list(readable_methylation_name.keys()),
                                              value_name="value",
                                              variable_name="methylation_type")

            # Filter out NaNs
            methyl_data = methyl_data.filter(pl.col("value").is_not_nan())
            self.position_df = methyl_data.select("position", "treatment", "methylation_type", "sample", "value").collect(streaming=True)

        methyl_data = self.position_df
        if boot_id is not None:
            methyl_data = self.generate_cross_validation_sets(methyl_data, "treatment", "sample", boot_id)

        # Mean
        methyl_data = methyl_data.group_by(["position", "treatment", "methylation_type"]).agg(pl.col("value").mean())

        # Convert to xarray
        return self.xarray_from_df(methyl_data)


    def get_gene_based_data(self, boot_id: int = None) -> xr.DataArray:
        if self.gene_df is None:
            gene_collection = GeneCollection(self.genome.gene_ids, self.genome)
            methyl_data = gene_collection.methylation_data

            # Filter samples
            methyl_data = methyl_data.with_columns(
                pl.col("sample").replace_strict(barcode_replicate_map, default=pl.first()).alias("treatment"))
            methyl_data = methyl_data.filter(pl.col("treatment").is_in(["top", "middle", "bottom"]))

            # Pivot the dataframe
            methyl_data = methyl_data.unpivot(index=["position", "treatment", "gene_callers_id", "sample"],
                                              on=list(readable_methylation_name.keys()),
                                              value_name="value",
                                              variable_name="methylation_type")

            # Filter out NaNs
            methyl_data = methyl_data.filter(pl.col("value").is_not_nan())

            self.gene_df = methyl_data.select("position", "treatment", "methylation_type", "sample", "gene_callers_id", "value").collect(streaming=True)

        methyl_data = self.gene_df
        if boot_id is not None:
            methyl_data = self.generate_cross_validation_sets(methyl_data, "treatment", "sample", boot_id)

        # Mean
        methyl_data = methyl_data.group_by(["position", "treatment", "methylation_type", "gene_callers_id"]).agg(pl.col("value").mean())

        # Convert to xarray
        return self.xarray_from_df(methyl_data)


    @staticmethod
    def generate_cross_validation_sets(df: pl.DataFrame, treatmeant_col: str, sample_col: str,
                                       boot_id: int) -> pl.DataFrame:
        # Get all possible combinations of replicate_labels and treatments
        all_permutations = list(product(
            *[df.filter(pl.col(treatmeant_col).eq(group)).get_column(sample_col).unique().to_list() for group in
              df.get_column(treatmeant_col).unique().to_list()]))

        if boot_id >= len(all_permutations):
            print(f"Max bootstraps is {len(all_permutations)}")
            boot_id = randint(0, len(all_permutations) - 1)

        # Get the combination of samples for this bootstrap
        combination = all_permutations[boot_id]
        df = df.filter(pl.col(sample_col).is_in(combination))
        return df


    def xarray_from_df(self, df: pl.DataFrame) -> xr.DataArray:
        df = df.to_pandas()
        assert df.columns[-1] == "value"

        tensor = xr.DataArray.from_series(
            df[df.columns].set_index(list(df.columns[:-1]))[df.columns[-1]]
        ).fillna(0)
        return tensor


class BarnacleModelManager:
    """Handles model fitting and cross-validation for Barnacle analysis."""

    def __init__(self, output_dir: Path, random_state_seed: int = 9481):
        self.rns = check_random_state(random_state_seed)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fit_models_to_replicates(self, replicate_labels, tensor, param_grid, boot_id, model_out, max_cpus):
        model_seed = self.rns.randint(2 ** 32)
        all_models, fitting_results, replicate_data = {}, {}, {}

        for rep in replicate_labels:
            models = [SparseCP(**params, random_state=model_seed) for params in param_grid]
            job_params = (
            models, [tensor.data] * len(models), [model_out] * len(models), [rep] * len(models), [{}] * len(models))

            with ProcessPoolExecutor(max_workers=max_cpus) as executor:
                fit_models = executor.map(self.fit_save_model, *job_params)

            all_models[rep] = list(fit_models)
            fitting_results[rep] = [self._extract_metrics(model, boot_id, rep, tensor) for model in all_models[rep]]
            replicate_data[rep] = tensor

        return all_models, fitting_results, replicate_data


    def fit_save_model(self, model: SparseCP, data: np.ndarray, path: Path, rep: str, fit_params: dict) -> SparseCP:
        if not path.exists():
            path.mkdir(parents=True)

        with open(path / 'model_parameters.txt', 'w') as f:
            f.write(json.dumps(model.__dict__, indent=4))

        model.fit_transform(data, return_losses=False, **fit_params)
        store_cp_tensor(model.decomposition_, path / f"fitted_model_{rep}_r{model.rank}_l{model.lambdas}.h5")
        return model


    def cross_validate_models(self, param_grid, boot_id, replicate_labels, all_models, all_tensors) -> list:
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
                    fms, css, scss = self._calculate_cv_metrics(all_models, modeled_rep, comparison_rep)
                    cv_results.append({
                        'bootstrap_id': boot_id,
                        'rank': params['rank'],
                        'lambda': params['lambdas'][0],
                        'modeled_replicate': modeled_rep,
                        'comparison_replicate': comparison_rep,
                        'n_components': nonzero_components(all_models[modeled_rep]),
                        'mean_gene_sparsity': (all_models[modeled_rep].factors[0] != 0).sum(axis=0).mean(),
                        'relative_sse': relative_sse(all_models[modeled_rep], all_tensors[comparison_rep]),
                        'fms_cv': fms,
                        'css_cv_factor0': css,
                        'scss_cv_factor0': scss
                    })
        return cv_results


    @staticmethod
    def _calculate_cv_metrics(cps, modeled_rep, comparison_rep):
        if modeled_rep < comparison_rep:
            fms = factor_match_score(cps[modeled_rep], cps[comparison_rep], consider_weights=False, allow_smaller_rank=True)
            css = cosine_similarity(cps[modeled_rep].factors[0], cps[comparison_rep].factors[0])
            scss = cosine_similarity((cps[modeled_rep].factors[0] != 0), (cps[comparison_rep].factors[0] != 0))
        else:
            fms = css = scss = np.nan
        return fms, css, scss


    @staticmethod
    def _extract_metrics(model, boot_id, rep, tensor):
        return {
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
                'candidate_fms': [
                    factor_match_score(model.decomposition_, c, consider_weights=False, allow_smaller_rank=True) for
                    c in model.candidates_],
                'candidate_sse': [relative_sse(c, tensor) for c in model.candidates_]
            }


class BarnacleVisualizer:
    """Visualization and reporting for Barnacle analysis results."""

    @staticmethod
    def plot_grid_search(result, rank_detail=None):
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


    @staticmethod
    def report_suggested_fms(result, rank):
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


class BarnacleManager:
    def __init__(self, genome: Genome, output_dir: Path):
        self.genome = genome
        self.data_manager = BarnacleDataManager(genome)
        self.model_manager = BarnacleModelManager(output_dir)
        self.visualizer = BarnacleVisualizer()

    def run(self, dataset: str, lambdas: list[float], ranks: list[int], n_bootstraps: int = 27, max_cpus: int = 30) -> dict:
        replicate_labels = ["A", "B", "C"]
        param_grid = sorted(list(ParameterGrid({
            'rank': ranks,
            'lambdas': [[i, 0.0, 0.0] for i in lambdas],
            'tol': [1e-5],
            'n_iter_max': [2000],
            'n_initializations': [5]
        })), key=lambda d: d['rank'])

        # Initialize ProcessPoolExecutor
        results = {}
        max_child_cpus = 3
        job_params = [range(n_bootstraps), [self.model_manager] * n_bootstraps, [self.data_manager] * n_bootstraps, [dataset] * n_bootstraps, [replicate_labels] * n_bootstraps,
                      [param_grid] * n_bootstraps, [max_child_cpus] * n_bootstraps]

        # Load the data on this thread first
        if dataset == "position":
             _ = self.data_manager.get_position_based_data(0)
        elif dataset == "gene":
             _ = self.data_manager.get_gene_based_data(0)
        else:
            raise ValueError(f"Invalid dataset: {dataset}. Must be 'position' or 'gene'")

        # Run bootstraps in parallel
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_cpus//3, mp_context=ctx) as executor:
            job_results = executor.map(process_bootstrap, *job_params)

        # As each future completes, store its result in the results dictionary
        for boot_id, result in job_results:
            results[boot_id] = result

        return results


    def visualize_results(self, results, rank_detail=None):
        self.visualizer.plot_grid_search(results, rank_detail)
        self.visualizer.report_suggested_fms(results, rank_detail)


# function to count number of nonzero components in a cp tensor
def nonzero_components(cp):
    accumulator = np.ones_like(cp.weights)
    for f in cp.factors:
        accumulator *= f.sum(axis=0)
        return (accumulator != 0.0).sum()


def process_bootstrap(boot_id, model_manager, data_manager, dataset, replicate_labels, param_grid, max_child_cpus):
    model_out = model_manager.output_dir / f"models_{boot_id}"

    if dataset == "position":
        tensor = data_manager.get_position_based_data(boot_id)
    elif dataset == "gene":
        tensor = data_manager.get_gene_based_data(boot_id)
    else:
        raise ValueError(f"Invalid dataset: {dataset}. Must be 'position' or 'gene'")

    print(f"Fitting models for bootstrap {boot_id}...")

    # Call model and store CV
    models, fitting_results, replicate_data = model_manager.fit_models_to_replicates(replicate_labels, tensor, param_grid, boot_id, model_out, max_child_cpus)
    cv_result = model_manager.cross_validate_models(param_grid, boot_id, replicate_labels, models, replicate_data)
    return boot_id, (fitting_results, cv_result)


if __name__ == "__main__":
    # Paramaters
    GENOME_NAME = "Pelagibacter_r-contigs"
    LAMBDAS = [0]  # Adjust lambdas as needed
    RANKS = [1, 2, 3, 4, 5, 6, 7, 8,9 ,10]  # Adjust ranks as needed
    N_BOOTSTRAPS = 10
    OUTPUT_DIR = Path(f'../../data/models/{GENOME_NAME}/')

    # Initialize genome and manager
    genome = Genome(GENOME_NAME)
    bm = BarnacleManager(genome, OUTPUT_DIR)

    # Run Barnacle cross-validation with the specified parameters
    result = bm.run(
        dataset="position",
        lambdas=LAMBDAS,
        ranks=RANKS,
        n_bootstraps=N_BOOTSTRAPS,
        max_cpus=30
    )

    # Save the results to a pickle file
    with open(OUTPUT_DIR / "cross_validation_result.pickle", 'wb') as file:
        dump(result, file)

    print("Cross-validation completed. Results saved.")

    # Load existing results if cross-validation was already completed
    # Uncomment this line if you want to load a saved result instead of re-running
    # with open(OUTPUT_DIR / "cross_validation_result.pickle", 'rb') as file:
    #     result = load(file)

    # Generate the visualization
    rank_detail = RANKS[0] if len(RANKS) == 0 else None
    bm.visualizer.plot_grid_search(result, rank_detail)
    bm.visualizer.report_suggested_fms(result, rank_detail)
    plt.show()
