#!/usr/bin/env python
# AUTHOR: Stephen Blaskowski
# CREATE DATE: 24 September 2023

# Script to determine optimal hyperparameters (number of components and
# sparsity coefficient) for the sparse tensor decomposition model of
# ProMo metabolite data, using parameter grid search.

from concurrent.futures import ProcessPoolExecutor
import datetime
import json
import numpy as np
from pathlib import Path
import pandas as pd
import scipy
from sklearn.model_selection import ParameterGrid
import tensorly as tl
from tensorly import check_random_state
from tensorly.cp_tensor import CPTensor
from barnacle import (
    SparseCP,
    simulated_sparse_tensor,
    visualize_3d_tensor,
    plot_factors_heatmap,
    recovery_relevance,
    pairs_precision_recall
)
from barnacle.tensors import SparseCPTensor
from tlab.cp_tensor import store_cp_tensor, load_cp_tensor
from tlviz.visualisation import optimisation_diagnostic_plots
from tlviz.model_evaluation import relative_sse, core_consistency
from tlviz.factor_tools import factor_match_score, cosine_similarity, degeneracy_score
from tlviz.multimodel_evaluation import similarity_evaluation
import xarray as xr


# function to return random replicate labelings
def generate_replicate_labels(sample_names, random_state=None, replicate_map=None):
    """Generates random replicate labels to align with an input vector of sample names.

    Parameters
    ----------
    sample_names : np.ndarray
        Input array of sample names. Must be sorted in ascending order.
    random_state : {None, int, numpy.random.RandomState}
        Default is None.
    replicate_map : {None, dict}
        Map of integer labels to preferred replicate labels.
        Example:
            {0:'A', 1:'B', 2:'C'}
        Default is None.

    Returns
    -------
    replicate_labels : np.ndarray
        Array of randomly generated replicate lables, to be aligned with input `sample_names`.

    """

    # check that input is a numpy array, and sample_names are sorted
    assert type(sample_names) is np.ndarray, "`sample_names` must be a numpy.ndarray"
    assert np.all(sample_names[:-1] <= sample_names[1:]), "`sample_names` must be sorted in ascending order"

    # Get counts of each sample name
    names, counts = np.unique(sample_names, return_counts=True)
    rns = check_random_state(random_state)

    # Generate replicate labels
    replicate_labels = [rns.choice(np.arange(counts.max()), size=c, replace=False) for c in counts]
    replicate_labels = np.concatenate(replicate_labels)

    # Map preferred replicate labels
    if replicate_map is not None:
        mapped_replicate_labels = [replicate_map[i] for i in replicate_labels]
        replicate_labels = np.array(mapped_replicate_labels)

    return replicate_labels


# function to separate out subtensors of each replicate
def separate_replicates(dataset, coordinates, data_variable, replicate_label='replicate'):
    '''Separates data from each replicate set into its own independent DataArray.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset with replicates
    coordinates : list of str
        Coordinates to be preserved in each replicate set.
    data_variable : str
        Name of data variable to be selected in each replicate set.
    replicate_label : str, default is 'replicate'
        Label of replicate field in `dataset`.

    Returns
    -------
    replicate_sets : dict of xarray.DataArrays
        Set of replicate DataArrays, each keyed on its replicate label.

    '''
    # get list of replicate labels
    replicates = np.unique(dataset[replicate_label].to_numpy())
    # pull out each replicate subset
    subsets = list()
    for rep in replicates:
        # pull out only data with of the particular replicate
        df = dataset.where(dataset[replicate_label] == rep, drop=True).to_dataframe().reset_index()
        # reform DataArray with specified coordinates and data_variable
        rep_da = xr.DataArray.from_series(df.set_index(coordinates)[data_variable])
        # add to dict of replicate subsets
        subsets.append(rep_da)
    # arrange in dictionary and return
    return dict(zip(replicates, subsets))


# function to select common indices between two datasets
def select_common_indices(dataset_1, dataset_2, coordinates):
    '''Finds common indices between two datasets.

    Parameters
    ----------
    dataset_1 : xarray.Dataset
        Dataset with common coordinates to be compared.
    dataset_2 : xarray.Dataset
        Dataset with common coordinates to be compared.
    coordinates : list of str
        Coordinates to be compared between `dataset_1` and `dataset_2`.

    Returns
    -------
    common_index_labels : list of numpy.Arrays
        Common indices' labels, one per coordinate provided.
    indices_1 : list of numpy.Arrays of ints
        Common integer indices, one per coordinate provided.
    indices_2 : list of numpy.Arrays of ints
        Common integer indices, one per coordinate provided.
    '''
    # initialize outputs
    if len(coordinates) > 1:
        common_index_labels = {}
        indices_1 = {}
        indices_2 = {}
    # loop through coordinates
    for coord in coordinates:
        # get shared coordinate labels
        shared_labels = np.intersect1d(
            dataset_1.indexes[coord],
            dataset_2.indexes[coord],
            assume_unique=True,
            return_indices=False
        )
        # get dataset 1 index
        _, index_1, _ = np.intersect1d(
            dataset_1.indexes[coord],
            shared_labels,
            assume_unique=True,
            return_indices=True
        )
        # get dataset 2 index
        _, index_2, _ = np.intersect1d(
            dataset_2.indexes[coord],
            shared_labels,
            assume_unique=True,
            return_indices=True
        )
        # store labels and indices
        if len(coordinates) > 1:
            common_index_labels[coord] = shared_labels
            indices_1[coord] = index_1
            indices_2[coord] = index_2

    # return results
    if len(coordinates) > 1:
        return common_index_labels, indices_1, indices_2
    else:
        return shared_labels, index_1, index_2


def fit_save_model(model, data, path, fit_params):
    '''Helper function that takes an instantiated model and data as input,
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
    '''
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
        store_cp_tensor(model.decomposition_, path / 'fitted_model.h5')
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


def load_existing_data(filepath, columns=None):
    if filepath.is_file():
        return pd.read_csv(filepath).to_dict('records')
    return pd.DataFrame(columns=columns if columns else []).to_dict('records')


def shuffle_dataset(output_dir, boot_id, dataset, shuffle_seed, replicate_col, group_cols):
    filepath_shuffle_ds = output_dir / f"dataset_bootstrap_{boot_id}.nc"

    # Import shuffled dataset if it exists
    if filepath_shuffle_ds.is_file():
        print(f"Importing DataSet discovered at:\n\t{filepath_shuffle_ds}", flush=True)
        return xr.open_dataset(filepath_shuffle_ds)

    # Make and save shuffled dataset if it doesn't exist
    # NOTE: This shuffling is done in a way so that all metabolite measurements
    # from the same replicate remain together. If instead it makes sense to
    # shuffle each set of metabolite replicates independently, we can do that.
    print('Shuffling DataSet replicate labels', flush=True)

    # pull out NormAbundance DataArray as Series
    abun_df = dataset.Abundance.to_series().reset_index().rename(columns={replicate_col: 'OldReplicate'})
    sample_df = dataset.Sample.to_series().reset_index().rename(columns={replicate_col: 'OldReplicate'})

    # make new "SampleName" field in Series with combined Treatment + control
    sample_df['GroupName'] = sample_df[group_cols].agg('_'.join, axis=1)
    sample_df = sample_df.sort_values('GroupName')

    # Automatically generate the replicate map: {0: 'A', 1: 'B', ...}
    unique_replicates = sorted(sample_df["OldReplicate"].unique())

    # use generate_replicate_labels() to get new replicate labels
    new_labels = generate_replicate_labels(
        sample_names=sample_df['GroupName'].to_numpy(),
        random_state=shuffle_seed,
        replicate_map=dict(enumerate(unique_replicates))
    )

    sample_df[replicate_col] = new_labels

    # map new replicate labels onto abundance data
    shuffle_abun_df = pd.merge(left=abun_df, right=sample_df, how='left', on=group_cols + ['OldReplicate'])

    # Rebuild the dataset with shuffled replicate labels
    restored_data_vars = {}

    for var_name, data_array in dataset.data_vars.items():
        original_dims = list(data_array.dims)
        restored_data_vars[var_name] = xr.DataArray().from_series(
            shuffle_abun_df[original_dims + [var_name]].drop_duplicates().set_index(original_dims)[var_name])

    # 4. Create the new dataset with restored DataArrays
    shuffle_ds = xr.Dataset(restored_data_vars, coords=dataset.coords)

    # Save random seed used for shuffling as dataset attribute
    shuffle_ds.attrs['shuffle_seed'] = shuffle_seed

    # Save replicate shuffled dataset to netCDF4 file
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)
    shuffle_ds.to_netcdf(filepath_shuffle_ds)

    return shuffle_ds


def start_grid_search(dataset, replicate_col, group_cols):
    # set random state
    seed = 9481
    rns = check_random_state(seed)

    print('\nBeginning parameter grid search\n', flush=True)

    # output directory and experiment parameters
    base_dir = Path('../data/models/')
    n_bootstraps = 100
    replicates = ['A', 'B', 'C']

    # define model grid search param
    model_params = {
        'rank': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'lambdas': [[i, 0.0, 0.0] for i in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]],
        # 'nonneg_modes': [[1, 2]],
        'tol': [1e-5],
        'n_iter_max': [2000],
        'n_initializations': [5]
    }

    # sort by rank to make parallelization more efficient
    param_grid = sorted(list(ParameterGrid(model_params)), key=lambda d: d['rank'])

    # set up output data records and locations
    filepath_fit_data = base_dir / 'fitting_data.csv'
    filepath_cv_data = base_dir / 'cv_data.csv'
    fitting_results = load_existing_data(filepath_fit_data)
    cv_results = load_existing_data(filepath_cv_data, columns=[
        'bootstrap_id', 'rank', 'lambda', 'modeled_replicate', 'comparison_replicate', 'replicate_pair',
        'n_components', 'mean_gene_sparsity', 'relative_sse', 'fms_cv', 'css_cv_factor0', 'scss_cv_factor0'
    ])


    # begin experiment
    for boot_id in range(n_bootstraps):
        shuffle_seed = rns.randint(2 ** 32)
        print(f"\nBootstrap: {boot_id} (seed={shuffle_seed})", flush=True)

        # directory and file paths
        output_dir = base_dir / f"bootstrap{boot_id}"

        shuffle_ds = shuffle_dataset(output_dir, boot_id, dataset, shuffle_seed, replicate_col, group_cols)

        # set up replicate subtensor data
        filepaths_reps = {}
        for rep in replicates:
            path = output_dir / 'replicate{}'.format(rep)
            if not path.is_dir():
                path.mkdir(parents=True)
            # collect all filepaths
            filepaths_reps[rep] = path / 'shuffled_replicate_{}.nc'.format(rep, rep)
        # check if all replicate dataarrays exist or not
        all_reps_saved = np.all([filepaths_reps[f].is_file() for f in replicates])
        # import replicate subtensors if the saved files exist
        if all_reps_saved:
            print('Importing replicate DataArrays discovered at:\n{}'.format(
                json.dumps({i: str(k) for i, k in filepaths_reps.items()}, indent=4)
            ), flush=True)
            replicate_sets = {}
            for rep in replicates:
                replicate_sets[rep] = xr.open_dataarray(filepaths_reps[rep])
        # otherwise separate out replicate sets from shuffled tensor
        else:
            print('Separating shuffled replicate DataArrays', flush=True)
            replicate_sets = separate_replicates(
                shuffle_ds,
                ['methylation_type', 'treatment', 'position'],
                'Abundance',
                replicate_label='replicate'
            )
            for rep in replicates:
                # save shuffled replicate data
                replicate_sets[rep].to_netcdf(filepaths_reps[rep])

        # fit grid search models to each replicate dataset
        for rep in replicates:
            # pull out shuffled replicate data
            tensor = replicate_sets[rep]

            # instantiate models and define output filepaths
            models = []
            dirpaths_models = []
            for params in param_grid:
                model_seed = rns.randint(2 ** 32)
                model_dir = output_dir / 'replicate{}/rank{}/lambda{}'.format(
                    rep, params['rank'], params['lambdas'][0]
                )
                filepath_fitted = model_dir / 'fitted_model.h5'
                # don't re-fit any models that have already been fitted
                if not filepath_fitted.is_file():
                    # instantiate parameterized model
                    models.append(SparseCP(**params, random_state=model_seed))
                    dirpaths_models.append(model_dir)

            # continue to next set of replicates if all models have already been fit
            if len(models) == 0:
                print(
                    """
                    Pre-existing models discovered, skipping fitting to following dataset:
                        bootstrap = {}
                        replicate = {}
                    """.format(boot_id, rep),
                    flush=True
                )
                continue
            else:
                print('\nFitting replicate {} models\n'.format(rep), flush=True)

            # assemble job parameters
            job_params = (
                models,
                [tensor.data for m in models],
                dirpaths_models,
                [{'threads': 1, 'verbose': 0} for m in models]
            )

            # run jobs
            executor = ProcessPoolExecutor(max_workers=20)
            fit_models = executor.map(fit_save_model, *job_params)

            # iterate through models in order
            for model in fit_models:
                # calculate metrics
                rank = model.rank
                lamb = model.lambdas[0]
                best_init = model._best_cp_index
                loss = model.loss_[-1]
                cvg_iter = len(model.loss_)
                sse = relative_sse(model.decomposition_, tensor)
                degeneracy = degeneracy_score(model.decomposition_)
                cc = core_consistency(model.decomposition_, tensor)
                monotonicity = np.all(np.diff(model.loss_) < 0)
                can_mon = [np.all(np.diff(l) < 0) for l in model.candidate_losses_]
                can_fms = [factor_match_score(model.decomposition_, c, consider_weights=False, allow_smaller_rank=True)
                           for c in model.candidates_]
                can_sse = [relative_sse(c, tensor) for c in model.candidates_]

                # record metrics
                fitting_results.append(
                    {
                        'datetime': datetime.datetime.now(),
                        'bootstrap_id': boot_id,
                        'replicate': rep,
                        'rank': rank,
                        'lambda': lamb,
                        'best_init': best_init,
                        'loss': loss,
                        'convergence_iterations': cvg_iter,
                        'sse': sse,
                        'degeneracy': degeneracy,
                        'core_consistency': cc,
                        'monotonicity': monotonicity,
                        'candidate_monotonicity': can_mon,
                        'candidate_fms': can_fms,
                        'candidate_sse': can_sse
                    }
                )

                # print some metrics
                print('rank:{}, lambda:{}, sse:{:.5}'.format(
                    rank,
                    lamb,
                    sse,
                ), flush=True)

                # save data
                fitting_df = pd.DataFrame(fitting_results)
                fitting_df.to_csv(filepath_fit_data, index=False)

            # shut down executor
            executor.shutdown()

    # collect cross validation results
    print('\nBeginning cross validataion calculations\n', flush=True)
    for boot_id in range(n_bootstraps):
        # set path of bootstrap data
        boot_path = base_dir / 'bootstrap{}'.format(boot_id)
        # read in shuffled replicate data
        rep_data = {}
        for rep in replicates:
            rep_path = boot_path / 'replicate{}'.format(rep)
            rep_data[rep] = xr.open_dataarray(rep_path / 'shuffled_replicate_{}.nc'.format(rep))
        # iterate through all parameter combos
        for params in param_grid:
            # get all the models
            cps = {}
            expt_path = 'rank{}/lambda{}'.format(params['rank'], params['lambdas'][0])
            for rep in replicates:
                cp_path = boot_path / 'replicate{}'.format(rep) / expt_path
                cps[rep] = load_cp_tensor(cp_path / 'fitted_model.h5')
            for modeled_rep in replicates:
                for comparison_rep in replicates:
                    print_string = 'bootstrap: {}, rank: {}, lambda: {}, modeled: {}, comparison: {}'.format(
                        boot_id,
                        params['rank'],
                        params['lambdas'][0],
                        modeled_rep,
                        comparison_rep
                    )
                    # check if comparison has already been calculated
                    record = cv_df.loc[(
                            (cv_df['bootstrap_id'] == boot_id) &
                            (cv_df['rank'] == params['rank']) &
                            (cv_df['lambda'] == params['lambdas'][0]) &
                            (cv_df['modeled_replicate'] == modeled_rep) &
                            (cv_df['comparison_replicate'] == comparison_rep)
                    )]
                    if len(record) >= 1:
                        print(
                            'Pre-existing record found, ' +
                            'skipping following comparison:\n\t{}'.format(print_string),
                            flush=True
                        )
                        continue
                    else:
                        print(print_string, flush=True)

                    # pull out data to be compared against
                    comparison_data = rep_data[comparison_rep]

                    # calculate fms & cosine similiary scores against other fit models
                    if modeled_rep < comparison_rep:
                        fms_cv = factor_match_score(
                            cps[modeled_rep],
                            cps[comparison_rep],
                            consider_weights=False,
                            allow_smaller_rank=True
                        )
                        css_cv = cosine_similarity(
                            cps[modeled_rep].factors[0],
                            cps[comparison_rep].factors[0]
                        )
                        scss_cv = cosine_similarity(
                            (cps[modeled_rep].factors[0] != 0),
                            (cps[comparison_rep].factors[0] != 0)
                        )
                    else:
                        # skip redundant and self comparisons
                        fms_cv = css_cv = scss_cv = np.nan
                    # calculate relative sse
                    rel_sse = relative_sse(cps[modeled_rep], comparison_data.data)
                    # keep results
                    cv_results.append(
                        {
                            'bootstrap_id': boot_id,
                            'rank': params['rank'],
                            'lambda': params['lambdas'][0],
                            'modeled_replicate': modeled_rep,
                            'comparison_replicate': comparison_rep,
                            'replicate_pair': '{}, {}'.format(modeled_rep, comparison_rep),
                            'n_components': nonzero_components(cps[modeled_rep]),
                            'mean_gene_sparsity':
                                (cps[modeled_rep].factors[0] != 0.0).sum(axis=0).mean(),
                            'relative_sse': rel_sse,
                            'fms_cv': fms_cv,
                            'css_cv_factor0': css_cv,
                            'scss_cv_factor0': scss_cv,
                        }
                    )
                    # store results in dataframe
                    cv_df = pd.DataFrame(cv_results)
        # save data
        cv_df.to_csv(filepath_cv_data, index=False)
