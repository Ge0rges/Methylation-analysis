import itertools
import tensorly as tl
import tlviz
from tlab.cp_tensor import load_cp_tensor
import numpy as np
import os
import pandas as pd
import seaborn as sns
import warnings
import xarray as xr
from src.Barnacle.barnacle_manager import BarnacleManager
from src.Objects.genome import Genome
from matplotlib import pyplot as plt
from pathlib import Path

from src.Barnacle.barnacle_manager import BarnacleManager

warnings.simplefilter(action='ignore', category=FutureWarning)

# set up data structures to store input data

# parameters
datapath = Path('../../data/models/pelagibacter_r-contigs/small_run/')
outdir = Path('../../data/models/pelagibacter_r-contigs/out/')
bootstraps = range(len(os.listdir(datapath)))
replicates = ['A', 'B', 'C']
params = {
    'rank': 3,
    'lambda': 0.5
}

# cp tensors as produced
cps = {
    rep: [] for rep in replicates
}

# cp tensors aligned to best representative, all samples present
aligned_cps = {
    rep: [] for rep in replicates
}
# import original tensor dataset for reference
barnacle_manager = BarnacleManager(Genome("Pelagibacter_r-contigs"))
df = barnacle_manager._get_genome_barnacle_format_by_position()
og_ds = barnacle_manager._xarray_from_df(df.collect(streaming=True))

# import all fitted models

for rep in replicates:
    for boot in bootstraps:
        rank = params['rank']
        lamb = params['lambda']
        # put together data path
        path_cp = f"model_{boot}/fitted_model_{rep}_r{rank}_l[{lamb}. 0.0, 0.0].h5"

        # store normalized cp tensor to cps
        cp = tl.cp_normalize(load_cp_tensor(datapath / path_cp))
        cps[rep].append(cp)

# find best representative reference cp tensor

results = []
for ref_rep, ref_boot in list(itertools.product(replicates, bootstraps)):
    reference_cp = cps[ref_rep][ref_boot]
    for comp_rep, comp_boot in list(itertools.product(replicates, bootstraps)):
        # no point in comparing to self
        if ref_rep == comp_rep and ref_boot == comp_boot:
            continue
        comparison_cp = cps[comp_rep][comp_boot]
        fms = tlviz.factor_tools.factor_match_score(
            reference_cp,
            comparison_cp,
            consider_weights=False
        )
        results.append({
            'reference_bootstrap': ref_boot,
            'reference_replicate': ref_rep,
            'comparison_bootstrap': comp_boot,
            'comparison_replicate': comp_rep,
            'fms': fms,
        })

fms_df = pd.DataFrame(results)

# summarize overall mean fms
fms_summary_df = fms_df.groupby([
    'reference_bootstrap',
    'reference_replicate'
]).agg(
    mean_fms=('fms', 'mean'),
    median_fms=('fms', 'median'),
    boot_count=('fms', 'count')
).reset_index()

# find the top mean for each comparison rank
best_ref = fms_summary_df.iloc[fms_summary_df.mean_fms.idxmax(), :]

# realign all models against best representative models

# permute reference cp so that components are in descending order of explaned variation
ref_cp = tlviz.factor_tools.permute_cp_tensor(
    cps[best_ref['reference_replicate']][best_ref['reference_bootstrap']],
    consider_weights=False
)

# realign all the other cp tensors against the best representative cp tensor
for rep in replicates:
    for boot in bootstraps:
        # permute components to line up with best representative reference cp
        aligned_cps[rep].append(tlviz.factor_tools.permute_cp_tensor(
            cps[rep][boot],
            reference_cp_tensor=ref_cp,
            consider_weights=False
        ))

# compile aligned model weights into xarray.Datasets

# set up data structures
component_labels = np.arange(params['rank']) + 1 # 1-based indexing for ease of communication
component_weights = []
metabo_weights = []
treatment_weights = []
timepoint_weights = []
sample_df = pd.DataFrame()
for boot in bootstraps:
    component_weights.append([])
    metabo_weights.append([])
    treatment_weights.append([])
    timepoint_weights.append([])
    # fetch shuffled tensor xr.DataSet
    ds = xr.open_dataset(datapath / f'model_{boot}/dataset_bootstrap_{boot}.nc')

    # pull out and save shuffled Sample data
    boot_sample_df = ds.Sample.to_series().reset_index()
    boot_sample_df['Bootstrap'] = boot
    sample_df = pd.concat([sample_df, boot_sample_df])
    for rep in replicates:
        # fetch aligned cp tensor
        cp = aligned_cps[rep][boot]
        # add component weights to list
        component_weights[boot].append(cp.weights)
        # add metabolite weights to list
        metabo_weights[boot].append(cp.factors[0].T)
        # add treatment weights to list
        treatment_weights[boot].append(cp.factors[1].T)
        # add timepoint weights to list
        timepoint_weights[boot].append(cp.factors[2].T)

# compile everything into an xarray.Dataset
ds = xr.Dataset(
    dict(
        ComponentWeights=xr.DataArray(
            np.array(component_weights),
            coords=[bootstraps, replicates, component_labels],
            dims=['Bootstrap', 'Replicate', 'Component']
        ),
        MethylationWeights=xr.DataArray(
            np.array(metabo_weights),
            coords=[bootstraps, replicates, component_labels, ds.methylation_type.data],
            dims=['Bootstrap', 'Replicate', 'Component', 'Methylation Type']
        ),
        TreatmentWeights=xr.DataArray(
            np.array(treatment_weights),
            coords=[bootstraps, replicates, component_labels, ds.Treatment.data],
            dims=['Bootstrap', 'Replicate', 'Component', 'Treatment']
        ),
        SampleWeights=xr.DataArray(
            np.array(timepoint_weights),
            coords=[bootstraps, replicates, component_labels, ds.Sample.data],
            dims=['Bootstrap', 'Replicate', 'Component', 'Sample']
        ),
        Sample=xr.DataArray.from_series(
            sample_df.set_index(['Bootstrap', 'Replicate', 'Treatment', 'Timepoint'])['Sample']
        ),
    )
)

# save Dataset as netCDF4 file
ds.to_netcdf(outdir / 'aligned-models.nc')
