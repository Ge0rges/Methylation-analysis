import numpy as np
import polars as pl
import pandas as pd
import polars.selectors as cs
from collections import OrderedDict
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter
from rpy2.robjects.packages import importr
from src.utilities.utils import readable_modification_name
import multiprocess as mp


def add_rao_score_by_gene(df: pl.DataFrame, samples: list[str], baseline: str | bool = False,
                          p_threshold: float = 0.05) -> pl.DataFrame:
    """
    Get the Rao score for each gene in the dataframe and keep only those that are statistically significant
    :param df: The methylation data with gene_callers_id
    :type df: pl.DataFrame
    :param samples: Samples is a list of sample strings to filter the df by first.
    :type samples: list[str]
    :param baseline: If baseline is false, do a string test, otherwise do a weak test with baseline as the index value of the baseline sample in samples to test agaisnt.
    :type baseline: str | bool
    :param p_threshold: The p-value threshold to use for the test.
    :type p_threshold: float
    :return: The dataframe with the Rao score added as a column.
    :rtype: pl.DataFrame
    """

    assert len(samples) > 1, "Cannot run rao score on 1 sample"

    # Get groups by gene_ids with the relevant samples and methylation data
    collect_samples = samples if isinstance(baseline, bool) else samples + [baseline]
    groups = (df.filter(pl.col("sample").is_in(collect_samples))
              .select("sample", "gene_callers_id", *list(readable_modification_name.keys()))
              .group_by("gene_callers_id"))

    def process_group(group_tuple):
        name, group = group_tuple  # Get the gene_id and the group df
        group = group.filter(pl.all_horizontal(cs.float().is_not_nan()))  # Remove NaNs

        # Only compare groups which have all the samples specified
        if group.get_column("sample").n_unique() == len(collect_samples):
            result = _willis_dmr_test_r(group.drop("gene_callers_id"), strong=(type(baseline) is bool), j=baseline)

            # No result, return None
            if result is None:
                return None

            return (group.get_column("gene_callers_id").item(0), result["test_stat"][0], result["p"][0] < p_threshold)

        # Didn't have all the samples
        return None

    # Dicts to store results
    score_dict = {}
    significance_dict = {}

    # Run the R code (willis test) on genes in parallel
    with mp.get_context("spawn").Pool(20) as p:
        for result in p.map(process_group, groups):
            if result is not None:
                score_dict[result[0]] = result[1]
                significance_dict[result[0]] = result[2]

    # Make the comparison string
    comp_str = "_vs_".join(samples)
    if type(baseline) is not bool:
        samples.remove(baseline)
        comp_str = f"{baseline}_vs_{'_'.join(samples)}"

    # Add the score and comparison to the df
    df_t = df.with_columns(
 pl.col("gene_callers_id").replace_strict(significance_dict, default=np.NAN, return_dtype=pl.Boolean).alias("test_result"),
        pl.col("gene_callers_id").replace_strict(score_dict, default=np.NAN).alias("rao_score"),
        pl.lit(comp_str).alias("comparison"))

    df = df_t.vstack(df) if "rao_score" in df.columns else df_t

    return df


def _willis_dmr_test_r(df: pl.DataFrame, strong: bool = True, j: str | bool = False) -> OrderedDict | None:
    """
    Run the raoBust multinomail test.

    :param df: The dataframe with the methylation data with the columns: name, sample, and methylation types.
    :type df: pl.DataFrame
    :return: The result dictionnarty from R.
    :rtype: OrderedDict
    """
    Y = df.drop("sample").to_numpy()
    X_dummies = pd.get_dummies(df["sample"], dtype=int)
    X = X_dummies.to_numpy()

    # Find the column for j
    if type(j) is str:
        j = X_dummies.columns.get_loc(j)

    # Call R function
    raobust = importr('raoBust')
    numpy2ri.activate()
    np_cv_rules = default_converter + numpy2ri.converter
    with np_cv_rules.context():
        try:
            return OrderedDict(raobust.multinom_test(X, Y, strong=strong, j=j, penalty=False, pseudo_inv=True))
        except:
            try:
                print("willis called multinom_test after error")
                return OrderedDict(raobust.multinom_test(X, Y, strong=strong, j=j, penalty=True, pseudo_inv=True))
            except:
                print(X)
                print(Y)
                print(strong)
                print(j)
                return None
