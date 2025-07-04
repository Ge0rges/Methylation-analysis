import itertools
import textwrap
import random
import polars as pl
import numpy as np
import math
import csv

from scipy import stats
from statsmodels.stats.multitest import multipletests


readable_modification_name = {"21839": "4mC", "m": "5mC", "a": "6mA", "Ncanonical_A": "A", "Ncanonical_C": "C"}
readable_methylation_name = {"21839": "4mC", "m": "5mC", "a": "6mA"}
methylation_base_map = {"21839": "C", "m": "C", "a": "A"}
base_methylation_map = {"C": ["21839", "m"], "A": ["a"]}

def truncate_label(label, max_length, max_lines):
    """Truncate labels to a maximum length and line count, adding an ellipsis if truncated."""

    # Hide extra alternatives
    i = 0
    result = label.split("!!!")[i]
    while i + 1 < len(label.split("!!!")) and len(result + "!!!" + label.split("!!!")[i + 1]) < max_length * max_lines:
        i += 1
        result += "!!!" + label.split("!!!")[i]

    result += " !!!..." if len(label.split("!!!")) > i + 1 else ""

    # Wrap the text
    lines = textwrap.wrap(result, max_length, break_long_words=False)
    result = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        result += "..."
    return result


def reshape_pileup_to_matrix_polars(methyl_data) -> pl.LazyFrame | None:
    # Add a name column
    position_cols = ["contig", "strand", "inclusive start position", "exclusive end position"]

    # Keep only what we need
    methyl_data = methyl_data.select(position_cols + ['modified base code and motif', "Nmod", "Ncanonical"])

    methyl_data = methyl_data.collect(streaming=True)
    if methyl_data.height == 0:
        return None

    pivot_df1 = methyl_data.pivot(index=position_cols, columns='modified base code and motif', values='Nmod')
    pivot_df2 = methyl_data.pivot(index=position_cols, columns='modified base code and motif', values='Ncanonical')

    # If there was no methylation of one type add Nulls
    for meth_type in readable_methylation_name.keys():
        if meth_type not in pivot_df1.columns:
            pivot_df1 = pivot_df1.with_columns(pl.lit(None).cast(pl.Int64).alias(meth_type))

    # Handle renaming columns to canonical
    for base, meth_group in base_methylation_map.items():
        assert len(meth_group) > 0, f"No methylation types specified for base {base} in base_methylation_map"

        if len(meth_group) == 1:
            if meth_group[0] not in pivot_df2.columns:
                pivot_df2 = pivot_df2.with_columns(pl.lit(None).cast(pl.Int64).alias("Ncanonical_" + base))
            else:
                pivot_df2 = pivot_df2.with_columns(pl.col(meth_group[0]).alias(f"Ncanonical_{base}"))
        else:
            # Create f"Ncanonical_{base}" with the first non-null value, row wise in columns of meth_group
            # This works because Ncanonical is the same for modifications of the same base.
            # This is required because not all modifications possible for a base are present for each instance of it.
            expr = None
            for col in meth_group:
                if col in pivot_df2.columns:
                    if expr is None:
                        expr = pl.col(col)
                    else:
                        expr = expr.fill_null(pl.col(col))

            # Add the expression as a new column
            if expr is None:
                pivot_df2 = pivot_df2.with_columns(pl.lit(None).cast(pl.Int64).alias(f"Ncanonical_{base}"))
            else:
                pivot_df2 = pivot_df2.with_columns(expr.alias(f"Ncanonical_{base}"))

    pivot_df2 = pivot_df2.select(*position_cols, *[f"Ncanonical_{base}" for base in base_methylation_map.keys()])
    pivot_df = pivot_df1.join(pivot_df2, on=position_cols, how='inner').lazy()

    # Select is needed to ensure order for vstack
    return pivot_df.select("contig", "strand", "inclusive start position", *readable_modification_name.keys())


def add_gene_caller_id(df: pl.LazyFrame, genes: pl.LazyFrame, keep_cols: list[str] = [],
                        include_intergenic: bool = False) -> pl.LazyFrame:
    """
    Add the gene caller id.
    """
    # Define the columns to keep eventually, including the new gene_callers_id
    og_columns = df.collect_schema().names() + keep_cols + ["gene_callers_id"]

    # Add a unique row ID to df to handle potential multiple matches later
    # and to facilitate joining back intergenic regions if needed.
    df = df.with_row_count(name="row_id")

    # Ensure both dataframes are sorted for join_asof
    # Sort df by contig, strand, and position
    df_sorted = df.sort("contig", "strand", "position")

    # Prepare and sort genes dataframe
    genes_sorted = genes.select("contig", "strand", "start", "stop", "gene_callers_id").sort("contig", "strand", "start")

    # Perform the asof join. Find the latest gene 'start' that is less than or equal to 'position'
    # for matching contig and strand.
    joined_df = df_sorted.join_asof(
        genes_sorted,
        left_on="position",
        right_on="start",
        by=["contig", "strand"],
        strategy="backward" # Find gene start <= position
    )

    # Filter the results to keep only rows where the position is actually within the gene boundaries
    # i.e., position >= start (implicit from join_asof) AND position < stop
    valid_overlaps = joined_df.filter(pl.col("position") < pl.col("stop"))

    if include_intergenic:
        # If including intergenic regions, perform a left join back to the original df (with row_id)
        # using the valid overlaps. Rows from df that didn't have a valid overlap will have nulls
        # for the gene columns.
        result = df.join(
            valid_overlaps.select("row_id", "gene_callers_id"), # Only need row_id and the gene id
            on="row_id",
            how="left"
        )
        # Fill null gene_callers_id with -1 for intergenic regions
        result = result.with_columns(pl.col("gene_callers_id").fill_null(-1))
    else:
        # If not including intergenic, just keep the valid overlaps
        result = valid_overlaps

    # If a position overlaps multiple genes (e.g., nested genes), join_asof might still produce multiple rows
    # for the same original row_id if multiple genes start at the same position.
    result = result.unique(subset=["row_id"], keep="first")
    
    return result.select(*og_columns)


def treatment_weighted_mean(df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame | pl.DataFrame:
    for base, meth_group in base_methylation_map.items():
        norm_columns = meth_group + ["Ncanonical_" + base]

        df = df.with_columns(pl.concat_list(*norm_columns).list.sum().alias(f"total_coverage_{base}"))
        df = df.with_columns((pl.col(meth_key) / pl.col(f"total_coverage_{base}")).alias(meth_key+"_fraction") for meth_key in norm_columns)

    df = df.group_by("contig", "position", "strand", "treatment").agg([
        ((pl.col(meth_key+"_fraction") * pl.col(f"total_coverage_{base}")).sum() / pl.col(f"total_coverage_{base}").sum()).alias(meth_key)
        for base, meth_group in base_methylation_map.items()
        for meth_key in (meth_group + ["Ncanonical_" + base])
    ])

    return df


def generate_cross_validation_sets(df: pl.DataFrame, unique_col: str, treatmeant_col: str, sample_col: str,
                                   boot_id: int) -> pl.DataFrame:
    # Get all possible combinations of replicate_labels and treatments
    all_permutations = list(itertools.product(
        *[df.filter(pl.col(treatmeant_col).eq(group)).get_column(sample_col).unique().to_list() for group in
          df.get_column(treatmeant_col).unique().to_list()]))
    if boot_id >= len(all_permutations):
        print(f"Max bootstraps is {len(all_permutations)}")
        boot_id = random.randint(0, len(all_permutations) - 1)

    # Keep only names (positions) that are in all samples
    labels_in_all_groups = df.group_by(unique_col).agg(pl.col(sample_col).n_unique().alias("unique_groups")).filter(
        pl.col("unique_groups") == df.get_column(sample_col).n_unique()).get_column(unique_col).to_list()
    df = df.filter(pl.col(unique_col).is_in(labels_in_all_groups))

    # Get the combination of samples for this bootstrap
    combination = all_permutations[boot_id]
    df = df.filter(pl.col(sample_col).is_in(combination))
    return df


def create_methylation_bins(num_bins: int, low: float = 0, high: float = 1) -> tuple[list[float], list[str]]:
    """
    Create bin boundaries and corresponding labels for methylation values.

    The function generates an extended set of boundaries (by adding two extra bins)
    and then removes the first and last bins so that the resulting bins are labeled
    consistently. For example, if num_bins=3, it returns three bin labels such as:
    ["0.30-0.60", "0.60-0.90", "0.90-1.20"] (depending on the low/high values).

    Parameters:
      num_bins: The desired number of bins (e.g., 3).
      low: The lower bound of the methylation range (default: 0).
      high: The upper bound of the methylation range (default: 1).

    Returns:
      A tuple (cut_points, bin_labels) where:
         - cut_points is a list of boundary values to be used with a cut function.
         - bin_labels is a list of string labels corresponding to each bin.
    """
    # Adjust the number of bins by adding two extra bins
    adjusted_bins = num_bins + 2
    cut_points = np.linspace(low, high, adjusted_bins - 1).tolist()

    bin_labels = []
    for i in range(adjusted_bins):
        if i == 0:
            bin_labels.append(f"<={cut_points[i]:.2f}")
        elif i == adjusted_bins - 1:
            bin_labels.append(f">{cut_points[i-1]:.2f}")
        else:
            bin_labels.append(f"{cut_points[i-1]:.2f}-{cut_points[i]:.2f}")

    # Remove the first and last bins to obtain the desired number of bins.
    return cut_points[1:-1], bin_labels[1:-1]


def parse_treatment_tsv(treatment_info_path):
    # Load treatment information mappings
    treatment_name_map = {}
    treatment_color_map = {}
    treatment_order_map = {}
    
    with open(treatment_info_path, mode='r') as file:
        reader = csv.reader(file, delimiter='\t')
        for i, row in enumerate(reader):
            if row == []:
                print(f"Warning: Empty row found in treatment info file at line {i}. Skipping.")
                continue
            
            if i==0:
                if row == ["treatment", "readable_treatment", "color", "order"]:
                    continue
                else:
                    raise ValueError("Treatment info TSV file must have header 'treatment', 'readable_treatment', 'color', and 'order' at {treatment_info}.")
            
            treatment_name_map[row[0]] = row[1]
            treatment_color_map[row[1]] = row[2].replace(" ", "")
            treatment_order_map[row[1]] = int(row[3])
    
    return treatment_name_map, treatment_color_map, treatment_order_map

def parse_barcode_tsv(barcode_treatment_sample_file):
    # Load barcode→treatment mappings
    barcode_sample_map = {}
    barcode_treatment_map = {}

    with open(barcode_treatment_sample_file, mode='r') as file:
        reader = csv.reader(file, delimiter='\t')
        for i, row in enumerate(reader):
            if i==0:
                if row == []:
                    print(f"Warning: Empty row found in treatment info file at line {i}. Skipping.")
                    continue
            
                if row == ['barcode', 'treatment', 'sample']:
                    continue
                else:
                    raise ValueError("Barcode treatment sample TSV file must have header 'barcode', 'treatment', and 'sample' at {barcode_treatment_sample_file}.")
            
            barcode_treatment_map[row[0]] = row[1]
            barcode_sample_map[row[0]] = row[2]
    
    return barcode_treatment_map, barcode_sample_map


def do_ks_test(motif, treatments, alpha): 
    # Define variables   
    group1_label, group2_label = treatments
    meth_col_name = motif.meth_type

    is_valid_filter = lambda x : pl.col(x).is_not_null() & pl.col(x).is_not_nan()
    group1_filter = pl.col("treatment").eq(group1_label)
    group2_filter = pl.col("treatment").eq(group2_label)
    promoter_filter = pl.col("distance_to_start").le(60)
    se_expr = (pl.col(meth_col_name).std() / pl.col(meth_col_name).len().sqrt()).alias("se")
    
    sort_cols = ["contig", "position", "strand"]
    grouping_cols = ["contig", "strand", "position", "treatment"]
    cols_for_calcs = [meth_col_name] + grouping_cols
    cols_for_promoter_calcs = ["distance_to_start"] + cols_for_calcs

    # Common rows
    def filter_common(df: pl.LazyFrame) -> pl.LazyFrame:
        common_rows = df.group_by(sort_cols).agg(pl.col("treatment").unique().len()).filter(pl.col("treatment") == 2).with_columns(pl.struct(sort_cols).alias("sort_key")).collect()
        df = df.filter(pl.struct(sort_cols).is_in(common_rows.get_column("sort_key")))
        return df
    
    # Get data
    normalized_data = motif.data(normalize=True).filter(group1_filter | group2_filter)
    raw_data = motif.data(normalize=False).filter(group1_filter | group2_filter)
    normalized_with_genes = motif.genome.nearest_gene_to_positions(normalized_data).filter(group1_filter | group2_filter)
    raw_with_genes = motif.genome.nearest_gene_to_positions(raw_data).filter(group1_filter | group2_filter)
    
    # Means
    means_df = filter_common(normalized_data.select(cols_for_calcs).filter(is_valid_filter(meth_col_name))).sort(sort_cols).collect()
    group1_means = means_df.filter(group1_filter).get_column(meth_col_name).to_numpy()
    group2_means = means_df.filter(group2_filter).get_column(meth_col_name).to_numpy()

    # Promoter means
    promoter_means_df = filter_common(normalized_with_genes.select(cols_for_promoter_calcs).filter(is_valid_filter(meth_col_name), promoter_filter)).sort(sort_cols).collect()
    group1_promoter_means = promoter_means_df.filter(group1_filter).get_column(meth_col_name).to_numpy()
    group2_promoter_means = promoter_means_df.filter(group2_filter).get_column(meth_col_name).to_numpy()

    # Standard error
    standard_error_df = filter_common(raw_data.select(cols_for_calcs).filter(is_valid_filter(meth_col_name)).group_by(grouping_cols).agg(se_expr).filter(is_valid_filter("se"))).sort(sort_cols).collect()
    group1_standard_error = standard_error_df.filter(group1_filter).sort(sort_cols).get_column("se").to_numpy()
    group2_standard_error = standard_error_df.filter(group2_filter).sort(sort_cols).get_column("se").to_numpy()

    # Standard error of promoter regions
    promoter_standard_error_df = filter_common(raw_with_genes.select(cols_for_promoter_calcs).filter(is_valid_filter(meth_col_name), promoter_filter).group_by(grouping_cols).agg(se_expr).filter(is_valid_filter("se"))).sort(sort_cols).collect()
    group1_promoter_standard_error = promoter_standard_error_df.filter(group1_filter).sort(sort_cols).get_column("se").to_numpy()
    group2_promoter_standard_error = promoter_standard_error_df.filter(group2_filter).sort(sort_cols).get_column("se").to_numpy()

    # Do tests
    means_pval, means_dval = ks_permutation_test(group1_means, group2_means)
    promoter_meanspval, promoter_means_dval = ks_permutation_test(group1_promoter_means, group2_promoter_means)
    standard_error_pval, standard_error_dval = ks_permutation_test(group1_standard_error, group2_standard_error)
    promoter_standard_error_pval, promoter_standard_error_dval = ks_permutation_test(group1_promoter_standard_error, group2_promoter_standard_error)
    
    # Store
    p_values = [means_pval, standard_error_pval, promoter_meanspval, promoter_standard_error_pval]
    d_values = [means_dval, standard_error_dval, promoter_means_dval, promoter_standard_error_dval]

    keys = ["means", "se", "means_pr", "se_pr"]
    valid_indices = []
    valid_p_values = []
    valid_d_values = []

    # Filter out None values and keep track of valid indices
    for i, p_value in enumerate(p_values):
        if p_value is not None:
            valid_p_values.append(p_value)
            valid_indices.append(i)
    
    for i, d_value in enumerate(d_values):
        if d_value is not None:
            valid_d_values.append(d_value)
    
    adj_pvals = {key+"_pval": None for key in keys}
    adj_dvals = {key: None for key in keys}
    if len(valid_p_values) == 0:
        return adj_pvals, adj_dvals
    
    # Apply multipletests only if we have valid p-values
    _, test_adj_pvals, _, _ = multipletests(valid_p_values, alpha=alpha, method='fdr_bh')
    
    # Update with adjusted p-values where available
    for i, orig_idx in enumerate(valid_indices):
        adj_pvals[keys[orig_idx]+"_pval"] = test_adj_pvals[i]
        adj_dvals[keys[orig_idx]] = valid_d_values[i]
    
    # Save Q-Q plots
    qq_plot_for_ks_data(group1_means, group2_means, motif.genome.output_dir, title=f"Q-Q Plot for {motif.meth_type} Means", group1_label=group1_label, group2_label=group2_label)
    qq_plot_for_ks_data(group1_promoter_means, group2_promoter_means, motif.genome.output_dir, title=f"Q-Q Plot for {motif.meth_type} Promoter Means", group1_label=group1_label, group2_label=group2_label)
    
    return adj_pvals, adj_dvals


def ks_permutation_test(data1, data2, n_permutations=10000):
    """
    Performs a two-sample permutation test based on the Kolmogorov-Smirnov statistic.

    This implements Stephen's null hypothesis: that both samples are drawn from the
    same underlying distribution. It calculates the KS statistic (D) for the observed
    data and compares it to a distribution of KS statistics generated by repeatedly
    permuting the combined data and splitting it back into two samples.

    Args:
        data1 (array-like): First sample data.
        data2 (array-like): Second sample data.
        n_permutations (int): The number of permutations to perform.

    Returns:
        float: The empirical p-value based on the permutations.
            Returns None if either sample has fewer than 2 data points.
    """
    # Convert to numpy arrays for easier handling
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    # KS test requires at least 2 points in each sample for meaningful comparison often,
    # although ks_2samp might handle 1. Let's stick to the original check.
    if len(data1) < 5 or len(data2) < 5:
        # Cannot perform meaningful KS test
        print("Unexpected data size for KS test. Returning None.")
        return None, None

    # Calculate the observed KS statistic (D)
    # ks_2samp returns a result object (or tuple in older scipy)
    # The first element or the .statistic attribute is the D value.
    observed_ks_value = stats.ks_2samp(data1, data2).statistic
    
    if np.isnan(observed_ks_value) or np.isinf(observed_ks_value):
        # If the observed KS value is NaN or infinite, we cannot proceed
        print("Observed KS value is NaN or infinite. Returning None.")
        return None, None

    # Combine the data for permutation
    combined_data = np.concatenate((data1, data2))
    n1 = len(data1)
    
    # Cap permutations
    n_permutations = min(math.comb(len(combined_data), n1), n_permutations)

    count_extreme = 0
    for _ in range(n_permutations):
        # Permute the combined data
        np.random.shuffle(combined_data)
        
        # Split into two samples
        permuted_sample1 = combined_data[:n1]
        permuted_sample2 = combined_data[n1:]

        # Calculate KS statistic for the permuted samples
        perm_ks_value = stats.ks_2samp(permuted_sample1, permuted_sample2).statistic

        if perm_ks_value >= observed_ks_value:
            count_extreme += 1

    return (count_extreme + 1) / (n_permutations + 1), observed_ks_value


def qq_plot_for_ks_data(group1_data, group2_data, out_dir, title='Q-Q Plot', group1_label='Group 1', group2_label='Group 2'):
    """
    Create a Q-Q plot comparing two datasets (same data used for Kolmogorov-Smirnov test).
    
    Args:
        group1_data (np.array): First group data (e.g., group1_means)
        group2_data (np.array): Second group data (e.g., group2_means)
        title (str): Title of the plot
        group1_label (str): Label for first group
        group2_label (str): Label for second group
    """
    # # Sort the data
    data1_sorted = group1_data #np.sort(group1_data)
    data2_sorted = group2_data #np.sort(group2_data)

    # Calculate quantiles
    quantiles = np.linspace(0, 1, min(len(data1_sorted), len(data2_sorted)))
    quantile_data1 = np.quantile(data1_sorted, quantiles)
    quantile_data2 = np.quantile(data2_sorted, quantiles)

    # Create the plot
    import matplotlib.pyplot as plt
    import os 
    
    plt.figure(figsize=(8, 6))
    plt.plot(quantile_data1, quantile_data2, 'o', alpha=0.7, label='Q-Q points')
    plt.plot([quantile_data1.min(), quantile_data1.max()], 
             [quantile_data1.min(), quantile_data1.max()], 
             'r--', label='y=x line (perfect match)')
    
    plt.xlabel(f'Quantiles of {group1_label}')
    plt.ylabel(f'Quantiles of {group2_label}')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs(out_dir / "qq_plots", exist_ok=True)
    plt.savefig(out_dir / "qq_plots" / f"{title.replace(' ', '_')}_{group1_label}_{group2_label}.pdf", dpi=300, bbox_inches='tight')