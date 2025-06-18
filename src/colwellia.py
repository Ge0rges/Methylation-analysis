import math 
import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import scipy.stats as stats
import seaborn as sns
from statsmodels.stats.multitest import multipletests

from src.objects.gene_collection import GeneCollection
from src.objects.genome import Genome
from src.objects.motif import Motif
from src.utilities.utils import readable_modification_name, create_methylation_bins

sns.set_theme(context="paper", style="whitegrid")

def plot_number_of_positions_by_coverage_colwellia(genome: Genome, motif: Motif, output_dir: Path) -> None:
    """
    Plot the number of positions by coverage for a given motif across all samples.
    """
    original_cov = genome.default_coverage
    
    # Get data for different coverages
    coverage_counts = []
    for cov in [1, 5, 10, 20, 30, 50, 100]:
        genome.default_coverage = cov
        df = motif.data().collect()
        df = df.group_by("treatment").agg(pl.struct(["contig", "strand", "position"]).unique().count().alias("count"))
        df = df.with_columns(pl.lit(cov).alias("coverage"))
        coverage_counts.append(df)
    
    coverage_counts = pl.concat(coverage_counts)
    
    # Plot barplot using seaborn
    _, ax = plt.subplots(figsize=(16, 16))
    sns.barplot(
        data=coverage_counts.to_pandas(),
        x="treatment",
        y="count",
        hue="coverage",
        legend='full',
        ax=ax
    )
    
    plt.xticks(rotation=90, ha='right')    
    plt.savefig(output_dir / f"{genome.readable_name}_{motif.readable_motif}_coverage_counts.pdf", format="pdf")
    
    # Get data for different coverages
    coverage_counts = []
    for cov in [1, 5, 10, 20, 30, 50, 100]:
        genome.default_coverage = cov
        df = motif.data(in_every_treatment=False).collect()
        df = df.group_by("treatment").agg(pl.struct(["contig", "strand", "position"]).unique().count().alias("count"))
        df = df.with_columns(pl.lit(cov).alias("coverage"))
        coverage_counts.append(df)
    
    coverage_counts = pl.concat(coverage_counts)
    
    # Plot barplot using seaborn
    _, ax = plt.subplots(figsize=(16, 16))
    sns.barplot(
        data=coverage_counts.to_pandas(),
        x="treatment",
        y="count",
        hue="coverage",
        legend='full',
        ax=ax
    )
    
    plt.title("Not in every treatment")    
    plt.xticks(rotation=90, ha='right')
    
    plt.savefig(output_dir / f"{genome.readable_name}_{motif.readable_motif}_coverage_counts_notevery treatment.pdf", format="pdf")
    
    genome.default_coverage = original_cov  # Reset to original coverage


def plot_whole_methylome_colwellia(genome: Genome, motif: Motif, output_dir: Path) -> None:
    """
    Plot the whole methylome (only for the motif's methylation type) across samples:
    - fraction = motif_type / (motif_type + canonical base)
    - x-axis = genome_position, color by sample
    - Creates separate subplots for each pairwise treatment comparison
    """
    df = motif.data()
    if df is None:
        return
    
    df = df.collect()
    df = genome.add_genome_relative_position(df).rename({"treatment": "Treatment"})
    
    if df.is_empty():
        return
    
    # Get all unique treatments and create pairwise combinations
    treatments = sorted(df.get_column("Treatment").unique().to_list(), key=genome.treatment_order_map.get)
    pairwise_comparisons = list(combinations(treatments, 2))
    
    # Calculate subplot grid dimensions
    n_comparisons = len(pairwise_comparisons)
    n_cols = min(4, n_comparisons)
    n_rows = math.ceil(n_comparisons / n_cols)
    
    # Create subplots
    _, axes = plt.subplots(n_rows, n_cols, figsize=(12*n_cols, 7*n_rows), constrained_layout=True)
    
    # Handle case where we have only one subplot
    if n_comparisons == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (treat1, treat2) in enumerate(pairwise_comparisons):
        ax = axes[idx]
        
        # Filter data for current treatment pair
        pair_df = df.filter(df["Treatment"].is_in([treat1, treat2]))
        
        # Create scatterplot for this pair
        sns.scatterplot(
            data=pair_df.to_pandas(),
            x="genome_position",
            y=motif.meth_type,
            hue="Treatment",
            style="Treatment",
            ax=ax,
            s=16,
            alpha=0.7,
            hue_order=[treat1, treat2],
            palette=[genome.treatment_color_map[treat1], genome.treatment_color_map[treat2]]
        )
        
        # Add polynomial regression lines for each treatment in the pair
        for j, treatment in enumerate([treat1, treat2]):
            treatment_df = pair_df.filter(pair_df["Treatment"] == treatment).to_pandas()
            
            if len(treatment_df) > 4:
                avg_df = treatment_df.groupby("genome_position")[motif.meth_type].mean().reset_index()
                
                if len(avg_df) > 4:
                    # Fit polynomial regression
                    x = avg_df["genome_position"].values
                    y = avg_df[motif.meth_type].values
                    z = np.polyfit(x, y, 4)
                    p = np.poly1d(z)
                    
                    # Generate smooth curve
                    x_smooth = np.linspace(x.min(), x.max(), 300)
                    y_smooth = p(x_smooth)
                    
                    # Plot the smoothed average line
                    ax.plot(x_smooth, y_smooth, color=genome.treatment_color_map[treatment], linewidth=2)
        
        ax.set_xlabel("Genome position (bp)")
        ax.set_ylabel(f"Methylation fraction")
        ax.set_title(f"{treat1} vs {treat2}")

        ax.legend(bbox_to_anchor=(1.05, 1), loc='lower left')
        ax.legend_.set_title(None)

    # Hide any unused subplots
    for idx in range(n_comparisons, len(axes)):
        axes[idx].set_visible(False)

    out_file = output_dir / f"{genome.readable_name}_whole_methylome_pairwise_{motif.readable_motif}.pdf"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"Saved PDF: {out_file}")


##############################################################################
# 2) METHYLATION DISTRIBUTION PER SAMPLE
##############################################################################

def plot_motif_methylation_distribution_colwellia(genome: Genome, motif: Motif, output_dir: Path) -> None:
    """
    Plot the distribution (histogram) of motif's methylation fraction in each sample,
    rather than the mean fraction.

    - fraction = meth / (meth + canonical)
    - We'll do a single figure with multiple histplot calls, or subplots, showing how
    many sites fall in each fraction bin per sample.
    """
    df = motif.data()
    
    if df is None:
        return
    
    df = df.collect().rename({"treatment": "Treatment"})
    if df.is_empty():
        return
    
    # One approach: single figure, color by treatment. Another approach: subplots per treatment.
    # Example: single figure, multiple histplot calls with "multiple='dodge'"
    _, ax = plt.subplots(figsize=(32, 20), constrained_layout=True)

    sns.histplot(
        data=df.to_pandas(),
        x=motif.meth_type,
        hue="Treatment",
        bins=20,
        multiple="dodge",
        edgecolor="white",
        ax=ax,
        kde=True,
        hue_order=sorted(df.get_column("Treatment").unique().to_list(), key=genome.treatment_order_map.get)
    )
    ax.set_xlabel(f"{readable_modification_name[motif.meth_type]} methylation fraction")
    ax.set_ylabel("Number of sites")
    ax.set_title(f"{genome.readable_name} - Distribution of {motif.motif} Methylation")

    out_file = output_dir / f"{genome.readable_name}_{motif.readable_motif}_methylation_distribution.pdf"
    plt.savefig(out_file, format="pdf") 
    plt.close()
    print(f"Saved PDF: {out_file}")


def plot_motif_distribution_stats_colwellia(genome: Genome, motif: Motif, output_dir: Path) -> None:
    """
    Plot the distribution statistics (mean, median, std) of motif's methylation fraction
    in each sample.
    """
    alpha = 0.01

    unique_treatment_names = set(genome.treatment_name_map.values())
    all_treatment_names = sorted(
        list(unique_treatment_names),
        key=lambda t: genome.treatment_order_map.get(t, str(t))
    )

    # Extract cycling steps and control steps from treatment names
    cycling_steps = []
    control_35ppt = []
    control_55ppt = []

    pattern = re.compile(r"(Cycling|35ppt control|55ppt control) S(\d+)")

    for treatment in all_treatment_names:
        match = pattern.match(treatment)
        if match:
            group, step = match.groups()
            step = int(step)
            if group == "Cycling":
                cycling_steps.append((treatment, step))
            elif group == "35ppt control":
                control_35ppt.append((treatment, step))
            elif group == "55ppt control":
                control_55ppt.append((treatment, step))

    # Sort by step number
    cycling_steps.sort(key=lambda x: x[1])
    control_35ppt.sort(key=lambda x: x[1])
    control_55ppt.sort(key=lambda x: x[1])
    
    # For each cycling step, find closest steps in each control
    closest_controls = {}
    for treatment, step in cycling_steps:
        closest_controls[treatment] = {
            "35ppt control": find_closest_step(step, control_35ppt),
            "55ppt control": find_closest_step(step, control_55ppt),
        }

    # Generate pairs of treatments using the sorted, unique names
    pairs = list(tuple(sorted(x)) for x in combinations(all_treatment_names, 2))
    results = {}

    # Get Stats
    for pair_treatments in pairs:
        adj_pvals, dvals = do_test(motif, pair_treatments, alpha=alpha)
        results[pair_treatments] = adj_pvals, dvals
        
    # Create DataFrame for Timeline
    timeline_data = []
    for cycling_treatment, controls in closest_controls.items():
        step_match = re.search(r'S(\d+)', cycling_treatment)
        if not step_match:
            continue
        step_num = int(step_match.group(1))
        
        for control_name, control_treatment in controls.items():
            if control_treatment is not None:
                pair = tuple(sorted([cycling_treatment, control_treatment]))
                if pair in results:
                    pvals, dvals = results[pair]
                    for stat_key in ['means', 'means_pr']:
                        dval = dvals.get(stat_key, np.nan)
                        pval = pvals.get(stat_key, 1.0)
                        timeline_data.append({
                            'Cycling Step': step_num,
                            'D-value': dval,
                            'Control': control_name,
                            'Significant': pval <= alpha,
                            'Stat Key': stat_key
                        })

    timeline_df = pd.DataFrame(timeline_data)

    # Make figure
    _, axes = plt.subplots(2, 2, figsize=(30, 60), constrained_layout=True)

    for i, stat_key in enumerate(["means", "means_pr"]):
        # Get axes
        dval_heatmap_ax = axes[i, 0]
        dval_timeline_ax = axes[i, 1]

        # Get matrices for heatmap
        pval_matrix = pd.DataFrame(np.nan, index=all_treatment_names, columns=all_treatment_names, dtype=float)
        dval_matrix = pd.DataFrame(np.nan, index=all_treatment_names, columns=all_treatment_names, dtype=float)

        for (treat1, treat2), (pval_dict, dval_dict) in results.items():            
            pval_matrix.loc[treat1, treat2] = pval_dict[stat_key]
            pval_matrix.loc[treat2, treat1] = pval_dict[stat_key]
            dval_matrix.loc[treat1, treat2] = dval_dict[stat_key]
            dval_matrix.loc[treat2, treat1] = dval_dict[stat_key]
        
        # Make pariwise heatmap
        mask = pval_matrix >= alpha
        annot = np.where(pval_matrix < alpha, dval_matrix.round(2), "X")
        
        sns.heatmap(
            dval_matrix,
            ax=dval_heatmap_ax,
            annot=annot,
            fmt="s",
            mask=mask,
            cmap="coolwarm_r",
            cbar_kws={'label': 'D-value', 'shrink': 0.8},
            vmin=0,
            vmax=1,
            linewidths=0.5,
        )
        title_stat_key = stat_key.replace('_', ' ').title()
        if "_pr" in stat_key: # Make "Pr" -> "Promoter"
            title_stat_key = title_stat_key.replace(" Pr", " (Promoter)")
        
        dval_heatmap_ax.set_title(title_stat_key)
        
        # Make timeline
        sns.scatterplot(
            data=timeline_df[timeline_df['Stat Key'] == stat_key],
            x='Cycling Step', 
            y='D-value',
            hue='Control', 
            style='Significant',
            markers={True: 'o', False: 'X'},
            s=24, ax=dval_timeline_ax
        )
        dval_timeline_ax.set_title(f'd-values for Cycling Steps vs Closest Controls ({stat_key})')
    
    # Save to PDF
    plt.savefig(output_dir / "motif_methylation_pvalue_heatmap.pdf", format="pdf")
    
    
# Function to find closest control step for a given cycling step
def find_closest_step(cycling_step, control_steps):
    control_step_nums = [step for _, step in control_steps]
    if not control_step_nums:
        return None
    closest_step_num = min(control_step_nums, key=lambda x: abs(x - cycling_step))
    # Find the treatment name for this step
    for treatment, step in control_steps:
        if step == closest_step_num:
            return treatment
    return None


def do_test(motif, treatments, alpha=0.01):    
    group1_label, group2_label = treatments
    meth_col_name = motif.meth_type

    # Start lazy frame, add gene info, filter invalid data
    normalized_lazy = motif.data(normalize=True)
    normalized_with_genes_lazy = motif.genome.nearest_gene_to_positions(normalized_lazy)

    # Define common filters
    is_valid_filter = pl.col(meth_col_name).is_not_null() & pl.col(meth_col_name).is_not_nan()
    group1_filter = pl.col("treatment").eq(group1_label)
    group2_filter = pl.col("treatment").eq(group2_label)
    promoter_filter = pl.col("distance_to_start").le(60)
    sort_cols = ["contig", "position", "strand"]

    # Apply validity filter and select only necessary columns early
    # We need: meth_col_name, treatment, distance_to_start, and sort columns
    cols_for_means = [meth_col_name, "treatment", "distance_to_start"] + sort_cols
    filtered_means_lazy = normalized_with_genes_lazy.filter(is_valid_filter).select(cols_for_means)

    # Collect the filtered/selected data ONCE for means calculation
    # Sorting is done *after* collection here for potentially better parallelization
    # during collection, but could be done lazily before collect too.
    collected_means_df = filtered_means_lazy.sort(sort_cols).collect()

    # Extract numpy arrays for means from the collected DataFrame
    group1_data = collected_means_df.filter(group1_filter).get_column(meth_col_name).to_numpy()
    group2_data = collected_means_df.filter(group2_filter).get_column(meth_col_name).to_numpy()

    # Extract numpy arrays for promoter means
    promoter_means_df = collected_means_df.filter(promoter_filter) # Filter the already collected DF
    group1_data_pr = promoter_means_df.filter(group1_filter).get_column(meth_col_name).to_numpy()
    group2_data_pr = promoter_means_df.filter(group2_filter).get_column(meth_col_name).to_numpy()

    # Start lazy frame for raw data, add gene info ONCE
    raw_lazy = motif.data(normalize=False)
    raw_with_genes_lazy = motif.genome.nearest_gene_to_positions(raw_lazy)

    # Apply validity filter and select necessary columns for SE calculation
    # We need: meth_col_name, treatment, distance_to_start, and grouping/sort columns
    grouping_cols = ["contig", "strand", "position", "treatment"]
    cols_for_se = [meth_col_name, "distance_to_start"] + grouping_cols
    filtered_se_lazy = raw_with_genes_lazy.filter(is_valid_filter).select(cols_for_se)

    # Collect the filtered/selected raw data ONCE for SE calculations
    collected_raw_df = filtered_se_lazy.collect()

    # Define the SE calculation expression
    se_expr = (pl.col(meth_col_name).std() / pl.col(meth_col_name).count().sqrt()).alias("se")

    # Calculate SE for all groups that have more than one row
    se_df_all = collected_raw_df.group_by(grouping_cols, maintain_order=True).agg(se_expr)
    se_df_all = se_df_all.filter(pl.col("se").is_not_null() & pl.col("se").is_not_nan())

    # Extract SE numpy arrays
    g1_se = se_df_all.filter(group1_filter).get_column("se").to_numpy()
    g2_se = se_df_all.filter(group2_filter).get_column("se").to_numpy()

    # Calculate SE for promoter regions by filtering the collected RAW data first
    se_df_pr = collected_raw_df.filter(promoter_filter).group_by(grouping_cols, maintain_order=True).agg(se_expr)

    # Extract promoter SE numpy arrays
    g1_se_pr = se_df_pr.filter(group1_filter).get_column("se").to_numpy()
    g2_se_pr = se_df_pr.filter(group2_filter).get_column("se").to_numpy()

    means_sig, means_d = ks_permutation_test(group1_data, group2_data)
    se_sig, se_d = ks_permutation_test(g1_se, g2_se) # Comparing distribution of standard errors

    # Promoter regions
    means_pr_sig, means_pr_d = ks_permutation_test(group1_data_pr, group2_data_pr)
    se_pr_sig, se_pr_d = ks_permutation_test(g1_se_pr, g2_se_pr) # Comparing distribution of standard errors in promoters
    
    p_values = [means_sig, se_sig, means_pr_sig, se_pr_sig]
    d_values = [means_d, se_d, means_pr_d, se_pr_d]
    
    # Create a mapping to keep track of indices and their corresponding keys
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
    
    adj_pvals = {key: None for key in keys}
    adj_dvals = {key: None for key in keys}
    if len(valid_p_values) == 0:
        return adj_pvals, adj_dvals
    
    # Apply multipletests only if we have valid p-values
    _, test_adj_pvals, _, _ = multipletests(valid_p_values, alpha=alpha, method='fdr_bh')
    
    # Update with adjusted p-values where available
    for i, orig_idx in enumerate(valid_indices):
        adj_pvals[keys[orig_idx]] = test_adj_pvals[i]
        adj_dvals[keys[orig_idx]] = valid_d_values[i]
    
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

    # KS test requires at least 2 points in each sample for meaningful comparison
    if len(data1) < 5 or len(data2) < 5:
        # Cannot perform meaningful KS test
        print("Unexpected data size for KS test. Returning None.")
        return None, None

    # Calculate the observed KS statistic (D)
    # ks_2samp returns a result object (or tuple in older scipy)
    # The first element or the .statistic attribute is the D value.
    observed_ks_value = stats.ks_2samp(data1, data2).statistic
    
    if np.isnan(observed_ks_value):
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


##############################################################################
# 3) DMR HEATMAP (ROWS = sample_a, COLUMNS = sample_b, VALUES = SCORE)
##############################################################################

def plot_dmr_scores_heatmap_colwellia(genome: Genome, motif: Motif, output_dir: Path) -> None:
    """
    Plot a heatmap of DMR scores for the specified motif across sample comparisons:
      - rows = sample_a
      - columns = sample_b
      - values = mean_dmr_score
    Each motif -> separate PDF.
    """
    dmr = motif.dmr_data.collect(streaming=True)
    if dmr.is_empty():
        print(f"No DMR data for motif {motif.motif}")
        return
    
    # Compute mean score per (sample_a, sample_b)
    pdf = (dmr.group_by(["treatment_a", "treatment_b"]).agg(pl.col("score").mean().alias("mean_dmr_score"))).to_pandas()

    # Make sure to unify sample names if needed, e.g. barcode -> treatment
    # If you want to rename sample_a, sample_b via genome.barcode_treatment_map:
    pdf["treatment_a"] = pdf["treatment_a"].replace(genome.barcode_treatment_map).replace(genome.treatment_name_map)
    pdf["treatment_b"] = pdf["treatment_b"].replace(genome.barcode_treatment_map).replace(genome.treatment_name_map)
    
    # Fill in values for inverse pairs
    
    
    # Get all unique treatments from both columns
    all_treatments_set = set(pdf["treatment_a"].unique()) | set(pdf["treatment_b"].unique())
    
    # Sort treatments based on genome.treatment_order_map
    # Fallback for sorting: if treatment not in map, use treatment name itself (converted to str for safety)
    sorted_treatments = sorted(
        list(all_treatments_set),
        key=lambda t: genome.treatment_order_map.get(t, str(t)) 
    )

    # Create pivot with sample_a as rows, sample_b as columns
    pivot_df = pdf.pivot(index="treatment_a", columns="treatment_b", values="mean_dmr_score")

    # Reindex to ensure the desired order and include all treatments, filling missing with NaN
    pivot_df = pivot_df.reindex(index=sorted_treatments, columns=sorted_treatments)

    _, ax = plt.subplots(figsize=(len(genome.default_treatments)*2, len(genome.default_treatments)*2), constrained_layout=True)
    sns.heatmap(pivot_df, cmap="viridis", annot=True, fmt=".2f", ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    ax.set_title(f"DMR Score Heatmap - {genome.readable_name} ({motif.motif})")
    ax.set_xlabel("Treatment B")
    ax.set_ylabel("Treatment A")

    out_file = output_dir / f"{genome.readable_name}_{motif.readable_motif}_dmr_heatmap.pdf"
    plt.savefig(out_file, format="pdf")
    plt.close()
    print(f"Saved PDF: {out_file}")


##############################################################################
# 4) PARALLEL CATEGORIES WITH CLEARER BIN LABELS & COLOR
##############################################################################

def plot_parallel_categories_methylation_colwellia(
    genome: Genome,
    motif: Motif,
    output_dir: Path,
    bins: int = 3
) -> None:
    """
    Create a Plotly parallel categories plot with clearer bin labels ("Low", "Medium", "High") 
    and color. For instance, we color by the first dimension's value.
    """
    df = motif.data()
    if df is None:
        return 
    df = df.collect()
    treatments = df.get_column("treatment").unique().to_list()
        
    if df.is_empty():
        return
    
    # Pivot by (contig, position, strand), columns = treatment, values = fraction_meth
    pivoted = df.pivot(
        index=["contig", "position", "strand"],
        on="treatment",
        values=motif.meth_type,
    )

    # Bin data
    cut_points, bin_labels = create_methylation_bins(bins)
    pivoted = pivoted.with_columns(pl.col(treatment).cut(cut_points, labels=bin_labels) for treatment in treatments).to_pandas()

    cat_to_num = {key: i for i, key in enumerate(bin_labels)}
    pivoted["meth_numeric"] = pivoted[treatments[0]].map(cat_to_num)

    sorted_treatments = sorted(treatments, key=genome.treatment_order_map.get)
    fig = px.parallel_categories(
        pivoted,
        dimensions=sorted_treatments,
        color="meth_numeric",
        color_continuous_scale=[
            (0.0, "blue"),
            (0.5, "orange"),
            (1.0, "red"),
        ],
        range_color=[0, 2]
    )
    
    fig.update_layout(title=f"Methylation state transitions in {genome.readable_name} across conditions", coloraxis_showscale=False)
    
    out_file = output_dir / f"{genome.readable_name}_{motif.readable_motif}_parallel_categories.html"
    fig.write_html(str(out_file))
    print(f"Saved HTML: {out_file}")

     

def extract_motif_data_all_transitions_colwellia(
    genome: Genome,
    motif: Motif,
    bins: int = 3,
) -> None:
    """
    Extract all motif data for each unique methylation
    transition across treatments. For each unique transition found in the data, this function
    outputs a separate CSV file (with gene annotations).

    The binning of methylation values is performed using a shared helper function to ensure
    consistency with the parallel categories plot.

    Parameters:
      genome: Genome object containing barcode/treatment maps and output info.
      motif: Motif object that provides the full motif data.
      bins: The number of bins used for methylation values (default: 3).

    Returns:
      None. (CSV files are written to genome.output_dir.)
    """
    # 1. Collect the full motif data and map samples to treatments.
    df = motif.data()
    if df is None:
        return
    
    df = df.collect()
    if df.is_empty():
        return

    # Add a composite key for later filtering.
    df = df.with_columns(
        pl.concat_str(
            [pl.col("contig").cast(str),
             pl.col("position").cast(str),
             pl.col("strand").cast(str)],
            separator="_"
        ).alias("composite_key")
    )

    # 2. Pivot the data so that each row (identified by contig, position, strand)
    #    has one column per treatment with methylation values given by motif.meth_type.
    pivoted = df.pivot(
        index=["contig", "position", "strand"],
        on="treatment",
        values=motif.meth_type,
    )

    # 3. Determine the unique treatments and sort them using genome.treatment_order_map.
    treatments = df["treatment"].unique().to_list()
    sorted_treatments = sorted(treatments, key=genome.treatment_order_map.get)

    # 4. Create bins for the methylation values using the helper function.
    cut_points, bin_labels = create_methylation_bins(bins)

    # 5. Bin the methylation values in each treatment column.
    pivoted = pivoted.with_columns(
        [pl.col(treatment).cut(cut_points, labels=bin_labels) for treatment in treatments]
    )

    # 6. Identify every unique transition present in the data.
    unique_transitions_df = pivoted.select(sorted_treatments).unique()

    # 7. For each unique transition, filter the data and output a CSV.
    result = []
    for transition_row in unique_transitions_df.iter_rows(named=True):
        # Build the transition tuple in the sorted treatment order.
        transitions = tuple(transition_row[col] for col in sorted_treatments)
        
        # Skip this for loop if the transitions has a None element
        if None in transitions:
            continue
        
        # Build the filter condition.
        condition = pl.lit(True)
        for col, val in zip(sorted_treatments, transitions):
            if val is None:
                condition &= pl.col(col).is_null()
            else:
                condition &= (pl.col(col) == val)
        
        filtered = pivoted.filter(condition)
        if filtered.is_empty():
            continue

        # 10. Annotate with gene information.
        data = genome.add_gene_caller_id(filtered.lazy(), include_intergenic=True).collect(streaming=True)
        gc = GeneCollection(data.get_column("gene_callers_id").unique().to_list(), genome)
        data = data.join(gc.get_function().collect(streaming=True), on="gene_callers_id", how="left")
        
        # Check if we can continue
        out_file = genome.output_dir / f"{genome.readable_name}_{motif.readable_motif}_motif_transition_{'_'.join(transitions)}.csv"
        if data.is_empty():
            # Print "No data for this transition" to output file
            with open(out_file, "w") as f:
                f.write(f"No data for transition {transitions}")
            print(f"Saved CSV for transition {transitions}: {out_file}")
            continue
        
        data = genome.nearest_gene_to_positions(data.lazy()).collect(streaming=True)
        gc = GeneCollection(data.get_column("gene_callers_id_start").unique().to_list(), genome)
        data = data.join(
            gc.get_function().collect(streaming=True),
            left_on="gene_callers_id_start",
            right_on="gene_callers_id",
            how="left",
            suffix="_start"
        )
        gc = GeneCollection(data.get_column("gene_callers_id_end").unique().to_list(), genome)
        data = data.join(
            gc.get_function().collect(streaming=True),
            left_on="gene_callers_id_end",
            right_on="gene_callers_id",
            how="left",
            suffix="_end"
        )

        # 11. Write the annotated data to CSV. The filename encodes the transition pattern.
        data.write_csv(out_file)
        print(f"Saved CSV for transition {transitions}: {out_file}")
        result.append(data)
    
    return pl.concat(result)


##############################################################################
# 5) EXTRACT DIFF METHYLATED GENES (WITH GENE FUNCTION)
##############################################################################

def extract_diff_methylated_genes_colwellia(
    genome: Genome,
    motif: Motif,
    top_n: int = 10
) -> pl.DataFrame:
    """
    Extract the top differentially methylated locations (by DMR 'score') for a motif,
    attach nearest/overlapping gene, and include the gene function in the final CSV.

    Returns a Polars DataFrame.
    """
    top_rows = motif.dmr_data.collect(streaming=True).sort("score", descending=True)
    if top_rows.is_empty():
        print(f"No DMR data for motif {motif.motif}")
        return

    # Now, we want to add the gene function from some table. Typically:
    # 1) Add gene_caller_id for direct hits
    # 2) For start/end, do the same and join gene function columns

    # Let's do 1) add gene_caller_id if needed
    data = genome.add_gene_caller_id(top_rows.lazy(), include_intergenic=True).collect(streaming=True)

    # Add function
    gc = GeneCollection(data.get_column("gene_callers_id").unique().to_list(), genome)
    data = data.join(gc.get_function().collect(streaming=True), on="gene_callers_id", how="left")

    # Add nearest gene if not in gene
    data = genome.nearest_gene_to_positions(data.lazy()).collect(streaming=True)

    # Add function of nearest genes
    gc = GeneCollection(data.get_column("gene_callers_id_start").unique().to_list(), genome)
    data = data.join(gc.get_function().collect(streaming=True), left_on="gene_callers_id_start", right_on="gene_callers_id", how="left", suffix="_start")
    gc = GeneCollection(data.get_column("gene_callers_id_end").unique().to_list(), genome)
    data = data.join(gc.get_function().collect(streaming=True), left_on="gene_callers_id_end", right_on="gene_callers_id", how="left", suffix="_end")
    
    # Logic to filter down the table by creating a ranking system
    # Create ranking columns based on priority criteria
    data = data.with_columns([
        # First priority: gene_callers_id is not -1 (has gene annotation)
        (pl.col("gene_callers_id") != -1).cast(pl.Int32).alias("has_gene"),
        
        # Create a sort priority column with clear priority rules
        pl.struct([
            # First priority: gene_callers_id is not -1 (has direct gene annotation)
            (pl.col("gene_callers_id") != -1),
            
            # Second priority: KOfam > COG20_FUNCTION > others for direct annotation
            pl.when(pl.col("source") == "KOfam").then(3)
              .when(pl.col("source") == "COG20_FUNCTION").then(2)
              .otherwise(1).alias("source_priority"),
            
            # Third priority: consistent annotations across direct and nearest genes
            (pl.col("source") == pl.col("source_start")) & (pl.col("source") == pl.col("source_end")),
            
            # Fourth priority: quality of nearest gene annotations
            pl.when(pl.col("source_start") == "KOfam").then(3)
              .when(pl.col("source_start") == "COG20_FUNCTION").then(2)
              .otherwise(1).alias("nearest_priority"),
            
            pl.when(pl.col("source_end") == "KOfam").then(3)
              .when(pl.col("source_end") == "COG20_FUNCTION").then(2)
              .otherwise(1).alias("nearest_priority_end")
        ]).alias("sort_priority")
    ])

    # Sort the data by priority within each position group
    data = data.sort("sort_priority", descending=True)
    
    # Group by position identifiers and take only the first row (highest priority) from each group
    data = data.group_by(["contig", "position", "strand"]).agg(
        pl.all().exclude(["contig", "position", "strand", "sort_priority", "has_gene"]).first()
    )
        
    # Take top_n rows by score if requested
    if top_n > 0 and data.height > top_n:
        data = data.sort("score", descending=True).head(top_n)
    
    # Write to CSV
    output_file = genome.output_dir / f"{genome.readable_name}_{motif.readable_motif}_top_diff_genes.csv"
    data.write_csv(output_file)
    
    return data


def write_basic_stats_colwellia(genome: Genome, motifs: list[Motif]):
    text = []
    for motif in motifs:
        site_count = motif.positions.unique(subset=["contig", "position", "strand"]).collect().height
        text.append(f"Number of sites: {site_count}\n")

        # Compute weighted fraction using treatment_weighted_mean
        df = motif.data()
        if df is None:
            return 
        
        df = df.collect()
        
        if df.is_empty():
            continue
        
        avg_fraction = df.select(pl.col(motif.meth_type).mean()).item()
        treatments = df.get_column("treatment").unique().to_list()
        for t in treatments:
            slice_df = df.filter(pl.col("treatment") == t)
            site_count_treatment = slice_df.unique(subset=["contig", "position", "strand"]).height
            text.append(f"Number of sites with data for {t}: {site_count_treatment}\n")
        text.append(f"Average {motif.meth_type} fraction: {avg_fraction}\n")

    text.append(f"Found motifs: {','.join([m.motif for m in motifs])}")
    
    out_file = genome.output_dir / f"{genome.readable_name}_motifs_basic_stats.txt"
    with open(out_file, "w") as f:
        f.writelines(text)
        
    print(f"Saved file: {out_file}")


def extract_consensus_genes_colwellia(genome: Genome, trans: pl.DataFrame, dmrs: pl.DataFrame, motif: Motif):
    # Get the intersection
    dmrs = dmrs.select("contig", "position", "strand", "score", "balanced_map_pvalue", "balanced_effect_size", "treatment_a", "treatment_b")
    consensus = trans.join(dmrs, how="inner", on=["contig", "position", "strand"])
    consensus.write_csv(genome.output_dir / f"{genome.readable_name}_{motif.readable_motif}_innerjoin_dmr_trans_genes.csv")