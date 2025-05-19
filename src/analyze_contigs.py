import polars as pl
import matplotlib.pyplot as plt
from src.objects.contig import Contig
from src.objects.gene_collection import GeneCollection
from src.utilities.utils import treatment_weighted_mean
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from matplotlib.colors import LogNorm
import math 

sns.set_theme(context="paper", style="whitegrid")

def plot_contig_motif_heatmap(contigs: list[Contig]):
    """
    Make a heatmap where each row is a contig. Each column is a motif string. 
    Show the average methylation fraction of that motif, in that contig, across all treatments.
    This is done by having one motif occur as many columns as there are treatments. 
    Draw a line across the X axis which highlights the sample.
    Color the y-ticks (contig names) by the taxonomy("g") of the contig
    """
    data = get_contigs_data(contigs, statistics=False)
    # Convert data list to a Polars DataFrame first
    df_partial = pl.DataFrame(data)

    if df_partial.is_empty():
        print(f"No data to plot in {contigs[0].parent_genome.readable_name}")
        return df_partial # Return the empty dataframe early

    # Get unique combinations and treatments
    unique_contig_motifs = df_partial.select("contig_name", "motif_string", "contig_taxonomy").unique()
    unique_treatments = df_partial.select("treatment").unique()
    df_complete = unique_contig_motifs.join(unique_treatments, how="cross")

    # Left join the partial data onto the complete grid
    # This ensures every (contig, motif, treatment) combination exists
    # Missing methylation_fraction values will be null
    df = df_complete.join(
        df_partial.select("contig_name", "motif_string", "treatment", "methylation_fraction"),
        on=["contig_name", "motif_string", "treatment"],
        how="left"
    )
    
    # Pivot the data with polars
    pivot_df = df.to_pandas().pivot(
        index="contig_name",
        columns=["motif_string", "treatment"],
        values="methylation_fraction"
    )
    
    # Create colors
    unique_taxonomies = df.get_column("contig_taxonomy").unique().to_list()
    pal = sns.color_palette(palette="tab20", n_colors=len(unique_taxonomies), as_cmap=False)
    lut = dict(zip(unique_taxonomies, pal))
    
    treatment_colors = df.with_columns(pl.col("treatment").replace_strict(contigs[0].parent_genome.treatment_color_map).alias("Treatment")).select("motif_string", "treatment", "Treatment")
    contig_colors = df.with_columns(pl.col("contig_taxonomy").replace_strict(lut).alias("Taxonomy")).select("contig_name", "Taxonomy")
    
    treatment_colors = treatment_colors.to_pandas().drop_duplicates().set_index(pivot_df.columns, drop=True).drop(columns=["motif_string", "treatment"])
    contig_colors = contig_colors.to_pandas().drop_duplicates("contig_name").set_index("contig_name")
        
    # Sort columns by motif_string first, then treatment
    pivot_df = pivot_df.sort_index(axis=1)

    # Sort rows by taxonomy classification
    contig_taxonomy_map = {row['contig_name']: row['contig_taxonomy'] for row in df.select('contig_name', 'contig_taxonomy').unique().to_dicts()}
    sorted_indices = sorted(pivot_df.index, key=lambda x: (str(contig_taxonomy_map.get(x, '')) == "Unclassified", str(contig_taxonomy_map.get(x, ''))))
    pivot_df = pivot_df.loc[sorted_indices]

    # Reindex row colors to match sorted rows
    contig_colors = contig_colors.reindex(pivot_df.index)
    
    # Reindex column colors to match sorted columns
    treatment_colors = treatment_colors.reindex(pivot_df.columns)
    
    # Create the heatmap without clustering to preserve sort order
    # Calculate dynamic figure size based on the number of rows and columns
    # Aim for roughly 0.3 inches per row and 0.3 inches per column for the heatmap itself
    # Add margins for labels, colorbars, and legends
    num_rows, num_cols = pivot_df.shape
    
    # Heuristic sizing: Adjust base factors and margins as needed for aesthetics
    row_height_factor = 1  # inches per row
    col_width_factor = 1   # inches per column
    height_margin = 3        # inches for x-axis labels, title, legends etc.
    width_margin = 3         # inches for y-axis labels, colorbar, legends etc.

    # Ensure a minimum size for readability, especially with few rows/columns
    fig_height = max(6, num_rows * row_height_factor + height_margin)
    fig_width = max(8, num_cols * col_width_factor + width_margin)

    g = sns.clustermap(
        pivot_df,
        figsize=(fig_width, fig_height), # Use dynamically calculated size
        row_colors=contig_colors,
        col_colors=treatment_colors,
        mask=pivot_df.isna(),
        cmap="coolwarm",
        row_cluster=False,  # Disable row clustering
        col_cluster=False,   # Disable column clustering
        linewidths=0.5, # Add faint lines between cells
        linecolor='lightgrey',
        vmin=0,
        vmax=1
    )
    
    # Modify x-axis labels to show only motifs, not treatments
    motifs = [label.get_text().split("-")[0] for label in g.ax_heatmap.get_xticklabels()]
    g.ax_heatmap.set_xticklabels(motifs)
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    g.ax_cbar.remove()

    # Create legends
    ordered_taxonomies = [contig_taxonomy_map[i] for i in pivot_df.index]
    ordered_taxonomies = pd.unique(ordered_taxonomies)  # preserve order of occurrence
    taxonomy_handles = [Patch(color=lut[t], label=t) for t in ordered_taxonomies]
    treatment_handles = [Patch(color=contigs[0].parent_genome.treatment_color_map[treatment], label=treatment) 
                        for treatment in df['treatment'].unique()]
    
    # Remove automatic figure legends to avoid overlap
    if g.figure.legends:
        for leg in g.figure.legends:
            leg.remove()
    
    # Get legends
    legend1 = g.fig.legend(handles=treatment_handles, title="Treatment")
    legend2 = g.fig.legend(handles=taxonomy_handles, title="Taxonomy")
    fig = g.fig
    
    # Get the figure renderer
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    
    # Get the bounding boxes of the legends in figure coordinates
    legend1_bbox = legend1.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    legend2_bbox = legend2.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    
    # Calculate the left position (centered with legends)
    plot_bbox = g.ax_heatmap.get_position()
    legend_width = legend2_bbox.width
    cbar_width = legend_width * 0.2  # Colorbar width is 1/5 of legend width
    left_pos = plot_bbox.x0 * 0.5  # Position on the left side of the plot
    
    # Position the first legend at the top
    legend1_height = legend1_bbox.height
    legend1_top = plot_bbox.y0 + plot_bbox.height  # Align with top of plot
    legend1_left = left_pos
    legend1.set_bbox_to_anchor([legend1_left, legend1_top], transform=fig.transFigure)
    
    # Position the second legend below the first
    legend2_height = legend2_bbox.height
    legend2_top = legend1_top - legend1_height
    legend2_left = left_pos
    legend2.set_bbox_to_anchor([legend2_left, legend2_top], transform=fig.transFigure)
    
    fig.canvas.draw()

    # Calculate colorbar height and position
    available_height = legend2_top - legend2_height - plot_bbox.y0  - 0.05# Space between bottom of legend2 and bottom of plot
    cbar_top = legend2_top - legend2_height
    
    # Adjust the colorbar axes
    cbar_left = left_pos - (legend_width - cbar_width)  # Center colorbar relative to legends
    cbar_height = min(available_height, 0.5 * plot_bbox.height)  # Limit height to not exceed plot bottom
    cbar_y = cbar_top - cbar_height - 0.05
    
    if cbar_height < 0:
        cbar_height = 0.5
    
    # Add color bar
    cbar_ax = g.fig.add_axes([cbar_left, cbar_y, cbar_width, cbar_height])
    cbar = plt.colorbar(g.ax_heatmap.collections[0], cax=cbar_ax, anchor = (cbar_left, cbar_y), orientation="vertical", label="Mean methylation fraction")
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar.ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
    cbar.ax.tick_params(axis='y', which='major')
    cbar.outline.set_visible(False)
        
    # Redraw the figure to apply changes
    fig.canvas.draw()

    # Set X axis title
    g.ax_heatmap.set_xlabel("Motif")
    
    if contigs[0].is_viral:
        g.ax_heatmap.set_ylabel("Viral contig")
    else:
        g.ax_heatmap.set_ylabel("Bacterial contig")

    g.savefig(f"{contigs[0].parent_genome.output_dir}/contig_motif_heatmap.svg", transparent=True)
    g.savefig(f"{contigs[0].parent_genome.output_dir}/contig_motif_heatmap.pdf")
    
    # Save a CSV of the data
    pivot_df.to_csv(f"{contigs[0].parent_genome.output_dir}/contig_motif_heatmap.csv")

    return df

def plot_contig_motif_heatmap_stats(contigs: list[Contig]):
    data = get_contigs_data(contigs, statistics=True)
    if len(data) == 0:
        return None
    
    # Convert data list to a Polars DataFrame first`1`
    df = pl.DataFrame(data).drop("treatment").unique()

    if df.is_empty():
        print(f"No data to plot in {contigs[0].parent_genome.readable_name}")
        return df # Return the empty dataframe early

    df = df.rename({"means": "Means", "se": "Standard error", "means_pr": "Promoter means", "se_pr": "Promoter standard error", "contig_taxonomy": "Taxonomy"})
    
    # Melt then pivot back
    statistic_columns = ["Means", "Standard error", "Promoter means", "Promoter standard error"]    
    df_long = df.melt(
        id_vars=["motif_string", "contig_name", "Taxonomy"],
        value_vars=statistic_columns,
        variable_name="statistic_key", # New column for the name of the statistic
        value_name="p_value"           # New column for the p-value itself
    ).unique()
    
    pivot_df = df_long.to_pandas().pivot(
        index="contig_name",
        columns=["motif_string", "statistic_key"],
        values="p_value"
    )
    
    # Row colors (Taxonomy) - Use df_long as it contains all original rows
    unique_taxonomies = df_long.get_column("Taxonomy").unique().sort().to_list()
    tax_pal = sns.color_palette(palette="tab20", n_colors=len(unique_taxonomies))
    tax_lut = dict(zip(unique_taxonomies, tax_pal))
    # Map contig names to taxonomy colors using info from df_long
    contig_tax_map = df_long.select("contig_name", "Taxonomy").unique().to_pandas().set_index("contig_name")["Taxonomy"]
    contig_colors = contig_tax_map.map(tax_lut)

    # Column colors (Statistic Key) - Use statistic_columns provided
    unique_statistics = sorted(statistic_columns) # Use the defined list
    stat_pal = sns.color_palette("Paired", n_colors=len(unique_statistics))
    stat_lut = dict(zip(unique_statistics, stat_pal))
    # Create a DataFrame matching the pivot table's columns multi-index
    col_multi_index = pivot_df.columns
    statistic_colors_df = pd.DataFrame(index=col_multi_index)
    statistic_colors_df["Statistic"] = statistic_colors_df.index.get_level_values("statistic_key").map(stat_lut)

    # --- 6. Sorting ---
    # Sort columns by motif_string first, then statistic_key (using the defined order if needed)
    pivot_df = pivot_df.sort_index(axis=1, level=["motif_string", "statistic_key"])

    # Sort rows by taxonomy classification (Unclassified last)
    # Refresh the contig_taxonomy_map from the potentially filtered/aggregated df_long or use contig_tax_map
    contig_taxonomy_map = contig_tax_map.to_dict()
    
    sorted_indices = sorted(
        pivot_df.index,
        key=lambda x: (str(contig_taxonomy_map.get(x, '')) == "Unclassified", str(contig_taxonomy_map.get(x, '')))
    )
    pivot_df = pivot_df.loc[sorted_indices]

    # Reindex row/column colors to match sorted data
    contig_colors = contig_colors.reindex(pivot_df.index)
    statistic_colors_df = statistic_colors_df.reindex(pivot_df.columns)
    
    # Create the heatmap without clustering to preserve sort order
    # Calculate dynamic figure size based on the number of rows and columns
    # Aim for roughly 0.3 inches per row and 0.3 inches per column for the heatmap itself
    # Add margins for labels, colorbars, and legends
    num_rows, num_cols = pivot_df.shape
    
    # Heuristic sizing: Adjust base factors and margins as needed for aesthetics
    row_height_factor = 1  # inches per row
    col_width_factor = 1   # inches per column
    height_margin = 3        # inches for x-axis labels, title, legends etc.
    width_margin = 3         # inches for y-axis labels, colorbar, legends etc.

    # Ensure a minimum size for readability, especially with few rows/columns
    fig_height = max(6, num_rows * row_height_factor + height_margin)
    fig_width = max(8, num_cols * col_width_factor + width_margin)

    g = sns.clustermap(
        pivot_df,
        figsize=(fig_width, fig_height), # Use dynamically calculated size
        row_colors=contig_colors,
        col_colors=statistic_colors_df,
        mask=pivot_df.isna(),
        cmap="viridis",
        row_cluster=False,  # Disable row clustering
        col_cluster=False,   # Disable column clustering
        linewidths=0.5, # Add faint lines between cells
        linecolor='lightgrey',
        vmin=0.01,
        vmax=1,
        norm=LogNorm()
    )

    # Modify x-axis labels to show only motifs, not treatments
    motifs = [label.get_text().split("-")[0] for label in g.ax_heatmap.get_xticklabels()]
    g.ax_heatmap.set_xticklabels(motifs)
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    g.ax_cbar.remove()

    # Ensure correct order and inclusion of all displayed taxonomies/statistics
    # Create legends
    ordered_taxonomies = [contig_taxonomy_map[i] for i in pivot_df.index]
    ordered_taxonomies = pd.unique(ordered_taxonomies)  # preserve order of occurrence
    taxonomy_handles = [Patch(color=tax_lut[t], label=t) for t in ordered_taxonomies]

    # Get unique statistic keys from the columns actually present in the pivot table
    ordered_statistics = pivot_df.columns.get_level_values("statistic_key").unique().tolist()
    statistic_handles = [Patch(color=stat_lut[s], label=s) for s in ordered_statistics if s in stat_lut]
    
    # Remove automatic figure legends to avoid overlap
    if g.figure.legends:
        for leg in g.figure.legends:
            leg.remove()
    
    # Get legends
    legend1 = g.fig.legend(handles=statistic_handles, title="Statistic")
    legend2 = g.fig.legend(handles=taxonomy_handles, title="Taxonomy")
    fig = g.fig
    
    # Get the figure renderer
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    
    # Get the bounding boxes of the legends in figure coordinates
    legend1_bbox = legend1.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    legend2_bbox = legend2.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    
    # Calculate the left position (centered with legends)
    plot_bbox = g.ax_heatmap.get_position()
    legend_width = legend2_bbox.width
    cbar_width = legend_width * 0.2  # Colorbar width is 1/5 of legend width
    left_pos = plot_bbox.x0 * 0.5  # Position on the left side of the plot
    
    # Position the first legend at the top
    legend1_height = legend1_bbox.height
    legend1_top = plot_bbox.y0 + plot_bbox.height  # Align with top of plot
    legend1_left = left_pos
    legend1.set_bbox_to_anchor([legend1_left, legend1_top], transform=fig.transFigure)
    
    # Position the second legend below the first
    legend2_height = legend2_bbox.height
    legend2_top = legend1_top - legend1_height
    legend2_left = left_pos
    legend2.set_bbox_to_anchor([legend2_left, legend2_top], transform=fig.transFigure)
    
    fig.canvas.draw()
    
    # Get the bounding box of the x-axis label in figure coordinates
    xaxis_label_bbox = g.ax_heatmap.xaxis.label.get_window_extent(fig.canvas.get_renderer())
    
    # The bottom of the colorbar should align with the bottom of the x-axis label
    cbar_bottom_y = xaxis_label_bbox.transformed(fig.transFigure.inverted()).y0
    
    # Calculate colorbar height and position
    available_height = legend2_top - legend2_height - cbar_bottom_y  - 0.05# Space between bottom of legend2 and bottom of plot
    cbar_top = legend2_top - legend2_height
    
    # Adjust the colorbar axes
    cbar_left = left_pos - (legend_width - cbar_width)  # Center colorbar relative to legends
    cbar_height = min(available_height, 0.5 * plot_bbox.height)  # Limit height to not exceed plot bottom
    cbar_y = cbar_top - cbar_height - 0.05
    
    if cbar_height < 0:
        cbar_height = 0.5
    
    # Add color bar
    cbar_ax = g.fig.add_axes([cbar_left, cbar_y, cbar_width, cbar_height])
    cbar = plt.colorbar(g.ax_heatmap.collections[0], cax=cbar_ax, anchor = (cbar_left, cbar_y), orientation="vertical", label="P-value")
    cbar.set_ticks([0.01, 0.1, 1])
    cbar.ax.set_yticklabels(['0.01', '0.1', '1'])
    cbar.ax.tick_params(axis='y', which='major')
    cbar.outline.set_visible(False)
        
    # Redraw the figure to apply changes
    fig.canvas.draw()

    # Set X axis title
    g.ax_heatmap.set_xlabel("Motif")
    
    if contigs[0].is_viral:
        g.ax_heatmap.set_ylabel("Viral contig")
    else:
        g.ax_heatmap.set_ylabel("Bacterial contig")

    g.savefig(f"{contigs[0].parent_genome.output_dir}/contig_motif_heatmap_stats.svg", transparent=True)
    g.savefig(f"{contigs[0].parent_genome.output_dir}/contig_motif_heatmap_stats.pdf")
    
    # Save a CSV of the data
    pivot_df.to_csv(f"{contigs[0].parent_genome.output_dir}/contig_motif_heatmap_stats.csv")

    return df


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
    collected_means_df = filtered_means_lazy.collect().sort(sort_cols)

    # Extract numpy arrays for means from the collected DataFrame
    group1_data = collected_means_df.filter(group1_filter).get_column(meth_col_name).to_numpy()
    group2_data = collected_means_df.filter(group2_filter).get_column(meth_col_name).to_numpy()

    # Extract numpy arrays for promoter means
    promoter_means_df = collected_means_df.filter(promoter_filter) # Filter the already collected DF
    group1_data_pr = promoter_means_df.filter(group1_filter).get_column(meth_col_name).to_numpy()
    group2_data_pr = promoter_means_df.filter(group2_filter).get_column(meth_col_name).to_numpy()

    del collected_means_df

    # Raw Data for Standard Error (SE) Distributions

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

    # Calculate SE for all groups
    se_df_all = collected_raw_df.group_by(grouping_cols).agg(se_expr)

    # Extract SE numpy arrays
    g1_se = se_df_all.filter(group1_filter).sort(sort_cols).get_column("se").to_numpy()
    g2_se = se_df_all.filter(group2_filter).sort(sort_cols).get_column("se").to_numpy()

    # Calculate SE for promoter regions by filtering the collected RAW data first
    se_df_pr = collected_raw_df.filter(promoter_filter).group_by(grouping_cols).agg(se_expr)

    # Extract promoter SE numpy arrays
    g1_se_pr = se_df_pr.filter(group1_filter).sort(sort_cols).get_column("se").to_numpy()
    g2_se_pr = se_df_pr.filter(group2_filter).sort(sort_cols).get_column("se").to_numpy()

    del collected_raw_df
    del se_df_all
    del se_df_pr

    means_sig = ks_permutation_test(group1_data, group2_data)
    se_sig = ks_permutation_test(g1_se, g2_se) # Comparing distribution of standard errors

    # Promoter regions
    means_sig_pr = ks_permutation_test(group1_data_pr, group2_data_pr)
    se_sig_pr = ks_permutation_test(g1_se_pr, g2_se_pr) # Comparing distribution of standard errors in promoters
    
    p_values = [means_sig, se_sig, means_sig_pr, se_sig_pr]

    # Create a mapping to keep track of indices and their corresponding keys
    keys = ["means", "se", "means_pr", "se_pr"]
    valid_indices = []
    valid_p_values = []

    # Filter out None values and keep track of valid indices
    for i, p_value in enumerate(p_values):
        if p_value is not None:
            valid_p_values.append(p_value)
            valid_indices.append(i)
    
    adj_pvals = {key: None for key in keys}
    if len(valid_p_values) == 0:
        return adj_pvals
    
    # Apply multipletests only if we have valid p-values
    reject, test_adj_pvals, _, _ = multipletests(valid_p_values, alpha=alpha, method='fdr_bh')
        
    # Update with adjusted p-values where available
    for i, orig_idx in enumerate(valid_indices):
        adj_pvals[keys[orig_idx]] = test_adj_pvals[i]
    
    return adj_pvals


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
        return None

    # Calculate the observed KS statistic (D)
    # ks_2samp returns a result object (or tuple in older scipy)
    # The first element or the .statistic attribute is the D value.
    observed_ks_value = stats.ks_2samp(data1, data2).statistic

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

    return (count_extreme + 1) / (n_permutations + 1)


def get_contigs_data(contigs: list[Contig], statistics):
    # Create a dataframe with columns: contig_name treatment motif_string methylation_fraction contig_taxonomy
    data = []

    for contig in contigs:
        for motif in contig.motifs:            
            
            # For contigs, we don't care about comparing exact positions, but general means across contig so include points not in ever sample and then filter by breadth.
            motif_df = motif.data()
            if motif_df is None:
                continue
            
            motif_df = motif_df.filter(pl.col(motif.meth_type).is_not_null(), pl.col(motif.meth_type).is_not_nan()).collect()

            # Add data with significance information
            i = 0
            treatments = motif_df.get_column("treatment").unique()
            for treatment in treatments:
                # For us to take a mean, there must be at least 20% of positions
                methylation_fractions = motif_df.filter(pl.col("treatment") == treatment).select(motif.meth_type)
                if len(methylation_fractions)/motif.positions.collect().height < 0.2:
                    continue
                
                i += 1
                methylation_fraction_mean = methylation_fractions.mean().item()
                data.append({
                    "contig_name": contig.contig_name,
                    "treatment": treatment,
                    "motif_string": motif.readable_motif,
                    "methylation_fraction": methylation_fraction_mean,
                    "contig_taxonomy": contig.taxonomy("c" if contig.is_viral else "f"),
                })
            
            # Calculate the statistical difference
            if i == 2 and statistics:
                adj_pvals = do_test(motif, treatments)
                
                data[-1].update(adj_pvals)
                data[-2].update(adj_pvals)
    
    return data


def extract_diff_methylated_genes_contigs(df, contigs: list[Contig]):
    """
    Get the DMRs that are significantly different between treatments.
    """
    # Get all the dmrs together
    all_data = []
    for contig in contigs:
        for motif in contig.motifs:
            # Let's do 1) add gene_caller_id if needed
            top_rows = motif.dmr_data.collect(streaming=True).sort("score", descending=True)
            if top_rows.is_empty():
                continue
            
            data = contig.parent_genome.add_gene_caller_id(top_rows.lazy(), include_intergenic=True).collect(streaming=True)
            
            # Add function
            gc = GeneCollection(data.get_column("gene_callers_id").unique().to_list(), contig.parent_genome)
            data = data.lazy().join(gc.get_function(), on="gene_callers_id", how="left")

            # Add nearest gene if not in gene
            data = contig.parent_genome.nearest_gene_to_positions(data).collect()

            # Add function of nearest genes
            gc = GeneCollection(data.get_column("gene_callers_id_start").unique().to_list(), contig.parent_genome)
            data = data.join(gc.get_function().collect(), left_on="gene_callers_id_start", right_on="gene_callers_id", how="left", suffix="_start")
            gc = GeneCollection(data.get_column("gene_callers_id_end").unique().to_list(), contig.parent_genome)
            data = data.join(gc.get_function().collect(), left_on="gene_callers_id_end", right_on="gene_callers_id", how="left", suffix="_end")
            
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
                
            all_data.append(data)
            
    # Write to CSV
    output_file = contig.parent_genome.output_dir / f"{contig.parent_genome.readable_name}_all_top_diff_genes.csv"
    if len(all_data) == 0:
        print(f"No data to write in {contig.parent_genome.readable_name}")
        return
    
    pl.concat(all_data).write_csv(output_file)    


def write_basic_stats_about_contigs(contigs: list[Contig], df):    
    contigs_with_motifs = [contig for contig in contigs if contig.motifs]
    motif_counts = [len(contig.motifs) for contig in contigs_with_motifs]    
    dmr_counts = [sum([motif.dmr_data.unique(subset=["contig", "position", "strand"]).collect(streaming=True).height for motif in contig.motifs]) for contig in contigs_with_motifs]    
    
    # Write those print statements to file
    with open(contigs[0].parent_genome.output_dir / "contig_stats.txt", "w") as f:
        f.write(f"Number of contigs: {len(contigs)}\n")
        f.write(f"Number of contigs with at least one motif: {len(contigs_with_motifs)}\n")
        f.write(f"Average number of motifs per contig: {sum(motif_counts) / len(motif_counts)}\n")
        f.write(f"Average number of DMRs per contig: {sum(dmr_counts) / len(dmr_counts)}\n")
    