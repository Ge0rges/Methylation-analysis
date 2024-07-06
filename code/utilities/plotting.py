import math
import seaborn
import vaex
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from utilities.utils import truncate_label, PlotMarker, group_methyl_data_by_genes

plt.style.use('ggplot')


def plot_pairwise_results(results, genome):
    # Create a list of sample combinations
    sample_combinations = results.keys()
    samples = list(set([sample for sample_pair in sample_combinations for sample in sample_pair]))

    # Create an empty DataFrame
    df = pd.DataFrame(columns=samples, index=samples)

    # Fill in the DataFrame with the results
    for sample1, sample2 in sample_combinations:
        df.loc[sample1, sample2] = results[(sample1, sample2)]
        df.loc[sample2, sample1] = results[(sample1, sample2)]

    # Replace the boolean values with 1s and 0s
    df = df.replace({np.nan: False}).astype(int)

    # Sort the datframe by row names
    df = df.sort_index(axis=0).sort_index(axis=1)

    # Plotting the heatmap
    sns.heatmap(df, annot=True, cmap='binary', cbar=False, linewidths=.5)
    plt.title(f"Pairwise similarity of {genome}")
    plt.tight_layout()
    plt.show()


def plot_all_sources_heatmaps(df, genome_name, heatmap_type="gene", fig_savepath="plots"):
    """
    Plot heatmaps in a figure where each subfigure corresponds to a genome, and each subplot within a subfigure corresponds to a source.
    In each heatmap, rows represent functions, columns represent samples, with cell values representing methylation measures.

    Parameters:
    df (dict): Dictionary with genome names as keys and respective DataFrame as values.
    genomes (list): List of genome names.
    """
    # Create figure for each genome
    sources = df['source'].unique()
    number_of_sources = len(sources)

    # Calculate the number of rows and columns for a square-like layout
    num_cols = math.ceil(math.sqrt(number_of_sources))
    num_rows = math.ceil(number_of_sources / num_cols)

    # Figure size based on mumber of sources (1 per column)
    fig = plt.figure(figsize=(40 * num_cols, 40 * num_rows), layout="constrained")

    # Set a title
    htype = "gene" if "dmr_by_gene" in heatmap_type else "nucleotide"
    fig.suptitle(f'Heatmaps for {genome_name} by {htype}', fontsize=60)

    # Create subplots for each source within a genome
    for index, source in enumerate(sorted(sources), start=1):
        ax = fig.add_subplot(num_rows, num_cols, index)

        # Filter data for the current source source and aggregate
        source_data = df[df['source'] == source]
        aggregated_data = source_data.groupby(['function', 'comparison']).agg({'score': 'mean'}).reset_index()

        plot_heatmap(aggregated_data.pivot(index='function', columns='comparison', values='score'), ax, source, index)

    # Save to file
    # plt.tight_layout()
    genome_savepath = f"{fig_savepath}/{heatmap_type}_{genome_name}.pdf"
    plt.savefig(genome_savepath, format='pdf', bbox_inches='tight')

    # Close the figure to free up memory
    plt.close(fig)


def plot_heatmap(heatmap_data, ax, source, index):
    if not heatmap_data.empty:
        # Create color palette
        cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)  #sns.color_palette("rocket", as_cmap=True)
        cmap.set_bad('lightgray')

        # Plot heatmap
        sns.heatmap(heatmap_data, cmap=cmap, annot=False, fmt=".2f", linewidths=1, linecolor='white',
                    cbar_kws={'shrink': 0.8}, ax=ax, square=True)

        # Add labels at the top and bottom of the color bar
        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel('Mean modkit score', fontsize=20)

        # Set title and axis labels
        ax.set_title(source, fontsize=40)
        ax.set_xlabel('Comparison', fontsize=30)
        ax.set_ylabel('Gene Function' if index == 1 else '', fontsize=30)

        # Truncate Y-axis labels
        num_y_labels = len(ax.get_yticklabels())
        ax_height_in = ax.get_window_extent().height / plt.gcf().dpi
        num_lines = max(1, int(ax_height_in / num_y_labels * 2))  # Adjust the multiplier as necessary

        y_labels = [truncate_label(lbl.get_text(), max_lines=num_lines) for lbl in ax.get_yticklabels()]
        ax.set_yticklabels(y_labels, rotation=0, ha='right', fontsize=20)

        # Orientate the X-acis labels
        x_labels = ax.get_xticklabels()
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=20)

    else:
        ax.set_title(f"No Data for {source}", fontsize=60)


def plot_methylation_levels_per_base(df, genome_name, coverage, fig_savepath="plots"):
    df.iloc[:, 0] = pd.factorize(df['name'])[0]
    methylation_types = df.columns[1:-1]

    # Normalize counts by coverage
    samples = df['sample'].unique()

    # Iterate through each methylation type and plot its values
    for methylation_type in methylation_types:

        plot_markers = [
            PlotMarker(shape='filled-circle', radius=1, color=[1, 0, 0]),
            PlotMarker(shape='filled-circle', radius=1, color=[0, 1, 0]),
            PlotMarker(shape='filled-circle', radius=1, color=[0, 0, 1]),
            PlotMarker(shape='filled-circle', radius=1, color=[1, 0.4470, 0.7410]),
            PlotMarker(shape='filled-circle', radius=1, color=[0, 0.4470, 1]),
            PlotMarker(shape='filled-circle', radius=1, color=[1, 0, 1]),
            PlotMarker(shape='filled-circle', radius=1, color=[0.2, 1, 0.2]),
            PlotMarker(shape='filled-circle', radius=1, color=[0.2, 0.2, 0.2]),
            PlotMarker(shape='filled-circle', radius=1, color=[0.5, 1, 0.2]),
            PlotMarker(shape='filled-circle', radius=1, color=[0.5, 1, 0.9]),
            PlotMarker(shape='filled-circle', radius=1, color=[1, 0.6, 0.2]),
            PlotMarker(shape='filled-circle', radius=1, color=[0.6, 0.2, 1]),
            PlotMarker(shape='filled-circle', radius=1, color=[0.2, 0.6, 1]),
            PlotMarker(shape='filled-circle', radius=1, color=[0.2, 1, 0.6]),
            PlotMarker(shape='filled-circle', radius=1, color=[0.6, 1, 0.2]),
            PlotMarker(shape='filled-circle', radius=1, color=[1, 0.2, 0.6]),
            PlotMarker(shape='filled-circle', radius=1, color=[0.6, 1, 0.6]),
            PlotMarker(shape='filled-circle', radius=1, color=[0.6, 0.6, 1]),
        ]

        # Plot all samples for this methylation type using Matplotlib
        for i, sample in enumerate(samples):
            df_i = vaex.from_arrays(x=np.ascontiguousarray(df[df['sample'] == sample]['name']),
                                    y=np.ascontiguousarray(df[df['sample'] == sample][methylation_type] / df[df['sample'] == sample][methylation_types].sum(axis=1))
                                    )
            df_i.my_viz.my_scatter(df_i.x, df_i.y, plot_markers[i])

        plt.title(f"{genome_name} - {coverage} - {methylation_type}")

        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=marker.color, markersize=5, linestyle=None) for marker in
                   plot_markers[:len(samples)]]
        plt.legend(handles, samples, title="Samples")

        plt.savefig(f"{fig_savepath}/{genome_name}_{coverage}_{methylation_type}_perbase.pdf", format='pdf')


def plot_methylation_levels_by_gene(df, genes, genome_name, coverage, fig_savepath="plots"):
    df = group_methyl_data_by_genes(pl.from_pandas(df).lazy(), pl.from_pandas(genes).lazy()).collect()

    for methylation_type in df.columns[1:-5]:
        # Create figure
        plots = [seaborn.boxplot, seaborn.violinplot, seaborn.boxenplot]

        for plot in plots:
            plt.figure(figsize=(40, 10))

            plot(df, x="range_id", y=methylation_type, hue="sample")

            plt.title(f"{genome_name} - {coverage} - {methylation_type}")

            plt.tight_layout()
            plt.savefig(f"{fig_savepath}/{genome_name}_{coverage}_{methylation_type}_{plot.__name__}.pdf", format='pdf')


# sort then take first 1000
def plot_methylation_levels_by_group(df, genome_name, coverage, fig_savepath="plots"):
    df['contig'] = df['name'].apply(lambda x: x.split('|')[0])
    df['start'] = pd.to_numeric(df['name'].apply(lambda x: x.split('|')[2]), downcast="integer")
    df['stop'] = pd.to_numeric(df['name'].apply(lambda x: x.split('|')[3]), downcast="integer")

    df.sort_values(by=["contig", "start", "stop"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Aggregate every 1000th rows on the same contig
    df['group'] = df.groupby('contig').cumcount() // 1000

    for methylation_type in df.columns[1:-5]:
        # Dataset wide mean and standard deviation
        df_mean = df.groupby("sample")[methylation_type].mean()
        df_std = df.groupby("sample")[methylation_type].std()

        # Group mean and standard deviation
        group_stats = df.groupby(['group', 'sample'])[methylation_type].agg(['mean', 'std']).reset_index()
        group_stats['sample_mean'] = group_stats['sample'].map(df_mean)
        group_stats['sample_std'] = group_stats['sample'].map(df_std)

        # Define the conditions for including groups
        n = 0.5
        conditions = (
                (group_stats['mean'] > n * group_stats['sample_mean']) |
                (group_stats['std'] > n * group_stats['sample_std']) |
                (group_stats['mean'] < group_stats['sample_mean'] / n) |
                (group_stats['std'] < group_stats['sample_std'] / n)
        )

        groups_to_include = group_stats[conditions]['group']
        df = df[df['group'].isin(groups_to_include)]

        # Create figure
        plots = [seaborn.boxplot, seaborn.violinplot, seaborn.boxenplot]

        for plot in plots:
            plt.figure(figsize=(40, 10))

            plot(df, x="group", y=methylation_type, hue="sample")

            plt.title(f"{genome_name} - {coverage} - {methylation_type}")

            plt.tight_layout()
            plt.savefig(f"{fig_savepath}/{genome_name}_{coverage}_{methylation_type}_{plot.__name__}.pdf", format='pdf')
