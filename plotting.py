import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import truncate_label

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
        cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)#sns.color_palette("rocket", as_cmap=True)
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
        y_labels = [truncate_label(lbl.get_text()) for lbl in ax.get_yticklabels()]
        ax.set_yticklabels(y_labels, rotation=0, ha='right', fontsize=20)

        # Orientate the X-acis labels
        x_labels = ax.get_xticklabels()
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=20)

    else:
        ax.set_title(f"No Data for {source}", fontsize=60)


def plot_methylation_levels_per_base(df, genome_name, fig_savepath="plots"):
    """
    Plot methylation levels per base for a genome.

    Parameters:
    df (pandas.DataFrame): DataFrame with methylation data.
    genome_name (str): Name of the genome.
    fig_savepath (str): Path to save the figure.
    """
    # Get the list of methylation types (excluding the "no methylation" count column)
    methylation_types = df.columns[1:-1]

    # Set up subplots
    num_subplots = len(methylation_types)
    fig, axs = plt.subplots(num_subplots, 1, figsize=(10, 5 * num_subplots))

    # Iterate through each methylation type and plot its values
    for i, methylation_type in enumerate(methylation_types):
        ax = axs[i]
        ax.set_title(methylation_type)

        # Plot all samples for this methylation type using Seaborn
        sns.lineplot(data=df, x='name', y=methylation_type, hue='sample', ax=ax)

        ax.set_xlabel('Genomic Position')
        ax.set_ylabel('Count')

    plt.tight_layout()
    plt.show()

    return
