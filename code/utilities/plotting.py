import math
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from utilities.utils import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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


def plot_all_sources_figure(df, genome_name, heatmap_type="gene", fig_savepath="plots", plot_function=None):
    """
    Plot heatmaps in a figure where each subfigure corresponds to a genome_name, and each subplot within a subfigure corresponds to a source.
    In each heatmap, rows represent functions, columns represent samples, with cell values representing methylation measures.

    Parameters:
    df (dict): Dictionary with genome_name names as keys and respective DataFrame as values.
    genomes (list): List of genome_name names.
    """
    # Create figure for each function
    sources = df['source'].unique()
    number_of_sources = len(sources)

    # Calculate the number of rows and columns for a square-like layout
    num_cols = max(1, math.ceil(math.sqrt(number_of_sources)))
    num_rows = math.ceil(number_of_sources / num_cols)

    # Figure size based on mumber of sources (1 per column)
    fig = plt.figure(figsize=(70 * num_cols, 70 * num_rows), layout="constrained")

    # Set a title
    region_type = "gene" if "dmr_by_gene" in heatmap_type else "nucleotide"
    plot_type = "Heatmap" if plot_function == plot_heatmap else "Unknown"
    genome_readable = genome_name.title().replace("_R-Contigs", " sp.")
    fig.suptitle(f'{plot_type} for {genome_readable} by {region_type}', fontsize=60)

    # Create subplots for each source within a genome_name
    for index, source in enumerate(sorted(sources), start=1):
        ax = fig.add_subplot(num_rows, num_cols, index)

        # Filter data for the current source source and aggregate
        source_data = df[df['source'] == source]
        plot_function(source_data, ax, source)

    # Save to file
    # plt.tight_layout()
    genome_savepath = f"{fig_savepath}/{plot_type}{region_type}_{genome_name}.svg"
    plt.savefig(genome_savepath, format='svg')

    # Close the figure to free up memory
    plt.close(fig)


def plot_heatmap(df, ax, source, horizontal=False):
    df = df.groupby(['function', 'comparison']).agg({'score': 'mean'}).reset_index()
    if horizontal:
        df = df.pivot(index='comparison', columns='function', values='score')
    else:
        df = df.pivot(index='function', columns='comparison', values='score')

    if not df.empty:
        # Create color palette
        cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)  #sns.color_palette("rocket", as_cmap=True)
        cmap.set_bad('lightgray')

        # Plot heatmap
        #q = df.stack().quantile(0.5)
        #df = df[df.ge(q).any(axis=1)]
        sns.heatmap(df, cmap=cmap, annot=False, fmt=".2f", linewidths=1, linecolor='white',
                    cbar_kws={'shrink': 0.8}, ax=ax, square=True)

        # Add labels at the top and bottom of the color bar
        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel('Mean modkit score', fontsize=20)

        # Set title and axis labels
        ax.set_title(source, fontsize=40)
        ax.set_xlabel('Comparison', fontsize=30)
        ax.set_ylabel('Gene Function', fontsize=30)

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
        ax.set_title(f"No heatmap data for {source}", fontsize=20)


def plot_gene_methylation_level_figure(df, genome_name, coverage, fig_savepath="plots"):
    methylation_types = df.columns[1:4]

    # Create figure and subplots
    n_types = len(methylation_types)
    fig, axes = plt.subplots(n_types*2, 1, figsize=(20, 5 * n_types), sharex=True, layout="constrained")

    for i, methylation_type in enumerate(methylation_types):
        ax_top = axes[i*2]
        ax_bottom = axes[i*2+1]

        plot_gene_methylation_level(ax_top, ax_bottom, df, methylation_type)

    cleaned_genome_name = genome_name.title().replace("_R-Contigs", " sp.")
    fig.suptitle(f"Mean gene methylation by type for {cleaned_genome_name}", fontsize=26)

    plt.savefig(f"{fig_savepath}/{genome_name}_{coverage}_gene.svg", format='svg')


def plot_gene_methylation_level(ax_top, ax_bottom, df, methylation_type):
    sns_plot_top = sns.lineplot(data=df, x="gene_id", y=methylation_type, hue="sample", ax=ax_top)
    sns_plot_bottom = sns.lineplot(data=df, x="gene_id", y=methylation_type, hue="sample", ax=ax_bottom)

    ax_top.set_title(f"Methylation type: {readable_methylation_name[methylation_type]}", fontsize=20)

    ax_top.set(ylabel="")
    ax_bottom.set(xlabel='Gene ID', ylabel=f"Number of observations")

    sns_plot_top.legend().set_title("Sample")
    sns_plot_bottom.legend().remove()

    try:
        ax_top.set_ylim(bottom=df.filter(pl.col(methylation_type) > 0).select(pl.col(methylation_type).quantile(0.99)).to_numpy()[0][0])
        ax_bottom.set_ylim(df.select(pl.min(methylation_type)).to_numpy()[0][0], 
                           df.filter(pl.col(methylation_type) > 0).select(pl.col(methylation_type).quantile(0.95)).to_numpy()[0][0])

    except ValueError:
        return

    sns.despine(ax=ax_top, bottom=True)
    
    # Diagonal lines for breakage of Y axis
    d = .0025
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    # Sort legend
    handles, labels = ax_top.get_legend_handles_labels()
    desired_order = ['Sackhole Top (40 cm)', 'Sackhole Middle (70 cm)', 'Sackhole Bottom (160 cm)']
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: desired_order.index(x[1]))
    handles, labels = zip(*sorted_handles_labels)
    ax_top.legend(handles, labels)


def annotate_heatmap_to_meth_level(fig, ax_top, ax_heatmap, composite_data, methylation_type):

    for i in range(composite_data.height):
        # Get the coordinates on the respective plots plot
        line_x, line_y = ax_top.transData.transform((composite_data.item(i, 'gene_id'), composite_data.item(i, methylation_type)))

        x_index = composite_data.get_column('function').unique().to_list().index(composite_data.item(i, "function"))
        y_index = composite_data.get_column('comparison').unique().to_list().index(composite_data.item(i, "comparison"))
        
        #print([i for i, x in enumerate(composite_data.get_column('function').to_list()) if x == composite_data.item(i, "function")])

        heatmap_x, heatmap_y = ax_heatmap.transData.transform((x_index,  y_index))

        # Create the arrow
        arrow = patches.FancyArrowPatch((heatmap_x, heatmap_y), (line_x, line_y),
                                        transform=fig.transFigure, color='red',
                                        arrowstyle='->', mutation_scale=15)
        fig.patches.append(arrow)
