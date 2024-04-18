import math
import seaborn as sns
import matplotlib.pyplot as plt


plt.style.use('ggplot')


def plot_all_sources_heatmaps(df, genome_name, heatmap_type="gene"):
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
    fig = plt.figure(figsize=(40 * num_cols, 30 * num_rows), layout="constrained")

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
    genome_savepath = f"plots/{heatmap_type}_{genome_name}.pdf"
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


def truncate_label(label, max_length=70):
    """Truncate labels to a maximum length, adding an ellipsis if truncated."""
    if len(label) > max_length:
        return label[:max_length] + '...'
    return label
