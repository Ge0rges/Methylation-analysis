import math
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list


plt.style.use('ggplot')


def scatterplot_methylation(df):
    df.drop_duplicates(subset=['sample', 'name'], inplace=True)
    df['fractions_C'] = df["fractions_a"] + df["fractions_h"]
    sns.scatterplot(data=df, x='fractions_C', y='fractions_m', hue='sample', alpha=0.5)
    plt.xlabel("Cytosine methylation")
    plt.ylabel("Adenine methylation")
    plt.title("Methylation percentage by gene")
    plt.show()


def barplot_methylation_level(df):
    df.drop_duplicates(subset=['sample', 'name'], inplace=True)

    # Create a new column that is the sum of all columns matching fractions_*
    # Calculate total methylation for each sample
    df['total_methylation'] = df.filter(like='counts_').sum(axis=1)
    total_methylation_by_sample = df.groupby('sample')['total_methylation'].sum().reset_index()

    # Plot the scatterplot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=total_methylation_by_sample, x='sample', y='total_methylation', alpha=0.5)
    plt.title("Sum of methylation percentages for all genes")

    plt.yscale('log')
    plt.show()


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
    htype = "gene" if "gene" in heatmap_type else "nucleotide"
    fig.suptitle(f'Heatmaps for {genome_name} by {htype}', fontsize=60)

    # Create subplots for each source within a genome
    for index, source in enumerate(sorted(sources), start=1):
        ax = fig.add_subplot(num_rows, num_cols, index)

        # Filter data for the current genome and source
        source_data = df[df['source'] == source]
        heatmap_data = source_data.pivot_table(index='function', columns='sample', values='fractions_m')

        # Plot it
        plot_heatmap(heatmap_data, ax, source, index)

    # Save to file
    # plt.tight_layout()
    genome_savepath = f"plots/{heatmap_type}_{genome_name}.pdf"
    plt.savefig(genome_savepath, format='pdf', bbox_inches='tight')

    # Close the figure to free up memory
    plt.close(fig)


def plot_heatmap(heatmap_data, ax, source, index):
    if not heatmap_data.empty:
        # Create a temporary DataFrame for clustering
        temp_heatmap_data = heatmap_data.fillna(0)

        # Check if columns have non-identical data
        if not all(temp_heatmap_data.nunique() <= 1):
            linkage_result = linkage(temp_heatmap_data.T, method='average')
            col_order = leaves_list(linkage_result)
            heatmap_data = heatmap_data.iloc[:, col_order]  # Apply the order to the original DataFrame

        # Create color palette
        cmap = sns.color_palette("coolwarm", as_cmap=True)
        cmap.set_bad('lightgray')

        # Plot heatmap
        sns.heatmap(heatmap_data, cmap=cmap, annot=False, fmt=".2f", linewidths=1, linecolor='white',
                    cbar_kws={'shrink': 0.8}, ax=ax, square=True)

        # Add labels at the top and bottom of the color bar
        cbar = ax.collections[0].colorbar
        cbar.ax.set_title("A is methylated", pad=10, fontsize=20)
        cbar.ax.set_ylabel('Mean methylation Level %', fontsize=20)
        cbar.ax.set_xlabel("B is methylated", labelpad=10, fontsize=20)

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
