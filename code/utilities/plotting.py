import math
import matplotlib
import numpy as np
import polars as pl
import seaborn as sns
from utilities.utils import *
import matplotlib.pyplot as plt
from adjustText import adjust_text
import matplotlib.patches as patches

plt.style.use('ggplot')


def plot_all_sources_figure(df: pl.DataFrame, genome_name, heatmap_type="gene", fig_savepath="plots",
                            plot_function=None):
    """
    Plot heatmaps in a figure where each subfigure corresponds to a genome_name, and each subplot within a subfigure corresponds to a source.
    In each heatmap, rows represent functions, columns represent samples, with cell values representing methylation measures.

    Parameters:
    merged_df (dict): Dictionary with genome_name names as keys and respective DataFrame as values.
    genomes (list): List of genome_name names.
    """
    # Create figure for each function
    sources = df.get_column('source').unique()
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
        source_data = df.filter(pl.col('source') == source)
        plot_function(source_data, ax, source)

    # Save to file
    # plt.tight_layout()
    genome_savepath = f"{fig_savepath}/{plot_type}{region_type}_{genome_name}.svg"
    plt.savefig(genome_savepath, format='svg')

    # Close the figure to free up memory
    plt.close(fig)


def plot_heatmap(df: pl.DataFrame, ax, source, fig=None, composite=False):
    if composite:
        df = df.to_pandas().pivot(columns="function", index="comparison", values="score")
    else:
        df = df.to_pandas().pivot(columns="comparison", index="function", values="score")

    if not df.empty:
        # Create color palette
        cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
        cmap.set_bad('lightgray')

        # Plot heatmap
        sns.heatmap(df, cmap=cmap, annot=composite, fmt=".2f", linewidths=1, linecolor='white',
                    cbar_kws={'shrink': 0.8}, ax=ax, square=(not composite), cbar=(not composite))

        # Set title and axis labels
        if composite:
            assert len(df.index) == 1, "Need to modify plotting code to support multiple heatmap rows"

            # Set labels
            ax.set_xlabel("Gene function")
            ax.set_ylabel("")

            # Truncate X-axis labels, move to top and rotate
            x_labels = [truncate_label(lbl.get_text(), max_length=25, max_lines=3) for lbl in ax.get_xticklabels()]
            ax.xaxis.tick_top()
            ax.set_xticklabels(x_labels, rotation=0, ha='center')

            # Create offset transform by 5 points in x direction
            dx = 0 / 72.
            dy = 10 / 72.
            offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

            # apply offset transform to all x ticklabels.
            for label in ax.xaxis.get_majorticklabels():
                label.set_transform(label.get_transform() + offset)

        else:
            # Add labels at the top and bottom of the color bar
            cbar = ax.collections[0].colorbar
            cbar.ax.set_ylabel('Mean modkit score', fontsize=20)

            # Set labels
            ax.set_title(source, fontsize=40)
            ax.set_xlabel('Comparison', fontsize=30)
            ax.set_ylabel('Gene Function', fontsize=30)

            # Truncate Y-axis labels
            num_y_labels = len(ax.get_yticklabels())
            ax_height_in = ax.get_window_extent().height / plt.gcf().dpi
            num_lines = max(1, int(ax_height_in / num_y_labels * 2))  # Adjust the multiplier as necessary

            y_labels = [truncate_label(lbl.get_text(), max_length=70, max_lines=num_lines) for lbl in
                        ax.get_yticklabels()]
            ax.set_yticklabels(y_labels, rotation=0, ha='right', fontsize=20)

            # Orientate the X-axis labels
            x_labels = ax.get_xticklabels()
            ax.set_xticklabels(x_labels, rotation=45, ha='center', fontsize=20)

    else:
        ax.set_title(f"No heatmap data for {source}", fontsize=20)
        if composite:
            ax.set_visible(False)


def plot_gene_methylation_level_figure(df: pl.DataFrame, genome_name, coverage, fig_savepath="plots"):
    methylation_types = list(readable_methylation_name.keys())

    # Create figure and subplots
    n_types = len(methylation_types)
    fig, axes = plt.subplots(n_types * 2, 1, figsize=(20, 5 * n_types), sharex=True, layout="constrained")

    for i, methylation_type in enumerate(methylation_types):
        ax_top = axes[i * 2]
        ax_bottom = axes[i * 2 + 1]

        plot_gene_methylation_level(ax_top, ax_bottom, df, methylation_type)

    cleaned_genome_name = genome_name.title().replace("_R-Contigs", " sp.")
    fig.suptitle(f"Mean gene methylation by type for {cleaned_genome_name}", fontsize=26)

    plt.savefig(f"{fig_savepath}/{genome_name}_{coverage}_gene.svg", format='svg')


def plot_gene_methylation_level(ax_top, ax_bottom, df, methylation_type, composite=False):
    sns_plot_bottom = sns.lineplot(data=df, x="gene_id", y=methylation_type, hue="sample", ax=ax_bottom,
                                   palette=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True))

    if composite:
        ax_bottom.set_title(f"Methylation type: {readable_methylation_name[methylation_type]}", fontsize=20)
        sns_plot_bottom.legend().set_title("Sample")

    else:
        sns_plot_top = sns.lineplot(data=df, x="gene_id", y=methylation_type, hue="sample", ax=ax_top,
                                    palette=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True))
        ax_top.set_title(f"Methylation type: {readable_methylation_name[methylation_type]}", fontsize=20)
        ax_top.set(xlabel="", ylabel="")
        sns_plot_top.legend().set_title("Sample")
        sns_plot_bottom.legend().remove()

        sns.despine(ax=ax_top, bottom=True)
        ax_top.set_xticks([])

        # Diagonal lines for breakage of Y axis
        d = .0025
        kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        kwargs.update(transform=ax_bottom.transAxes)
        ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    ax_bottom.set(xlabel='Gene ID', ylabel=f"Coverage normalized mean methylation fraction")

    try:
        ax_bottom.set_ylim(df.select(pl.min(methylation_type)).item(),
                           df.filter(pl.col(methylation_type) > 0).select(
                               pl.col(methylation_type).quantile(0.95)).item())
        if not composite:
            ax_top.set_ylim(
                bottom=df.filter(pl.col(methylation_type) > 0).select(pl.col(methylation_type).quantile(0.99)).item())

    except ValueError:
        return

    # Sort legend
    legend_ax = ax_bottom if composite else ax_top
    handles, labels = legend_ax.get_legend_handles_labels()
    desired_order = ['Sackhole Top (40 cm)', 'Sackhole Middle (70 cm)', 'Sackhole Bottom (160 cm)']
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: desired_order.index(x[1]))
    handles, labels = zip(*sorted_handles_labels)
    legend_ax.legend(handles, labels)


def plot_mean_gene_methylation_level(ax, df):
    plot = sns.lineplot(data=df, x="gene_id", y="total_methylation", hue="sample", style="sample", ax=ax,
                        palette=["#3982b8", "#6abce2", "#235284"], alpha=0.5)
    ax.set_title(f"Mean methylation level by brine horizon", fontsize=18)
    plot.legend().set_title("Sample")

    ax.set(xlabel='Gene ID', ylabel=f"Normalized methylation fraction")

    ax.set_ylim(0, 1)

    # Sort legend
    handles, labels = ax.get_legend_handles_labels()
    desired_order = ['Sackhole Top (40 cm)', 'Sackhole Middle (70 cm)', 'Sackhole Bottom (160 cm)']
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: desired_order.index(x[1]))
    handles, labels = zip(*sorted_handles_labels)
    ax.legend(handles, labels)


def plot_gene_methylation_level_diff(ax, df, diff_string):
    plot = sns.lineplot(data=df, x="gene_id", y="methylation_level", hue="methylation_type", style="methylation_type",
                        ax=ax,
                        palette=sns.color_palette("colorblind", n_colors=df.get_column("methylation_type").n_unique()),
                        alpha=0.5)

    ax.set_title(f"Mean methylation difference by methylation type: {diff_string}", fontsize=18)
    plot.legend().set_title("Methylation type")

    ax.set(xlabel='Gene ID', ylabel=f"Normalized methylation fraction")

    ax.set_ylim(-0.5, 0.5)


def annotate_meth_level_with_score_function_table(annotate_ax, table_ax, df: pl.DataFrame, function_source: str, score_col: str, comparison: str, show_significance=True):
    # If it's None, pick the best function
    if function_source is None:
        filtered_df = df.select("test_result", "source").filter(pl.col("test_result") == True)
        grouped_counts = filtered_df.group_by("source").count()
        func_source_df = grouped_counts.top_k(1, by="count").select("source")
        if func_source_df.height > 0:
            function_source = func_source_df.item()
        else:
            function_source = ""

    # Adding annotations for each gene_id
    texts = []
    df = df.filter(pl.col("comparison").eq(comparison) & pl.col(score_col).is_not_nan() & pl.col("source").eq(function_source))
    df = df.select("function", score_col, "gene_id", "test_result")
    table_data = df.select('function', score_col, "test_result")
    if not show_significance:
        table_data = df.select('function', score_col)
    
    table_data = table_data.unique().top_k(10, by=score_col).to_numpy()

    if len(table_data) == 0:
        return

    for i, row in enumerate(table_data):
        genes = df.filter(pl.col('function').eq(row[0]) & pl.col(score_col).eq(row[1])).get_column("gene_id").unique().to_list()
        for gene in genes:
            max_y = -np.inf
            for line in annotate_ax.lines:
                x_data = line.get_xdata()
                y_data = line.get_ydata()

                if len(x_data) == 0:
                    continue

                # Find the closest index to the desired x value
                idx = (np.abs(x_data - gene)).argmin()

                # Update max_y if the current y value is greater
                max_y = max(max_y, y_data[idx])
            
            label = str(i+1) + "*" if show_significance and row[2] else str(i+1)
            texts.append(annotate_ax.text(gene, max_y, label, fontsize=12, color='red'))

    # Add texts
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='r'), ax=annotate_ax, min_arrow_len=0)

    # Creating a table to show function and score
    if table_ax is not None:
        for i in table_data:
            i[0] = truncate_label(i[0], max_length=50, max_lines=2)
            i[1] = f"{i[1]:.1f}"
            if show_significance:
                i[2] = str(i[2])

        colLabels = [function_source.replace("_", " "), f"{'Rao' if score_col else 'Modkit'} score"]
        colLabels = colLabels + ["Significant?"] if show_significance else colLabels
        table = table_ax.table(cellText=table_data,
                               colLabels=colLabels,
                               rowLabels=[str(i + 1) for i in range(len(table_data))],
                               loc='center',
                               cellLoc='center',
                               colColours=["lightblue"] * len(colLabels))

        table.auto_set_column_width([0, 1, 2])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(2,  2.3)

        comp_str = comparison.replace("_vs_", " and ").replace("_", ", ")
        table_ax.set_title(f"Top {len(table_data)} differentially methylated {function_source.replace('_', ' ')} \n functions between {comp_str}", fontsize=18)
