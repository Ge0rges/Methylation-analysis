import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pylab as plt
import os

from src.utilities.data_loading import get_coverage
from src.utilities.utils import metagenome_study, read_counts
from matplotlib.colors import LogNorm
from pathlib import Path

sns.set_theme(context="poster", style="white")

readable_sample_name = metagenome_study[0]
barcode_replicate_map = metagenome_study[1]

def plot_coverage():
    """
    Plot the coverage metrics.
    :return: Saves a file
    :rtype: None
    """

    # Load data
    coverage = get_coverage(Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/"))).collect().to_pandas()
    coverage.rename(inplace=True, columns=barcode_replicate_map)
    coverage.rename(inplace=True, columns=readable_sample_name)
    
    # Remove metagenome_assembly row in Genome
    coverage = coverage[coverage['Genome'] != "metagenome_assembly"]

    # Clean the mag names
    coverage['Genome'] = coverage['Genome'].str.title()
    coverage.columns = coverage.columns.str.title()
    coverage['Genome'] = coverage['Genome'].str.replace("_R-Contigs", " sp.")
    
    # Rename column Control_barcode04 to Control and exclude Core-* columns
    coverage.rename(inplace=True, columns={"Control_Barcode04": "Control"})
    coverage = coverage[coverage.columns[~coverage.columns.str.contains("Core-")]]

    # Format it
    coverage.set_index(coverage['Genome'], inplace=True)
    coverage.drop(columns=['Genome'], inplace=True)

    # Mean same samples
    coverage = coverage.groupby(by=coverage.columns, axis=1).sum()

    # Make a plot
    fig, axes = plt.subplots(1, 1, figsize=(15, 15))

    # Change the order of the X axis so that the samples are ordered alphabetically
    coverage = coverage.reindex(sorted(coverage.columns), axis=1)

    # Heatmap with MAG name as rows, samples as columns, and coverage as numbers
    sns.heatmap(coverage[coverage.mean().sort_values().index], annot=True, ax=axes, square=True, cbar_kws={"shrink": 0.5}, fmt=".2f", norm=LogNorm())

    axes.set_title("Coverage Heatmap", fontsize=20)
    axes.set_xlabel("Samples", fontsize=16)
    axes.set_ylabel("MAG names", fontsize=16)

    plt.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../plots/mag_coverage.pdf"), format='pdf', bbox_inches='tight')


def plot_mag_eval():
    """
    Plot MAG contamiantion and redundancy.
    :return: Saves a file
    """
    # Load data
    checkm2_quality = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/mag_eval/checkm2.tsv"), sep="\t", header=0)
    anvio_quality = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/mag_eval/anvio.tsv"), sep="\t", header=0)

    # Merge the quality dataframes
    checkm2_quality['Name'] = checkm2_quality['Name'].str.replace("_r-contigs", "_r")
    quality = pd.merge(checkm2_quality, anvio_quality, left_on="Name", right_on="bins", how="inner")

    # Clean the mag names
    quality['Name'] = quality['Name'].str.title()
    quality['Name'] = quality['Name'].str.replace("_R", " sp.")
    quality.rename(inplace=True, columns={"percent_completion": "Anvi'o Completeness",
                                          "percent_redundancy": "Anvi'o Redundancy",
                                          "Completeness": "CheckM2 Completeness",
                                          "Contamination": "CheckM2 Redundancy"})


    # Format it
    bar_chart_data = pd.melt(quality[["Name", "CheckM2 Completeness", "CheckM2 Redundancy",
                                      "Anvi'o Completeness", "Anvi'o Redundancy"]],
                             id_vars="Name", var_name="Metric", value_name="percent")

    # Define colors and hatches
    colors = sns.color_palette(["lightgreen", "lightcoral", "lightgreen", "lightcoral"])

    # Makea plot
    fig, axes = plt.subplots(1, 1, figsize=(27, 10), layout="constrained")

    #  Bar chart with Completeness and Contamination percentages
    sns.barplot(data=bar_chart_data, x='Name', y='percent', hue="Metric", palette=colors, ax=axes)

    # Apply hatch patterns to each group of four bars
    for i, bar in enumerate(axes.patches):
        group = i // len(bar_chart_data['Name'].unique())
        if 4 > group > 1:
            bar.set_hatch("//")

        elif group == 4 and len(bar_chart_data['Metric'].unique()) > i - len(bar_chart_data['Metric'].unique()) * len(bar_chart_data['Name'].unique()) >= len(bar_chart_data['Metric'].unique()) - 2:
            bar.set_hatch("//")

    # Set the format for the bar labels
    for i in range(0, len(axes.containers)):
        axes.bar_label(axes.containers[i], fontsize=10, fmt="%.2f")

    axes.axhline(y=10, color='red', linestyle='--', linewidth=1)  # 10% line
    axes.axhline(y=90, color='green', linestyle='--', linewidth=1)  # 90% line

    axes.set_title("MAG quality assessment", fontsize=20)
    axes.set_xlabel("MAG names", fontsize=16)
    axes.set_ylabel("Percentage", fontsize=16)
    axes.yaxis.grid(False)
    axes.legend(loc='center right')

    # # Table with the MAG name and Taxonomy classification
    # taxonomy = pd.read_csv("data/mag_eval/gtdbtk.tsv", sep="\t", header=0)
    # table_data = taxonomy[['user_genome', 'classification']]
    # table = axes[2].table(cellText=table_data.values, colLabels=table_data.columns, loc='center')
    # table.set_fontsize(10)
    # table.scale(1.2, 1.2)
    # axes[2].axis('off')  # Hide the axes for the table

    # Display the figure save based on curent file path
    plt.savefig( os.path.join(os.path.dirname(os.path.realpath(__file__)), "../plots/mag_eval.pdf"), format='pdf', bbox_inches='tight')


def read_count_plot():
    """
    Barplot of read counts
    """
    # Some modifications for cores
    barcode_replicate_map["barcode11"] = "top"
    barcode_replicate_map["barcode12"] = "bottom"
    barcode_replicate_map["barcode13"] = "ocean interface"
    barcode_replicate_map["barcode14"] = "middle"

    readable_sample_name["barcode11"] = "IC3-1"
    readable_sample_name["barcode12"] = "IC3-2"
    readable_sample_name["barcode13"] = "IC3-3"
    readable_sample_name["barcode14"] = "IC3-4"
    readable_sample_name["barcode04"] = "Control"

    for key, value in readable_sample_name.items():
        readable_sample_name[key] = value.split("-")[0]

    # Replace the key in read_counts with the value in readable_sample_name
    df = pl.from_dict({"barcode": read_counts.keys(), "Read count": read_counts.values()})
    df = df.with_columns(pl.col("barcode").replace(readable_sample_name).alias("Sample"))

    df = df.with_columns(pl.col("barcode").replace(barcode_replicate_map).alias("Sea-ice horizon"))
    df = df.sort("Sample")
    df = df.to_pandas()

    hue_order = ["top", "middle", "bottom", "ocean interface", "control"]

    fig, axes = plt.subplots(figsize=(20, 10))

    g = sns.barplot(data=df, x="Sample", y="Read count", hue="Sea-ice horizon", hue_order=hue_order, ax=axes)
    plt.title("Read counts per sample")

    new_labels = ['Top', 'Middle', "Bottom", "Ice-Ocean interface", "Control"]
    for t, l in zip(g.legend().texts, new_labels):
        t.set_text(l)

    # Make y axis log
    plt.yscale("log")
    
    plt.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../plots/read_counts.pdf"), format='pdf', bbox_inches="tight")


if __name__ == "__main__":
    plot_mag_eval()
    plot_coverage()
    read_count_plot()
