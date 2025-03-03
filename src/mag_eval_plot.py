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

def plot_coverage(coverm_path, output_dir):
    """
    Plot the coverage metrics.
    :return: Saves a file
    :rtype: None
    """

    # Load data
    coverage = get_coverage(coverm_path).collect().to_pandas()
    coverage.rename(inplace=True, columns=barcode_replicate_map)
    coverage.rename(inplace=True, columns=readable_sample_name)
    
    # Remove metagenome_assembly row in Genome
    coverage = coverage[coverage['Genome'] != "metagenome_assembly"]

    # Clean the mag names
    coverage['Genome'] = coverage['Genome'].str.split('__bin', expand=True)[0].str.split("__", expand=True)[:][0] + "__" + coverage['Genome'].str.split('__bin', expand=True)[0].str.split("__", expand=True)[:][1]
    coverage.columns = coverage.columns.str.title()
    
    # Rename column Control_barcode04 to Control and exclude Core-* columns
    coverage.rename(inplace=True, columns={"Control_Barcode04": "Control"})
    coverage = coverage[coverage.columns[~coverage.columns.str.contains("Core-")]]

    # Format it
    coverage.set_index(coverage['Genome'], inplace=True)
    coverage.drop(columns=['Genome'], inplace=True)

    # Mean same samples
    coverage = coverage.groupby(by=coverage.columns, axis=1).sum()

    # Make a plot
    _, axes = plt.subplots(1, 1, figsize=(60, 60), layout="constrained")

    # Change the order of the X axis so that the samples are ordered alphabetically
    coverage = coverage.reindex(sorted(coverage.columns), axis=1)

    # Heatmap with MAG name as rows, samples as columns, and coverage as numbers
    sns.heatmap(coverage[coverage.mean().sort_values().index], annot=True, ax=axes, square=True, cbar_kws={"shrink": 0.5}, fmt=".2f", norm=LogNorm())

    axes.set_title("Coverage Heatmap", fontsize=20)
    axes.set_xlabel("Samples", fontsize=16)
    axes.set_ylabel("MAG names", fontsize=16)

    plt.savefig(output_dir / "mag_coverage.pdf", format='pdf')


def plot_mag_qual(checkm_tsv, output_dir):
    """
    Plot MAG contamiantion and redundancy.
    :return: Saves a file
    """
    # Load data
    quality = pd.read_csv(checkm_tsv, sep="\t", header=0)
    
    quality = quality[quality['Name'] != "metagenome_assembly"]
    quality = quality[quality['Name'] != "viruses"]

    # Clean the mag names
    quality['Name'] = quality['Name'].str.split('__bin', expand=True)[0].str.title()
    quality.rename(inplace=True, columns={"Completeness": "CheckM2 Completeness", "Contamination": "CheckM2 Redundancy"})

    # Format it
    bar_chart_data = pd.melt(quality[["Name", "CheckM2 Completeness", "CheckM2 Redundancy"]], id_vars="Name", var_name="Metric", value_name="percent")

    # Define colors and hatches
    colors = sns.color_palette(["lightgreen", "lightcoral"])

    # Makea plot
    _, axes = plt.subplots(1, 1, figsize=(90, 29), layout="constrained")

    #  Bar chart with Completeness and Contamination percentages
    sns.barplot(data=bar_chart_data, x='Name', y='percent', hue="Metric", palette=colors, ax=axes)

    # Set the format for the bar labels
    for i in range(0, len(axes.containers)):
        axes.bar_label(axes.containers[i], fontsize=10, fmt="%.2f")

    axes.axhline(y=10, color='red', linestyle='--', linewidth=1)  # 10% line
    axes.axhline(y=30, color='orange', linestyle='--', linewidth=1)  # 30% line
    axes.axhline(y=90, color='green', linestyle='--', linewidth=1)  # 90% line

    axes.set_title("MAG quality assessment", fontsize=20)
    axes.set_xlabel("MAG names", fontsize=16)
    axes.set_ylabel("Percentage", fontsize=16)
    axes.yaxis.grid(False)
    axes.legend(loc='center right')

    # Display the figure save based on curent file path
    plt.savefig(output_dir / "mag_eval.pdf", format='pdf')


def plot_microbemod(microbemod_tsv, output_dir):
    """
    Barplot by of number of genes by RM type, and enzyme type (methyltransferase, restriction enzyme)
    MicrobeMod TSV Example:
    Operon	Gene	System Type	Gene type	HMM	Evalue	REBASE homolog	Homolog identity(%)	Homolog methylation	Homolog motif
    Singleton #1	c_000000073274_55	RM_Type_II	MT	Type_II_MTases_FAM_3	9e-104	M.PgiNP1I	99.438	m6A	GANTC
    """
    # Load data
    microbemod = pd.read_csv(microbemod_tsv, sep="\t", header=0)

    # Keep only Gene type which are RE, MT, or IIG
    microbemod = microbemod[microbemod['Gene type'].str.contains("MT|RE")]        
    microbemod['Gene type'] = microbemod['Gene type'].str.replace("MT", "Methyltransferase").str.replace("RE", "Restriction enzyme")#.str.replace("IIG", "Type IIG RM system")
    microbemod['System type'] = microbemod['System Type'].str.split("_", expand=True)[2]
    microbemod['System type'] = pd.Categorical(microbemod['System type'], ['I','II', 'III','IV'])

    # Make a plot
    _, axes = plt.subplots(figsize=(10, 20), layout="constrained")

    # Bar chart with number of genes by RM type, and enzyme type (methyltransferase, restriction enzyme)
    sns.histplot(data=microbemod, x='System type', hue="Gene type", ax=axes, stat="count", discrete=True, multiple="dodge", shrink=0.8)

    # Set the format for the bar labels
    for i in range(0, len(axes.containers)):
        axes.bar_label(axes.containers[i], fmt="%d")

    axes.set_title("Count of RM genes in the metagenome")
    axes.set_xlabel("System type")
    axes.set_ylabel("Count")
    axes.yaxis.grid(False)

    # Display the figure save based on curent file path
    plt.savefig(output_dir / "microbemod.pdf")
    

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
    read_count_plot()
