import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pylab as plt
import os

from src.utilities.data_loading import get_coverage
from src.utilities.utils import read_counts
from matplotlib.colors import LogNorm

sns.set_theme(context="poster", style="white")


def plot_coverage(coverm_path, output_dir, treatment_name_map, barcode_treatment_map, treatment_order_map):
    """
    Plot the coverage metrics.
    :return: Saves a file
    :rtype: None
    """

    # Load data using Polars
    coverage = get_coverage(coverm_path).collect()
    coverage = coverage.unpivot(index="Genome", variable_name="sample", value_name="coverage")
    
    # Get treatments
    coverage = coverage.with_columns(pl.col("sample").replace(barcode_treatment_map).replace(treatment_name_map).alias("Treatment"))
    coverage = coverage.group_by("Treatment", "Genome").agg(pl.col("coverage").mean())
    
    # Exclude columns that contain "core" and metagenome
    coverage = coverage.filter(~pl.col("Genome").str.contains("metagenome"), ~pl.col("Treatment").str.contains("core"))

    # Change the order of the X axis so that the samples are ordered alphabetically
    coverage = coverage.pivot(on="Treatment", values="coverage", index="Genome").to_pandas()
    
    # Set index and reorder
    coverage = coverage.set_index("Genome", drop=True)
    coverage = coverage.reindex([k for k in sorted(treatment_order_map, key=lambda x: x[1]) if k in coverage.columns], axis="columns")

    # Sort by coverage
    coverage = coverage.sort_values(by=[coverage.columns[i] for i in [1,2,3]], ascending=False)
    
    # Create a heatmap with MAG names as rows, samples as columns, and coverage as numbers
    _, ax = plt.subplots(figsize=(30, 60), constrained_layout=True)
    sns.heatmap(
        coverage,
        annot=True,
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.5},
        fmt=".2f",
        norm=LogNorm()
    )
    
    ax.set_title("Coverage Heatmap")
    ax.set_xlabel("Samples")
    ax.set_ylabel("MAG names")

    plt.savefig(output_dir / "mag_coverage.pdf", format='pdf')


def plot_mag_qual(checkm_tsv, output_dir):
    """
    Plot MAG contamiantion and redundancy.
    :return: Saves a file
    """
    # Load data
    quality = pd.read_csv(checkm_tsv, sep="\t", header=0)
    
    quality = quality[quality['Name'] != "viruses"]
    quality = quality[~quality['Name'].str.contains("metagenome")]

    # Clean the mag names
    quality.rename(inplace=True, columns={"Completeness": "CheckM2 Completeness", "Contamination": "CheckM2 Redundancy"})

    # Format it
    bar_chart_data = pd.melt(quality[["Name", "CheckM2 Completeness", "CheckM2 Redundancy"]], id_vars="Name", var_name="Metric", value_name="percent")

    # Sort by completeness
    bar_chart_data = bar_chart_data.sort_values(by="percent", ascending=False)
    
    # Define colors and hatches
    colors = sns.color_palette(["lightgreen", "lightcoral"])

    # Makea plot
    _, axes = plt.subplots(1, 1, figsize=(300, 30), layout="constrained")

    #  Bar chart with Completeness and Contamination percentages
    sns.barplot(data=bar_chart_data, x='Name', y='percent', hue="Metric", palette=colors, ax=axes)

    # Set the format for the bar labels
    for i in range(0, len(axes.containers)):
        axes.bar_label(axes.containers[i], fontsize=10, fmt="%.2f")

    axes.axhline(y=10, color='red', linestyle='--', linewidth=1)  # 10% line
    axes.axhline(y=30, color='orange', linestyle='--', linewidth=1)  # 30% line
    axes.axhline(y=90, color='green', linestyle='--', linewidth=1)  # 90% line
    
    # Add ticks every 10%
    axes.set_yticks(range(0, 101, 10))

    axes.set_title("MAG quality assessment")
    axes.set_xlabel("MAG names")
    axes.set_ylabel("Percentage")
    axes.yaxis.grid(False)
    axes.legend(loc='center right')

    # Display the figure save based on curent file path
    plt.savefig(output_dir / "mag_eval.pdf", format='pdf')


def plot_microbemod(microbemod_tsv, output_dir, name):
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

    axes.set_title(f"Count of RM genes in {name}")
    axes.set_xlabel("System type")
    axes.set_ylabel("Count")
    axes.yaxis.grid(False)

    # Display the figure save based on curent file path
    plt.savefig(output_dir / f"microbemod_{name}.pdf")
    

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
