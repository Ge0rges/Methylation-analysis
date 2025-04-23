import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pylab as plt
import os
import glob
from pathlib import Path

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
    coverage = coverage.sort_values(by=[coverage.columns[i] for i in [0, 1, 2]], ascending=False)
    
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
    
    # Define expected categories and ensure columns are categorical *before* grouping
    system_types = ['I', 'II', 'III', 'IV']
    gene_types = ["Methyltransferase", "Restriction enzyme"]
    microbemod['System type'] = pd.Categorical(microbemod['System type'], categories=system_types, ordered=True)
    microbemod['Gene type'] = pd.Categorical(microbemod['Gene type'], categories=gene_types, ordered=True)

    # Calculate counts for all defined category combinations.
    plot_data = microbemod.groupby(['System type', 'Gene type'], observed=False).size().reset_index(name='count')

    # Make a plot
    _, axes = plt.subplots(figsize=(8, 9), layout="constrained")

    # Bar chart with number of genes by RM type, and enzyme type (methyltransferase, restriction enzyme)
    sns.barplot(data=plot_data, y="count", x='System type', hue="Gene type", ax=axes, hue_order=["Methyltransferase", "Restriction enzyme"])
    
    # Set the format for the bar labels
    for i in range(0, len(axes.containers)):
        axes.bar_label(axes.containers[i], fmt="%d")

    axes.set_xlabel("System type")
    axes.set_ylabel(f"Count of genes in {name}")
    axes.get_legend().set_title("")
    axes.yaxis.grid(False)

    # Display the figure save based on curent file path
    plt.savefig(output_dir / f"microbemod_{name}.pdf")


def cross_microbemod_identified_motifs(microbemod_tsv: Path, methylation_data_dir: Path, output_dir: Path):
    motif_dfs: list = []
    pattern: Path = methylation_data_dir / "motifs" / "*_motifs.tsv"
    
    for file in glob.glob(str(pattern)):
        # Extract contig name from filename (assumes format X_<contig name>_motifs.tsv)
        base = os.path.basename(file)
        parts = base.split("_")
        contig_name = f"{parts[1]}_{parts[2]}"
        
        # Read the TSV file using Polars and select the desired columns
        try: 
            df = pl.read_csv(file, separator="\t", schema_overrides=[pl.String, pl.String, pl.Int64, pl.String]).select(["mod_code", "motif", "offset"])
            
            # If empty, add manual rows
            if df.is_empty():
                df = pl.DataFrame({
                    "mod_code": ["No motif identified"],
                    "motif": ["No motif identified"],
                    "offset": [-1]
                })
            
        except pl.exceptions.NoDataError:
            df: pl.DataFrame = pl.DataFrame({
            "mod_code": ["No data"],
            "motif": ["No data"],
            "offset": [-1]
        })
        
        # Add the contig name as a new column
        df = df.with_columns(pl.lit(contig_name).alias("Contig name"))
        motif_dfs.append(df)
        
    motif_table: pl.DataFrame = pl.concat(motif_dfs)
    
    # Group by contig name
    motif_table = motif_table.group_by("Contig name").agg(
        pl.col("mod_code"),
        pl.col("motif"),
        pl.col("offset").cast(pl.String)
    )
    
    motif_table = motif_table.with_columns(
        pl.col("mod_code").list.join(",").alias("mod_codes"),
        pl.col("motif").list.join(",").alias("De novo motifs"),
        pl.col("offset").list.join(",").alias("offsets")
    ).select("Contig name", "mod_codes", "De novo motifs", "offsets")

    # Process microbemod file
    rm_genes_df: pl.DataFrame = pl.read_csv(microbemod_tsv, separator="\t")
    rm_genes_df = rm_genes_df.with_columns(("c_" + pl.col('Gene').str.split('_').list.get(1)).alias("Contig name"))
    
    # Group by contig name to aggregate:
    # - Count of genes.
    # - Unique, sorted gene types joined by commas.
    # - Unique, sorted homolog motifs (dropping nulls) joined by commas.
    rm_genes_df = rm_genes_df.group_by("Contig name").agg([
        pl.count("Gene").alias("Number of genes"),
        pl.col("Gene type"),
        pl.col("Homolog motif"),
        pl.col("REBASE homolog"),
        pl.col("Homolog methylation")
    ])
    
    # Join the gene types and homolog motifs into comma-separated strings
    rm_genes_df = rm_genes_df.with_columns(
        pl.col("Gene type").list.join(",").alias("Gene types"),
        pl.col("Homolog motif").list.join(",").alias("Homolog motifs"),
        pl.col("REBASE homolog").list.join(",").alias("REBASE homologs"),
        pl.col("Homolog methylation").list.join(",").alias("Homolog methylation")
    ).select("Contig name", "Number of genes", "Gene types", "Homolog motifs", "REBASE homologs", "Homolog methylation")

    # Join the motif and gene tables on "contig name"
    result = motif_table.join(rm_genes_df, on="Contig name", how="full").select("Contig name", "De novo motifs", "Homolog motifs", "Number of genes", "Gene types", "REBASE homologs", "Homolog methylation", "mod_codes", "offsets")

    # Write the resulting DataFrame as a TSV
    result.write_csv(output_dir / "motifs_vs_microbemod.tsv", separator="\t")


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
