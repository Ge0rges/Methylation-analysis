import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path

from src.objects.genome import Genome
from src.objects.gene_collection import GeneCollection
from src.objects.motif import Motif
from src.utilities.utils import methylation_base_map, treatment_weighted_mean, readable_modification_name
import numpy as np

##############################################################################
# 1) WHOLE METHYLOME (unchanged logic)
##############################################################################

def plot_whole_methylome(
    genome: Genome,
    motif: Motif,
    output_dir: Path,
) -> None:
    """
    Plot the whole methylome (only for the motif’s methylation type) across samples:
      - fraction = motif_type / (motif_type + canonical base)
      - x-axis = genome_position, color by sample
    """
    df = motif.data(normalize=False).with_columns(pl.col("sample").replace_strict(genome.barcode_treatment_map).replace_strict(genome.treatment_name_map).alias("treatment"))
    df = treatment_weighted_mean(df).collect(streaming=True)
    df = genome.add_genome_relative_position(df).rename({"treatment": "Treatment"})

    _, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        
    sns.scatterplot(
        data=df.to_pandas(),
        x="genome_position",
        y=motif.meth_type,
        hue="Treatment",
        ax=ax,
        s=4,
        alpha=0.7,
        hue_order=sorted(df.get_column("Treatment").unique().to_list(), key=genome.treatment_order_map.get)
    )

    ax.set_xlabel("Genome position (bp)")
    ax.set_ylabel(f"Fraction of {readable_modification_name[motif.meth_type]} methylation")
    ax.set_title(f"{genome.readable_name} - Whole Methylome ({motif.motif})")
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize=8)

    out_file = output_dir / f"{genome.readable_name}_whole_methylome_{motif.motif}.pdf"
    plt.savefig(out_file, format="pdf")
    plt.close()
    print(f"Saved PDF: {out_file}")


##############################################################################
# 2) METHYLATION DISTRIBUTION PER SAMPLE
##############################################################################

def plot_motif_methylation_distribution(
    genome: Genome,
    motif: Motif,
    output_dir: Path,
) -> None:
    """
    Plot the distribution (histogram) of motif's methylation fraction in each sample,
    rather than the mean fraction.

    - fraction = meth / (meth + canonical)
    - We'll do a single figure with multiple histplot calls, or subplots, showing how
      many sites fall in each fraction bin per sample.
    """
    df = motif.data(normalize=False).collect(streaming=True)

    # Replace sample → treatment name
    df = df.with_columns(pl.col("sample").replace_strict(genome.barcode_treatment_map).replace_strict(genome.treatment_name_map).alias("treatment"))

    # For each positiomn take a mean
    df = treatment_weighted_mean(df).rename({"treatment": "Treatment"})

    # One approach: single figure, color by treatment. Another approach: subplots per treatment.
    # Example: single figure, multiple histplot calls with "multiple='dodge'"
    _, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    sns.histplot(
        data=df.to_pandas(),
        x=motif.meth_type,
        hue="Treatment",
        bins=20,
        multiple="dodge",
        edgecolor="white",
        ax=ax,
        kde=True,
        hue_order=sorted(df.get_column("Treatment").unique().to_list(), key=genome.treatment_order_map.get)
    )
    ax.set_xlabel(f"{readable_modification_name[motif.meth_type]} methylation fraction")
    ax.set_ylabel("Count of sites")
    ax.set_title(f"{genome.readable_name} - Distribution of {motif.motif} Methylation")

    out_file = output_dir / f"{genome.readable_name}_{motif.motif}_methylation_distribution.pdf"
    plt.savefig(out_file, format="pdf")
    plt.close()
    print(f"Saved PDF: {out_file}")


##############################################################################
# 3) DMR HEATMAP (ROWS = sample_a, COLUMNS = sample_b, VALUES = SCORE)
##############################################################################

def plot_dmr_scores_heatmap(
    genome: Genome,
    motif: Motif,
    output_dir: Path
) -> None:
    """
    Plot a heatmap of DMR scores for the specified motif across sample comparisons:
      - rows = sample_a
      - columns = sample_b
      - values = mean_dmr_score
    Each motif -> separate PDF.
    """
    dmr = motif.dmr_data.collect(streaming=True)

    # Compute mean score per (sample_a, sample_b)
    pdf = (dmr.group_by(["treatment_a", "treatment_b"]).agg(pl.col("score").mean().alias("mean_dmr_score"))).to_pandas()

    # Make sure to unify sample names if needed, e.g. barcode -> treatment
    # If you want to rename sample_a, sample_b via genome.barcode_treatment_map:
    pdf["treatment_a"] = pdf["treatment_a"].replace(genome.barcode_treatment_map).replace(genome.treatment_name_map)
    pdf["treatment_b"] = pdf["treatment_b"].replace(genome.barcode_treatment_map).replace(genome.treatment_name_map)

    # Create pivot with sample_a as rows, sample_b as columns
    pivot = pdf.pivot(index="treatment_a", columns="treatment_b", values="mean_dmr_score")

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    sns.heatmap(pivot, cmap="viridis", annot=True, fmt=".2f", ax=ax)

    ax.set_title(f"DMR Score Heatmap - {genome.readable_name} ({motif.motif})")
    ax.set_xlabel("Treatment B")
    ax.set_ylabel("Treatment A")

    out_file = output_dir / f"{genome.readable_name}_{motif.motif}_dmr_heatmap.pdf"
    plt.savefig(out_file, format="pdf")
    plt.close()
    print(f"Saved PDF: {out_file}")


##############################################################################
# 4) PARALLEL CATEGORIES WITH CLEARER BIN LABELS & COLOR
##############################################################################

def plot_parallel_categories_methylation(
    genome: Genome,
    motif: Motif,
    output_dir: Path,
    bins: int = 3
) -> None:
    """
    Create a Plotly parallel categories plot with clearer bin labels ("Low", "Medium", "High") 
    and color. For instance, we color by the first dimension's value.
    """
    df = motif.data(normalize=False).collect(streaming=True)
    df = df.with_columns(pl.col("sample").replace_strict(genome.barcode_treatment_map).replace_strict(genome.treatment_name_map).alias("treatment"))
    treatments = df.get_column("treatment").unique().to_list()
    
    df = treatment_weighted_mean(df)
    
    # Pivot by (contig, position, strand), columns = treatment, values = fraction_meth
    pivoted = df.pivot(
        index=["contig", "position", "strand"],
        on="treatment",
        values=motif.meth_type,
    )

    # Bin data
    bins += 2 # We will remove the first and last
    cut_points = np.linspace(0, 1, bins - 1).tolist()
    bin_labels = []
    for i in range(bins):
        if i == 0:
            bin_labels.append(f"<={cut_points[i]:.2f}")
        elif i == bins - 1:
            bin_labels.append(f">{cut_points[i-1]:.2f}")
        else:
            bin_labels.append(f"{cut_points[i-1]:.2f}-{cut_points[i]:.2f}")
    
    cut_points = cut_points[1:-1]
    bin_labels = bin_labels[1:-1]
    pivoted = pivoted.with_columns(pl.col(treatment).cut(cut_points, labels=bin_labels) for treatment in treatments).to_pandas()

    cat_to_num = {key: i for i, key in enumerate(bin_labels)}
    pivoted["meth_numeric"] = pivoted[treatments[0]].map(cat_to_num)

    sorted_treatments = sorted(treatments, key=genome.treatment_order_map.get)
    fig = px.parallel_categories(
        pivoted,
        dimensions=sorted_treatments,
        color="meth_numeric",
        color_continuous_scale=[
            (0.0, "blue"),
            (0.5, "orange"),
            (1.0, "red"),
        ],
        range_color=[0, 2]
    )
    
    fig.update_layout(title=f"Methylation state transitions in {genome.readable_name} across conditions", coloraxis_showscale=False)
    
    out_file = output_dir / f"{genome.readable_name}_{motif.motif}_parallel_categories.html"
    fig.write_html(str(out_file))
    print(f"Saved HTML: {out_file}")


##############################################################################
# 5) EXTRACT DIFF METHYLATED GENES (WITH GENE FUNCTION)
##############################################################################

def extract_diff_methylated_genes(
    genome: Genome,
    motif: Motif,
    top_n: int = 10
) -> pl.DataFrame:
    """
    Extract the top differentially methylated locations (by DMR 'score') for a motif,
    attach nearest/overlapping gene, and include the gene function in the final CSV.

    Returns a Polars DataFrame.
    """
    top_rows = motif.dmr_data.collect(streaming=True).sort("score", descending=True).head(top_n)

    # Now, we want to add the gene function from some table. Typically:
    # 1) Add gene_caller_id for direct hits
    # 2) For start/end, do the same and join gene function columns

    # Let's do 1) add gene_caller_id if needed
    data = genome.add_gene_caller_id(top_rows.lazy(), include_intergenic=True).collect(streaming=True)

    # Add function
    gc = GeneCollection(data.get_column("gene_callers_id").unique().to_list(), genome)
    data = data.join(gc.get_function().collect(streaming=True), on="gene_callers_id", how="left")

    # Add nearest gene if not in gene
    data = genome.nearest_gene_to_positions(data)

    # Add function of nearest genes
    gc = GeneCollection(data.get_column("gene_callers_id_start").unique().to_list(), genome)
    data = data.join(gc.get_function().collect(streaming=True), left_on="gene_callers_id_start", right_on="gene_callers_id", how="left", suffix="_start")
    gc = GeneCollection(data.get_column("gene_callers_id_end").unique().to_list(), genome)
    data = data.join(gc.get_function().collect(streaming=True), left_on="gene_callers_id_end", right_on="gene_callers_id", how="left", suffix="_end")

    # Write to CSV
    data.write_csv(genome.output_dir / f"{genome.readable_name}_{motif.motif}_top_diff_genes.csv")
