import numpy as np

from data_manager import *
import seaborn as sns
import matplotlib.pyplot as plt
from utilities.utils import *

sns.set_theme(context="talk", style="white", font_scale=3)


def run_analysis():
    genome = Genome("Pelagibacter_r-contigs")
    gene = Gene(2538688, genome)  # 2195033

    print(f"RBS is {gene.rbs_motif} located at {gene.rbs_motif_position} and start is {gene.start_codon}")
    print(f"Gene start {gene.sequence[:13]}")
    plot_gene_promoter_start(gene)


def plot_gene_promoter_start(gene):
    # Build filter for the region of interest
    region_start = gene.start - (gene.rbs_motif_position + len(gene.rbs_motif)*2) if gene.rbs_motif else gene.start - (
                gene.rbs_spacer_length + 12)
    region_end = gene.start + 10
    strand = "+" if gene.strand else "-"
    promoter_start_filter = (pl.col("chrom").eq(gene.contig) &
                             pl.col("inclusive start position").ge(region_start) &
                             pl.col("exclusive end position").le(region_end) &
                             pl.col("strand").eq(strand))

    methyl_data = gene.genome.load_region_methylation_data(region_filter=promoter_start_filter)
    methyl_data = methyl_data.with_columns(
        pl.col('name').str.split(by='|').list.get(2).cast(pl.Int64).sub(gene.start).alias("position")).drop("name")

    sequence = gene.get_flanking_sequence(0, (gene.start - region_end, gene.start - region_start))
    print(f"Got sequence {sequence}")
    plot_whole_gene(methyl_data, sequence, (region_start - gene.start, region_end - gene.start))


def plot_whole_gene(gene_data, sequence, sequence_range):
    # Plot whole gene
    data = gene_data.with_columns(pl.col('sample').replace(barcode_replicate_map)).collect(streaming=True)
    # data = data.sort(["sample", "position"]).with_columns(
    #     pl.col("total_methylation").rolling_mean(20, min_periods=1).over("sample").alias("total_methylation"))

    hue_order = ["top", "middle", "bottom"]
    fig, axes = plt.subplots(4, 1, figsize=(40, 50), layout="constrained")

    # Plot methylation
    long_form = data.unpivot(on=list(readable_methylation_name.keys()),
                             index=["sample", "position"],
                             variable_name="methylation_type",
                             value_name="methylation_fraction").filter(pl.col("methylation_fraction").gt(0))

    sns.scatterplot(long_form.to_pandas(), x="position", y="methylation_fraction", hue="sample", style="methylation_type", ax=axes[3], hue_order=hue_order)
    # Plot the sequence as X ticks
    axes[3].set_xticks(np.linspace(sequence_range[0], sequence_range[1], len(sequence)))
    axes[3].set_xticklabels(sequence)

    for i, meth_type in enumerate(readable_methylation_name.keys()):
        df = data.filter(pl.col(meth_type).gt(0))
        sns.scatterplot(df.to_pandas(), x="position", y=meth_type, hue="sample", ax=axes[i], hue_order=hue_order)

        # Plot the sequence as X ticks
        axes[i].set_xticks(np.linspace(sequence_range[0], sequence_range[1], len(sequence)))
        axes[i].set_xticklabels(sequence)

    plt.show()


if __name__ == "__main__":
    run_analysis()
