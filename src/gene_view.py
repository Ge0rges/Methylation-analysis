import numpy as np
from src.Objects.gene_collection import GeneCollection
from src.Objects.gene import Gene
import seaborn as sns
import matplotlib.pyplot as plt
from utilities.utils import *
from src.Objects.genome import Genome
from platform import system


sns.set_theme(context="poster", style="white")


def plot_all_promoters():
    genome = Genome("Pelagibacter_r-contigs")
    gene_collection = GeneCollection(genome.gene_ids, genome)
    methyl_data = gene_collection.load_flanking_methylation_data(0, (-25, 10))

    # Get DF for all genes
    data = methyl_data.with_columns(pl.col('sample').replace(barcode_replicate_map))
    data = data.rename(readable_methylation_name).rename({"sample": "Sample", "position": "Position"})
    data = data.with_columns(pl.col('Sample').replace(readable_sample_name)).collect(streaming=True)

    # All types plot
    long_form = data.unpivot(on=list(readable_methylation_name.values()),
                             index=["Sample", "Position"],
                             variable_name="Methylation type",
                             value_name="Normalized methylation fraction")
    long_form = long_form.filter(pl.col("Normalized methylation fraction").gt(0))

    # Get DF for promoter distribution
    promoter_positions = gene_collection.rbs_motif_and_relative_position.drop("gene_callers_id").to_pandas()

    hue_order = [readable_sample_name["top"], readable_sample_name["middle"], readable_sample_name["bottom"]]
    fig, axes = plt.subplots(5, 1, figsize=(15, 35), layout="constrained", sharex=True)

    # Promoter position distribution plot
    sns.histplot(promoter_positions, ax=axes[0], kde=(len(promoter_positions) > 1))
    axes[0].set_title(f"Proportion of genes with RBS motif: {len(promoter_positions)} / {len(gene_collection.ids)}")

    # Per type plot
    for i, meth_type in enumerate(readable_methylation_name.values()):
        i += 1
        df = data.filter(pl.col(meth_type).gt(0))
        sns.lineplot(df.to_pandas(), x="Position", y=meth_type, hue="Sample", ax=axes[i], hue_order=hue_order)

    sns.lineplot(long_form.to_pandas(), x="Position", y="Normalized methylation fraction", hue="Sample",
                    style="Methylation type", ax=axes[4], hue_order=hue_order)

    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig("../plots/plots_5/all_genes.pdf", format="pdf")

    return


def plot_gene_promoter_start():
    genome = Genome("Pelagibacter_r-contigs")
    gene = Gene.from_id(2538688, genome)
    print(f"Gene is {gene.contig} at {gene.start} with length {gene.length} and strand {gene.strand}")
    print(f"RBS is {gene.rbs_motif} located at {gene.rbs_motif_position} and start is {gene.start_codon_sequence}")
    print(f"Gene start {gene.sequence[:13]}")

    # Build filter for the region of interest
    relative_start = gene.rbs_motif_position - len(gene.rbs_motif) * 2 if gene.rbs_motif else -(gene.rbs_spacer_length[1] + 12)
    relative_end = 10

    methyl_data = gene.load_flanking_methylation_data(0, (relative_start, relative_end))
    sequence = gene.get_flanking_sequence(0, (relative_start, relative_end))

    # Plot whole gene
    data = methyl_data.with_columns(pl.col('sample').replace(barcode_replicate_map))
    data = data.rename(readable_methylation_name).rename({"sample": "Sample", "position": "Position"})
    data = data.with_columns(pl.col('Sample').replace(readable_sample_name)).collect(streaming=True)

    long_form = (data.unpivot(on=list(readable_methylation_name.values()),
                             index=["Sample", "Position"],
                             variable_name="Methylation type",
                             value_name="Normalized methylation fraction")
                     .filter(pl.col("Normalized methylation fraction").gt(0)))


    hue_order = [readable_sample_name["top"], readable_sample_name["middle"], readable_sample_name["bottom"]]
    ticks = np.linspace(relative_start, relative_end, len(sequence))
    fig, axes = plt.subplots(4, 1, figsize=(15, 25), layout="constrained", sharex=True, sharey=True)

    # Per type plot
    for i, meth_type in enumerate(readable_methylation_name.values()):
        df = data.filter(pl.col(meth_type).gt(0)).to_pandas()
        # If the DF is empty, plot a test and turn off axis
        if df.empty:
            axes[i].set_title(f"No {meth_type} methylation")
            axes[i].axis('off')
        else:
            sns.scatterplot(df, x="Position", y=meth_type, hue="Sample", ax=axes[i], hue_order=hue_order, s=81 * 4)

    # All types plot
    sns.scatterplot(long_form.to_pandas(), x="Position", y="Normalized methylation fraction", hue="Sample",
                    style="Methylation type", ax=axes[3], hue_order=hue_order, s=81 * 4)

    # Plot the sequence as X ticks
    axes[3].set_xticks(ticks)
    axes[3].set_xticklabels(sequence)

    # Set different colors for start and RBS
    for i, label in enumerate(axes[3].get_xticklabels()):
        if 0 <= ticks[i] < 3:  # Start codon
            label.set_color('green')
        elif gene.rbs_motif_position - len(gene.rbs_motif) < ticks[i] <= gene.rbs_motif_position:  # RBS motif
            label.set_color('orange')

    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig("../plots/plots_5/gene_promoter_methylation.pdf", format="pdf")

    return


if __name__ == "__main__":
    plot_gene_promoter_start()
    plot_all_promoters()
