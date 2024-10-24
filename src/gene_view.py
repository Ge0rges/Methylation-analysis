import numpy as np
from Bio.SeqRecord import SeqRecord
from Cython.Compiler.Nodes import relative_position

from data_manager import *
import seaborn as sns
import matplotlib.pyplot as plt
from utilities.utils import *

sns.set_theme(context="talk", style="white", font_scale=3)


def plot_all_promoters():
    genome = Genome("Pelagibacter_r-contigs")
    gene_collection = GeneCollection(genome.gene_ids, genome)
    methyl_data = gene_collection.load_flanking_methylation_data(0, (-25, 25))

    # Plot whole gene
    data = methyl_data.with_columns(pl.col('sample').replace(barcode_replicate_map))
    data = data.rename(readable_methylation_name)
    data = data.rename({"sample": "Sample", "position": "Position"})
    data = data.with_columns(pl.col('Sample').replace(readable_sample_name)).collect(streaming=True)

    hue_order = [readable_sample_name["top"], readable_sample_name["middle"], readable_sample_name["bottom"]]
    fig, axes = plt.subplots(5, 1, figsize=(40, 50), layout="constrained")

    # Promoter position distribution plot
    promoter_positions = gene_collection.rbs_motif_and_relative_position.drop("gene_callers_id").to_pandas()
    sns.histplot(promoter_positions, ax=axes[0], bins=50, kde=True)

    # Add text on plot which shows proportion of genes with RBS motif
    total_genes = len(genome.gene_ids)
    rbs_genes = len(promoter_positions)
    axes[0].text(0, 0, f"Proportion of genes with RBS motif: {rbs_genes} / {total_genes}")

    # Per type plot
    for i, meth_type in enumerate(readable_methylation_name.values()):
        i += 1
        df = data.filter(pl.col(meth_type).gt(0))
        sns.lineplot(df.to_pandas(), x="Position", y=meth_type, hue="Sample", ax=axes[i],
                     hue_order=hue_order)  #, s=81 * 4)

    # All types plot
    long_form = data.unpivot(on=list(readable_methylation_name.values()),
                             index=["Sample", "Position"],
                             variable_name="Methylation type",
                             value_name="Normalized methylation fraction")
    long_form = long_form.filter(pl.col("Normalized methylation fraction").gt(0))

    sns.lineplot(long_form.to_pandas(), x="Position", y="Normalized methylation fraction", hue="Sample",
                 style="Methylation type", ax=axes[4], hue_order=hue_order)  #, s=81 * 4)

    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig("../plots/plots_5/all_genes.pdf", format="pdf")

    return


def plot_gene_promoter_start():
    genome = Genome("Pelagibacter_r-contigs")
    gene = Gene(2538688, genome)
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
    data = data.rename(readable_methylation_name)
    data = data.rename({"sample": "Sample", "position": "Position"})
    data = data.with_columns(pl.col('Sample').replace(readable_sample_name)).collect(streaming=True)

    hue_order = [readable_sample_name["top"], readable_sample_name["middle"], readable_sample_name["bottom"]]
    ticks = np.linspace(relative_start, relative_end, len(sequence))
    fig, axes = plt.subplots(4, 1, figsize=(40, 50), layout="constrained")

    # Per type plot
    for i, meth_type in enumerate(readable_methylation_name.values()):
        df = data.filter(pl.col(meth_type).gt(0))
        sns.scatterplot(df.to_pandas(), x="Position", y=meth_type, hue="Sample", ax=axes[i], hue_order=hue_order,
                        s=81 * 4)

        # Plot the sequence as X ticks
        axes[i].set_xticks(ticks)
        axes[i].set_xticklabels(sequence)

        # Set different colors for certain characters
        for j, label in enumerate(axes[i].get_xticklabels()):
            if 0 <= ticks[j] < 3:  # Start codon
                label.set_color('green')
            elif gene.rbs_motif_position - len(gene.rbs_motif) < ticks[j] <= gene.rbs_motif_position:  # RBS motif
                label.set_color('orange')

    # All types plot
    long_form = data.unpivot(on=list(readable_methylation_name.values()),
                             index=["Sample", "Position"],
                             variable_name="Methylation type",
                             value_name="Normalized methylation fraction").filter(
        pl.col("Normalized methylation fraction").gt(0))

    sns.scatterplot(long_form.to_pandas(), x="Position", y="Normalized methylation fraction", hue="Sample",
                    style="Methylation type", ax=axes[3], hue_order=hue_order, s=81 * 4)

    # Plot the sequence as X ticks
    axes[3].set_xticks(ticks)
    axes[3].set_xticklabels(sequence)

    # Set different colors for certain characters
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
