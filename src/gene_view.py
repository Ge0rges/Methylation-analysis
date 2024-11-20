import numpy as np
from src.objects.gene_collection import GeneCollection
from src.objects.gene import Gene
import seaborn as sns
import matplotlib.pyplot as plt
from utilities.utils import *
from src.objects.genome import Genome
from platform import system


sns.set_theme(context="poster", style="white")


def plot_genes_regions(gene_collection: GeneCollection, relative_position: int = 0, relative_start: int = -40, relative_end: int = 20):
    # Keep genes that have a start if relative_start is negative
    keep_ids = (gene_collection.is_start_missing.filter(pl.col("partial_begin").eq(False))
                .collect(streaming=True).get_column("gene_callers_id").to_list())
    gene_collection = GeneCollection(keep_ids, genome)

    # Get the data
    methyl_data = gene_collection.load_flanking_methylation_data(relative_position,
                                                                 (relative_start, relative_end),
                                                                 common_only=True)

    # Get DF for all genes
    data = methyl_data.with_columns(pl.col('sample').replace(barcode_replicate_map))
    data = data.rename(readable_methylation_name).rename({"sample": "Sample", "position": "Position"})
    data = data.with_columns(pl.col('Sample').replace(readable_sample_name)).collect(streaming=True)

    if data.height == 0:
        print(f"No data for genes in genome {gene_collection.genome.name} in region {relative_position} +- {relative_start}, {relative_end}")
        return

    # All types plot
    long_form = data.unpivot(on=list(readable_methylation_name.values()),
                             index=["Sample", "Position"],
                             variable_name="Methylation type",
                             value_name="Normalized methylation fraction")

    # Get the most frequent nucleotide at each position
    sequence = gene_collection.get_flanking_sequence(relative_position, (relative_start, relative_end))
    sequence = sequence.with_columns(pl.col("sequence").str.split("")).explode("sequence")
    sequence = sequence.with_columns((pl.cum_count("gene_callers_id") - 1 + relative_start).over("gene_callers_id").alias("position"))
    sequence_str = sequence.group_by("position").agg(pl.col("sequence").mode().first().alias("mode"))
    sequence_str = sequence_str.sort("position").collect().get_column("mode").str.join("").item()

    # Get DF for promoter distribution
    promoter_positions = gene_collection.pribnow_box_position_and_sequence.drop("gene_callers_id", "pribnow_box_sequence").filter(pl.col("pribnow_box_position").is_not_null()).collect().to_pandas()

    hue_order = [readable_sample_name["top"], readable_sample_name["middle"], readable_sample_name["bottom"]]
    fig, axes = plt.subplots(7, 1, figsize=(int((relative_end - relative_start) * 0.4), 45), layout="constrained", sharex=True)
    fig.suptitle(f"All gene promoter methylation in {genome.readable_name}", fontsize=28)

    # Promoter position distribution plot
    sns.histplot(promoter_positions, ax=axes[0], kde=(len(promoter_positions) > 1))
    axes[0].set_title(f"Proportion of genes with pribnox box: {len(promoter_positions)} / {len(gene_collection.ids)}")

    # Per type plot
    for i, meth_type in enumerate(readable_methylation_name.values()):
        i += 1
        sns.lineplot(data.to_pandas(), x="Position", y=meth_type, hue="Sample", ax=axes[i], hue_order=hue_order)

    sns.lineplot(long_form.to_pandas(), x="Position", y="Normalized methylation fraction", hue="Sample",
                 style="Methylation type", ax=axes[4], hue_order=hue_order)

    # Plot number of data points at each position
    sns.histplot(data, x="Position", hue="Sample", ax=axes[5], discrete=True, multiple="stack", hue_order=hue_order)

    # Plot distribution of nucleotides
    nucleotide_freq = sequence.select("sequence", "position").rename({"sequence": "Nucleotide"}).collect(streaming=True).to_pandas()
    sns.histplot(nucleotide_freq, x="position", hue="Nucleotide", ax=axes[6], discrete=True, multiple="stack", hue_order=["A", "T", "G", "C"], palette="Paired")

    # Plot the sequence as X ticks
    ticks = np.linspace(relative_start, relative_end, len(sequence_str))
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(sequence_str)

    # Draw line at  0
    for ax in axes:
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)

    # Highlight start codon
    for i, label in enumerate(axes[-1].get_xticklabels()):
        if 0 <= ticks[i] < 3:  # Start codon
            label.set_color('green')

    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig(genome.plot_dir / "all_genes.pdf", format="pdf")

    return


def plot_gene_region(gene: Gene, relative_position: int = 0, relative_start: int = -40, relative_end: int = 40):
    print(f"Gene is {gene.contig} at {gene.start} with length {gene.length} and strand {gene.strand}")
    print(f"RBS is {gene.rbs_motif} located at {gene.rbs_motif_position} and start is {gene.start_codon_sequence}")
    print(f"Pribnows box is {gene.pribnow_box_sequence} located at {gene.pribnow_box_position}")
    print(f"Minus 35 box is {gene.minus_35_sequence} located at {gene.minus_35_position}")
    print(f"Gene start {gene.sequence[:13]}")

    # Get region of intereset
    methyl_data = gene.load_flanking_methylation_data(relative_position, (relative_start, relative_end))
    sequence = gene.get_flanking_sequence(relative_position, (relative_start, relative_end))

    # Plot whole gene
    data = methyl_data.with_columns(pl.col('sample').replace(barcode_replicate_map).replace(readable_sample_name))
    data = data.rename(readable_methylation_name).rename({"sample": "Sample", "position": "Position"})
    data = data.collect(streaming=True)

    if data.height == 0:
        print(f"No data for {gene_collection.ids} in genome {gene_collection.genome.name} in region {relative_position} +- {relative_start}, {relative_end}")
        return

    long_form = (data.unpivot(on=list(readable_methylation_name.values()),
                             index=["Sample", "Position"],
                             variable_name="Methylation type",
                             value_name="Normalized methylation fraction")
                 .filter(pl.col("Normalized methylation fraction").gt(0)))


    hue_order = [readable_sample_name["top"], readable_sample_name["middle"], readable_sample_name["bottom"]]
    ticks = np.linspace(relative_start, relative_end, len(sequence))
    fig, axes = plt.subplots(4, 1, figsize=(int((relative_end - relative_start) * 0.4), 25), layout="constrained", sharex=True, sharey=True)

    fig.suptitle(f"Gene {gene.id} methylation - {gene.genome.readable_name}", fontsize=16)

    # Per type plot
    for i, meth_type in enumerate(readable_methylation_name.values()):
        df = data.filter(pl.col(meth_type).gt(0)).to_pandas()
        # If the DF is empty, plot a test and turn off axis
        if df.empty:
            axes[i].set_title(f"No {meth_type} methylation")
            axes[i].axis('off')
        else:
            sns.pointplot(df, x="Position", y=meth_type, hue="Sample", ax=axes[i], hue_order=hue_order,
                          native_scale=True, linestyles="None")


    # All types plot
    sns.scatterplot(long_form.to_pandas(), x="Position", y="Normalized methylation fraction", hue="Sample",
                    style="Methylation type", ax=axes[3], hue_order=hue_order)

    # Plot the sequence as X ticks
    axes[3].set_xticks(ticks)
    axes[3].set_xticklabels(sequence)

    # Set different colors for start and RBS
    for i, label in enumerate(axes[3].get_xticklabels()):
        if 0 <= ticks[i] < 3:  # Start codon
            label.set_color('green')

        if gene.rbs_motif is not None:
            if gene.rbs_motif_position - len(gene.rbs_motif) < ticks[i] <= gene.rbs_motif_position:  # RBS motif
                label.set_color('orange')

        if gene.pribnow_box_position is not None:
            if gene.pribnow_box_position - len(gene.pribnow_box_sequence) < ticks[i] <= gene.pribnow_box_position:
                label.set_color('red')

        if gene.minus_35_position is not None:
            if gene.minus_35_position - len(gene.minus_35_sequence) < ticks[i] <= gene.minus_35_position:
                label.set_color('purple')

    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig(genome.plot_dir / f"{gene.id}_promoter_methylation.pdf", format="pdf")

    return


def identify_interesting_genes(genome: Genome):
    all_genes = GeneCollection(genome.gene_ids, genome)
    all_data = all_genes.load_flanking_methylation_data(0, (-40, 40))
    all_data = all_data.with_columns(pl.col('sample').replace(barcode_replicate_map))

    # Get the ones that are DMRed
    dmr_ids = []
    s = all_data.collect(streaming=True).get_column("sample").unique().to_list()
    if "top" in s and "bottom" in s:
        dmr_result = (all_genes.is_significantly_different_between_samples(all_data, ["top", "bottom"], False)
                      .filter(pl.col("test_result").eq(True)))
        dmr_ids = dmr_result.get_column("gene_callers_id").to_list()

        if len(dmr_ids) == 0:
            print(f"No DMRed genes for {genome.name}")
            return []

        dmr_genes = GeneCollection(dmr_ids, genome)

        # Get entropy of promoter
        entropies = dmr_genes.get_entropy_for_region(0, (-40, 20)).join(dmr_result, on="gene_callers_id")
        if entropies.height > 0:
            entropies.select("gene_callers_id", "entropy", "base", "sample").sort("gene_callers_id", "base", "entropy", descending=True).write_csv(genome.plot_dir / "entropy_genes.csv")

        # Get functions
        dmr_genes = (dmr_genes.get_function()
                     .select("gene_callers_id", "function", "rao_score", "source", "test_result").unique() 
                     .sort("entropy", "rao_score", descending=True))

        # Write to CSV
        dmr_genes.sink_csv(genome.plot_dir / "dmred_genes_rao_entropy.csv")

    return dmr_ids


if __name__ == "__main__":
    import os
    from pathlib import Path

    data_path = Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/methylation_data/"))
    for methylation_path in data_path.iterdir():
        if methylation_path.is_dir():
            Genome._Genome__methylation_data_dir = methylation_path

            for name in Genome.valid_genome_names():
                if "metagenome" in name:
                    continue

                genome = Genome(name)
                gene_collection = GeneCollection(genome.gene_ids, genome)

                print(f"Plotting all gene start for {name}")
                plot_genes_regions(gene_collection, 0, -40, 10)

                if "Pelagibacter" in name or "polaribacter" in name:
                    print(f"Getting interesting genes for {name}")
                    interesting_ids = identify_interesting_genes(genome)

                    #for gene_id in interesting_ids:
                    #    print(f"Plotting gene {gene_id} for {name}")
                    #    plot_gene_region(Gene(gene_id, genome), 0, -40, 10)

                print(f"Done with {genome.name}")

