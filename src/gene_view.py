import numpy as np
from src.objects.gene_collection import GeneCollection
from src.objects.gene import Gene
import seaborn as sns
import matplotlib.pyplot as plt
from utilities.utils import *
from src.objects.genome import Genome
from platform import system


sns.set_theme(context="poster", style="white")


def plot_all_gene_starts(genome: Genome):
    relative_start, relative_end = -40, 20

    gene_collection = GeneCollection(genome.gene_ids, genome)

    # Keep genes that have a start
    keep_ids = (gene_collection.is_start_missing.filter(pl.col("partial_begin").eq(False))
                .collect(streaming=True).get_column("gene_callers_id").to_list())
    gene_collection = GeneCollection(keep_ids, genome)

    # Get the data
    methyl_data = gene_collection.load_flanking_methylation_data(0, (relative_start, relative_end))

    # Get DF for all genes
    data = methyl_data.with_columns(pl.col('sample').replace(barcode_replicate_map))
    data = data.filter(pl.col("sample").is_in(["top", "middle", "bottom"]))
    data = data.rename(readable_methylation_name).rename({"sample": "Sample", "position": "Position"})
    data = data.with_columns(pl.col('Sample').replace(readable_sample_name)).collect(streaming=True)

    # All types plot
    long_form = data.unpivot(on=list(readable_methylation_name.values()),
                             index=["Sample", "Position"],
                             variable_name="Methylation type",
                             value_name="Normalized methylation fraction")

    # Get the most frequent nucleotide at each position
    sequence = gene_collection.get_flanking_sequence(0, (relative_start, relative_end))
    sequence = sequence.with_columns(pl.col("sequence").str.split("")).explode("sequence")
    sequence = sequence.with_columns((pl.cum_count("gene_callers_id") - 1 + relative_start).over("gene_callers_id").alias("position"))
    sequence_str = sequence.group_by("position").agg(pl.col("sequence").mode().first().alias("mode"))
    sequence_str = sequence_str.sort("position").collect().get_column("mode").str.join("").item()

    # Get DF for promoter distribution
    promoter_positions = gene_collection.pribnow_box_position_and_sequence.drop("gene_callers_id", "pribnow_box_sequence").filter(pl.col("pribnow_box_position").is_not_null()).collect().to_pandas()

    hue_order = [readable_sample_name["top"], readable_sample_name["middle"], readable_sample_name["bottom"]]
    fig, axes = plt.subplots(7, 1, figsize=(15, 45), layout="constrained", sharex=True)
    fig.suptitle(f"All gene promoter methylation in {genome.readable_name}", fontsize=28)

    # Promoter position distribution plot
    sns.histplot(promoter_positions, ax=axes[0], kde=(len(promoter_positions) > 1))
    axes[0].set_title(f"Proportion of genes with motif: {len(promoter_positions)} / {len(gene_collection.ids)}")

    # Per type plot
    for i, meth_type in enumerate(readable_methylation_name.values()):
        i += 1
        sns.lineplot(data.to_pandas(), x="Position", y=meth_type, hue="Sample", ax=axes[i], hue_order=hue_order)

        # Draw a vertical line at 0
        axes[i].axvline(x=0, color='black', linestyle='--', alpha=0.7)

    sns.lineplot(long_form.to_pandas(), x="Position", y="Normalized methylation fraction", hue="Sample",
                 style="Methylation type", ax=axes[4], hue_order=hue_order)
    axes[4].axvline(x=0, color='black', linestyle='--', alpha=0.7)

    # Plot number of data points at each position
    sns.histplot(data, x="Position", hue="Sample", ax=axes[5], discrete=True, multiple="stack")

    # Plot distribution of nucleotides
    nucleotide_freq = sequence.select("sequence", "position").rename({"sequence": "Nucleotide"}).collect(streaming=True).to_pandas()
    sns.histplot(nucleotide_freq, x="position", hue="Nucleotide", ax=axes[6], discrete=True, multiple="stack")

    # Plot the sequence as X ticks
    ticks = np.linspace(relative_start, relative_end, len(sequence_str))
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(sequence_str)

    # Highlight start codon
    for i, label in enumerate(axes[-1].get_xticklabels()):
        if 0 <= ticks[i] < 3:  # Start codon
            label.set_color('green')

    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig(genome.plot_dir / "all_genes.pdf", format="pdf")

    return


def plot_gene_start(genome: Genome, gene_id):
    gene = Gene.from_id(gene_id, genome)
    print(f"Gene is {gene.contig} at {gene.start} with length {gene.length} and strand {gene.strand}")
    print(f"RBS is {gene.rbs_motif} located at {gene.rbs_motif_position} and start is {gene.start_codon_sequence}")
    print(f"Pribnows box is {gene.pribnow_box_sequence} located at {gene.pribnow_box_position}")
    print(f"Minus 35 box is {gene.minus_35_sequence} located at {gene.minus_35_position}")
    print(f"Gene start {gene.sequence[:13]}")

    # Build filter for the region of interest
    relative_start, relative_end = -40, 40

    methyl_data = gene.load_flanking_methylation_data(0, (relative_start, relative_end))
    sequence = gene.get_flanking_sequence(0, (relative_start, relative_end))

    # Plot whole gene
    data = methyl_data.with_columns(pl.col('sample').replace(barcode_replicate_map).replace(readable_sample_name))
    data = data.rename(readable_methylation_name).rename({"sample": "Sample", "position": "Position"})
    data = data.collect(streaming=True)

    long_form = (data.unpivot(on=list(readable_methylation_name.values()),
                             index=["Sample", "Position"],
                             variable_name="Methylation type",
                             value_name="Normalized methylation fraction")
                 .filter(pl.col("Normalized methylation fraction").gt(0)))


    hue_order = [readable_sample_name["top"], readable_sample_name["middle"], readable_sample_name["bottom"]]
    ticks = np.linspace(relative_start, relative_end, len(sequence))
    fig, axes = plt.subplots(4, 1, figsize=(20, 25), layout="constrained", sharex=True, sharey=True)

    fig.suptitle(f"Gene {gene_id} methylation - {genome.readable_name}", fontsize=16)

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

    # Get operons of interest
    operons = genome.gene_cassettes
    operon_of_interest = []
    for gene_collection in operons:

        # Get list of functions of each gene in the operon
        functions = gene_collection.get_function().collect()
        number_of_genes_in_operon = len(gene_collection.ids)

        if 5 > number_of_genes_in_operon > 1:
            continue

        # Figure out if the function for KEGG_Brite is the same
        if functions.filter(pl.col("source").eq("COG20_CATEGORY")).get_column("function").n_unique() == 1 and functions.filter(pl.col("source").eq("COG20_CATEGORY")).height == number_of_genes_in_operon:
            operon_of_interest.append(gene_collection)

    # Get every gene with a promoter
    all_genes = GeneCollection(genome.gene_ids, genome)
    promoter_present = (all_genes.pribnow_box_position_and_sequence.filter(pl.col("pribnow_box_position").is_not_null())
                        .collect(streaming=True)
                        .get_column("gene_callers_id").to_list())
    promoter_genes = GeneCollection(promoter_present, genome)
    promoters_data = promoter_genes.load_flanking_methylation_data(0, (-40, 0))

    # Change sample names
    promoters_data = promoters_data.with_columns(pl.col('sample').replace(barcode_replicate_map))

    # Get the ones that are DMRed
    s = promoters_data.collect(streaming=True).get_column("sample").unique().to_list()
    dmr_genes = promoter_genes
    if "top" in s and "bottom" in s:
        dmr_result = promoter_genes.is_significantly_different_between_samples(promoters_data, ["top", "bottom"], False)
        dmr_ids = dmr_result.filter(pl.col("test_result").eq(True)).get_column("gene_callers_id").to_list()
        dmr_genes = GeneCollection(dmr_ids, genome)

    # Get genes with biggest difference in methylation between top and bottom, by meth type
    mod_names = list(readable_modification_name.keys())
    meth_sum = promoters_data.group_by("gene_callers_id", "sample").agg(pl.col(*mod_names).mean())
    samples_to_compare = ["top", "bottom"]
    meth_sum = meth_sum.filter(pl.col("sample").is_in(samples_to_compare))
    meth_sum = meth_sum.sort("gene_callers_id", "sample")

    meth_diffs: list[GeneCollection] = []
    for mod_name in mod_names:
        diff = meth_sum.group_by("gene_callers_id", maintain_order=True).agg(pl.col(mod_name).diff()).collect()
        diff = diff.top_k(5, by=pl.col(mod_name).list.last().abs())
        diff = GeneCollection(diff.get_column("gene_callers_id").to_list(), genome)
        meth_diffs.append(diff)

    # Return ids present in more than one list
    return set(dmr_genes.ids).intersection(*[d.ids for d in meth_diffs])


if __name__ == "__main__":
    for name in Genome.valid_genome_names():
        if "metagenome" in name:
            continue

        genome = Genome(name)

        print(f"Plotting all gene start for {name}")
        plot_all_gene_starts(genome)
        print(f"Getting interesting genes for {name}")
        interesting_ids = identify_interesting_genes(genome)

        for gene_id in interesting_ids:
            print(f"Plotting gene {gene_id} for {name}")
            plot_gene_start(genome, gene_id)

