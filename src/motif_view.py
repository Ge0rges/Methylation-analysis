from src.objects import Genome, GeneCollection
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from platform import system
import numpy as np

from utilities.utils import readable_sample_name, barcode_replicate_map, readable_methylation_name
from src.objects.motif import Motif

sns.set_theme(context="poster", style="white")


def motif_methylated_frequency(genome: Genome, motif: Motif):
    # Mean the methylation fraction for each motif
    data = (motif.data
            .filter(pl.col("Treatment").is_not_null() & (pl.col(motif.meth_type).is_not_null() | pl.col(motif.canonical_base).is_not_null()))
            .filter(pl.col("Treatment").n_unique().eq(3).over("contig", "strand", "position"))
            .group_by("contig", "strand", "position", "Treatment")
            .agg(pl.col(motif.meth_type).mean()))

    data = (data.unpivot(on=motif.meth_type, index=["Treatment", "contig", "strand", "position"],
                         variable_name="Methylation type",
                         value_name="Normalized methylation fraction")
            .filter(pl.col("Normalized methylation fraction").is_not_null() & pl.col("Normalized methylation fraction").is_not_nan()))

    # For each one get counts methylated and unmethylated
    data = data.with_columns(pl.col("Normalized methylation fraction").lt(0.5).alias("Methylated"),
                             pl.col("Treatment").replace(readable_sample_name))

    hue_order = [readable_sample_name["top"], readable_sample_name["middle"], readable_sample_name["bottom"]]
    fig = plt.subplots(figsize=(16, 12))
    sns.histplot(data.to_pandas(), x="Methylated", hue="Treatment", hue_order=hue_order, stat="count", multiple="dodge", element="bars")
    plt.suptitle(f"{genome.name} methylation motifs")

    plt.tight_layout()
    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig(genome.plot_dir / "motif_methylated_frequency.pdf", format="pdf")


def motif_view(genome: Genome, motif: Motif):
    # Add function
    data = genome.add_gene_caller_id(motif.data.lazy(), include_intergenic=True).collect(streaming=True)
    gc = GeneCollection(data.get_column("gene_callers_id").unique().to_list(), genome)
    data = data.join(gc.get_function().collect(streaming=True), on="gene_callers_id", how="left")

    # Add nearest gene if not in gene
    data = genome.nearest_gene_to_positions(data)

    # Write to CSV
    data.write_csv(genome.plot_dir / f"{motif}_motif_view.csv")


def number_of_positions_switched(genome: Genome, motif: Motif):
    data = (motif.data
            .filter(pl.col("Treatment").is_not_null() & (pl.col(motif.meth_type).is_not_null() | pl.col(motif.canonical_base).is_not_null()))
            .filter(pl.col("Treatment").n_unique().eq(3).over("contig", "strand", "position"))
            .group_by("contig", "strand", "position", "Treatment")
            .agg(pl.col(motif.meth_type).mean()))

    # Binarize meth_type and canonical_base values
    data = data.with_columns((pl.col(motif.meth_type) > 0.5).alias("binarized_meth_type"))

    # Assuming `data` is the DataFrame with the required columns
    # Filter data for "bottom" and "top" treatments
    bottom_data = data.filter(pl.col("Treatment").eq("bottom"))
    top_data = data.filter(pl.col("Treatment").eq("top"))

    # Ensure the data is aligned by some key, e.g., sample_id, to compare "bottom" and "top"
    # Here, we assume there's a common column "sample_id" to align both treatments
    aligned_data = bottom_data.join(top_data, on=["contig", "strand", "position"], suffix="_top")

    # Create a contingency table for binarized_meth_type
    transition_counts = aligned_data.select([
        pl.col("binarized_meth_type").alias("bottom"),
        pl.col("binarized_meth_type_top").alias("top")
    ]).to_pandas().pivot_table(
        index="bottom",
        columns="top",
        aggfunc="size",
        fill_value=0
    )

    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(transition_counts, annot=True, fmt="g", cmap="coolwarm", cbar_kws={'label': 'Transition Count'})
    plt.title(f"Transition Heatmap (Bottom vs. Top) in {genome.readable_name}")
    plt.xlabel("Top")
    plt.ylabel("Bottom")

    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig(genome.plot_dir / f"motif{motif.motif}_positions_switched.pdf", format="pdf")


def plot_genes_regions(motif: Motif, gene_collection: GeneCollection, relative_position: int = 0, relative_start: int = -40,
                       relative_end: int = 20):
    # Keep genes that have a start if relative_start is negative
    keep_ids = (gene_collection.is_start_missing.filter(pl.col("partial_begin").eq(False))
                .collect(streaming=True).get_column("gene_callers_id").to_list())
    gene_collection = GeneCollection(keep_ids, genome)

    # Get the data
    methyl_data = gene_collection.load_flanking_methylation_data(relative_position,
                                                                 (relative_start, relative_end),
                                                                 common_only=True)

    # Filter to motifs
    methyl_data = methyl_data.join(motif.positions.lazy(), on=["contig", "strand", "position"], how="inner")

    # Get DF for all genes
    data = methyl_data.with_columns(pl.col('sample').replace(barcode_replicate_map))
    data = data.rename(readable_methylation_name).rename({"sample": "Sample", "position": "Position"})
    data = data.with_columns(pl.col('Sample').replace(readable_sample_name)).collect(streaming=True)

    if data.height == 0:
        print(
            f"No data for genes in genome {gene_collection.genome.name} in region {relative_position} +- {relative_start}, {relative_end}")
        return

    # All types plot
    long_form = (data.unpivot(on=list(readable_methylation_name.values()),
                              index=["Sample", "Position"],
                              variable_name="Methylation type",
                              value_name="Normalized methylation fraction")
                 .filter(pl.col("Normalized methylation fraction").is_not_null() &
                         pl.col("Normalized methylation fraction").is_not_nan()))

    # Get the most frequent nucleotide at each position
    sequence = gene_collection.get_flanking_sequence(relative_position, (relative_start, relative_end))
    sequence = sequence.with_columns(pl.col("sequence").str.split("")).explode("sequence")
    sequence = sequence.with_columns(
        (pl.cum_count("gene_callers_id") - 1 + relative_start).over("gene_callers_id").alias("position"))
    sequence_str = sequence.group_by("position").agg(pl.col("sequence").mode().first().alias("mode"))
    sequence_str = sequence_str.sort("position").collect().get_column("mode").str.join("").item()

    # Get DF for promoter distribution
    promoter_positions = gene_collection.pribnow_box_position_and_sequence.drop("gene_callers_id",
                                                                                "pribnow_box_sequence").filter(
        pl.col("pribnow_box_position").is_not_null()).collect().to_pandas()

    hue_order = [readable_sample_name["top"], readable_sample_name["middle"], readable_sample_name["bottom"]]
    fig, axes = plt.subplots(7, 1, figsize=(int((relative_end - relative_start) * 0.4), 45),
                             layout="constrained", sharex=True)
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
    nucleotide_freq = sequence.select("sequence", "position").rename({"sequence": "Nucleotide"}).collect(
        streaming=True).to_pandas()
    sns.histplot(nucleotide_freq, x="position", hue="Nucleotide", ax=axes[6], discrete=True, multiple="stack",
                 hue_order=["A", "T", "G", "C"], palette="Paired")

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


if __name__ == "__main__":
    import os
    from pathlib import Path

    data_path = Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/methylation_data/"))
    for methylation_path in data_path.iterdir():
        if methylation_path.is_dir():
            Genome._Genome__methylation_data_dir = methylation_path

            for name in Genome.valid_genome_names():
                if "metagenome" in name or "brevundimonas" in name:
                    continue

                genome = Genome(name)

                # Motif
                motifs = genome.motifs

                for motif in motifs:

                    # Print percent of positions with no data
                    data = motif.data.get_column(motif.meth_type)
                    for treatment in motif.data.get_column("Treatment").unique().to_list():
                        if treatment is None:
                            d = motif.data.filter(pl.col("Treatment").is_null())
                        else:
                            d = motif.data.filter(pl.col("Treatment").eq(treatment))

                        if d.height == 0:
                            print(f"No data for {motif.motif} in {treatment}")
                            continue
                        print(f"Motif {motif.motif} in {treatment} has {d.null_count().get_column(motif.meth_type).item() / d.height * 100:.2f}% positions with no data")

                    motif_methylated_frequency(genome, motif)
                    number_of_positions_switched(genome, motif)
                    gene_collection = GeneCollection(genome.gene_ids, genome)
                    plot_genes_regions(motif, gene_collection, relative_position=0, relative_start=-40, relative_end=10)
                    motif_view(genome, motif)
