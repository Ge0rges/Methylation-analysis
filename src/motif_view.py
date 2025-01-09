from src.objects import Genome, GeneCollection
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from platform import system
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
                             pl.col("Treatment").replace(genome._barcode_treatment_map))

    hue_order = [genome._barcode_replicate_map[x] for x in genome._default_treatments]
    plt.subplots(figsize=(16, 12))
    sns.histplot(data.to_pandas(), x="Methylated", hue="Treatment", hue_order=hue_order, stat="count", multiple="dodge", element="bars")
    plt.suptitle(f"{genome.name} methylation motifs")

    plt.tight_layout()
    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig(genome.plot_dir / "motif_methylated_frequency.pdf", format="pdf")
    plt.close()


def number_of_positions_switched(genome: Genome, motif: Motif):
    data = (motif.data
            .filter(pl.col("Treatment").is_not_null() & (pl.col(motif.meth_type).is_not_null() | pl.col(motif.canonical_base).is_not_null()))
            .filter(pl.col("Treatment").n_unique().eq(3).over("contig", "strand", "position"))
            .group_by("contig", "strand", "position", "Treatment")
            .agg(pl.col(motif.meth_type).mean()))

    # Binarize meth_type and canonical_base values
    # Categorize values into low <25%), middle (20-80%), and high (>75%) into column binarized_meth_type
    data = data.with_columns(pl.when(pl.col(motif.meth_type).lt(0.25))
                             .then(pl.lit("low"))
                             .otherwise(pl.when(pl.col(motif.meth_type).gt(0.75))
                                        .then(pl.lit("high"))
                                        .otherwise(pl.lit("middle")))
                             .alias("binarized_meth_type"))

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
    sns.heatmap(transition_counts, annot=True, fmt="g", cmap="coolwarm", cbar_kws={'label': 'Number of positions'})
    plt.title(f"Transition map in {genome.readable_name}")

    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig(genome.plot_dir / f"{motif.motif}_positions_switched.pdf", format="pdf")


def annotate_switched_positions(genome: Genome, motif: Motif):
    # Filter down to data that has switched
    data = (motif.data
            .filter(pl.col("Treatment").is_not_null() & (
                pl.col(motif.meth_type).is_not_null() | pl.col(motif.canonical_base).is_not_null()))
            .filter(pl.col("Treatment").n_unique().eq(3).over("contig", "strand", "position"))
            .group_by("contig", "strand", "position", "Treatment")
            .agg(pl.col(motif.meth_type).mean()))

    # Categorize values into low <25%), middle (20-80%), and high (>75%) into column binarized_meth_type
    data = data.with_columns(pl.when(pl.col(motif.meth_type).lt(0.25))
                             .then(pl.lit("low"))
                             .otherwise(pl.when(pl.col(motif.meth_type).gt(0.75))
                                        .then(pl.lit("high"))
                                        .otherwise(pl.lit("middle")))
                             .alias("binarized_meth_type"))

    # Assuming `data` is the DataFrame with the required columns
    # Filter data for "bottom" and "top" treatments
    bottom_data = data.filter(pl.col("Treatment").eq("bottom"))
    top_data = data.filter(pl.col("Treatment").eq("top"))

    # Ensure the data is aligned by some key, e.g., sample_id, to compare "bottom" and "top"
    # Here, we assume there's a common column "sample_id" to align both treatments
    aligned_data = bottom_data.join(top_data, on=["contig", "strand", "position"], suffix="_top")

    # Filter down to switched positions
    switched_positions = aligned_data.filter(pl.col("binarized_meth_type").ne(pl.col("binarized_meth_type_top")))

    if switched_positions.height == 0:
        print(f"No switched positions for {motif.motif}")
        return

    # Add function
    data = genome.add_gene_caller_id(switched_positions.lazy(), include_intergenic=True).collect(streaming=True)
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
    data.write_csv(genome.plot_dir / f"{motif.motif}_motif_view.csv")


if __name__ == "__main__":
    import os
    from pathlib import Path

    data_path = Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/methylation_data/"))
    for methylation_path in data_path.iterdir():
        if methylation_path.is_dir():
            Genome._Genome__methylation_data_dir = methylation_path

            for name in Genome.valid_genome_names():
                if "metagenome" in name or "brevundimonas" in name or "microbemod" in name:
                    continue

                genome = Genome(name)
                print(f"Processing {genome.name}")

                # Motif
                motifs = genome.motifs

                if len(motifs) == 0:
                    print(f"No motifs for {genome.name}")
                    continue

                for motif in motifs:

                    # Print percent of positions with no data
                    data = motif.data.get_column(motif.meth_type)
                    if len(data) == 0:
                        print(f"No data for {motif.motif} in any treatment")
                        continue

                    for treatment in motif.data.get_column("Treatment").unique().to_list():
                        if treatment is None:
                            d = motif.data.filter(pl.col("Treatment").is_null())
                        else:
                            d = motif.data.filter(pl.col("Treatment").eq(treatment))

                        if d.height == 0:
                            print(f"No data for {motif.motif} in {treatment}")
                            continue
                        print(f"Motif {motif.motif} in {treatment} has {d.null_count().get_column(motif.meth_type).item() / d.height * 100:.2f}% of positions with no data")

                    motif_methylated_frequency(genome, motif)
                    number_of_positions_switched(genome, motif)
                    annotate_switched_positions(genome, motif)
