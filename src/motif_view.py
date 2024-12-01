from pygments.lexer import include

from src.objects import Genome, GeneCollection
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from platform import system

from utilities.utils import (readable_methylation_name, barcode_replicate_map, readable_sample_name,
                             generate_possible_sequences)


sns.set_theme(context="poster", style="white")


def motif_methylated_frequency(genome: Genome, motifs: list[str]):
    data = genome.load_all_methylation_data(common_only=True)
    data = data.with_columns(pl.col('sample').replace(barcode_replicate_map).alias("Treatment"))

    # Get sequence
    data = genome.add_sequence_around_position(data, 2, 2)
    data = data.select("contig", "position", *list(readable_methylation_name.keys()), "Sequence", "Treatment")

    # Make it presentable
    data = data.with_columns(pl.col('Treatment').replace(readable_sample_name)).rename(readable_methylation_name).collect(streaming=True)

    # Long form the data
    data = (data.unpivot(on=list(readable_methylation_name.values()),
                         index=["Treatment", "Sequence"],
                         variable_name="Methylation type",
                         value_name="Normalized methylation fraction")
                .filter(pl.col("Normalized methylation fraction").is_not_null() &
                        pl.col("Normalized methylation fraction").is_not_nan()))

    # Filter motifs
    dfs = []
    for motif in motifs:
        possible_motifs = generate_possible_sequences(motif)
        df = data.filter(pl.col("Sequence").is_in(possible_motifs))
        dfs.append(df)
    data = pl.concat(dfs)

    # For each one get counts methylated and unmethylated
    data = data.with_columns(pl.col("Normalized methylation fraction").lt(0.5).alias("Methylated"))

    hue_order = [readable_sample_name["top"], readable_sample_name["middle"], readable_sample_name["bottom"]]
    g = sns.displot(data, x="Sequence", y="Methylated", row="Methylation type", hue="Treatment",
                    height=12, aspect=2, hue_order=hue_order, binwidth=0.1, kind="hist", stat="proportion")
    g.fig.suptitle(f"{genome.name} methylation motifs")

    plt.tight_layout()
    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig(genome.plot_dir / "motif_methylated_frequency.pdf", format="pdf")


def motif_view(genome: Genome, motif: str):
    data = genome.load_all_methylation_data(common_only=True)
    data = data.with_columns(pl.col('sample').replace(barcode_replicate_map).alias("Treatment"))

    # Get sequence
    data = genome.add_sequence_around_position(data, 2, 2)
    data = data.select("contig", "position", "strand", *list(readable_methylation_name.keys()), "Sequence", "Treatment")

    data = data.with_columns(pl.col('Treatment').replace(readable_sample_name)).rename(readable_methylation_name)

    # Make compatible list of motifs
    motifs = generate_possible_sequences(motif)

    # Filter motif
    data = data.filter(pl.col("Sequence").is_in(motifs))

    # Add function
    data = genome.add_gene_caller_id(data, include_intragenic=True).collect(streaming=True)
    gc = GeneCollection(data.get_column("gene_callers_id").unique().to_list(), genome)
    data = data.join(gc.get_function().collect(streaming=True), on="gene_callers_id", how="left")

    # Add nearest gene if not in gene
    data = genome.nearest_gene_to_positions(data)

    # Write to CSV
    data.write_csv(genome.plot_dir / f"{motif}_motif_view.csv")


if __name__ == "__main__":
    import os
    from pathlib import Path

    data_path = Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/methylation_data/"))
    for methylation_path in data_path.iterdir():
        if methylation_path.is_dir():
            Genome._Genome__methylation_data_dir = methylation_path

            for name in Genome.valid_genome_names():
                if "Pelagibacter" not in name:
                    continue

                motifs = ["GANTC"]
                genome = Genome(name)
                # motif_methylated_frequency(genome, motifs)
                motif_view(genome, motifs[0])
