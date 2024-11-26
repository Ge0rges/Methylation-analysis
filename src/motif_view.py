from src.objects import Genome
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from platform import system

from utilities.utils import readable_methylation_name, barcode_replicate_map, readable_sample_name


sns.set_theme(context="poster", style="white")


def motif_methylation(genome: Genome):
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

    # Get frequency of each sequence
    popular_motifs = (data.group_by(["Sequence", "Methylation type"])
                          .agg(pl.len())
                          .group_by("Methylation type")
                          .agg(pl.col("Sequence").top_k_by(pl.col("len"), 10)))

    # Filter data for only the most popular motifs per methylation type
    dfs = []
    for meth in readable_methylation_name.values():
        pop_motifs = popular_motifs.filter(pl.col("Methylation type") == meth).get_column("Sequence").to_list()[0]
        df = data.filter(pl.col("Sequence").is_in(pop_motifs) & pl.col("Methylation type").eq(meth))
        dfs.append(df)

    data = pl.concat(dfs)

    # Plot the strand in two seperate columns, one row per methylation type
    hue_order = [readable_sample_name["top"], readable_sample_name["middle"], readable_sample_name["bottom"]]
    g = sns.displot(data, x="Sequence", y="Normalized methylation fraction", row="Methylation type", col="Treatment",
                    height=12, aspect=2, hue_order=hue_order, binwidth=0.1, kind="hist", stat="count")
    g.fig.suptitle(f"{genome.name} methylation motifs")

    plt.tight_layout()
    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig(genome.plot_dir / "motif_methylated_frequency.pdf", format="pdf")


def motif_frequency(genome: Genome):
    data = genome.load_all_methylation_data(common_only=True)
    data = data.with_columns(pl.col('sample').replace(barcode_replicate_map).alias("Treatment"))

    # Get sequence
    data = genome.add_sequence_around_position(data, 2, 2)
    data = data.select("contig", "position", *list(readable_methylation_name.keys()), "Sequence", "Treatment")

    data = data.with_columns(pl.col('Treatment').replace(readable_sample_name)).rename(readable_methylation_name)

    # Long form the data
    data = (data.unpivot(on=list(readable_methylation_name.values()),
                         index=["Treatment", "Sequence"],
                         variable_name="Methylation type",
                         value_name="Normalized methylation fraction")
                .filter(pl.col("Normalized methylation fraction").is_not_null() &
                        pl.col("Normalized methylation fraction").is_not_nan() &
                        pl.col("Normalized methylation fraction").gt(0))
                .select("Treatment", "Sequence", "Methylation type")
                .collect(streaming=True))

    # Plot the strand in two seperate columns, one row per methylation type
    hue_order = [readable_sample_name["top"], readable_sample_name["middle"], readable_sample_name["bottom"]]
    g = sns.displot(data, x="Sequence", row="Methylation type", hue="Treatment", height=36, aspect=2, hue_order=hue_order,
                    kind="hist", stat="count")
    g.fig.suptitle(f"{genome.name} methylation motifs frequency")

    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig(genome.plot_dir / "motif_frequency.pdf", format="pdf")


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
                motif_methylation(genome)
                motif_frequency(genome)
                exit()
