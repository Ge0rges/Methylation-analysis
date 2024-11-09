from src.objects import Genome
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

from src.utilities.data_loading import get_genomic_sequence
from utilities.utils import readable_methylation_name, barcode_replicate_map, readable_sample_name


sns.set_theme(context="poster", style="white")


def show_motifs():
    genome = Genome("Pelagibacter_r-contigs")

    data = genome.load_all_methylation_data()

    # Preprocess the data. Sort, rename, filter, and make position absolute to genome.
    data = data.sort("strand", "contig", "position", descending=False)
    data = data.with_columns(pl.col('sample').replace(barcode_replicate_map))
    data = data.filter(pl.col("sample").is_in(["top", "middle", "bottom"]))

    # Add sequence
    threshold = 0.8
    data = data.filter(pl.any_horizontal(pl.col(*readable_methylation_name.keys())).ge(threshold))

    # Get full sequences first and their length
    sequences = {}
    com_sequences = {}
    for key, value in get_genomic_sequence(genome.name).items():
        sequences[key] = str(value.seq)
        com_sequences[key] = str(value.seq.complement())
    sequences = {"contig": sequences.keys(), "sequence": sequences.values(), "complement_sequence": com_sequences.values()}
    sequences = pl.from_dict(sequences, schema=["contig", "sequence", "complement_sequence"]).lazy()

    data = data.join(sequences, on="contig")
    data = data.with_columns(pl.when(pl.col("strand"))
                             .then(pl.col("sequence").str.slice(pl.col("position"), 2))
                             .otherwise(pl.col("complement_sequence").str.slice(pl.col("position"), 2)).alias("Sequence"))
    data = data.drop("sequence", "complement_sequence")

    # Make it presentable
    data = data.with_columns(pl.col('sample').replace(readable_sample_name))
    data = data.rename(readable_methylation_name).rename({"sample": "Sample", "strand": "Strand"})

    # Get contigs cumsum
    data = genome.add_genome_relative_position(data).rename({"genome_position": "Position"}).drop("position")
    data = data.collect(streaming=True)

    # Long form the data
    data = (data.unpivot(on=list(readable_methylation_name.values()),
                         index=["Sample", "Position", "Strand", "Sequence"],
                         variable_name="Methylation type",
                         value_name="Normalized methylation fraction")
                .filter(pl.col("Normalized methylation fraction").ge(threshold)))

    # # Plot the strand in two seperate columns, one row per methylation type
    # hue_order = [readable_sample_name["top"], readable_sample_name["middle"], readable_sample_name["bottom"]]
    # g = sns.relplot(data, x="Position", y="Normalized methylation fraction", col="Methylation type", row="Strand",
    #                 hue="Sample", height=8, aspect=2, row_order=[True, False], hue_order=hue_order)
    #
    # # Draw vertical lines at contig limits
    # for ax in g.axes.flatten():
    #     val = list(offsets.values())[1:]
    #     ymin, ymax = ax.get_ylim()
    #     ax.vlines(x=val, color='gray', linestyle='-', ymin=[ymin]*len(val), ymax=[ymax]*len(val))
    #
    # plt.show()

    result = {}
    motifs = data.get_column("Sequence").unique().to_list()
    for motif in motifs:
        if not motif in result.keys():
            result[motif] = 1
        else:
            result[motif] += 1

    print(result)


if __name__ == "__main__":
    show_motifs()
