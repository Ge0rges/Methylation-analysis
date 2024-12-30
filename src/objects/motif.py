from __future__ import annotations
from functools import cached_property
import polars as pl
from typing import TYPE_CHECKING
import re
import Bio.Data.IUPACData as bd
import itertools
from src.utilities.utils import barcode_replicate_map, methylation_base_map

if TYPE_CHECKING:  # Only for type hints
    from genome import Genome


class Motif(object):

    def __init__(self, genome: Genome, motif: str):
        self.genome: Genome = genome
        self.motif: str = motif
        self.meth_type: str = None
        self.offset: int = None
        self.frac_mod: float = None
        self.high_count: int = None
        self.low_count: int = None
        self.mid_count: int = None

        self.strings: list[str] = generate_possible_sequences(motif)


    @cached_property
    def canonical_base(self):
        return f"Ncanonical_{methylation_base_map[self.meth_type]}"


    @classmethod
    def load_from_modkit(cls, genome: Genome) -> list[Motif]:
        motifs = pl.read_csv(str(genome._methylation_data_dir / "all_motifs-5-motifs.tsv"), separator="\t", has_header=True)

        # Create a Motif object for each row
        motif_objs = []
        for row in motifs.iter_rows(named=True):
            motif = Motif(genome, row["motif"])

            # Header: mod_code	motif	offset	frac_mod	high_count	low_count	mid_count
            motif.meth_type = row["mod_code"]
            motif.motif = row["motif"]
            motif.offset = row["offset"]
            motif.frac_mod = row["frac_mod"]
            motif.high_count = row["high_count"]
            motif.low_count = row["low_count"]
            motif.mid_count = row["mid_count"]

            motif_objs.append(motif)

        return motif_objs


    @cached_property
    def positions(self) -> pl.DataFrame:
        # Return a table with contig position and strand for each motif in the sequence
        contigs = []
        positions = []
        strands = []

        for contig, seqrecord in self.genome.sequence.items():
            pos_strand = str(seqrecord.seq)
            neg_strand = str(seqrecord.seq.complement())

            # Find motif in positive strand
            pattern = re.compile('|'.join(map(re.escape, self.strings)))
            pos_positions = [match.start() + self.offset for match in pattern.finditer(pos_strand)]
            neg_positions = [match.start() + self.offset for match in pattern.finditer(neg_strand)]

            # Add to lists
            contigs.extend([contig] * len(pos_positions))
            positions.extend(pos_positions)
            strands.extend([True] * len(pos_positions))

            contigs.extend([contig] * len(neg_positions))
            positions.extend(neg_positions)
            strands.extend([False] * len(neg_positions))

        # Make a dataframe and return it
        return pl.DataFrame({
            "contig": contigs,
            "position": positions,
            "strand": strands,
            "motif": [self.motif] * len(contigs)
        })


    @cached_property
    def data(self) -> pl.DataFrame:
        # Get all the data for the motif
        data_filter = (self.positions.select("contig", "position", "strand")
                                     .with_columns(pl.col("position").alias("filter_end")))
        data_filter = data_filter.rename({"contig": "filter_contig", "position": "filter_start", "strand": "filter_strand"})
        data = self.genome.load_region_methylation_data(in_every_treatment=True, region_filter=data_filter)

        data = data.with_columns(pl.col('sample').replace(barcode_replicate_map).alias("Treatment"))

        # Get sequence
        data = (data.select("contig", "position", "strand", self.meth_type, self.canonical_base, "Treatment", "sample")
                    .collect(streaming=True))

        # Combine with all known positions
        if self.positions.height == 0:
            try:
                assert data.height == 0
            except AssertionError:
                print(f"Motif {self.motif} has no positions but has data")

            return pl.DataFrame()

        return data


def number_of_positions_switched(genome: Genome):
    data = genome.load_all_methylation_data(normalize=True, in_every_treatment=True, treatments=["top", "bottom"]).collect(
        streaming=True)
    if data.height == 0:
        print(f"No data for {genome.name}")
        return

    # Make a binary decision on methylation state at a positon
    data = data.with_columns(pl.col("sample").replace(barcode_replicate_map).alias("treatment")).sort("treatment")
    data = data.group_by("contig", "strand", "position", "treatment").agg(pl.col(*readable_methylation_name.keys()))

    # Binarize
    data = data.with_columns(pl.col(*readable_methylation_name.keys()).list.mean() > 0.5)

    for key, meth_group in base_methylation_map.items():
        data = data.with_columns(pl.any_horizontal(*meth_group).alias(key))

    # Count the number of transitions between each possibility in a treatment
    data = data.sort("treatment").group_by("contig", "strand", "position", maintain_order=True).agg(
        pl.col("A").alias("A_bottom"),
        pl.col("C").alias("C_bottom"),
        pl.col("A").shift(-1).alias("A_top"),
        pl.col("C").shift(-1).alias("C_top")
    ).explode(pl.col("A_bottom"), pl.col("C_bottom"), pl.col("A_top"), pl.col("C_top"))

    # Count transitions
    Atransition_counts = data.group_by(["A_bottom", "A_top"]).len().rename({"len": "transition_count"})
    Ctransition_counts = data.group_by(["C_bottom", "C_top"]).len().rename({"len": "transition_count"})

    # Convert to pandas and then pivot
    Atransition_counts = Atransition_counts.to_pandas().pivot(index="A_bottom", columns="A_top",
                                                              values="transition_count")
    Ctransition_counts = Ctransition_counts.to_pandas().pivot(index="C_bottom", columns="C_top",
                                                              values="transition_count")

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(Atransition_counts, annot=True, fmt="g", cmap="coolwarm", cbar_kws={'label': 'Transition Count'})
    plt.title(f"Transition Heatmap (A_bottom vs. A_top) in {genome.readable_name}")
    plt.xlabel("A_top")
    plt.ylabel("A_bottom")

    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig(genome.plot_dir / "A_positions_switched.pdf", format="pdf")

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(Ctransition_counts, annot=True, fmt="g", cmap="coolwarm", cbar_kws={'label': 'Transition Count'})
    plt.title(f"Transition Heatmap (C_bottom vs. C_top) in {genome.readable_name}")
    plt.xlabel("C_top")
    plt.ylabel("C_bottom")
    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig(genome.plot_dir / "C_positions_switched.pdf", format="pdf")


def generate_possible_sequences(seq):
    """return list of all possible sequences given an ambiguous DNA input"""
    d = bd.ambiguous_dna_values
    return list(map("".join, itertools.product(*map(d.get, seq))))
