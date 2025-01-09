from __future__ import annotations
from functools import cached_property
import polars as pl
from typing import TYPE_CHECKING
import re
import Bio.Data.IUPACData as bd
import itertools
from src.utilities.utils import methylation_base_map

if TYPE_CHECKING:  # Only for type hints
    from genome import Genome


class Motif(object):

    def __init__(self, genome: Genome, motif: str):
        self.genome: Genome = genome
        self.motif: str = motif
        self.meth_type: str = ""
        self.offset: int = -1
        self.frac_mod: float = -1
        self.high_count: int = -1
        self.low_count: int = -1
        self.mid_count: int = -1

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
    def positions(self) -> pl.LazyFrame:
        # Return a table with contig position and strand for each motif in the sequence
        contigs = []
        positions = []
        strands = []
        motifs = []

        for contig, seqrecord in self.genome.sequence.items():
            pos_strand = str(seqrecord.seq)
            neg_strand = str(seqrecord.seq.complement())

            # Find motif in positive strand
            pattern = re.compile('|'.join(map(re.escape, self.strings)))

            pos_iter = list(pattern.finditer(pos_strand))
            pos_positions = [match.start() + self.offset for match in pos_iter]
            pos_motifs = [match[0] for match in pos_iter]

            neg_iter = list(pattern.finditer(neg_strand))
            neg_positions = [match.start() + self.offset for match in neg_iter]
            neg_motifs = [match[0] for match in neg_iter]

            # Add to lists
            contigs.extend([contig] * len(pos_positions))
            positions.extend(pos_positions)
            strands.extend([True] * len(pos_positions))
            motifs.extend(pos_motifs)

            contigs.extend([contig] * len(neg_positions))
            positions.extend(neg_positions)
            strands.extend([False] * len(neg_positions))
            motifs.extend(neg_motifs)

        # Make a dataframe and return it
        return pl.DataFrame({
            "contig": contigs,
            "position": positions,
            "strand": strands,
            "motif": motifs
        }).lazy()


    @cached_property
    def data(self) -> pl.DataFrame:
        # Get all the data for the motif
        data_filter = (self.positions.select("contig", "position", "strand").with_columns(pl.col("position").alias("filter_end")))
        data_filter = data_filter.rename({"contig": "filter_contig", "position": "filter_start", "strand": "filter_strand"})
        data = self.genome.load_region_methylation_data(in_every_treatment=True, region_filter=data_filter)

        data = data.with_columns(pl.col('sample').replace(self.genome._barcode_treatment_map).alias("Treatment"))

        # Get sequence
        data = data.select("contig", "position", "strand", self.meth_type, self.canonical_base, "Treatment", "sample")

        return data.collect(streaming=True)


def generate_possible_sequences(seq):
    """return list of all possible sequences given an ambiguous DNA input"""
    d = bd.ambiguous_dna_values
    return list(map("".join, itertools.product(*map(d.get, seq))))
