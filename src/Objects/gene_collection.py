from __future__ import annotations

from functools import lru_cache, cached_property
from src.utilities.utils import add_gene_caller_id, readable_methylation_name
from Bio import SeqRecord
from src.utilities.data_loading import get_genomic_sequence, get_genes_polars
import polars as pl

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # Only for type hints
    from gene import Gene
    from genome import Genome

try:
    from _statistics import add_rao_score_by_gene
except:
    pass


space_dict = {"3-4bp": (3, 4), "5-10bp": (5, 10), "11-12bp": (11, 12), "13-15bp": (13, 15), "11bp": (11, 11),
              "12bp": (12, 12), "5bp": (5, 5), "7bp": (7, 7), "15bp": (15, 15), "9bp": (9, 9), "4bp": (4, 4),
              "8bp": (8, 8), "14bp": (14, 14), "10bp": (10, 10), "13bp": (13, 13), "3bp": (3, 3), "6bp": (6, 6),
              None: None}

rbs_motifs = {"GGA/GAG/AGG": ["GGA", "GAG", "AGG"],
              "3Base/5BMM": ["GGA", "GAG", "AGG", "AGAAG", "AGTAG", "AGGAG", "AGCAG", "GGAGG", "GGTGG", "GGGGG", "GGCGG"],
              "4Base/6BMM": ["AGGA", "GGAG", "GAGG", "AGAAGG", "AGTAGG", "AGGAGG", "AGCAGG", "AGGAGG", "AGGGGG", "AGGTGG", "AGGCGG"],
              "AGxAG": ["AGAAG", "AGTAG", "AGGAG", "AGCAG"],
              "AGxAGG/AGGxGG": ["AGAAGG", "AGTAGG", "AGGAGG", "AGCAGG", "AGGAGG", "AGGGGG", "AGGTGG", "AGGCGG"],
              "GGxGG": ["GGAGG", "GGTGG", "GGGGG", "GGCGG"],
              "AGGAG(G)/GGAGG": ["AGGAG", "AGGAGG", "GGAGG"],
              "AGGA": ["AGGA"],
              "AGGA/GGAG/GAGG": ["AGGA", "GGAG", "GAGG"],
              "GGAG/GAGG": ["GGAG", "GAGG"],
              "AGGAG/GGAGG": ["AGGAG", "GGAGG"],
              "AGGAG": ["AGGAG"],
              "GGAGG": ["GGAGG"],
              "AGGAGG": ["AGGAGG"],
              "AAAA": ["AAAA"],
              "AAAAA": ["AAAAA"],
              "AATAA": ["AATAA"],
              "TAA": ["TAA"],
              "AAA": ["AAA"],
              "AAT": ["AAT"],
              "AAAT": ["AAAT"],
              "AAAAT": ["AAAAT"],
              "TAAA": ["TAAA"],
              "TAAAA": ["TAAAA"],
              None: [None]
}


class GeneCollection(object):

    def __init__(self, ids: list[int], genome: Genome):
        self.ids: list[int] = ids
        self.genome: Genome = genome

        self._load_data()


    def _load_data(self) -> None:
        self.gene_caller_df: pl.LazyFrame = get_genes_polars(self.genome._data_dir).filter(
            pl.col("gene_callers_id").is_in(self.ids))
        self.functional_df: pl.lazyframe = pl.scan_csv(f"{self.genome._data_dir}/function-calls.txt", separator="\t").filter(
            pl.col("gene_callers_id").is_in(self.ids))


    def __getitem__(self, item) -> Gene:
        from gene import Gene
        return Gene(self.ids[item], self)


    def __len__(self):
        return len(self.ids)


    @cached_property
    def methylation_data(self) -> pl.LazyFrame:
        # TODO: This loading step could be made more efficient by pushing down the filtering to the data loading step
        # That would require building a regional filter for each gene, and then concatenate those expressions
        methylation_data: pl.LazyFrame = self.genome.load_all_methylation_data()

        # Split name into coordinates
        methylation_data = methylation_data.with_columns(
            contig=pl.col('name').str.split(by='|').list.get(0),
            strand=pl.col('name').str.split(by='|').list.get(1),
            start=pl.col('name').str.split(by='|').list.get(2).cast(pl.Int64),
            end=pl.col('name').str.split(by='|').list.get(3).cast(pl.Int64)
        )

        methylation_data = add_gene_caller_id(methylation_data, self.gene_caller_df)
        methylation_data = methylation_data.drop("start", "end", "strand", "contig")
        methylation_data = methylation_data.filter(pl.col("gene_callers_id").is_in(self.ids))

        # Do a group by gene_callers_id and then do a subtraction of the start position
        methylation_data = methylation_data.join(self.start, on="gene_callers_id")
        methylation_data = methylation_data.with_columns(pl.col('name').str.split(by='|').list.get(2).cast(pl.Int64).sub(pl.col("start")).alias("position")).drop("name")
        return methylation_data


    @cached_property
    def contig(self) -> pl.LazyFrame:
        return self.gene_caller_df.select("gene_callers_id", "contig")


    @cached_property
    def start(self) -> pl.LazyFrame:
        """Coordinate on forward strand. Position relative to the contig. Start is inclusive"""
        return self.gene_caller_df.select("gene_callers_id", "start")


    @cached_property
    def stop(self) -> pl.LazyFrame:
        """Coordinate on forward strand. Position relative to the contig. Stop is exclusive."""
        return self.gene_caller_df.select("gene_callers_id", "stop")


    @cached_property
    def strand(self) -> pl.LazyFrame:
        df = self.gene_caller_df.select("gene_callers_id", "direction").with_columns(pl.col("direction").eq("+"))
        df = df.rename({"direction": "strand"})
        return df


    @cached_property
    def source(self) -> pl.LazyFrame:
        return self.functional_df.select("gene_callers_id", "source")


    @cached_property
    def sequence(self) -> pl.LazyFrame:
        return self.gene_caller_df.select("gene_callers_id", "sequence")


    @cached_property
    def length(self) -> pl.LazyFrame:
        df = self.start.join(self.stop, on="gene_callers_id")
        # This is correct because prodigal indexes at 0, and stop is exclusive
        df = df.with_columns(pl.col("stop").sub(pl.col("start")).alias("length"))
        df = df.select("gene_callers_id", "length")
        return df


    @cached_property
    def is_start_missing(self) -> pl.LazyFrame:
        return self.gene_caller_df.select("gene_callers_id", "partial_begin")


    @cached_property
    def is_end_missing(self) -> pl.LazyFrame:
        return self.gene_caller_df.select("gene_callers_id", "partial_end")


    @cached_property
    def start_codon_sequence(self) -> pl.LazyFrame:
        return self.gene_caller_df.select("gene_callers_id", "start_type")


    @cached_property
    def start_codon_position(self) -> pl.LazyFrame:
        """Position of the first nucleotide of the start codon relative to the contig"""
        df = self.gene_caller_df.select("gene_callers_id")
        df = df.with_columns(pl.when(pl.col("strand"))
                             .then(pl.col("start"))
                             .otherwise(pl.col("stop") - 1)  # Stop is exclusive
                             .alias("start_codon_position"))
        return df


    @cached_property
    def stop_codon_sequence(self) -> pl.LazyFrame:
        df = self.sequence.with_columns(pl.lit("sequence").str.slice(-3).alias("stop_codon"))
        return df.select("gene_callers_id", "stop_codon_sequence")


    @cached_property
    def stop_codon_position(self) -> pl.LazyFrame:
        """Position of the first nucleotide of the stop codon relative to the contig"""
        df = self.gene_caller_df.select("gene_callers_id")
        df = df.with_columns(pl.when(pl.col("strand"))
                             .then(pl.col("stop") - 3) # Stop is exclusive, and codon is 3
                             .otherwise(pl.col("start"))
                             .alias("stop_codon_position"))
        return df


    @cached_property
    def candidate_rbs_motifs(self) -> pl.LazyFrame:
        df = self.gene_caller_df.select("gene_callers_id", "rbs_motif")
        df = df.with_columns(pl.col("rbs_motif").replace_strict(rbs_motifs, return_dtype=pl.List(pl.String)).alias("candidate_rbs_motifs"))
        df = df.select("gene_callers_id", "candidate_rbs_motifs")
        return df


    @cached_property
    def rbs_motif_and_relative_position(self) -> pl.DataFrame:

        def find_rbs_motif_position(row):
            candidate_rbs_motifs: list[str] = row[1]
            rbs_spacer_length: list[int, int] = row[2]
            flanking: str = row[3]

            if rbs_spacer_length[0] is None:
                assert candidate_rbs_motifs[0] is None

            if candidate_rbs_motifs[0] is None:
                return None, None
            else:
                for candidate_motif in candidate_rbs_motifs:
                    try:
                        # - 1 is needed because index gives first character and length is total length
                        # - rbs_spacer_length[1] - 12 is needed to get position relative to the gene start
                        pos = flanking.index(candidate_motif) + len(candidate_motif) - 1 - rbs_spacer_length[1] - 12
                        rbs_motif: str = candidate_motif
                        return pos, rbs_motif

                    except ValueError:
                        continue

            raise ValueError

        df = self.candidate_rbs_motifs.join(self.rbs_spacer_length, on="gene_callers_id")
        df = df.join(self.get_flanking_sequence(0, (-pl.col("rbs_spacer_length").list.last() - 12, 0)), on="gene_callers_id")
        df = df.collect(streaming=True)

        # map_rows requires return_dtype and can't specify different types
        result = {"rbs_motif_position": [], "rbs_motif": []}
        for row in df.iter_rows():
            res, rbs_motif = find_rbs_motif_position(row)
            result["rbs_motif_position"].append(res)
            result["rbs_motif"].append(rbs_motif)

        motifs_df = pl.from_dict(result)
        return df.hstack(motifs_df).select("rbs_motif_position", "rbs_motif", "gene_callers_id")


    @cached_property
    def rbs_spacer_length(self) -> pl.LazyFrame:
        df = self.gene_caller_df.select("gene_callers_id", "rbs_spacer")
        df = df.with_columns(pl.col("rbs_spacer").replace_strict(space_dict, return_dtype=pl.List(pl.Int16)).alias("rbs_spacer_length"))

        return df.select("gene_callers_id", "rbs_spacer_length")


    @cached_property
    def gc_content(self) -> pl.LazyFrame:
        return self.gene_caller_df.select("gene_callers_id", "gc_cont")


    @lru_cache
    def is_significantly_different_between_samples(self, samples: list[str] = None,
                                                   baseline: str | bool = False) -> pl.DataFrame:
        if samples is None:
            samples = ["top", "bottom"]
        return add_rao_score_by_gene(self.methylation_data.collect(streaming=True), samples, baseline)


    @lru_cache
    def get_function(self, source: str | list[str] | None = None) -> pl.LazyFrame:
        if source is None:
            return self.functional_df.select("gene_callers_id", "function")
        elif type(source) is list:
            return self.functional_df.filter(pl.col("source").is_in(source)).select("gene_callers_id", "function")
        else:
            return self.functional_df.filter(pl.col("source").eq(source)).select("gene_callers_id", "function")


    def get_flanking_sequence(self, relative_position: int,
                              seq_range: tuple[int, int] | tuple[pl.Expr, int] | tuple[pl.Expr, pl.Expr] | tuple[
                                  int, pl.Expr]) -> pl.LazyFrame:
        """
        Extracts the sequence from the contig around a relative position in the gene.
        All coordinates are relative to the gene's translation direction.

        Parameters:
            relative_position (int): Position relative to the gene start (zero-based).
            seq_range (Tuple[int, int]): (start_offset, end_offset) around the relative position.

        Returns:
            pl.LazyFrame: A dataframe with the sequence in the "sequence" column
        """
        start_offset, end_offset = seq_range
        if type(start_offset) is int and type(end_offset) is int:
            assert start_offset <= end_offset, "Give a gene slice sequence_range in order"

        # Get the Df we need
        df = self.gene_caller_df.select("contig", "start", "stop", "direction", "gene_callers_id")
        df = df.with_columns(pl.col("direction").eq("+")).rename({"direction": "strand"})

        # Add RBS if needed, filter for RBS spacer length non-null
        if type(start_offset) is pl.Expr or type(end_offset) is pl.Expr:
            df = df.join(self.rbs_spacer_length, on="gene_callers_id")  # In case, expr uses RBS spacer length
            df = df.filter(pl.col("rbs_spacer_length").is_not_null())

        # Get full sequences first and their length
        sequences = {}
        for key, value in get_genomic_sequence(self.genome.name).items():
            sequences[key] = str(value.seq)
        sequences = {"contig": sequences.keys(), "sequence": sequences.values()}
        sequences = pl.from_dict(sequences, schema=["contig", "sequence"]).lazy()

        df = df.join(sequences, on="contig")

        # Figure out coordinate
        df = df.with_columns(pl.when(pl.col("strand"))
                             .then(pl.col("start").add(relative_position + start_offset))
                             .otherwise(pl.col("stop").sub(relative_position + end_offset + 1)).alias("start_pos"))
        df = df.with_columns(pl.when(pl.col("strand"))
                             .then(pl.col("start").add(relative_position + end_offset))
                             .otherwise(pl.col("stop").sub(relative_position + start_offset + 1)).alias("end_pos"))

        # Bound the slice range
        df = df.with_columns(pl.when(pl.col("start_pos") < 0).then(0).otherwise(pl.col("start_pos")).alias("start_pos"))
        df = df.with_columns(pl.when(pl.col("end_pos") < 0).then(0).otherwise(pl.col("end_pos")).alias("end_pos"))

        # Slice sequence
        # This is needed because when.then runs all then masks, which causes a negative value error
        # In slice the params are (start, length), start is included in length so dragonfruit (4, 3) gives "onf"
        # So + 1 is needed to make this an inclusive range
        df1 = df.filter(pl.col("strand")).with_columns(pl.col("sequence").str.slice(pl.col("start_pos"), pl.col("end_pos") - pl.col("start_pos") + 1))
        df2 = df.filter(pl.col("strand").eq(False)).with_columns(pl.col("sequence").str.slice(pl.col("start_pos"), pl.col("end_pos") - pl.col("start_pos") + 1))
        df = pl.concat([df1, df2])

        # Take the reverse complement if strand is negative
        df = df.with_columns(pl.when(pl.col("strand"))
                             .then(pl.col("sequence"))
                             .otherwise(pl.col("sequence").map_elements(lambda x: str(SeqRecord.Seq(x).reverse_complement()), return_dtype=pl.String)))


        return df.select("gene_callers_id", "sequence")


    @lru_cache
    def load_flanking_methylation_data(self, relative_position: int = 0, meth_range: (int, int) = (-10, 10)) -> pl.LazyFrame:
        """
        Extracts the methylationd data from the contig around a relative position in the gene.
        All coordinates are relative to the gene's translation direction.

        Parameters:
            relative_position (int): Position relative to the gene start (zero-based).
            seq_range (Tuple[int, int]): (start_offset, end_offset) around the relative position inclusive.

        Returns:
            pl.LazyFrame: A dataframe with the sequence in the "sequence" column
        """

        start_offset, end_offset = meth_range
        assert start_offset <= end_offset, "Give a gene slice sequence_range in order"

        # Get the gene information we need
        df = self.gene_caller_df.select("gene_callers_id", "contig", "start", "stop", "direction")
        df = df.with_columns(pl.col("direction").eq("+")).rename({"direction": "strand"})

        # Figure out coordinates for each gene. The + 1's are needed because stop is exclusive, and we want inclusive.u
        df = df.with_columns(pl.when(pl.col("strand"))
                             .then(pl.col("start").add(relative_position + start_offset))
                             .otherwise(pl.col("stop").sub(relative_position + end_offset + 1)).alias("region_start"))

        df = df.with_columns(pl.when(pl.col("strand"))
                             .then(pl.col("start").add(relative_position + end_offset))
                             .otherwise(pl.col("stop").sub(relative_position + start_offset + 1)).alias("region_end"))

        # Get methylation data
        methyl_data = self.genome.load_all_methylation_data().with_columns(
            contig=pl.col('name').str.split(by='|').list.get(0),
            strand=pl.col('name').str.split(by='|').list.get(1).eq("+"),
            position=pl.col('name').str.split(by='|').list.get(2).cast(pl.Int64),
        )

        # Filter methylation data
        methyl_data = methyl_data.join_where(df,
                                             pl.col("contig").eq(pl.col("contig_right")),
                                             pl.col("strand").eq(pl.col("strand_right")),
                                             pl.col("position").ge(pl.col("region_start")),
                                             pl.col("position").le(pl.col("region_end")))

        # Take the reverse complement if strand is negative
        methyl_data = methyl_data.with_columns(pl.when(pl.col("strand"))
                                               .then(pl.col("position").sub(pl.col("start") + relative_position))
                                               .otherwise(pl.col("stop") - relative_position - 1 - pl.col("position")).alias("position"))

        return methyl_data.select("gene_callers_id", "position", "total_methylation", "sample", *readable_methylation_name.keys())

