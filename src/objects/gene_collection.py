from __future__ import annotations
from functools import lru_cache, cached_property
from src.utilities.utils import add_gene_caller_id, readable_modification_name
from Bio import SeqRecord
from src.utilities.data_loading import get_dataset_genes
import polars as pl
import os
from typing import TYPE_CHECKING
import glob
import subprocess
from pathlib import Path
import shutil

if TYPE_CHECKING:  # Only for type hints
    from gene import Gene
    from genome import Genome

try:
    from src.utilities.raobust import add_rao_score_by_gene
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
              "TATAA": ["TATAA"],
              "AAAAAA": ["AAAAAA"],
              None: [None]
}


class GeneCollection(object):

    def __init__(self, ids: list[int], genome: Genome):
        self.ids: list[int] = ids
        self.genome: Genome = genome
        self._index = 0

        self._load_data()


    def _load_data(self) -> None:
        self.gene_caller_df: pl.LazyFrame = get_dataset_genes(self.genome).filter(
            pl.col("gene_callers_id").is_in(self.ids))
        self.functional_df: pl.lazyframe = pl.scan_csv(f"{self.genome._methylation_data_dir}/../function-calls.txt", separator="\t").filter(
            pl.col("gene_callers_id").is_in(self.ids))


    def __getitem__(self, item) -> Gene | list[Gene]:
        from src.objects.gene import Gene
        if isinstance(item, slice):
            return [Gene(self.ids[i]) for i in self.ids[item]]
        return Gene(self.ids[item])


    def __len__(self):
        return len(self.ids)


    @cached_property
    def methylation_data(self) -> pl.LazyFrame:
        # Figure out coordinates
        region_filter = (self.gene_caller_df
                         .select("gene_callers_id", "contig", "start", "stop", "strand")
                         .with_columns(pl.col("stop").sub(1))  # in gene caller df stop is exclusive, in filter it is inclusive
                         .rename({"contig": "filter_contig", "strand": "filter_strand",
                                  "start": "filter_start", "stop": "filter_end"
                                  }))

        methylation_data: pl.LazyFrame = self.genome.load_region_methylation_data(region_filter=region_filter)

        methylation_data = add_gene_caller_id(methylation_data, self.gene_caller_df)
        methylation_data = methylation_data.drop("position", "strand", "contig")
        methylation_data = methylation_data.filter(pl.col("gene_callers_id").is_in(self.ids))

        # Do a group by gene_callers_id and then do a subtraction of the start position
        methylation_data = methylation_data.join(self.start, on="gene_callers_id")
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
        df = self.gene_caller_df.select("gene_callers_id", "strand")
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
        return self.gene_caller_df.select("gene_callers_id", "start_codon_sequence")


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
        print(df.collect().get_column("rbs_motif").to_list())
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
        df = df.join(self.get_flanking_sequence(0, (-pl.col("rbs_spacer_length").list.max() - 12, 0)), on="gene_callers_id")
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
        print(df.collect().get_column("rbs_spacer").to_list())
        df = df.with_columns(pl.col("rbs_spacer").replace_strict(space_dict, return_dtype=pl.List(pl.Int16)).alias("rbs_spacer_length"))

        return df.select("gene_callers_id", "rbs_spacer_length")


    @cached_property
    def pribnow_box_position_and_sequence(self) -> pl.LazyFrame:
        search_window_length = 20
        search_sequence = "TATAAT"
        seqs = self.get_flanking_sequence(0, (-search_window_length, 0))
        df = seqs.with_columns(pl.col('sequence').str.find(search_sequence, literal=True).add(-search_window_length + len(search_sequence) - 1).alias("pribnow_box_position"))
        df = df.with_columns(pl.when(pl.col("pribnow_box_position").is_not_null())
                             .then(pl.lit(search_sequence))
                             .otherwise(pl.lit(None)).alias("pribnow_box_sequence"))

        return df.select("gene_callers_id", "pribnow_box_position", "pribnow_box_sequence")


    @cached_property
    def minus_35_position_and_sequence(self) -> pl.LazyFrame:
        search_window_length = 40
        search_sequence = "TTGACA"
        seqs = self.get_flanking_sequence(0, (-search_window_length, 0))
        df = seqs.with_columns(pl.col('sequence').str.find(search_sequence, literal=True).add(
            -search_window_length + len(search_sequence) - 1).alias("minus_35_position"))
        df = df.with_columns(pl.when(pl.col("minus_35_position").is_not_null())
                             .then(pl.lit(search_sequence))
                             .otherwise(pl.lit(None)).alias("minus_35_sequence"))

        return df.select("gene_callers_id", "minus_35_position", "minus_35_sequence")


    @cached_property
    def gc_content(self) -> pl.LazyFrame:
        return self.gene_caller_df.select("gene_callers_id", "gc_cont")


    def is_significantly_different_between_samples(self, df: pl.LazyFrame, samples: list[str], baseline: str | bool) -> pl.DataFrame:
        df = df.collect(streaming=True)
        assert all(sample in df.get_column("sample").unique().to_list() for sample in samples)

        df = add_rao_score_by_gene(df, samples, baseline)

        return df.select("gene_callers_id", "test_result", "comparison", "rao_score")


    @lru_cache
    def get_function(self, source: str | list[str] | None = None) -> pl.LazyFrame:
        if source is None:
            return self.functional_df.select("gene_callers_id", "function", "source")
        elif type(source) is list:
            return self.functional_df.filter(pl.col("source").is_in(source)).select("gene_callers_id", "function")
        else:
            return self.functional_df.filter(pl.col("source").eq(source)).select("gene_callers_id", "function")


    def get_flanking_sequence(self, relative_position: int,
                              seq_range: tuple[int, int] | tuple[pl.Expr, int] | tuple[pl.Expr, pl.Expr] | tuple[
                                  int, pl.Expr]) -> pl.LazyFrame:
        """
        Extracts the sequence from the contig around a relative position in the gene.
        All coordinates are relative to the gene's transcription direction.

        Parameters:
            relative_position (int): Position relative to the gene start (zero-based).
            seq_range (Tuple[int, int]): (start_offset, end_offset) around the relative position.

        Returns:
            pl.LazyFrame: A dataframe with the sequence in the "sequence" column, reverse complemented when needed.
        """
        start_offset, end_offset = seq_range
        if type(start_offset) is int and type(end_offset) is int:
            assert start_offset <= end_offset, "Give a gene slice seq_range in order"

        # Get the Df we need
        df = self.gene_caller_df.select("contig", "start", "stop", "strand", "gene_callers_id")

        # Add RBS if needed, filter for RBS spacer length non-null
        if type(start_offset) is pl.Expr or type(end_offset) is pl.Expr:
            df = df.join(self.rbs_spacer_length, on="gene_callers_id")  # In case, expr uses RBS spacer length
            df = df.filter(pl.col("rbs_spacer_length").is_not_null())

        # Get full sequences first and their length
        sequences = {}
        for key, value in self.genome.sequence.items():
            sequences[key] = str(value.seq)
        sequences = {"contig": sequences.keys(), "sequence": sequences.values()}
        sequences = pl.from_dict(sequences, schema=["contig", "sequence"]).lazy()

        df = df.join(sequences, on="contig")

        # If relative position is negative handle that as being from gene end
        if relative_position < 0:
            relative_position = pl.col("stop") + relative_position + 1  # -1 should be the end

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
    def load_flanking_methylation_data(self, relative_position: int, meth_range: tuple[int, int] | tuple[pl.Expr, int] | tuple[pl.Expr, pl.Expr] | tuple[int, pl.Expr], triplicates_only: bool = True, common_only: bool = False) -> pl.LazyFrame:
        """
        Extracts the methylationd data from the contig around a relative position in the gene.
        All coordinates are relative to the gene's tran direction.

        Parameters:
            relative_position (int): Position relative to the gene start (zero-based).
            seq_range (Tuple[int, int]): (start_offset, end_offset) around the relative position inclusive.

        Returns:
            pl.LazyFrame: A dataframe with the sequence in the "sequence" column
        """

        start_offset, end_offset = meth_range
        if type(start_offset) is int and type(end_offset) is int:
            assert start_offset <= end_offset, "Give a gene slice meth-range in order"

        # Get the gene information we need
        df = self.gene_caller_df.select("gene_callers_id", "contig", "start", "stop", "strand")

        # Add RBS if needed, filter for RBS spacer length non-null
        if type(start_offset) is pl.Expr or type(end_offset) is pl.Expr:
            df = df.join(self.rbs_motif_and_relative_position.lazy(), on="gene_callers_id")  # In case, expr uses RBS spacer length
            df = df.filter(pl.col("rbs_motif_position").is_not_null())

        # If relative position is negative handle that as being from gene end
        if relative_position < 0:
            relative_position = pl.col("stop") + relative_position + 1  # -1 should be the end

        # Figure out coordinates for each gene. The + 1's are needed because stop is exclusive, and we want inclusive.
        df = df.with_columns(pl.when(pl.col("strand"))
                             .then(pl.col("start").add(relative_position + start_offset))
                             .otherwise(pl.col("stop").sub(relative_position + end_offset + 1)).alias("filter_start"))

        df = df.with_columns(pl.when(pl.col("strand"))
                             .then(pl.col("start").add(relative_position + end_offset))
                             .otherwise(pl.col("stop").sub(relative_position + start_offset + 1)).alias("filter_end"))

        # Get methylation data and filter using the filter df we built
        region_filter = df.select("contig", "strand", "filter_start", "filter_end")
        region_filter = region_filter.rename({"contig": "filter_contig", "strand": "filter_strand"})

        methyl_data = self.genome.load_region_methylation_data(region_filter=region_filter, triplicates_only=triplicates_only, common_only=common_only)

        # Add gene info
        methyl_data = methyl_data.join_where(df,
                                             pl.col("contig").eq(pl.col("contig_right")),
                                             pl.col("strand").eq(pl.col("strand_right")),
                                             pl.col("position").ge(pl.col("filter_start")),
                                             pl.col("position").le(pl.col("filter_end")))

        # Take the reverse complement if strand is negative
        methyl_data = methyl_data.with_columns(pl.when(pl.col("strand"))
                                               .then(pl.col("position").sub(pl.col("start") + relative_position))
                                               .otherwise(pl.col("stop") - relative_position - 1 - pl.col("position")).alias("position"))

        return methyl_data.select("gene_callers_id", "position", "sample", *readable_modification_name.keys())


    def get_entropy_for_region(self, relative_position: int, range: tuple[int, int] | tuple[pl.Expr, int] | tuple[pl.Expr, pl.Expr] | tuple[int, pl.Expr]) -> pl.DataFrame:
        start_offset, end_offset = range
        if type(start_offset) is int and type(end_offset) is int:
            assert start_offset <= end_offset, "Give a gene slice meth-range in order"

        # Get the gene information we need
        df = self.gene_caller_df.select("gene_callers_id", "contig", "start", "stop", "strand")

        # Add RBS if needed, filter for RBS spacer length non-null
        if type(start_offset) is pl.Expr or type(end_offset) is pl.Expr:
            df = df.join(self.rbs_motif_and_relative_position.lazy(),
                         on="gene_callers_id")  # In case, expr uses RBS spacer length
            df = df.filter(pl.col("rbs_motif_position").is_not_null())

        # If relative position is negative handle that as being from gene end
        if relative_position < 0:
            relative_position = pl.col("stop") + relative_position + 1  # -1 should be the end

        # Figure out coordinates for each gene. The + 1's are needed because stop is exclusive, and we want inclusive.
        df = df.with_columns(pl.when(pl.col("strand"))
                             .then(pl.col("start").add(relative_position + start_offset))
                             .otherwise(pl.col("stop").sub(relative_position + end_offset + 1)).alias("filter_start"))

        df = df.with_columns(pl.when(pl.col("strand"))
                             .then(pl.col("start").add(relative_position + end_offset))
                             .otherwise(pl.col("stop").sub(relative_position + start_offset + 1)).alias("filter_end"))

        # Get methylation data and filter using the filter df we built
        region_filter = df.select("contig", "strand", "filter_start", "filter_end", "gene_callers_id").collect(streaming=True)

        # Iterate over rows and execute modkit command
        bam_files = [Path(f) for f in glob.glob(str(self.genome._bam_dir / "*.bam"))]
        results = []
        for row in region_filter.iter_rows(named=True):
            # Construct the BED3 or BED4 string dynamically
            bed_entry = f"{row['contig']}\t{row['filter_start']}\t{row['filter_end']}\n"

            # Save the BED entry to a file
            bed_file_path = os.path.join(self.genome._bam_dir, f"{row['contig']}_{row['filter_start']}_{row['filter_end']}.bed")
            with open(bed_file_path, 'w') as bed_file:
                bed_file.write(bed_entry)

            for mod_bam in bam_files:
                sample = mod_bam.stem
                for base in ["A", "C"]:
                    out = os.path.join(self.genome._bam_dir, f"{row['contig']}_{row['filter_start']}_{row['filter_end']}_out")
                    # Construct the modkit entropy command
                    cmd = ["modkit",
                           "entropy",
                           "--threads", "8",
                           "--regions", bed_file_path,
                           "--base", base,
                           "--in-bam", mod_bam,
                           "--ref", self.genome.genome_path,
                           "-o", out
                    ]

                    # Execute the command and capture output
                    try:
                        process = subprocess.run(cmd, capture_output=True, check=True)

                        # Entropy in reality is mean_entropy
                        schema = ["contig", "start", "end", "region_name", "entropy", "strand", "median_entropy", "min_entropy", "max_entropy", "mean_num_reads", "min_num_reads", "max_num_reads", "successful_window_count", "failed_window_count"]
                        try:
                            df = pl.read_csv(out + "/regions.bed", separator="\t", has_header=False, new_columns=schema)
                            df = df.with_columns(pl.lit(row['gene_callers_id']).cast(pl.Int64).alias("gene_callers_id"),
                                                 pl.lit(base).alias("base"), pl.col("entropy").cast(pl.Float64),
                                                 pl.lit(sample).alias("sample"))
                            df = df.filter(pl.col("strand").eq(row["strand"])).select("entropy", "gene_callers_id", "start", "end", "strand", "base", "sample")
                            results.append(df)

                            shutil.rmtree(out)

                        except Exception as e:
                            if type(e) is pl.exceptions.NoDataError:
                                print("empty csv")
                                continue

                            print(f"Error parsing output: {e}")
                            print(f"Got std: {process.stdout}")
                            raise Exception

                    except subprocess.CalledProcessError as e:
                        print(f"Error running command for row {row}: {e.stderr}")
                        if "length is 1" in str(e.stderr):
                            continue
                        raise Exception

            # Delete the bed file
            os.remove(bed_file_path)

            # Concat results
            if len(results) == 0:
                return pl.DataFrame()

            results = pl.concat(results)
            return results

