from functools import lru_cache, cached_property
from src.utilities.utils import normalize_data_by_pileup, add_gene_caller_id, readable_methylation_name, readable_modification_name
from utilities.data_loading import *
from Bio import SeqRecord
from platform import system

try:
    from _statistics import get_rao_score, add_rao_score_by_gene
except:
    pass

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"../../methylation_data/methylation_5")
if system() == "Darwin":
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"../data/methylation_data/methylation_5")

min_coverage_default = 5

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


class Genome(object):

    def __init__(self, name: str):
        self._data_dir: str = data_dir
        if not self._is_valid_genome_name(name):
            raise ValueError(f"Genome {name} not found in the data directory.")

        self.name: str = name

    def _is_valid_genome_name(self, name: str) -> bool:
        # Check if genome exists in the data directory
        return os.path.exists(os.path.join(self._data_dir, name))

    @cached_property
    def sequence(self) -> dict[str, SeqRecord]:
        return get_genomic_sequence(self.name)

    @cached_property
    def gene_ids(self) -> list[int]:
        bed_files = [f for f in glob.glob(os.path.join(self._data_dir, self.name, "*.bed")) if
                     '-bedgraph' not in os.path.basename(f)]

        all_data = []
        for bed_file in bed_files:
            # Load the data for the positions that overlap only
            methyl_data = get_pileup_polars(bed_file).select("chrom", "inclusive start position",
                                                             "exclusive end position", "strand")
            methyl_data = methyl_data.rename({"chrom": "contig", "inclusive start position": "start",
                                              "exclusive end position": "end", })  #"strand": "direction"})

            all_data.append(methyl_data)

        all_data = pl.concat(all_data)
        all_genes = get_genes_polars(self._data_dir)
        gene_ids = add_gene_caller_id(all_data, all_genes).select("gene_callers_id").unique().collect(
            streaming=True).get_column("gene_callers_id").to_list()
        return gene_ids

    @lru_cache
    def load_all_methylation_data(self, coverage: int = min_coverage_default) -> pl.LazyFrame:
        return self.load_region_methylation_data(coverage, None)


    def load_region_methylation_data(self, coverage: int = min_coverage_default,
                                     region_filter: pl.Expr | None = None) -> pl.LazyFrame:
        # Get all the bed files for this genome
        bed_files = [f for f in glob.glob(os.path.join(self._data_dir, self.name, "*.bed")) if
                     '-bedgraph' not in os.path.basename(f)]

        all_data = []
        for bed_file in bed_files:
            # Load the data for the positions that overlap only
            methyl_data = get_pileup_polars(bed_file)
            if region_filter is not None:
                methyl_data = methyl_data.filter(region_filter)

            methyl_data = utils.reshape_pileup_to_matrix_polars(methyl_data)
            if methyl_data is None:
                continue

            # Add sample column
            sample_name = os.path.basename(bed_file).split(".")[0]
            methyl_data = methyl_data.with_columns(sample=pl.lit(sample_name))

            all_data.append(methyl_data)

        result = pl.concat(all_data)

        # Filter for coverage of at least 5 and no full Null/NaN values
        modification_types = list(readable_modification_name.keys())
        result = (result.filter(
            pl.any_horizontal(pl.col(modification_types).is_not_null() & pl.col(modification_types).is_not_nan()) &
            pl.concat_list(modification_types).list.sum().ge(coverage)))

        # Normalize to fraction
        result = normalize_data_by_pileup(result)

        # Create total methylation column
        methylation_types = list(readable_methylation_name.keys())
        result = result.with_columns(pl.concat_list(methylation_types).list.sum().alias("total_methylation"))

        return result


class Gene(object):

    def __init__(self, id: int, genome: Genome):
        self.id: int = id
        self._data_dir: str = data_dir
        self.genome: Genome = genome
        self.rbs_motif: str = ""

        self._load_data()
        self._load_gene_methylation_data()
        _ = self.rbs_motif_position  # Used to set self.rbs_motif

    @lru_cache
    def _load_data(self) -> None:
        self.gene_caller_df: pl.LazyFrame = get_genes_polars(self._data_dir).filter(
            pl.col("gene_callers_id").eq(self.id))
        self.functional_df: pl.lazyframe = pl.scan_csv(f"{self._data_dir}/function-calls.txt", separator="\t").filter(
            pl.col("gene_callers_id").eq(self.id))

    @lru_cache
    def _load_gene_methylation_data(self, coverage: int = min_coverage_default) -> None:
        strand = "+" if self.strand else "-"
        filter = (pl.col("chrom").eq(self.contig) &
                  pl.col("inclusive start position").ge(self.start) &
                  pl.col("exclusive end position").le(self.stop) &
                  pl.col("strand").eq(strand))

        self.methylation_data: pl.LazyFrame = self.genome.load_region_methylation_data(coverage, region_filter=filter)
        self.methylation_data = self.methylation_data.with_columns(
            pl.col('name').str.split(by='|').list.get(2).cast(pl.Int64).sub(self.start).alias("position")).drop("name")

    @cached_property
    def contig(self) -> str:
        return self.gene_caller_df.select("contig").collect(streaming=True).item()

    @cached_property
    def start(self) -> int:
        return self.gene_caller_df.select("start").collect(streaming=True).item()

    @cached_property
    def stop(self) -> int:
        return self.gene_caller_df.select("stop").collect(streaming=True).item()

    @cached_property
    def strand(self) -> bool:
        return self.gene_caller_df.select("direction").collect(streaming=True).item() == "+"

    @cached_property
    def sources(self) -> list[str]:
        return self.functional_df.select("source").collect(streaming=True).to_list()

    @cached_property
    def sequence(self) -> SeqRecord:
        record_from_fasta = self.get_flanking_sequence(0, (0, self.length - 1))
        record_from_gene_caller = self.gene_caller_df.select("sequence").collect(streaming=True).item()
        assert str(record_from_fasta) == record_from_gene_caller
        return record_from_fasta

    @cached_property
    def length(self) -> int:
        return self.stop - self.start

    @cached_property
    def is_end_missing(self) -> bool:
        return self.gene_caller_df.select("partial_begin").collect(streaming=True).item()

    @cached_property
    def is_start_missing(self) -> bool:
        return self.gene_caller_df.select("partial_end").collect(streaming=True).item()

    @cached_property
    def start_codon(self) -> str | None:
        if self.is_start_missing:
            return None
        start_str = self.gene_caller_df.select("start_type").collect(streaming=True).item()
        start_seq = self.sequence[:3]
        assert start_str in ["ATG", "GTG", "TTG"], f"Start codon {start_str} unknown"
        assert start_str == start_seq, f"Start codon from gene caller and sequence don't match"
        return start_str

    @cached_property
    def start_codon_position(self) -> int | None:
        if self.is_start_missing:
            return None
        assert self.sequence.index(self.start_codon) == 0
        return 0

    @cached_property
    def stop_codon(self) -> str | None:
        if self.is_end_missing:
            return None

        stop_codon = self.sequence[-3:]
        assert stop_codon in ["TAG", "TAA", "TGA"], f"Stop codon {stop_codon} not found at end of sequence"
        return stop_codon

    @cached_property
    def stop_codon_position(self) -> int | None:
        if self.is_end_missing:
            return None

        assert self.sequence[self.length - 3:] == self.stop_codon
        return self.length - 3

    @cached_property
    def candidate_rbs_motifs(self) -> str | None:
        motif = self.gene_caller_df.select("rbs_motif").collect(streaming=True).item()
        return rbs_motifs[motif]

    @cached_property
    def rbs_motif_position(self) -> int | None:
        if self.rbs_spacer_length is None or self.candidate_rbs_motifs[0] is None:
            return None
        else:
            for candidate_motif in self.candidate_rbs_motifs:
                try:
                    right_flank = -self.rbs_spacer_length[1] - len(candidate_motif) * 2
                    flanking = self.get_flanking_sequence(0, (right_flank, 25))
                    res = flanking.index(candidate_motif)+len(candidate_motif) + right_flank
                    self.rbs_motif: str = candidate_motif
                    return res

                except ValueError:
                    continue

        raise ValueError

    @cached_property
    def rbs_spacer_length(self) -> tuple | None:
        spacer_str = self.gene_caller_df.select("rbs_spacer").collect(streaming=True).item()
        return space_dict[spacer_str]

    @cached_property
    def gc_content(self) -> float:
        return self.gene_caller_df.select("gc_cont").collect(streaming=True).item()

    @lru_cache
    def is_significantly_different_between_samples(self, samples: list[str] = None,
                                                   baseline: str | bool = False) -> bool:
        if samples is None:
            samples = ["top", "bottom"]
        _, is_diff, _ = get_rao_score(self.methylation_data, samples, baseline)
        return is_diff


    @lru_cache
    def get_function(self, source: str) -> str:
        return self.functional_df.filter(pl.col("source").eq(source)).select("function").collect(streaming=True).item()


    @lru_cache
    def get_mean_methylation(self, sample: str = None, position: int | tuple[int] = None, methylation_type: str = None):
        """
        Gets the mean methylation value for this gene.

        If position is specified, restricts itself to that position (or inclusive sequence_range if tuple). Else, entire gene.
        If sample is specified, restricts itself to that sample. Else, all samples.
        If methylation_type is specified, restricts itself to that type. Else, all types.

        :param sample:
        :type sample:
        :param position:
        :type position:
        :param methylation_type:
        :type methylation_type:
        :return:
        :rtype:
        """
        mean_data = self.methylation_data

        # Sample
        if sample:
            mean_data = mean_data.filter(pl.col("sample").eq(sample))
        mean_data = mean_data.drop("sample")

        # Position
        if isinstance(position, int):
            mean_data = mean_data.filter(pl.col("position").eq(position))

        elif isinstance(position, tuple):
            mean_data = mean_data.filter(pl.col("position").ge(position[0]) &
                                         pl.col("position").le(position[1]))
        mean_data = mean_data.drop("position")

        # Methylation type
        meth_columns = utils.readable_methylation_name.keys()
        if methylation_type:
            mean_data = mean_data.select(methylation_type)
            meth_columns.remove(methylation_type)
            mean_data = mean_data.drop(meth_columns)
        else:
            mean_data = mean_data.with_columns(pl.concat_list(meth_columns).list.sum().alias("total_methylation"))
            mean_data = mean_data.drop(meth_columns)

        # Now take the mean
        return mean_data.mean().collect(streaming=True).item()


    @lru_cache
    def get_flanking_sequence(self, relative_position: int, seq_range: (int, int)) -> SeqRecord.Seq:
        """
        Extracts the sequence from the contig around a relative position in the gene.

        Parameters:
            relative_position (int): Position relative to the gene start (zero-based).
            seq_range (Tuple[int, int]): (start_offset, end_offset) around the relative position.

        Returns:
            Seq: The extracted sequence.
        """
        sequence = get_genomic_sequence(self.genome.name)[self.contig].seq
        seq_length = len(sequence)
        start_offset, end_offset = seq_range
        assert start_offset <= end_offset, "Give a gene slice sequence_range in order"

        if self.strand:  # Positive strand
            # Calculate genomic position of the relative position
            genomic_position = self.start + relative_position

            # Calculate start and end positions
            start_pos = genomic_position + start_offset
            end_pos = genomic_position + end_offset

            # Ensure positions are within sequence bounds
            start_pos = max(0, start_pos)
            end_pos = min(seq_length, end_pos)

            # Extract sequence
            seq_slice = sequence[start_pos:end_pos + 1]  # +1 because our sequence_range is inclusive

        else:  # Negative strand
            # Calculate genomic position of the relative position
            genomic_position = self.stop - relative_position

            # Calculate start and end positions
            start_pos = genomic_position - start_offset
            end_pos = genomic_position - end_offset

            # Ensure positions are within sequence bounds
            start_pos = min(seq_length - 1, start_pos)
            end_pos = max(0, end_pos)

            # Extract sequence
            seq_slice = sequence[end_pos - 1:start_pos]  # -1 because our sequence_range is inclusive
            seq_slice = seq_slice.reverse_complement()

        return seq_slice


    @lru_cache
    def load_flanking_methylation_data(self, relative_position: int = 0, meth_range: (int, int) = (-10, 10),
                                       coverage: int = min_coverage_default, ) -> pl.LazyFrame:

        start_offset, end_offset = meth_range
        assert start_offset <= end_offset, "Give a gene slice sequence_range in order"

        if self.strand:
            # Create a filter for the region
            region_start = self.start + relative_position + start_offset
            region_end = self.start + relative_position + end_offset
            region_filter = (pl.col("chrom").eq(self.contig) &
                             pl.col("inclusive start position").ge(region_start) &
                             pl.col("exclusive end position").le(region_end) &
                             pl.col("strand").eq("+"))

            # Load the methylation data
            methylation_data = self.genome.load_region_methylation_data(coverage, region_filter)
            methylation_data = methylation_data.with_columns(
                pl.col('name').str.split(by='|').list.get(2).cast(pl.Int64).sub(self.start + relative_position).alias(
                    "position"))

        else:
            # Create a filter for the region
            region_start = self.stop - relative_position - start_offset
            region_end = self.stop - relative_position - end_offset
            region_filter = (pl.col("chrom").eq(self.contig) &
                             pl.col("inclusive start position").ge(region_end) &
                             pl.col("exclusive end position").le(region_start) &
                             pl.col("strand").eq("-"))

            # Load the methylation data
            methylation_data = self.genome.load_region_methylation_data(coverage, region_filter)
            methylation_data = methylation_data.with_columns(pl.lit(self.stop - relative_position).sub(
                pl.col('name').str.split(by='|').list.get(2).cast(pl.Int64)).alias("position"))

        return methylation_data.drop("name")


class GeneCollection(object):

    def __init__(self, ids: list[int], genome: Genome):
        self._data_dir: str = data_dir
        self.ids: list[int] = ids
        self.genome: Genome = genome

        self._load_data()


    def _load_data(self) -> None:
        self.gene_caller_df: pl.LazyFrame = get_genes_polars(self._data_dir).filter(
            pl.col("gene_callers_id").is_in(self.ids))
        self.functional_df: pl.lazyframe = pl.scan_csv(f"{self._data_dir}/function-calls.txt", separator="\t").filter(
            pl.col("gene_callers_id").is_in(self.ids))


    @cached_property
    def methylation_data(self) -> pl.LazyFrame:
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
        return self.gene_caller_df.select("gene_callers_id", "start")


    @cached_property
    def stop(self) -> pl.LazyFrame:
        return self.gene_caller_df.select("gene_callers_id", "stop")


    @cached_property
    def strand(self) -> pl.LazyFrame:
        df = self.gene_caller_df.select("gene_callers_id", "direction").with_columns(pl.col("direction").eq("+"))
        df = df.rename({"direction": "strand"})
        return df


    @cached_property
    def sources(self) -> pl.LazyFrame:
        return self.functional_df.select("gene_callers_id", "source")


    @cached_property
    def sequence(self) -> pl.LazyFrame:
        return self.gene_caller_df.select("gene_callers_id", "sequence")


    @cached_property
    def length(self) -> pl.LazyFrame:
        df = self.start.join(self.stop, on="gene_callers_id")
        df = df.with_columns(pl.col("stop").sub(pl.col("start")).alias("length"))
        df = df.select("gene_callers_id", "length")
        return df


    @cached_property
    def is_end_missing(self) -> pl.LazyFrame:
        return self.gene_caller_df.select("gene_callers_id", "partial_begin")


    @cached_property
    def is_start_missing(self) -> pl.LazyFrame:
        return self.gene_caller_df.select("gene_callers_id", "partial_end")


    @cached_property
    def start_codon(self) -> pl.LazyFrame:
        return self.gene_caller_df.select("gene_callers_id", "start_type")


    @cached_property
    def start_codon_position(self) -> pl.LazyFrame:
        df = self.gene_caller_df.select("gene_callers_id")
        df = df.with_columns(pl.lit(0).alias("start_codon_position"))
        return df


    @cached_property
    def stop_codon(self) -> pl.LazyFrame:
        df = self.sequence.with_columns(pl.lit("sequence").str.slice(-3).alias("stop_codon"))
        return df.select("gene_callers_id", "stop_codon")


    @cached_property
    def stop_codon_position(self) -> pl.LazyFrame:
        df = self.length.with_columns(pl.col("length").sub(3).alias("stop_codon_position"))
        df = df.select("gene_callers_id", "stop_codon_position")
        return df


    @cached_property
    def candidate_rbs_motifs(self) -> pl.LazyFrame:
        df = self.gene_caller_df.select("gene_callers_id", "rbs_motif")
        df = df.with_columns(pl.col("rbs_motif").replace_strict(rbs_motifs, return_dtype=pl.List(pl.String)).alias("candidate_rbs_motif"))
        df = df.select("gene_callers_id", "candidate_rbs_motif")
        return df


    @cached_property
    def rbs_motif_and_position(self) -> pl.DataFrame:

        def find_rbs_motif_positon(row):
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
                        res = flanking.index(candidate_motif) + len(candidate_motif) - rbs_spacer_length[1] - 12
                        rbs_motif: str = candidate_motif
                        return res, rbs_motif

                    except ValueError:
                        continue

            raise ValueError

        df = self.candidate_rbs_motifs.join(self.rbs_spacer_length, on="gene_callers_id")
        df = df.join(self.get_flanking_sequence(0, (-pl.col("rbs_spacer_length").list.last() - 12, 0)), on="gene_callers_id")
        df = df.collect(streaming=True)

        # map_rows requires return_dtype and can't specify different types
        result = {"rbs_motif_position": [], "rbs_motif": []}
        for row in df.iter_rows():
            res, rbs_motif = find_rbs_motif_positon(row)
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
    def get_function(self, source: str) -> pl.LazyFrame:
        return self.functional_df.filter(pl.col("source").eq(source)).select("gene_callers_id", "function")


    def get_flanking_sequence(self, relative_position: int,
                              seq_range: tuple[int, int] | tuple[pl.Expr, int] | tuple[pl.Expr, pl.Expr] | tuple[
                                  int, pl.Expr]) -> pl.LazyFrame:
        """
        Extracts the sequence from the contig around a relative position in the gene.

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
                             .otherwise(pl.col("stop").sub(relative_position + start_offset)).alias("start_pos"))
        df = df.with_columns(pl.when(pl.col("strand"))
                             .then(pl.col("start").add(relative_position + end_offset))
                             .otherwise(pl.col("stop").sub(relative_position + end_offset)).alias("end_pos"))

        # Bound the slice range
        df = df.with_columns(pl.when(pl.col("start_pos") < 0).then(0).otherwise(pl.col("start_pos")).alias("start_pos"))
        df = df.with_columns(pl.when(pl.col("end_pos") < 0).then(0).otherwise(pl.col("end_pos")).alias("end_pos"))

        # Slice sequence
        # This is needed because when.then runs all then masks, which causes a negative value error
        df1 = df.filter(pl.col("strand")).with_columns(pl.col("sequence").str.slice(pl.col("start_pos"), pl.col("end_pos") + 1 - pl.col("start_pos")))
        df2 = df.filter(pl.col("strand").eq(False)).with_columns(pl.col("sequence").str.slice(pl.col("end_pos")-1, pl.col("start_pos")+1 - pl.col("end_pos")))
        df = pl.concat([df1, df2])

        # Take the reverse complement if strand is negative
        df = df.with_columns(pl.when(pl.col("strand"))
                             .then(pl.col("sequence"))
                             .otherwise(pl.col("sequence").map_elements(lambda x: str(SeqRecord.Seq(x).reverse_complement()), return_dtype=pl.String)))


        return df.select("gene_callers_id", "sequence")


    @lru_cache
    def load_flanking_methylation_data(self, relative_position: int = 0,
                                       meth_range: (int, int) = (-10, 10), coverage: int = min_coverage_default) -> pl.LazyFrame:

        start_offset, end_offset = meth_range
        assert start_offset <= end_offset, "Give a gene slice sequence_range in order"

        # Get the gene information we need
        df = self.gene_caller_df.select("gene_callers_id", "contig", "start", "stop", "direction")
        df = df.with_columns(pl.col("direction").eq("+")).rename({"direction": "strand"})

        # Figure out coordinates for each gene
        df = df.with_columns(pl.when(pl.col("strand"))
                             .then(pl.col("start").add(relative_position + start_offset))
                             .otherwise(pl.col("stop").sub(relative_position + end_offset)).alias("region_start"))
        df = df.with_columns(pl.when(pl.col("strand"))
                             .then(pl.col("start").add(relative_position + end_offset))
                             .otherwise(pl.col("stop").sub(relative_position + start_offset)).alias("region_end"))

        # Get methylation data
        methyl_data = self.genome.load_all_methylation_data(coverage).with_columns(
            contig=pl.col('name').str.split(by='|').list.get(0),
            strand=pl.col('name').str.split(by='|').list.get(1).eq("+"),
            position=pl.col('name').str.split(by='|').list.get(2).cast(pl.Int64),
        )

        # Filter methylation data
        methyl_data = methyl_data.join_where(df,
                                             pl.col("strand").eq(pl.col("strand_right")),
                                             pl.col("position").ge(pl.col("region_start")),
                                             pl.col("position").le(pl.col("region_end")))

        # Take the reverse complement if strand is negative
        methyl_data = methyl_data.with_columns(pl.when(pl.col("strand"))
                                               .then(pl.col("position").sub(pl.col("start") + relative_position))
                                               .otherwise(pl.col("stop") - relative_position - pl.col("position")).alias("position"))

        return methyl_data.select("gene_callers_id", "position", "total_methylation", "sample", *readable_methylation_name.keys())
