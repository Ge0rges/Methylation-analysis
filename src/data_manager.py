from src.utilities.utils import normalize_data_by_pileup, add_gene_caller_id
from utilities.data_loading import *
from Bio import SeqRecord
import src.utilities.utils as utils
import platform

try:
    from _statistics import get_rao_score
except:
    pass

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"../../methylation_data/methylation_5")
if platform.system() == "Darwin":
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"../data/methylation_data/methylation_5")

min_coverage_default = 5


class Genome(object):

    def __init__(self, name: str):
        self._data_dir: str = data_dir
        if not self._is_valid_genome_name(name):
            raise ValueError(f"Genome {name} not found in the data directory.")

        self.name: str = name


    def _is_valid_genome_name(self, name: str) -> bool:
        # Check if genome exists in the data directory

        return os.path.exists(os.path.join(self._data_dir, name))

    @property
    def sequence(self) -> SeqRecord:
        return get_genomic_sequence(self.name)


    @property
    def gene_ids(self) -> list[int]:
        bed_files = [f for f in glob.glob(os.path.join(self._data_dir, self.name, "*.bed")) if
                     '-bedgraph' not in os.path.basename(f)]

        all_data = []
        for bed_file in bed_files:
            # Load the data for the positions that overlap only
            methyl_data = get_pileup_polars(bed_file).select("chrom", "inclusive start position", "exclusive end position", "strand")
            methyl_data = methyl_data.rename({"chrom": "contig", "inclusive start position": "start",
                                              "exclusive end position": "end", "strand": "direction"})

            all_data.append(methyl_data)

        all_data = pl.concat(all_data)
        all_genes = get_genes_polars(self._data_dir)
        gene_ids = add_gene_caller_id(all_data, all_genes).select("gene_callers_id").unique().collect(streaming=True).get_column("gene_callers_id").to_list()
        return gene_ids


    def load_all_methylation_data(self, coverage: int = min_coverage_default) -> pl.LazyFrame:
        return self.load_region_methylation_data(coverage, None)


    def load_region_methylation_data(self, coverage: int = min_coverage_default, region_filter: pl.Expr | None = None) -> pl.LazyFrame:
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
        modification_types = list(utils.readable_modification_name.keys())
        result = (result.filter(
            pl.any_horizontal(pl.col(modification_types).is_not_null() & pl.col(modification_types).is_not_nan()) &
            pl.concat_list(modification_types).list.sum().ge(coverage)))

        # Normalize to fraction
        result = normalize_data_by_pileup(result)

        # Create total methylation column
        methylation_types = list(utils.readable_methylation_name.keys())
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


    def _load_data(self) -> None:
        self.gene_caller_df: pl.LazyFrame = get_genes_polars(self._data_dir).filter(
            pl.col("gene_callers_id").eq(self.id)).collect(streaming=True).lazy()
        self.functional_df: pl.lazyframe = pl.scan_csv(f"{self._data_dir}/function-calls.txt", separator="\t").filter(
            pl.col("gene_callers_id").eq(self.id)).collect(streaming=True).lazy()


    def _load_gene_methylation_data(self, coverage: int = min_coverage_default) -> None:
        strand = "+" if self.strand else "-"
        filter = (pl.col("chrom").eq(self.contig) &
                  pl.col("inclusive start position").ge(self.start) &
                  pl.col("exclusive end position").le(self.stop) &
                  pl.col("strand").eq(strand))

        self.methylation_data: pl.LazyFrame = self.genome.load_region_methylation_data(coverage, region_filter=filter)
        self.methylation_data = self.methylation_data.with_columns(pl.col('name').str.split(by='|').list.get(2).cast(pl.Int64).sub(self.start).alias("position")).drop("name")


    @property
    def contig(self) -> str:
        return self.gene_caller_df.select("contig").collect(streaming=True).item()


    @property
    def start(self) -> int:
        return self.gene_caller_df.select("start").collect(streaming=True).item()


    @property
    def stop(self) -> int:
        return self.gene_caller_df.select("stop").collect(streaming=True).item()


    @property
    def strand(self) -> bool:
        return self.gene_caller_df.select("direction").collect(streaming=True).item() == "+"


    @property
    def sources(self) -> list[str]:
        return self.functional_df.select("source").collect(streaming=True).to_list()


    @property
    def sequence(self) -> SeqRecord:
        record_from_fasta = get_genomic_sequence(self.genome.name)[self.contig][self.start:self.stop].seq
        if not self.strand:
            record_from_fasta = record_from_fasta.reverse_complement()
        record_from_gene_caller = self.gene_caller_df.select("sequence").collect(streaming=True).item()
        assert str(record_from_fasta) == record_from_gene_caller
        return record_from_fasta


    @property
    def length(self) -> int:
        return self.stop - self.start


    @property
    def is_end_missing(self) -> str:
        return self.gene_caller_df.select("partial_begin").collect(streaming=True).item()


    @property
    def is_start_missing(self) -> str:
        return self.gene_caller_df.select("partial_end").collect(streaming=True).item()


    @property
    def start_codon(self) -> str | None:
        if self.is_start_missing:
            return None
        start_str = self.gene_caller_df.select("start_type").collect(streaming=True).item()
        start_seq = self.sequence[:3]
        assert start_str in ["ATG", "GTG", "TTG"], f"Start codon {start_str} unknown"
        assert start_str == start_seq, f"Start codon from gene caller and sequence don't match"
        return start_str


    @property
    def start_codon_position(self) -> int | None:
        if self.is_start_missing:
            return None
        assert self.sequence.index(self.start_codon) == 0
        return 0


    @property
    def stop_codon(self) -> str | None:
        if self.is_end_missing:
            return None

        stop_codon = self.sequence[-3:]
        assert stop_codon in ["TAG", "TAA", "TGA"], f"Stop codon {stop_codon} not found at end of sequence"
        return stop_codon


    @property
    def stop_codon_position(self) -> int | None:
        if self.is_end_missing:
            return None

        assert self.sequence[self.length - 3:] == self.stop_codon
        return self.length - 3


    @property
    def candidate_rbs_motifs(self) -> str | None:
        rbs_motifs = {"GGA/GAG/AGG": ["GGA", "GAG", "AGG"],
                      "3Base/5BMM": "",  # ["3Base", "5BMM"],
                      "4Base/6BMM": "",  # ["4Base", "6BMM"],
                      "AGxAG": ["AGAAG", "AGTAG", "AGGAG", "AGCAG"],
                      "GGxGG": ["GGAGG", "GGTGG", "GGGGG", "GGCGG"],
                      "AGGAG(G)/GGAGG": ["AGGAG", "AGGAGG", "GGAGG"],
                      "AGGA": ["AGGA"],
                      "AGGA/GGAG/GAGG": ["AGGA", "GGAG", "GAGG"],
                      "GGAG/GAGG": ["GGAG", "GAGG"],
                      "AGGAG/GGAGG": ["AGGAG", "GGAGG"],
                      "AGGAG": ["AGGAG"],
                      "GGAGG": ["GGAGG"],
                      "AGGAGG": ["AGGAGG"],
                      None: None
                      }

        motif = self.gene_caller_df.select("rbs_motif").collect(streaming=True).item()
        return rbs_motifs[motif]


    @property
    def rbs_motif_position(self) -> int | None:
        if self.rbs_spacer_length is None:
            return None
        flanking = self.get_flanking_sequence(0, (-self.rbs_spacer_length[1] * 2, 25))

        if self.candidate_rbs_motifs is None or self.candidate_rbs_motifs == "":
            return None
        else:
            for candidate_motif in self.candidate_rbs_motifs:
                try:
                    res = flanking.index(candidate_motif)
                    self.rbs_motif: str = candidate_motif
                    return res

                except ValueError:
                    continue

        raise ValueError


    @property
    def rbs_spacer_length(self) -> tuple | None:
        spacer_str = self.gene_caller_df.select("rbs_spacer").collect(streaming=True).item()
        space_dict = {"3-4bp": (3, 4), "5-10bp": (5, 10), "11-12bp": (11, 12), "13-15bp": (13, 15), None: None}
        return space_dict[spacer_str]


    @property
    def gc_content(self) -> float:
        return self.gene_caller_df.select("gc_cont").collect(streaming=True).item()


    def is_significantly_different_between_samples(self, samples: list[str] = None, baseline: str | bool = False):
        if samples is None:
            samples = ["top", "bottom"]
        _, is_diff, _ = get_rao_score(self.methylation_data, samples, baseline)
        return is_diff


    def get_function(self, source: str) -> str:
        return self.functional_df.filter(pl.col("source").eq(source)).select("function").collect(streaming=True).item()


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


    def get_flanking_sequence(self, relative_position: int, range: (int, int)) -> SeqRecord.Seq:
        """
        Extracts the sequence from the contig around a relative position in the gene.

        Parameters:
            relative_position (int): Position relative to the gene start (zero-based).
            range (Tuple[int, int]): (start_offset, end_offset) around the relative position.

        Returns:
            Seq: The extracted sequence.
        """
        sequence = get_genomic_sequence(self.genome.name)[self.contig].seq
        seq_length = len(sequence)
        start_offset, end_offset = range
        assert start_offset < end_offset, "Give a gene slice sequence_range in order"

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
            seq_slice = sequence[start_pos:end_pos+1]  # +1 because our sequence_range is inclusove and python is exclusive

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
            seq_slice = sequence[end_pos:start_pos+1]  # +1 because our sequence_range is inclusove and python is exclusive
            seq_slice = seq_slice.reverse_complement()

        return seq_slice
