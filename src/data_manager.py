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

        self._load_all_methylation_data()

    def _load_all_methylation_data(self, coverage: int = min_coverage_default):
        all_genes = get_genes_polars(self._data_dir)
        methyl_data = load_combined_methyl_data_for_genome_polars(self.name, self._data_dir, coverage=coverage)
        self.methyl_df: pl.LazyFrame = add_gene_caller_id(methyl_data, all_genes)

    def _is_valid_genome_name(self, name: str) -> bool:
        # Check if genome exists in the data directory
        return os.path.exists(os.path.join(self._data_dir, name))

    @property
    def sequence(self) -> SeqRecord:
        return get_genomic_sequence(self.name, False)

    @property
    def gene_ids(self) -> list[int]:
        return self.methyl_df.select("gene_callers_id").unique().collect(streaming=True).get_column("gene_callers_id").to_list()


class Gene(object):

    def __init__(self, id: int, genome: Genome):
        self.id: int = id
        self._data_dir: str = data_dir
        self.genome: Genome = genome

        self._load_data()
        self._load_all_methylation_data()

    def _load_data(self) -> None:
        self.position_df = get_genes_polars(self._data_dir, False).filter(pl.col("gene_callers_id").eq(self.id))
        self.functional_df = pl.scan_csv(f"{self._data_dir}/function-calls.txt", separator="\t").filter(pl.col("gene_callers_id").eq(self.id))

    def _load_all_methylation_data(self, coverage: int = min_coverage_default) -> None:
        # Get all the bed files for this genome
        bed_files = [f for f in glob.glob(os.path.join(self._data_dir, self.genome.name, "*.bed")) if '-bedgraph' not in os.path.basename(f)]

        all_data = []
        for bed_file in bed_files:
            # Load the data for the positions that overlap only
            strand = "+" if self.strand else "-"
            methyl_data = get_pileup_polars(bed_file).filter(pl.col("chrom").eq(self.contig) &
                                                             pl.col("inclusive start position").ge(self.start) &
                                                             pl.col("exclusive end position").le(self.stop) &
                                                             pl.col("strand").eq(strand))

            methyl_data = utils.reshape_pileup_to_matrix_polars(methyl_data)
            if methyl_data is None:
                continue

            # Add sample column
            sample_name = os.path.basename(bed_file).split(".")[0]
            methyl_data = methyl_data.with_columns(sample=pl.lit(sample_name))

            # Drop the name column in favor of a relative position column
            methyl_data = methyl_data.with_columns(pl.col('name').str.split(by='|').list.get(2).cast(pl.UInt32).sub(self.start).alias("position")).drop("name")

            all_data.append(methyl_data)

        self.methylation_data: pl.LazyFrame = pl.concat(all_data)

        # Filter for coverage of at least 5 and no full Null/NaN values
        methylation_types = list(utils.readable_modification_name.keys())
        self.methylation_data = (self.methylation_data.filter(pl.any_horizontal(pl.col(methylation_types).is_not_null() & pl.col(methylation_types).is_not_nan()) &
                                                              pl.concat_list(methylation_types).list.sum().ge(coverage)))

        # Normalize to fraction
        self.methylation_data = normalize_data_by_pileup(self.methylation_data)

        # Create total methylation column
        self.methylation_data = self.methylation_data.with_columns(pl.concat_list(methylation_types).list.sum().alias("total_methylation"))

    @property
    def contig(self) -> str:
        return self.position_df.select("contig").collect(streaming=True).item()

    @property
    def start(self) -> int:
        return self.position_df.select("start").collect(streaming=True).item()

    @property
    def stop(self) -> int:
        return self.position_df.select("stop").collect(streaming=True).item()

    @property
    def strand(self) -> bool:
        return self.position_df.select("direction").collect(streaming=True).item() == "+"

    @property
    def sources(self) -> list[str]:
        return self.functional_df.select("source").collect(streaming=True).to_list()

    @property
    def sequence(self) -> SeqRecord:
        return get_genomic_sequence(self.genome.name, not self.strand)[self.contig][self.start:self.stop]

    @property
    def length(self) -> int:
        return self.stop - self.start

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

        If position is specified, restricts itself to that position (or inclusive range if tuple). Else, entire gene.
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
