import polars as pl
import os
from src.utilities.utils import normalize_data_by_pileup, add_gene_caller_id, readable_methylation_name, readable_modification_name, reshape_pileup_to_matrix_polars
from src.utilities.data_loading import get_pileup_polars, get_genomic_sequence, get_genes_polars
from platform import system
from functools import lru_cache, cached_property
import glob
from Bio import SeqRecord


class Genome(object):
    __min_coverage_default = 5
    __data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"../../../methylation_data/methylation_5")
    if system() == "Darwin":
        __data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"../../data/methylation_data/methylation_5")

    def __init__(self, name: str):
        self._data_dir: str = Genome.__data_dir
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
                                              "exclusive end position": "end", })

            all_data.append(methyl_data)

        all_data = pl.concat(all_data)
        all_genes = get_genes_polars(self._data_dir)
        gene_ids = add_gene_caller_id(all_data, all_genes).select("gene_callers_id").unique().collect(
            streaming=True).get_column("gene_callers_id").to_list()
        return gene_ids


    @lru_cache
    def load_all_methylation_data(self, coverage: int = __min_coverage_default, normalize: bool = True) -> pl.LazyFrame:
        return self.load_region_methylation_data(coverage, None, normalize)


    @lru_cache
    def load_region_methylation_data(self, coverage: int = __min_coverage_default,
                                     region_filter: pl.Expr | None = None, normalize: bool = True) -> pl.LazyFrame:
        # Get all the bed files for this genome
        bed_files = [f for f in glob.glob(os.path.join(self._data_dir, self.name, "*.bed")) if
                     '-bedgraph' not in os.path.basename(f)]

        all_data = []
        for bed_file in bed_files:
            # Load the data for the positions that overlap only
            methyl_data = get_pileup_polars(bed_file)
            if region_filter is not None:
                methyl_data = methyl_data.filter(region_filter)

            methyl_data = reshape_pileup_to_matrix_polars(methyl_data)
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
        if normalize:
            result = normalize_data_by_pileup(result)

        # Create total methylation column
        methylation_types = list(readable_methylation_name.keys())
        result = result.with_columns(pl.concat_list(methylation_types).list.sum().alias("total_methylation"))

        # Seperate name
        result = result.with_columns(
            contig=pl.col('name').str.split(by='|').list.get(0),
            strand=pl.col('name').str.split(by='|').list.get(1).eq("+"),
            position=pl.col('name').str.split(by='|').list.get(2).cast(pl.Int64),
        ).drop("name")
        return result


    def add_genome_relative_position(self, df: pl.LazyFrame) -> pl.LazyFrame:
        # Get contigs cumsum
        sequences = get_genomic_sequence(self.name)
        contigs = list(sequences.keys())
        contigs.sort()
        cum_sum = 0
        offsets = {}
        for key in contigs:
            offsets[key] = cum_sum
            cum_sum += len(sequences[key])

        # Convert position to absolute
        return df.with_columns(pl.col("position").add(pl.col("contig").replace_strict(offsets)).alias("genome_position"))


    def add_gene_caller_id(self, df: pl.LazyFrame) -> pl.LazyFrame:
        genes = get_genes_polars(self._data_dir)
        return add_gene_caller_id(df, genes)
