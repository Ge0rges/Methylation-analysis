import os
from src.utilities.utils import *
from src.utilities.data_loading import get_pileup, get_dataset_genes
from platform import system
from functools import lru_cache, cached_property
import glob
from Bio import SeqRecord, SeqUtils
from src.objects.gene_collection import GeneCollection
from pathlib import Path
import numpy as np
from Bio import SeqIO


class Genome(object):

    __min_coverage_default = 10
    __default_treatments = ["top", "middle", "bottom"]
    __methylation_data_dir = Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data/methylation_data/methylation_5"))

    if system() == "Darwin":
        __methylation_data_dir = Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data/methylation_data/methylation_5"))

    def __init__(self, name: str):
        self._methylation_data_dir: Path = Genome.__methylation_data_dir
        if not self._is_valid_genome_name(name):
            raise ValueError(f"Genome {name} not found in the data directory.")

        self.name: str = name
        self.readable_name: str = name.capitalize().replace("_r-contigs", " sp.")
        self.plot_dir: Path = Path(f"../plots/{self.name}")
        self.plot_dir.mkdir(exist_ok=True, parents=True)
        self.genome_path: Path = Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data", "mags", f"{self.name}.fna"))


    @classmethod
    def valid_genome_names(cls) -> list[str]:
        # Check if genome exists in the data directory
        return [name for name in os.listdir(cls.__methylation_data_dir) if os.path.isdir(cls.__methylation_data_dir / name)]


    def _is_valid_genome_name(self, name: str) -> bool:
        # Check if genome exists in the data directory
        return os.path.exists(os.path.join(self._methylation_data_dir, name))


    @cached_property
    def sequence(self) -> dict[str, SeqRecord]:
        """
        Read genomic sequence data from file.

        :param path: Path to .fasta file.
        :type path: str
        :return: Dataframe of file data
        :rtype: pandas.DataFrame
        """
        fasta_dict = SeqIO.index(str(self.genome_path), "fasta")

        # Remove contigs not in this genome
        my_contigs = self.gene_caller_df.select("contig").unique().collect(streaming=True).get_column("contig").to_list()
        fasta_dict = {k: v for k, v in fasta_dict.items() if k in my_contigs}

        return fasta_dict


    @cached_property
    def gc_content(self):
        gcs = np.asarray([SeqUtils.gc_fraction(x.seq, ambiguous="weighted") for x in self.sequence.values()])
        return gcs.mean()


    @cached_property
    def gene_caller_df(self) -> pl.LazyFrame:
        return get_dataset_genes(self).filter(pl.col("gene_callers_id").is_in(self.gene_ids))


    @cached_property
    def gene_ids(self) -> list[int]:
        # This works because modkit takes a reference and then does pileup one area within that reference only.
        bed_files = [Path(f) for f in glob.glob(os.path.join(self._methylation_data_dir, self.name, "*.bed")) if
                     '-bedgraph' not in os.path.basename(f)]

        all_data = []
        for bed_file in bed_files:
            # Load the data for the positions that overlap only
            methyl_data = get_pileup(bed_file).select("contig", "inclusive start position",
                                                             "exclusive end position", "strand")
            methyl_data = methyl_data.rename({"inclusive start position": "position", "exclusive end position": "end"})

            all_data.append(methyl_data)

        all_data = pl.concat(all_data)

        all_genes = get_dataset_genes(self)
        gene_ids = add_gene_caller_id(all_data, all_genes).select("gene_callers_id").unique().collect(
            streaming=True).get_column("gene_callers_id").to_list()
        return gene_ids


    def load_all_methylation_data(self, coverage: int = __min_coverage_default, normalize: bool = True,
                                  treatments: list[str] = __default_treatments, triplicates_only: bool = True,
                                  common_only: bool = False) -> pl.LazyFrame:
        return self.load_region_methylation_data(coverage, None, normalize, treatments, triplicates_only,
                                                 common_only)


    def load_region_methylation_data(self, coverage: int = __min_coverage_default,
                                     region_filter: pl.Expr | pl.LazyFrame | None = None, normalize: bool = True,
                                     treatments: list[str] = __default_treatments, triplicates_only: bool = True,
                                     common_only: bool = False) -> pl.LazyFrame | None:
        # Get all the bed files for this genome
        bed_files = [Path(f) for f in glob.glob(str(self._methylation_data_dir / self.name / "*.bed")) if
            '-bedgraph' not in os.path.basename(f)]

        all_data = []
        for bed_file in bed_files:
            # Load only asked treamtent samples
            sample_name = os.path.basename(bed_file).split(".")[0]
            treatment = barcode_replicate_map[sample_name]
            if treatments is not None and treatment not in treatments:
                continue

            # Load the data for the positions that overlap only
            methyl_data = get_pileup(bed_file)
            if isinstance(region_filter, pl.Expr):
                methyl_data = methyl_data.filter(region_filter)

            elif isinstance(region_filter, pl.LazyFrame):
                og_columns = methyl_data.collect_schema().names()
                methyl_data = methyl_data.join_where(region_filter,
                                                     pl.col("contig").eq(pl.col("filter_contig")),
                                                     pl.col("strand").eq(pl.col("filter_strand")),
                                                     pl.col("inclusive start position").ge(pl.col("filter_start")),
                                                     pl.col("inclusive start position").le(pl.col("filter_end")))
                methyl_data = methyl_data.select(*og_columns).unique()  # Unique is needed when a positon is in more than one region filter

            elif region_filter is not None:
                raise ValueError("Region filter must be of type pl.Expr, pl.LazyFrame, or None.")

            methyl_data = reshape_pileup_to_matrix_polars(methyl_data)
            if methyl_data is None:
                continue

            # Filter for coverage and no full Null/NaN values
            modification_types = list(readable_modification_name.keys())
            methyl_data = (methyl_data.filter(
                pl.any_horizontal(pl.col(modification_types).is_not_null() &
                                  pl.col(modification_types).is_not_nan()) &
                pl.concat_list(modification_types).list.sum().ge(coverage)))

            # Add sample column
            methyl_data = methyl_data.with_columns(sample=pl.lit(sample_name))

            all_data.append(methyl_data)

        if len(all_data) == 0:
            print(f"No data found for {self.name}")
            return None

        result = pl.concat(all_data)

        # Rename
        result = result.rename({"inclusive start position": "position"})

        #result = result.filter(pl.col("contig").eq("contig_112356"), pl.col("strand").eq(False), pl.col("position").eq(260)).collect()

        # Keep only positions that occur in triplicate within a treatment
        # Keep only positions that are in all samples
        if common_only:
            og_columns = result.collect_schema().names()
            triplicate_positions = (result.group_by("contig", "strand", "position")
                                    .agg(pl.col("sample").n_unique().alias("sample_count"))
                                    .filter(pl.col("sample_count").eq(len(treatments) * 3)))

            result = (result.join(triplicate_positions, on=["contig", "strand", "position"], how="inner")
                      .select(*og_columns))

        elif triplicates_only:
            og_columns = result.collect_schema().names()
            triplicate_positions = result.with_columns(pl.col("sample").replace_strict(barcode_replicate_map).alias("treatment"))
            triplicate_positions = (triplicate_positions.group_by("contig", "strand", "position", "treatment")
                                    .agg(pl.col("sample").n_unique().alias("treatment_count"), pl.col("sample"))
                                    .explode("sample")
                                    .filter(pl.col("treatment_count").eq(3)))

            result = (result.join(triplicate_positions, on=["contig", "strand", "position", "sample"], how="inner")
                      .select(*og_columns))

        # Normalize to fraction
        if normalize:
            result = normalize_data_by_pileup(result)

        return result


    def add_genome_relative_position(self, df: pl.LazyFrame) -> pl.LazyFrame:
        # Get contigs cumsum
        contigs = list(self.sequence.keys())
        contigs.sort()
        cum_sum = 0
        offsets = {}
        for key in contigs:
            offsets[key] = cum_sum
            cum_sum += len(self.sequence[key])

        # Convert position to absolute
        return df.with_columns(pl.col("position").add(pl.col("contig").replace_strict(offsets, return_dtype=pl.UInt64)).alias("genome_position"))


    def add_gene_caller_id(self, df: pl.LazyFrame) -> pl.LazyFrame:
        genes = get_dataset_genes(self)
        return add_gene_caller_id(df, genes)


    @cached_property
    def gene_cassettes(self) -> list[GeneCollection]:
        # Get genes with a pribnow box
        gene_collection = GeneCollection(self.gene_ids, self)
        all_genes = gene_collection.gene_caller_df.select("gene_callers_id", "contig", "start", "strand")

        # Get genes with promoters
        promoter_genes = self.genes_with_promoter.pribnow_box_position_and_sequence.select("pribnow_box_position", "gene_callers_id")
        promoter_genes = promoter_genes.with_columns(pl.col("gene_callers_id").alias("operon_id"))

        # Merge to know which gene has a promoter
        df = all_genes.join(promoter_genes, on="gene_callers_id", how="left")

        # Group by strand, and contig, then sort by start position and foward the operon ID
        df = df.sort("strand", "contig", "start", descending=False)
        df = (df.group_by("strand", "contig", maintain_order=True)
              .agg(pl.col("operon_id").forward_fill(), pl.col("gene_callers_id"))
              .explode("gene_callers_id", "operon_id"))
        df = (df.filter(pl.col("operon_id").is_not_null())  # Aggregation happens on null also
              .group_by("operon_id")
              .agg(pl.col("gene_callers_id"))
              .filter(pl.col("gene_callers_id").list.len().gt(1)))

        gcs = [GeneCollection(ids, self) for ids in df.collect(streaming=True).get_column("gene_callers_id").to_list()]

        return gcs


    @cached_property
    def genes_with_promoter(self) -> GeneCollection:
        gene_collection = GeneCollection(self.gene_ids, self)
        return GeneCollection(gene_collection.pribnow_box_position_and_sequence
                              .filter(pl.col("pribnow_box_position").is_not_null())
                              .collect(streaming=True).get_column("gene_callers_id").to_list(), self)


if __name__ == "__main__":
    genome = Genome("Pelagibacter_r-contigs")
    df = genome.load_all_methylation_data().collect().filter(pl.col("sample").replace_strict(barcode_replicate_map).is_in(["top", "middle", "bottom"]))
    meth_types = list(readable_methylation_name.keys())

    print(df.unique(["contig", "strand", "position"]).height)

    data = (df.select(*meth_types, "contig", "strand", "position")
     .filter(pl.any_horizontal(pl.col(meth_types).is_not_null() & pl.col(meth_types).is_not_nan())).unique(["contig", "strand", "position"]))

    print(data.height)

    # Keep only names (positions) that are in all samples
    labels_in_all_groups = (df.group_by("contig", "strand", "position")
                            .agg(pl.col("sample").n_unique().alias("unique_groups"))
                            .filter(pl.col("unique_groups") == df.get_column("sample").n_unique())
                            .select("contig", "strand", "position"))

    print(labels_in_all_groups.height)


