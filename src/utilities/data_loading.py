from __future__ import annotations
import os
import glob
import polars as pl
import src.utilities.utils as utils
from pathlib import Path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.objects.genome import Genome


def get_pileup(path: Path) -> pl.LazyFrame:
    """
    Read pileup data from a file.

    :param path: Path to .bed file.
    :type path: str
    :return: Dataframe of file data
    :rtype: pandas.DataFrame
    """
    try:
        pileup = pl.scan_csv(path, separator="\t", has_header=False,
                         new_columns=["contig", "inclusive start position", "exclusive end position",
                                      "modified base code and motif", "score", "strand", "start position2",
                                      "end position2", "color", "Nvalid_cov", "fraction modified", "Nmod", "Ncanonical",
                                      "Nother_mod", "Ndelete", "Nfail", "Ndiff", "Nnocall"])
    except pl.exceptions.NoDataError:
        # Return an empty dataframe with the same columns
        pileup = pl.DataFrame(schema={
            "contig": pl.Utf8,
            "inclusive start position": pl.Int64,
            "exclusive end position": pl.Int64,
            "modified base code and motif": pl.Utf8,
            "strand": pl.Utf8,
            "Nvalid_cov": pl.Int64,
            "fraction modified": pl.Float64,
            "Nmod": pl.Int64,
            "Ncanonical": pl.Int64,
            "Nother_mod": pl.Int64,
            "Ndelete": pl.Int64,
            "Nfail": pl.Int64,
            "Ndiff": pl.Int64,
            "Nnocall": pl.Int64
        }).lazy()
        return pileup

    # Drop redundant columns
    pileup = pileup.drop("score", "start position2", "end position2", "color")

    # Convert strand to bool
    pileup = pileup.with_columns(pl.col("strand").replace_strict({"+": True, "-": False}).cast(pl.Boolean))

    return pileup


def get_dataset_genes(genome: Genome) -> pl.LazyFrame:
    """
    Parameters:
    data_dir (str): The path to the data directory.
    genome_name (str): Genome name.

    Returns:
    pandas.DataFrame: DataFrame with gene functions.
    """
    gene_calls = pl.scan_csv(genome.gene_calls_path, separator="\t")

    # Map direction to +/-
    gene_calls = gene_calls.with_columns(pl.col("direction").replace_strict({"f": True, "r": False}))
    gene_calls = gene_calls.rename({"direction": "strand", "start_type": "start_codon_sequence"})

    return gene_calls


def get_dmrs(path: Path) -> (pl.LazyFrame, bool):
    """
    Read DMRs (Differentially Methylated Regions) data from a file, replaces the fractions and count columns with
    individual columns for each methylation type. Adds a column called comparison to note the samples comapred.

    Parameters:
    path (str): Path to the file.

    Returns:
    pandas.DataFrame: DataFrame with DMRs data.
    """
    try:
        dmrs = pl.scan_csv(path, separator="\t", has_header=False,
                        new_columns=['chrom', 'start', 'end', 'name', 'score', 'samplea_counts', 'samplea_total',
                              'sampleb_counts', 'sampleb_total', 'samplea_fractions', 'sampleb_fractions',
                              'samplea_percent_modified', 'sampleb_percent_modified'],
                        schema_overrides={'start': int, 'end': int, 'score': float,
                               'samplea_total': int, 'sampleb_total': int,
                               'samplea_counts': str, 'sampleb_counts': str,
                               'samplea_fractions': str, 'sampleb_fractions': str,
                               'samplea_percent_modified': float, 'sampleb_percent_modified': float})

    except pl.exceptions.NoDataError:
        return pl.LazyFrame(), True

    # Remove the string columns
    dmrs = dmrs.drop('samplea_counts', 'sampleb_counts', 'samplea_fractions', 'sampleb_fractions')

    # Add a column to note the comparison done in this DMR
    sample_a_name, sample_b_name = os.path.basename(path).replace('.bed', '').split('_')
    sample_a_name = utils.barcode_replicate_map[sample_a_name]
    sample_b_name = utils.barcode_replicate_map[sample_b_name]
    dmrs = dmrs.with_columns(comparison=pl.lit(f"{sample_a_name}_vs_{sample_b_name}"))

    return dmrs, False


def get_dmrs_for_genome(data_dir: Path, genome_name, dmr_type) -> pl.LazyFrame | None:
    # Get bed files
    bed_files = glob.glob(data_dir / genome_name / dmr_type / "*.bed")
    bed_files = [file for file in bed_files if not file.endswith('-bedgraph.bed')]
    if len(bed_files) == 0:
        return None

    # Get all the methylation data from the bed files
    dmrs = []
    for bed_file in bed_files:
        dmr, empty = get_dmrs(bed_file)
        if not empty:
            dmrs.append(dmr)

    # Concatenate the list of merged_df for this sample into a single dataframe
    dmrs = pl.concat(dmrs).rename({"chrom": "contig"}).collect().lazy()

    return dmrs


def get_sample_metadata(data_dir: Path) -> pl.DataFrame:
    """
    Load the sample metadata from an Excel file.

    Args:
    file_path (str): The path to the Excel file containing the sample metadata.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded sample metadata.
    """
    metadata_df = pl.read_excel(data_dir / "sample_metadata.xlsx")
    return metadata_df


def get_coverage(data_dir: Path, genome_name=None, agg=False) -> pl.LazyFrame:
    """
    Load the coverage data from a file. If aggregate, returns the mean coverage for the group.
    """
    coverage = pl.scan_csv(data_dir, separator="\t")

    # Replace the coverage column names based on dictionnary mapping
    coverage = coverage.rename(lambda x: x.replace(".fastq Mean", ""))

    # Get specific genome_name
    if genome_name is not None:
        coverage = coverage.filter(pl.col("Genome") == genome_name)

    # Aggregate by sample groups
    if agg:
        # Create a dictionary to map sample groups to barcodes
        sample_group_barcodes = {}

        for barcode, group in utils.barcode_replicate_map.items():
            if barcode not in coverage.collect_schema().names():
                continue

            if group not in sample_group_barcodes.keys():
                sample_group_barcodes[group] = [barcode]
            else:
                sample_group_barcodes[group].append(barcode)

        # Calculate mean coverage for each sample group and add to the new DataFrame
        for group, barcodes in sample_group_barcodes.items():
            coverage = coverage.with_columns(pl.concat_list(*barcodes).list.mean().alias(group))

        return coverage.select(*list(sample_group_barcodes.keys()), "Genome")

    return coverage


def load_methylation_data(
    genome: Genome,
    bed_files: list[Path],
    in_every_treatment: bool,
    triplicates_only: bool,
    treatments: list[str] | None = None,
    region_filter: pl.Expr | pl.LazyFrame | None = None,
    normalize: bool = True,
) -> pl.LazyFrame | None:
    """
    :param bed_files: List of .bed file paths to load.
    :param coverage: Coverage threshold.
    :param barcode_treatment_map: Maps a sample name -> treatment name.
    :param barcode_replicate_map: Maps a sample name -> replicate or something similar.
    :param treatments: List of treatments to include; if None, include all.
    :param region_filter: Either a pl.Expr, a pl.LazyFrame, or None.
    :param triplicates_only: If True, keep only triplicate positions.
    :param in_every_treatment: If True, keep only positions present in *all* treatments.
    :param normalize: Whether to normalize the data to fractions.
    :return: A polars.LazyFrame or None if no data was found.
    """
    assert len(bed_files) > 0, "No bed files provided."
    assert genome.default_coverage > 0, "Coverage must be greater than 0."
    
    # Collect all the data from each bed file
    all_data = []
    for bed_file in bed_files:
        sample_name = bed_file.stem
        
        # Skip sample if not in treatments
        if genome.barcode_treatment_map and treatments:
            treatment = genome.barcode_treatment_map.get(sample_name)
            if treatment not in treatments:
                continue
        
        # Load pileup
        methyl_data = get_pileup(bed_file)

        # Apply region filter if provided
        if isinstance(region_filter, pl.Expr):
            methyl_data = methyl_data.filter(region_filter)
            
        elif isinstance(region_filter, pl.LazyFrame):
            og_columns = methyl_data.collect_schema().names()

            # Join ASOF
            methyl_data.sort = methyl_data.sort(["contig", "strand", "inclusive start position"], descending=False)
            region_filter = region_filter.sort(["filter_contig", "filter_strand", "filter_start"], descending=False)
            methyl_data = methyl_data.join_asof(region_filter,
                                                left_on="inclusive start position",
                                                right_on="filter_start",

                                                # By columns guarrantee equality
                                                by_left=["contig", "strand"],
                                                by_right=["filter_contig", "filter_strand"],

                                                # filter_start <= inclusive start because of backward strategy
                                                # Takes the last key that satisfies this inequality
                                                # Which is good since preceeding genes will be filtered out
                                                strategy="backward"
                                                )

            # Do a filter for the end
            methyl_data = methyl_data.filter(pl.col("inclusive start position") <= pl.col("filter_end"))
            methyl_data = methyl_data.select(*og_columns)

            # Compare the dataframes
            assert True not in methyl_data.select(*og_columns).collect().is_duplicated(), "Duplicated data found."
                
        elif region_filter is not None:
            raise ValueError("region_filter must be pl.Expr, pl.LazyFrame, or None.")

        # Reshape the data
        methyl_data = utils.reshape_pileup_to_matrix_polars(methyl_data)
        if methyl_data is None:  # Data was empty before reshape upon collecting
            continue

        # Filter for coverage and removes null/NaN values
        modification_types = list(utils.readable_modification_name.keys())
        methyl_data = (
            methyl_data.filter(
                pl.any_horizontal(
                    pl.col(modification_types).is_not_null() &
                    pl.col(modification_types).cast(pl.Float64, strict=False).is_not_nan()
                )
                & pl.concat_list(modification_types).list.sum().ge(genome.default_coverage)
            )
        )

        # Add a sample column
        methyl_data = methyl_data.with_columns(sample=pl.lit(sample_name))

        all_data.append(methyl_data)

    if len(all_data) == 0:
        return None

    # Concat and rename
    result = pl.concat(all_data).rename({"inclusive start position": "position"})

    # Keep only positions that are in all samples
    if in_every_treatment and triplicates_only:
        og_columns = result.collect_schema().names()
        triplicate_positions = (result.group_by("contig", "strand", "position")
                                .agg(pl.col("sample").n_unique().alias("sample_count"))
                                .filter(pl.col("sample_count").eq(len(treatments) * 3)))

        result = (result.join(triplicate_positions, on=["contig", "strand", "position"], how="inner")
                    .select(*og_columns))

    # Keep only positions that occur in triplicate within a treatment
    elif triplicates_only:
        og_columns = result.collect_schema().names()
        triplicate_positions = result.with_columns(pl.col("sample").replace_strict(genome.barcode_replicate_map).alias("treatment"))
        triplicate_positions = (triplicate_positions.group_by("contig", "strand", "position", "treatment")
                                .agg(pl.col("sample").n_unique().alias("treatment_count"), pl.col("sample"))
                                .explode("sample")
                                .filter(pl.col("treatment_count").eq(3)))

        result = (result.join(triplicate_positions, on=["contig", "strand", "position", "sample"], how="inner")
                    .select(*og_columns))

    # Keep any position that occurs at least once in all treatments
    elif in_every_treatment:
        assert treatments is not None, "Treatments must be provided if in_every_treatment is True."
        og_columns = result.collect_schema().names()
        triplicate_positions = result.with_columns(pl.col("sample").replace_strict(genome.barcode_treatment_map).alias("treatment"))
        triplicate_positions = (triplicate_positions.group_by("contig", "strand", "position")
                                .agg(pl.col("treatment").n_unique().alias("treatment_count"), pl.col("sample"))
                                .explode("sample")
                                .filter(pl.col("treatment_count").eq(len(treatments))))

        result = (result.join(triplicate_positions, on=["contig", "strand", "position", "sample"], how="inner")
                    .select(*og_columns))

    # Normalize
    if normalize:
        result = result.with_columns(pl.col("sample").replace_strict(genome.barcode_treatment_map).replace_strict(genome.treatment_name_map).alias("treatment"))
        result = utils.treatment_weighted_mean(result)

    return result
