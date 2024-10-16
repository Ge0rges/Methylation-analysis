import os
import glob
import polars as pl
from Bio import SeqIO
import src.utilities.utils as utils


def get_pileup_polars(path) -> pl.LazyFrame:
    """
    Read pileup data from a file.

    :param path: Path to .bed file.
    :type path: str
    :return: Dataframe of file data
    :rtype: pandas.DataFrame
    """
    pileup = pl.scan_csv(path, separator="\t", has_header=False,
                         new_columns=["chrom", "inclusive start position", "exclusive end position",
                                      "modified base src and motif", "score", "strand", "start position2",
                                      "end position2", "color", "Nvalid_cov", "fraction modified", "Nmod", "Ncanonical",
                                      "Nother_mod", "Ndelete", "Nfail", "Ndiff", "Nnocall"])

    # Drop redundant columns
    pileup = pileup.drop("score", "start position2", "end position2", "color")

    return pileup


def load_combined_methyl_data_for_genome_polars(genome_name, data_dir, coverage=None) -> pl.LazyFrame:
    """
    Load the methyl data from every sample into a matrix.

    :param genome_name: Folder name of the genome_name.
    :type genome_name: str
    :param data_dir: Path to the data directory.
    :type data_dir: str
    :param common_locations: Exclude locations that are not common to all samples
    :type common_locations: bool
    :return: Dataframe of the combined methyl data.
    :rtype: pd.DataFrame
    """
    # Load the methyl_dfs from the bed files
    bed_files = [f for f in glob.glob(os.path.join(data_dir, genome_name, "*.bed")) if
                 '-bedgraph' not in os.path.basename(f)]

    if len(bed_files) == 0:
        print(f"No pileup bed files found for {data_dir}/{genome_name}")
        return None

    dfs = []
    for i, bed_file in enumerate(bed_files):
        methyl_data = get_pileup_polars(bed_file)

        methyl_data = utils.reshape_pileup_to_matrix_polars(methyl_data)

        # Add sample column
        sample_name = os.path.basename(bed_file).split(".")[0]
        methyl_data = methyl_data.with_columns(sample=pl.lit(sample_name))

        dfs.append(methyl_data)

    # Concat everything together
    dfs = pl.concat(dfs)

    # Split name
    dfs = dfs.with_columns(
        contig=pl.col('name').str.split(by='|').list.get(0),
        strand=pl.col('name').str.split(by='|').list.get(1),
        start=pl.col('name').str.split(by='|').list.get(2).cast(pl.UInt32),
        end=pl.col('name').str.split(by='|').list.get(3).cast(pl.UInt32)
    )

    # Filter for coverage
    if coverage is not None:
        methylation_types = list(utils.readable_methylation_name.keys()) + ["Ncanonical"]
        dfs = dfs.filter(pl.concat_list(methylation_types).list.sum().ge(coverage))
        dfs = dfs.filter(pl.any_horizontal(pl.col(methylation_types).is_not_null() & pl.col(methylation_types).is_not_nan()))

    return dfs


def get_genes_polars(data_dir, drop_extras=True) -> pl.LazyFrame:
    """
    Parameters:
    data_dir (str): The path to the data directory.
    genome_name (str): Genome name.

    Returns:
    pandas.DataFrame: DataFrame with gene functions.
    """
    gene_calls = pl.scan_csv(f"{data_dir}/gene-calls.txt", separator="\t")
    if drop_extras:
        gene_calls = gene_calls.drop("source", "version", "partial", "call_type")

    # Map direction to +/-
    gene_calls = gene_calls.with_columns(pl.col("direction").str.replace_many(["f", "r"],["+", "-"]))

    return gene_calls


def get_dmrs_from_file_polars(path) -> (pl.LazyFrame, bool):
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
    sample_a_name = utils.barcode_sample_map[sample_a_name]
    sample_b_name = utils.barcode_sample_map[sample_b_name]
    dmrs = dmrs.with_columns(comparison=pl.lit(f"{sample_a_name}_vs_{sample_b_name}"))

    return dmrs, False


def get_dmrs_for_genome_polars(data_dir, genome_name, dmr_type) -> pl.LazyFrame:
    # Get bed files
    bed_files = glob.glob(os.path.join(os.path.join(data_dir, genome_name, dmr_type), "*.bed"))
    bed_files = [file for file in bed_files if not file.endswith('-bedgraph.bed')]
    if len(bed_files) == 0:
        return None

    # Get all the methylation data from the bed files
    dmrs = []
    for bed_file in bed_files:
        dmr, empty = get_dmrs_from_file_polars(bed_file)
        if not empty:
            dmrs.append(dmr)

    # Concatenate the list of merged_df for this sample into a single dataframe
    dmrs = pl.concat(dmrs).rename({"chrom": "contig"}).collect().lazy()

    return dmrs


def get_sample_metadata(data_dir) -> pl.DataFrame:
    """
    Load the sample metadata from an Excel file.

    Args:
    file_path (str): The path to the Excel file containing the sample metadata.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded sample metadata.
    """
    file_path = os.path.join(data_dir, "sample_metadata.xlsx")
    metadata_df = pl.read_excel(file_path)
    return metadata_df


def get_coverage(data_dir, genome_name=None, agg=False) -> pl.LazyFrame:
    """
    Load the coverage data from a file. If aggregate, returns the mean coverage for the group.
    """
    coverage = pl.scan_csv(os.path.join(data_dir, "mag_eval/coverm.tsv"), separator="\t")

    # Replace the coverage column names based on dictionnary mapping
    coverage = coverage.rename(lambda x: x.replace(".fastq Mean", ""))

    # Get specific genome_name
    if genome_name is not None:
        coverage = coverage.filter(pl.col("Genome") == genome_name)

    # Aggregate by sample groups
    if agg:
        # Create a dictionary to map sample groups to barcodes
        sample_group_barcodes = {}

        for barcode, group in utils.barcode_sample_map.items():
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


def get_coordinated_functions_polars(data_dir) -> pl.LazyFrame:
    """
    Read gene caller and functions from seperate files then intersect.

    Parameters:
    data_dir (str): The path to the data directory.
    genome_name (str): Genome name.

    Returns:
    pandas.DataFrame: DataFrame with gene functions.
    """
    # Load data
    function_calls = pl.scan_csv(f"{data_dir}/function-calls.txt", separator="\t")
    gene_calls = get_genes_polars(data_dir)

    # Merge using efficient indexing
    coordinated_functions = gene_calls.join(function_calls, on='gene_callers_id')

    return coordinated_functions


def get_genomic_sequence(genome_name, reverse=False) -> dict:
    """
    Read genomic sequence data from a file.

    :param path: Path to .fasta file.
    :type path: str
    :return: Dataframe of file data
    :rtype: pandas.DataFrame
    """
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data", "mags", f"{genome_name}.fna")
    fasta_file = SeqIO.parse(path, "fasta")
    fasta_dict = {}
    for record in fasta_file:
        if reverse:
            fasta_dict[record.id] = record.seq.reverse_complement
        else:
            fasta_dict[record.id] = record.seq

    return fasta_dict
