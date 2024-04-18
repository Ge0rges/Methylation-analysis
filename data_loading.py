import os
from utils import *


def get_pileup(path):
    """
    Read pileup data from a file.

    :param path: Path to .bed file.
    :type path: str
    :return: Dataframe of file data
    :rtype: pandas.DataFrame
    """
    pileup = pd.read_csv(path, sep="\t", header=None, names=["chrom", "inclusive start position", "exclusive end position", "modified base code and motif", "score", "strand", "start position2", "end position2", "color", "Nvalid_cov", "fraction modified", "Nmod", "Ncanonical", "Nother_mod", "Ndelete", "Nfail", "Ndiff", "Nnocall"])

    # Drop redundant columns
    pileup.drop(columns=["score" "start position2", "end position2", "color"], inplace=True)

    return pileup


def get_dmrs(path):
    """
    Read DMRs (Differentially Methylated Regions) data from a file, replaces the fractions and count columns with
    individual columns for each methylation type. Adds a column called comparison to note the samples comapred.

    Parameters:
    path (str): Path to the file.

    Returns:
    pandas.DataFrame: DataFrame with DMRs data.
    """
    dmrs = pd.read_csv(path, sep="\t", header=None,
                       names=['chrom', 'start', 'end', 'name', 'score', 'samplea_counts', 'samplea_total',
                              'sampleb_counts', 'sampleb_total', 'samplea_fractions', 'sampleb_fractions', 'samplea_percent_modified', 'sampleb_percent_modified'])

    # Convert columns to specified data types
    dmrs = dmrs.astype({'start': 'int', 'end': 'int', 'score': 'float',
                        'samplea_total': 'int', 'sampleb_total': 'int',
                        'samplea_counts': 'str', 'sampleb_counts': 'str',
                        'samplea_fractions': 'str', 'sampleb_fractions': 'str',
                        'samplea_percent_modified': 'float', 'sampleb_percent_modified': 'float'})

    # Seperate out the samplex_counts and_fractions columns
    dmrs = expand_pivot_merge_sample_strings(dmrs, 'samplea_counts')
    dmrs = expand_pivot_merge_sample_strings(dmrs, 'sampleb_counts')
    dmrs = expand_pivot_merge_sample_strings(dmrs, 'samplea_fractions')
    dmrs = expand_pivot_merge_sample_strings(dmrs, 'sampleb_fractions')

    # Remove the string columns
    dmrs.drop(columns=['samplea_counts', 'sampleb_counts', 'samplea_fractions', 'sampleb_fractions'], inplace=True)

    # Add a column to note the comparison done in this DMR
    sample_a_name, sample_b_name = os.path.basename(path).replace('.bed', '').split('_')
    dmrs["comparison"] = f"{sample_a_name}_VS_{sample_b_name}"

    return dmrs


def get_sample_metadata(data_dir):
    """
    Load the sample metadata from an Excel file.

    Args:
    file_path (str): The path to the Excel file containing the sample metadata.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded sample metadata.
    """
    file_path = os.path.join(data_dir, "sample_metadata.xlsx")
    metadata_df = pd.read_excel(file_path)
    return metadata_df


def get_coordinated_functions(data_dir, genome_name):
    """
    Read gene caller and functions from seperate files then intersect.

    Parameters:
    genome_name (str): Genome name.

    Returns:
    pandas.DataFrame: DataFrame with gene functions.
    """
    # Load data
    gene_calls = pd.read_csv(f"{data_dir}/{genome_name}/gene-calls.txt", sep=",").drop(columns=["source"])
    function_calls = pd.read_csv(f"{data_dir}/{genome_name}/function-calls.txt", sep="\t")

    # Ensure efficient data types
    gene_calls['gene_callers_id'] = gene_calls['gene_callers_id'].astype('int32')
    function_calls['gene_callers_id'] = function_calls['gene_callers_id'].astype('int32')

    # Merge using efficient indexing
    coordinated_functions = pd.merge(gene_calls, function_calls, on='gene_callers_id')
    coordinated_functions['e_value'] = coordinated_functions['e_value'].astype(float)

    return coordinated_functions
