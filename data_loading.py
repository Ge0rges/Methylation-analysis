import os
import glob
from utils import *
from Bio import SeqIO


def get_pileup(path) -> pd.DataFrame:
    """
    Read pileup data from a file.

    :param path: Path to .bed file.
    :type path: str
    :return: Dataframe of file data
    :rtype: pandas.DataFrame
    """
    pileup = pd.read_csv(path, sep="\t", header=None, names=["chrom", "inclusive start position", "exclusive end position", "modified base code and motif", "score", "strand", "start position2", "end position2", "color", "Nvalid_cov", "fraction modified", "Nmod", "Ncanonical", "Nother_mod", "Ndelete", "Nfail", "Ndiff", "Nnocall"])

    # Drop redundant columns
    pileup.drop(columns=["score", "start position2", "end position2", "color"], inplace=True)

    return pileup


def get_dmrs(path) -> pd.DataFrame:
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
    sample_a_name = barcode_sample_map[sample_a_name.title()]
    sample_b_name = barcode_sample_map[sample_b_name.title()]
    dmrs["comparison"] = f"{sample_a_name}_VS_{sample_b_name}"

    return dmrs


def get_sample_metadata(data_dir) -> pd.DataFrame:
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


def get_coordinated_functions(data_dir, genome_name) -> pd.DataFrame:
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


def get_genomic_sequence(genome_name) -> dict:
    """
    Read genomic sequence data from a file.

    :param path: Path to .fasta file.
    :type path: str
    :return: Dataframe of file data
    :rtype: pandas.DataFrame
    """
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "mags", f"{genome_name}.fna")
    fasta_file = SeqIO.parse(path, "fasta")
    fasta_dict = {}
    for record in fasta_file:
        fasta_dict[record.id] = str(record.seq)

    return fasta_dict


def load_combined_methyl_data_for_genome(genome_name, data_dir, common_locations) -> pd.DataFrame:
    """
    Load the methyl data from every sample into a matrix.

    :param genome_name: Folder name of the genome.
    :type genome_name: str
    :param data_dir: Path to the data directory.
    :type data_dir: str
    :param common_locations: Exclude locations that are not common to all samples
    :type common_locations: bool
    :return: Dataframe of the combined methyl data.
    :rtype: pd.DataFrame
    """
    # Check to see if CSV file exists for this genome
    try:
        combined_methyl_data = pd.read_csv(f"{data_dir}/{genome_name}/combined_methyl_data.csv")

    except NotADirectoryError:
        return pd.DataFrame()

    except FileNotFoundError:
        # Load the methyl_dfs from the bed files
        bed_files = glob.glob(os.path.join(os.path.join(data_dir, genome_name), "*.bed"))

        if len(bed_files) == 0:
            print(f"No pileup bed files found for {data_dir}/{genome_name}")
            return pd.DataFrame()

        methyl_dfs = []
        for i, bed_file in enumerate(bed_files):
            methyl_data = get_pileup(bed_file)
            methyl_data = reshape_pileup_to_matrix(methyl_data, genome_name)
            methyl_data["sample"] = os.path.basename(bed_file).split(".")[0]

            methyl_dfs.append(methyl_data)

        # Build matrix for statistical testing
        combined_methyl_data = pd.concat(methyl_dfs, ignore_index=True)

        # Save this dataframe
        combined_methyl_data.to_csv(f"{data_dir}/{genome_name}/combined_methyl_data.csv", index=False)

    # Keep in each dataframes only the names that are common to all dataframes
    name_sets = []
    if common_locations:
        # Get the index of common names
        common_index = None
        for i, group in enumerate(combined_methyl_data.groupby("sample")):
            group.set_index("name", inplace=True)

            if i == 0:
                common_index = group.index
            else:
                common_index = common_index.intersection(group.index)

        # Keep only the common indices and set name back to a column
        combined_methyl_data = combined_methyl_data.loc[common_index].reset_index(names="name")

        # Check that every dataframe has the same names
        name_sets = [group['name'] for group in combined_methyl_data.groupby("sample")]
        assert all([np.array_equal(name_set, name_sets[0]) for name_set in
                    name_sets]), "Not all methyl methyl_dfs have the same regions"

        assert len(combined_methyl_data["name"]) == len(name_sets[0]) * combined_methyl_data["sample"].nunique()
        assert combined_methyl_data["name"].nunique() == len(name_sets[0])


    # Check that first column is name and last is sample
    assert combined_methyl_data.columns[0] == "name" and combined_methyl_data.columns[-1] == "sample", "Columns are not in the expected order"

    # Set all columns but the first to be integer types
    for col in combined_methyl_data.columns[1:-1]:
        combined_methyl_data[col] = combined_methyl_data[col].astype(int)

    return combined_methyl_data
