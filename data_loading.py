import os
import numpy as np
import pandas as pd

def sum_counts(count_str):
    """
    Sum numeric counts from a string column in a DataFrame.

    Parameters:
    count_str (pandas.Series): A pandas Series with string values containing numeric counts.

    Returns:
    pandas.Series: Sum of counts for each row.
    """
    # Extract numeric values from the string and convert them to numeric type
    numbers = pd.to_numeric(count_str.str.extractall(r'(\d+)')[0])
    return numbers.groupby(level=0).sum()


def expand_pivot_merge_sample_strings(df, column_name):
    # Split the string, explode into separate rows, and create a new DataFrame
    exploded = df[column_name].str.split(',').explode()
    temp_df = exploded.str.split(':', expand=True)
    temp_df.columns = ['letter', 'value'] if len(temp_df.columns) == 2 else ['letter']

    # Handle rows where value is missing (no colon in the string)
    temp_df['value'] = temp_df['value'].astype(float) if 'value' in temp_df else np.nan

    # Pivot the table
    pivot = temp_df.pivot_table(index=temp_df.index, columns='letter', values='value', aggfunc='first')

    # Rename the columns
    pivot.columns = [f'{column_name}_{col}' for col in pivot.columns]

    # Merge the pivot table with the original DataFrame
    return df.join(pivot)


def get_dmrs(path):
    """
    Read DMRs (Differentially Methylated Regions) data from a file.

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

    return dmrs


def get_sample_df_from_dmr(path):
    # If bed file is empty skip
    if os.stat(path).st_size == 0:
        return pd.DataFrame()

    sample_a_name, sample_b_name = os.path.basename(path).replace('.bed', '').split('_')
    dmrs = get_dmrs(path)

    dfs = []
    for sample, op_sample in zip(["samplea_", "sampleb_"], ["sampleb_", "samplea_"]):

        df = dmrs.copy()
        df = df[df.columns.drop(list(df.filter(regex=op_sample)))]

        sample_name = sample_a_name if "samplea_" in sample else sample_b_name
        op_sample_name = sample_b_name if "samplea_" in sample else sample_a_name
        df['sample'] = sample_name

        df.rename(columns={col: col.replace(sample, '') for col in df.columns}, inplace=True)
        df.rename(columns={"score": 'modkit_score_' + op_sample_name}, inplace=True)

        dfs.append(df)

    return dfs


def get_sample_metadata(file_path):
    """
    Load the sample metadata from an Excel file.

    Args:
    file_path (str): The path to the Excel file containing the sample metadata.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded sample metadata.
    """
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
