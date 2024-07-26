import textwrap
import numpy as np
import pandas as pd
import polars as pl
from itertools import combinations
import utilities.data_loading as dl


readable_methylation_name = {"21839": "4mC", "a": "6mA", "m": "5mC"}

readable_sample_name = {"barcode01": "S2-1",
                        "barcode02": "S2-2",
                        "barcode03": "S2-3",
                        "barcode04": "Control",
                        "barcode05": "S3-1",
                        "barcode06": "S3-2",
                        "barcode07": "S3-3",
                        "barcode08": "S4-1",
                        "barcode09": "S4-2",
                        "barcode10": "S4-3",
                        "barcode11": "IC3-1 (30 cm)",
                        "barcode12": "IC3-2 (160 cm)",
                        "barcode13": "IC3-3 (205 cm)",
                        "barcode14": "IC3-4 (70 cm)",
                        "top": "Sackhole Top (40 cm)",
                        "bottom": "Sackhole Bottom (160 cm)",
                        "middle": "Sackhole Middle (70 cm)",
                        "control": "Control",
                        "core-40": "Ice core 40 cm",
                        "core-160": "Ice core 160 cm",
                        "core-205": "Ice core 205 cm",
                        'core-70': "Ice core 70 cm"
}

barcode_sample_map = {"barcode01": "top",
                      "barcode02": "middle",
                      "barcode03": "bottom",
                      "barcode04": "control",
                      "barcode05": "top",
                      "barcode06": "middle",
                      "barcode07": "bottom",
                      "barcode08": "top",
                      "barcode09": "middle",
                      "barcode10": "bottom",
                      "barcode11": "core-40",
                      "barcode12": "core-160",
                      "barcode13": "core-205",
                      "barcode14": "core-70",
                      "top": "top",
                      "middle": "middle",
                      "bottom": "bottom"
}


read_counts = {
    "barcode01": 1093788,
    "barcode02": 296042,
    "barcode03": 5812056,
    "barcode04": 57626,
    "barcode05": 344880,
    "barcode06": 180208,
    "barcode07": 1056185,
    "barcode08": 178883,
    "barcode09": 1776313,
    "barcode10": 1163651,
    "barcode11": 41324,
    "barcode12": 591165,
    "barcode13": 39685,
    "barcode14": 96793,
}


def sum_counts(count_str) -> pd.Series:
    """
    Sum numeric counts from a string column in a DMR DataFrame.

    Parameters:
    count_str (pandas.Series): A pandas Series with string values containing numeric counts.

    Returns:
    pandas.Series: Sum of counts for each row.
    """
    # Extract numeric values from the string and convert them to numeric type
    numbers = pd.to_numeric(count_str.str.extractall(r'(\d+)')[0])
    return numbers.groupby(level=0).sum()


def truncate_label(label, max_length=70, max_lines=4):
    """Truncate labels to a maximum length and line count, adding an ellipsis if truncated."""

    # Hide extra alternatives
    i = 0
    result = label.split("!!!")[i]
    while i+1 < len(label.split("!!!")) and len(result + "!!!" + label.split("!!!")[i+1]) < max_length * max_lines:
        i += 1
        result += "!!!" + label.split("!!!")[i]

    result += " !!!..." if len(label.split("!!!")) > i+1 else ""

    # Wrap the text
    lines = textwrap.wrap(result, max_length, break_long_words=False)
    result = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        result += "..."
    return result


def expand_pivot_merge_sample_strings(df, column_name) -> pd.DataFrame:
    """
    Expand the methylation string column into separate rows, pivot the table,
    and merge it back with the original DataFrame.
    """
    # Check if dataframe has no data then return it
    if df.empty:
        return df

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


def select_best_annotation_row(group) -> pd.DataFrame:
    """
    Helper function to select the row with the lowest e-value per group, and least ambiguous annotation
    indicating the best functional match.

    :param group: A group of rows from the merged functional annotation DMR DataFrame.
    """
    # Add a temporary column to sort by the number of '!!!' splits in accession
    group['accession_split_len'] = group['accession'].apply(lambda x: len(x.split('!!!')))
    return group.sort_values(by=['e_value', 'accession_split_len']).iloc[0]


def call_function_pairwise(df, function):
    samples = df['sample'].unique()
    sample_combinations = list(combinations(samples, 2))

    results = {}
    for i, (sample1, sample2) in enumerate(sample_combinations):

        sample_pair = None
        if type(df) == pl.DataFrame or type(df) == pl.LazyFrame:
            sample_pair = df.filter(pl.col("sample").is_in([sample1, sample2]))

        elif type(df) == pd.DataFrame:
            sample_pair = df[df['sample'].isin([sample1, sample2])]

        results[sample1, sample2] = function(sample_pair)
        print(f"Done with {i+1}/{len(sample_combinations)}")

    return results


def reshape_pileup_to_matrix(methyl_data, genome_name) -> pd.DataFrame:
    """
    Reshape the methyl_data dataframe to a matrix where each row is a region and each column is a modification type

    :param methyl_data: Dataframe of a bedfile where regions are single nucleotides as returned by get_pileup()
    :type methyl_data: pandas.Dataframe
    :return: A reshaped dataframe where each row is a region and each column is a modification type
    :rtype: pandas.Dataframe
    """
    # Assert for each row that exclusive end position - inclusive start position == 1
    assert all((methyl_data['exclusive end position'] - methyl_data[
        'inclusive start position']) == 1), "Given regions larger than a single nucleotide"

    # Merge columns chrom, inclusive start position, and exclusive end position into a single column called 'name' for easy comparison
    methyl_data['name'] = methyl_data['chrom'] + '|' + methyl_data['strand'] + "|" + methyl_data[
        'inclusive start position'].astype(str) + "|" + methyl_data['exclusive end position'].astype(str)

    # Check no negative valuesin Ndiff and Nvalid_cov
    assert all(methyl_data[methyl_data['Ndiff'] >= 0]) and all(methyl_data[methyl_data['Nvalid_cov'] >= 0]) and all(methyl_data[methyl_data['Ncanonical'] >= 0])

    # Drop rows where Ndiff is larger than Nvalid_cov
    methyl_data = methyl_data[methyl_data['Ndiff'] < methyl_data['Nvalid_cov']].copy()

    # Make sure we are handling supported data
    assert set(methyl_data['modified base code and motif'].unique()).issubset({'a', 'm', '21839'}), \
        f"Unexpected values found: {set(methyl_data['modified base code and motif'].unique()) - {'a', 'm', '21839'} }"

    # Handle different nucleotide types called by keeping group with largest Nvalid_cov
    mod_base_map = {"a": "A", "m": "C", "21839": "C"}
    methyl_data['mod_group'] = methyl_data['modified base code and motif'].map(mod_base_map)

    # Assert that every name has the same total coverage
    assert methyl_data.groupby('name')['Nvalid_cov'].nunique().all() == 1, "Nvalid_cov is not unique for each region"

    # Group by name and mod_group and keep the group with the largest Nvalid_cov
    grouped = methyl_data.groupby(['name', 'mod_group'])
    max_valid_cov = grouped['Nvalid_cov'].transform('max')
    base_corrected_df = methyl_data[methyl_data['Nvalid_cov'] == max_valid_cov]

    assert base_corrected_df.groupby('name')['mod_group'].nunique().max() == 1, "There are multiple nucleotide types called for the same region"

    # Create a new dataframe where there is a row per name, and a column per diffferent value in
    # 'modified base code and motif' where the value of that column is the value in 'Nmod'
    pivot_df = base_corrected_df.pivot_table(index='name', columns='modified base code and motif', values='Nmod',
                                             fill_value=0)
    pivot_df.reset_index(inplace=True)
    pivot_df = pivot_df.merge(base_corrected_df[['name', 'Ncanonical']], on='name', how='left')
    pivot_df.drop_duplicates(inplace=True)

    # Ensure all values are positive
    assert (pivot_df.iloc[:, 1:] >= 0).all().all(), "Not all methylation values are positive"

    # Assert that every name in methyl_data appears at least once in pivot_df, and only once in pivot_df
    assert set(methyl_data['name']) == set(pivot_df['name']), "Not all regions were conserved"
    assert pivot_df['name'].nunique() == len(pivot_df['name']), "There are duplicate regions in pivot_df"

    # Assert that for rows with the same name the sum of modifications in pivot_df is equal to Nvalid_cov in methyl_data
    assert all(pivot_df.groupby('name').sum().sum(axis=1) == methyl_data.groupby('name')['Nvalid_cov'].first()), "Sum of modifications in pivot_df is not equal to Nvalid_cov in methyl_data"

    return pivot_df


def group_methyl_data_by_genes(df: pl.LazyFrame, genes: pl.LazyFrame) -> pl.LazyFrame:
    """
    Aggregate methylation data by genes.

    :param df: The methylation data as a count table
    :type df: pd.Dataframe
    :param genes: The genes as a list of ranges
    :type genes: pd.Dataframe
    :param aggregate: A list of aggregation functions to apply to the columns e.g. [pl.min, pl.max, pl.mean, pl.sum]
    :type aggregate: list
    :return: The aggregated methylation data.
    :rtype: pd. Dataframe
    """

    df = df.with_columns(
        contig=pl.col('name').str.split(by='|').list.get(0),
        start=pl.col('name').str.split(by='|').list.get(2).cast(pl.UInt32),
        stop=pl.col('name').str.split(by='|').list.get(3).cast(pl.UInt32)
    )


    a = df.collect().select('contig').unique().get_column('contig').to_list()
    b = genes.collect().select('contig').unique().get_column('contig').to_list()

    assert all(g1 in b for g1 in a), "Not all contigs are in this genome_name."
    del a, b

    # Create a unique identifier for each range in ranges dataframe
    genes = genes.with_row_index('range_id')

    # Merge df with ranges based on conditions
    df_merged = df.join(genes, on='contig')

    # Filter rows where df start and end values are within range start and end.
    # Gene range is inclusive of end, modkit bed is not.
    df_filtered = df_merged.filter((pl.col('start') >= pl.col('start_right')) & (pl.col('stop') < pl.col('stop_right')))

    # Clean
    result = df_filtered.sort(by=['contig', 'start_right'])
    result = result.with_columns(
            pl.col("range_id").rank("dense").alias("gene_id") - 1
    )
    result = result.drop(['contig', 'start', 'start_right', 'stop', 'stop_right', 'range_id'])

    return result


def normalize_data_for_methylation_level(df, genes, genome_name, aggregate=False):
    if aggregate:
        df = df.with_columns(pl.col('sample').replace(barcode_sample_map))

    # Normalize to coverage
    coverages = dl.get_coverage("../data/", genome_name, agg=aggregate).drop(columns="Genome").to_dict("records")[0]
    methylation_types = df.collect_schema().names()[1:4]

    for key, value in coverages.items():
        if value == 0 and key in df.select("sample").unique():
            print(f"Coverage for {key} is 0")

    df = df.with_columns(pl.col(methylation_types) / pl.col('sample').replace_strict(coverages))

    # Rename samples
    df = df.with_columns(pl.col('sample').replace(readable_sample_name))

    return df


def add_functional_annotations(dmrs, data_dir, genome_name):
    """
    Add functional annotations to DMRs (Differentially Methylated Regions) by matching DMR positions with
    genomic annotations to find overlaps. Drops the "partial" column from the merged DataFrame.
    """
    # Load functional annotations for the specified genome_name from a data directory
    functions = dl.get_coordinated_functions(data_dir, genome_name)

    # Rename 'start' column to 'start_y' to avoid name clash and remove 'gene_callers_id' from function columns
    func_cols = [col if col != "start" else "start_y" for col in functions.columns]
    func_cols.remove("gene_callers_id")

    # Merge DMR data with functional annotations based on contig name
    merged_df = pd.merge(dmrs, functions, how='left', left_on='chrom', right_on='contig')

    # Define a condition for DMRs that actually overlap the annotated regions
    condition = (merged_df['start_x'] >= merged_df['start_y']) & (merged_df['end'] <= merged_df['stop'])

    # For rows not meeting the condition, clear out irrelevant function columns and set default values
    merged_df.loc[~condition, func_cols] = pd.NA
    merged_df.loc[~condition, "gene_callers_id"] = -1
    merged_df.loc[~condition, "source"] = "Unannotated"
    merged_df.loc[~condition, "function"] = "Unknown"

    # Remove duplicate entries from the merged DataFrame
    merged_df.drop_duplicates(inplace=True)

    # Convert the 'accession' column to string format for processing l
    merged_df['accession'] = merged_df['accession'].astype(str)

    # Apply the helper function on groups defined by unique columns, and remove the temporary column afterwards
    unique_cols = ['name', 'gene_callers_id', 'source']
    merged_df = (merged_df.groupby(unique_cols, as_index=False)
                 .apply(select_best_annotation_row, include_groups=False)
                 .drop('accession_split_len', axis=1))

    # Ensure that all unique names from the DMRs are still present after merging and that there are no duplicate groups
    assert set(merged_df["name"].unique()) == set(dmrs["name"].unique())
    assert merged_df.shape[0] == merged_df.groupby(unique_cols).ngroups, "There are duplicate groups in the result."

    # Clean up by dropping columns that are no longer needed
    merged_df.drop(columns=["contig", "start_y", "stop", "e_value"], inplace=True)

    return merged_df
