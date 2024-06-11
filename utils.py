import textwrap
import numpy as np
import pandas as pd


barcode_sample_map = {"Barcode01": "S2-1",
                      "Barcode02": "S2-2",
                      "Barcode03": "S2-3",
                      "Barcode04": "Control",
                      "Barcode05": "S3-1",
                      "Barcode06": "S3-2",
                      "Barcode07": "S3-3",
                      "Barcode08": "S4-1",
                      "Barcode09": "S4-2",
                      "Barcode10": "S4-3",
                      "Barcode11": "IC3-1 (30 cm)",
                      "Barcode12": "IC3-2 (160 cm)",
                      "Barcode13": "IC3-3 (205 cm)",
                      "Barcode14": "IC3-4 (70 cm)",
                      "Top": "Sackhole Top (40 cm)",
                      "Bottom": "Sackhole Bottom (160cm)",
                      "Middle": "Sackhole Middle (70 cm)",
                      "Control": "Control"
}

read_counts = {
    "Barcode01": 1093788,
    "Barcode02": 296042,
    "Barcode03": 5812056,
    "Barcode04": 57626,
    "Barcode05": 344880,
    "Barcode06": 180208,
    "Barcode07": 1056185,
    "Barcode08": 178883,
    "Barcode09": 1776313,
    "Barcode10": 1163651,
    "Barcode11": 41324,
    "Barcode12": 591165,
    "Barcode13": 39685,
    "Barcode14": 96793,
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
    while i+1 < len(label.split("!!!")) and len(result + label.split("!!!")[i+1]) < max_length * max_lines:
        i += 1
        result += label.split("!!!")[i]

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
    methyl_data['name'] = methyl_data['chrom'] + '_' + methyl_data['strand'] + ":" + methyl_data[
        'inclusive start position'].astype(str) + "-" + methyl_data['exclusive end position'].astype(str)

    # Drop rows where Ndiff is larger than Nvalid_cov
    methyl_data = methyl_data[methyl_data['Ndiff'] < methyl_data['Nvalid_cov']].copy()

    # Handle different nucleotide types called by keeping group with largest Nvalid_cov
    mod_base_map = {"a": "A", "m": "C", "21839": "C"}
    methyl_data['mod_group'] = methyl_data['modified base code and motif'].map(mod_base_map)

    # Assert that every name has the same total coverage
    assert methyl_data.groupby('name')['Nvalid_cov'].nunique().all() == 1, "Nvalid_cov is not unique for each region"

    # Group by name and mod_group and keep the group with the largest Nvalid_cov
    grouped = methyl_data.groupby(['name', 'mod_group'])
    max_valid_cov = grouped['Nvalid_cov'].transform('max')
    base_corrected_df = methyl_data[methyl_data['Nvalid_cov'] == max_valid_cov]

    # Check that for each name, there is only mod_group value
    assert base_corrected_df.groupby('name')[
               'mod_group'].nunique().max() == 1, "There are multiple nucleotide types called for the same region"

    # Create a new dataframe where there is a row per name, and a column per diffferent value in
    # 'modified base code and motif' where the value of that column is the value in 'Nmod'
    pivot_df = base_corrected_df.pivot_table(index='name', columns='modified base code and motif', values='Nmod',
                                             fill_value=0)
    pivot_df.reset_index(inplace=True)
    pivot_df = pivot_df.merge(base_corrected_df[['name', 'Ncanonical']], on='name', how='left')
    pivot_df.drop_duplicates(inplace=True)

    # Assert that every name in methyl_data appears at least once in pivot_df, and only once in pivot_df
    assert set(methyl_data['name']) == set(pivot_df['name']), "Not all regions were conserved"
    assert pivot_df['name'].nunique() == len(pivot_df['name']), "There are duplicate regions in pivot_df"

    # Assert that for rows with the same name the sum of modifications in pivot_df is equal to Nvalid_cov in methyl_data
    if genome_name != "test":
        assert all(pivot_df.groupby('name').sum().sum(axis=1) == methyl_data.groupby('name')['Nvalid_cov'].first()), "Sum of modifications in pivot_df is not equal to Nvalid_cov in methyl_data"

    return pivot_df
