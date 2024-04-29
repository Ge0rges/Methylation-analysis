import pandas as pd
import numpy as np

barcode_sample_map = {"Barcode01": "S2-1", "Barcode02": "S2-2", "Barcode03": "S2-3",
                      "Barcode04": "Control", "Barcode05": "S3-1", "Barcode06": "S3-2",
                      "Barcode07": "S3-3", "Barcode08": "S4-1", "Barcode09": "S4-2",
                      "Barcode10": "S4-3", "Barcode11": "IC3-1 (30 cm)", "Barcode12": "IC3-2 (160 cm)",
                      "Barcode13": "IC3-3 (205 cm)", "Barcode14": "IC3-4 (70 cm)", "Top": "Sackhole Top (40 cm)",
                      "Bottom": "Sackhole Bottom (160cm)", "Middle": "Sackhole Middle (70 cm)", "Control": "Control"}

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

def sum_counts(count_str):
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


def expand_pivot_merge_sample_strings(df, column_name):
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


def select_best_annotation_row(group):
    """
    Helper function to select the row with the lowest e-value per group, and least ambiguous annotation
    indicating the best functional match.

    :param group: A group of rows from the merged functional annotation DMR DataFrame.
    """
    # Add a temporary column to sort by the number of '!!!' splits in accession
    group['accession_split_len'] = group['accession'].apply(lambda x: len(x.split('!!!')))
    return group.sort_values(by=['e_value', 'accession_split_len']).iloc[0]
