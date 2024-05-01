import os
import glob
import pandas as pd
from data_loading import get_pileup


def reshape_bed_df_to_matrix(methyl_data):
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
    assert methyl_data.groupby('name')['Nvalid_cov'].nunique().all() == 1, "Nvalid_cov is not unique for each region"
    assert all(pivot_df.groupby('name').sum().sum(axis=1) == methyl_data.groupby('name')['Nvalid_cov'].first()), \
        "Sum of modifications in pivot_df is not equal to Nvalid_cov in methyl_data"

    return pivot_df


def run_willis_analysis(genome_name, dmr_type, data_dir, fig_savepath="plots"):
    """
    Run the Willis DMR analysis for a specific genome, DMR type, and source.

    :param genome: Folder name of the genome.
    :type genome: str
    :param dmr_type: Either dmr_by_gene or dmr_by_position, which is also the folder name
    :type dmr_type: str
    :param source: Either KEGG or COG for the functional annotation source.
    :type source: str
    :return: Returns the methyl_data dataframe and saves a PDF of the plot.
    :rtype: pandas.DataFrame
    """

    # Load the data from the bed files
    bed_files = glob.glob(os.path.join(os.path.join(data_dir, genome_name), "*.bed"))
    if len(bed_files) == 0:
        return

    for bed_file in bed_files:
        methyl_data = get_pileup(bed_file)
        methyl_data = reshape_bed_df_to_matrix(methyl_data)

        # Save the dataframe to a CSV
        savepath = os.path.join(data_dir, genome_name, f"{os.path.basename(bed_file).replace('.bed', '_reshaped.csv')}")
        methyl_data.to_csv(savepath, index=False)



if __name__ == "__main__":
    # For each folder in the data directory
    print("Running Willis DMR analysis at coverage 5")
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/methylation_5")

    run_willis_analysis("polaribacter_r-contigs", "dmr_by_gene", data_dir, fig_savepath="plots_5")

    # for genome in os.listdir(data_dir):
    #     # Run the DMR analysis for the genome
    #     run_willis_analysis(genome, "dmr_by_gene", data_dir, fig_savepath="plots_5")
