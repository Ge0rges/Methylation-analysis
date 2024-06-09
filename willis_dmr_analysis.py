import os
import glob
from data_loading import get_pileup
from _statistics import *
from itertools import combinations
from plotting import plot_pairwise_results


def reshape_bed_df_to_matrix(methyl_data, genome_name):
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


def load_methyl_data(genome_name, data_dir):
    # Check to see if CSV file exists for this genome
    try:
        combined_methyl_data = pd.read_csv(f"{data_dir}/{genome_name}/combined_methyl_data.csv")

    except FileNotFoundError:
        # Load the methyl_dfs from the bed files
        bed_files = glob.glob(os.path.join(os.path.join(data_dir, genome_name), "*.bed"))
        if len(bed_files) == 0:
            print(f"No bed files found for {data_dir}/{genome_name}")
            return

        methyl_dfs = []
        for i, bed_file in enumerate(bed_files):
            methyl_data = get_pileup(bed_file)
            methyl_data = reshape_bed_df_to_matrix(methyl_data, genome_name)
            methyl_data["sample"] = os.path.basename(bed_file).split(".")[0]

            # Set the index to be name
            methyl_data.set_index("name", inplace=True)

            methyl_dfs.append(methyl_data)

        # Keep in each dataframes only the names that are common to all dataframes
        common_index = methyl_dfs[0].index
        for df_i in methyl_dfs[1:]:
            common_index = common_index.intersection(df_i.index)
        methyl_dfs = [df.loc[common_index] for df in methyl_dfs]

        # Set name back to a column
        methyl_dfs = [df.reset_index(names="name") for df in methyl_dfs]

        # Check that every dataframe has the same names
        name_sets = [df['name'].unique() for df in methyl_dfs]
        assert all([np.array_equal(name_set, name_sets[0]) for name_set in
                    name_sets]), "Not all methyl methyl_dfs have the same regions"

        # Build matrices for statistical testing
        combined_methyl_data = pd.concat(methyl_dfs, ignore_index=True)

        assert len(combined_methyl_data["name"]) == len(name_sets[0]) * len(methyl_dfs)
        assert combined_methyl_data["name"].nunique() == len(name_sets[0])

        # Save this dataframe
        combined_methyl_data.to_csv(f"{data_dir}/{genome_name}/combined_methyl_data.csv", index=False)

    # Check that first column is name and last is sample
    assert combined_methyl_data.columns[0] == "name" and combined_methyl_data.columns[-1] == "sample", "Columns are not in the expected order"

    # Set all columns but the first to be integer types
    for col in combined_methyl_data.columns[1:-1]:
        combined_methyl_data[col] = combined_methyl_data[col].astype(int)

    return combined_methyl_data


def pairwise_epigenomes(combined_methyl_data, function):
    # Perform some logistic regression
    samples = combined_methyl_data['sample'].unique()
    sample_combinations = list(combinations(samples, 2))

    results = {}
    for i, (sample1, sample2) in enumerate(sample_combinations):
        sample_pair = combined_methyl_data[combined_methyl_data['sample'].isin([sample1, sample2])]
        results[sample1, sample2] = function(sample_pair)
        print(f"Done with {i+1}/{len(sample_combinations)}")

    return results


def run_analysis(genome_name, dmr_type, data_dir, fig_savepath="plots"):
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

    # Load the data
    combined_methyl_data = load_methyl_data(genome_name, data_dir)

    # Paired t-test
    #plot_pairwise_results(pairwise_epigenomes(combined_methyl_data, paired_t_test), genome_name + " using paired t-test")

    # Keep first 100 rows of each sample
    # combined_methyl_data = combined_methyl_data.groupby('sample').head(100)
    # print("WARNING: in debug mode cropped data")

    # Rao score
    plot_pairwise_results(pairwise_epigenomes(combined_methyl_data, logistic_regression_pvalue), genome_name + " using statsmodels score")


if __name__ == "__main__":
    # For each folder in the data directory
    print("Running Willis DMR analysis at coverage 5")
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/methylation_5")
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    run_analysis("34H_compare_5", "dmr_by_gene", data_dir, fig_savepath="plots_5")

    # for genome in folders:
    #     # Run the DMR analysis for the genome
    #     print(f"Running analysis for {genome}")
    #     run_analysis(genome, "dmr_by_gene", data_dir, fig_savepath="plots_5")
