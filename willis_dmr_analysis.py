import os
from _statistics import *
from itertools import combinations
from plotting import plot_pairwise_results
from data_loading import load_combined_methyl_data_for_genome


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
    combined_methyl_data = load_combined_methyl_data_for_genome(genome_name, data_dir, common_locations=False)

    print("Got combined methyl data")

    # Paired t-test
    #plot_pairwise_results(pairwise_epigenomes(combined_methyl_data, paired_t_test), genome_name + " using paired t-test")

    # Keep first 100 rows of each sample
    combined_methyl_data = combined_methyl_data[combined_methyl_data['sample'].isin(["top", "bottom"])]
    if not combined_methyl_data.empty:
        willis_dmr_test(combined_methyl_data)

    # Rao score
    #plot_pairwise_results(pairwise_epigenomes(combined_methyl_data, logistic_regression_pvalue), genome_name + " using statsmodels score")


if __name__ == "__main__":
    # For each folder in the data directory
    print("Running Willis DMR analysis at coverage 5")
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../methylation_5")
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    run_analysis("polaribacter_r-contigs", "dmr_by_gene", data_dir, fig_savepath="plots_5")

    # for genome in folders:
    #     # Run the DMR analysis for the genome
    #     print(f"Running analysis for {genome}")
    #     run_analysis(genome, "dmr_by_gene", data_dir, fig_savepath="plots_5")
