import os
import polars as pl
from _statistics import *
from utilities.utils import barcode_sample_map
from utilities.data_loading import load_combined_methyl_data_for_genome_polars


def run_analysis(genome_name, dmr_type, data_dir, fig_savepath="plots"):
    """
    Run the Willis DMR analysis for a specific genome_name, DMR type, and source.

    :param genome_name: Folder name of the genome_name.
    :type genome_name: str
    :param dmr_type: Either dmr_by_gene or dmr_by_position, which is also the folder name
    :type dmr_type: str
    :param source: Either KEGG or COG for the functional annotation source.
    :type source: str
    :return: Returns the methyl_data dataframe and saves a PDF of the plot.
    :rtype: pandas.DataFrame
    """

    # Load the data
    combined_methyl_data = load_combined_methyl_data_for_genome_polars(genome_name, data_dir, common_locations=False).collect()

    # Keep two samples and try to run the logistic regression
    combined_methyl_data = combined_methyl_data.with_columns(pl.col("sample").replace(barcode_sample_map, default=pl.first()))
    combined_methyl_data = combined_methyl_data.filter(pl.col("sample").is_in(["top", "middle", "bottom"]))

    # Run the Willis DMR test on each group of same "name" rows
    groups = combined_methyl_data.filter(pl.len().over("name") == 9).group_by(["name"])
    for name, group in groups:
        result = willis_dmr_test_r(group)
        print(result)


if __name__ == "__main__":
    # For each folder in the data directory
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "/Users/GeorgesKanaan/Desktop/methylation_data/methylation_5_agg")
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    run_analysis("polaribacter_r-contigs", "dmr_by_gene", data_dir, fig_savepath="../plots/plots_5")
    run_analysis("Pelagibacter_r-contigs", "dmr_by_gene", data_dir, fig_savepath="../plots/plots_5")

    # for genome_name in folders:
    #     # Run the DMR analysis for the genome_name
    #     print(f"Running analysis for {genome_name}")
    #     run_analysis(genome_name, "dmr_by_gene", data_dir, fig_savepath="plots/plots_5")
