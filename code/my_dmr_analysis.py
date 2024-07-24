import os
import polars as pl
from _statistics import *
from utilities.utils import group_methyl_data_by_genes, barcode_sample_map
from utilities.data_loading import load_combined_methyl_data_for_genome, get_genes
from utilities.data_loading_polars import load_combined_methyl_data_for_genome_polars


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
    combined_methyl_data = load_combined_methyl_data_for_genome_polars(genome_name, data_dir, common_locations=False).collect()

    # # Try chi2 test
    # genes = pl.from_pandas(get_genes(data_dir, genome_name)[['contig', 'start', 'stop']].drop_duplicates()).lazy()
    # df = group_methyl_data_by_genes(combined_methyl_data, genes)
    # result = {}
    # for gene in df.select('name').unique().collect():
    #     result[gene] = pearson_chi_squared(df.filter(pl.col('name') == gene))

    # Keep two samples and try to run the logistic regression
    combined_methyl_data = combined_methyl_data.with_columns(pl.col("sample").replace(barcode_sample_map, default=pl.first()))
    combined_methyl_data = combined_methyl_data.filter(pl.col("sample").is_in(["top", "middle", "bottom"]))

    # Run the Willis DMR test on each group of same "name" rows
    groups = combined_methyl_data.filter(pl.len().over("name") == 9).group_by(["name"])
    print(combined_methyl_data.filter(pl.len().over("name") == 9))
    for name, group in groups:
        print(f"Running {group}")
        result = willis_dmr_test_r(group)
        print(result)

    # if not combined_methyl_data.empty:
    #     logistic_regression_pvalue(combined_methyl_data.collect().to_pandas())

    # Rao score
    # plot_pairwise_results(call_function_pairwise(combined_methyl_data, logistic_regression_pvalue), genome_name + " using statsmodels score")


if __name__ == "__main__":
    # For each folder in the data directory
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../methylation_data/methylation_5")
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    run_analysis("polaribacter_r-contigs", "dmr_by_gene", data_dir, fig_savepath="../plots/plots_5")
    run_analysis("Pelagibacter_r-contigs", "dmr_by_gene", data_dir, fig_savepath="../plots/plots_5")

    for genome in folders:
         # Run the DMR analysis for the genome
         print(f"Running analysis for {genome}")
         run_analysis(genome, "dmr_by_gene", data_dir, fig_savepath="plots/plots_5")
