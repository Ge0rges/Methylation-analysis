from utilities.plotting import *
from utilities.data_loading import *
from utilities.data_loading_polars import load_combined_methyl_data_for_genome_polars
from utilities.utils import group_and_normalize_data_for_methylation_level


def run_meth_level_plots(genome_name, data_dir, coverage, fig_savepath="plots"):
    """
    Run the DMR analysis for a specific genome_name, DMR type, and source.

    :param genome_name: Folder name of the genome_name.
    :type genome_name: str
    :param data_dir: Either dmr_by_gene or dmr_by_position, which is also the folder name
    :type data_dir: str
    :param fig_savepath: Folder in which to save figures
    :type fig_savepath: str
    :return: Returns the methyl_data dataframe and saves a PDF of the plot.
    :rtype: pandas.DataFrame
    """
    # Plot methylation levels per base
    methyl_data = load_combined_methyl_data_for_genome_polars(genome_name, data_dir, common_locations=False)
    genes = get_genes(data_dir, genome_name)[['contig', 'start', 'stop']].drop_duplicates()
    df = group_and_normalize_data_for_methylation_level(methyl_data, genes, genome_name, ("agg" in coverage))
    df = df.filter(pl.col("sample").is_in(["top", "middle", "bottom"]))

    print(f"Plotting methyl data for {genome_name}")
    plot_gene_methylation_level_figure(df, genome_name, coverage, fig_savepath=fig_savepath)

    return


if __name__ == "__main__":
    print("Plotting methylation level for 5_agg")
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../methylation_data/methylation_5_agg")
    for genome in os.listdir(data_dir):
        run_meth_level_plots(genome, data_dir, "5_agg", fig_savepath="../plots/plots_5_agg")

    print("Plotting methylation level at coverage 10_agg")
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../methylation_data/methylation_10_agg")
    for genome in os.listdir(data_dir):
        run_meth_level_plots(genome, data_dir, "10_agg", fig_savepath="../plots/plots_10_agg")
