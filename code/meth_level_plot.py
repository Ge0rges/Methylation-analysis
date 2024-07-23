from utilities.plotting import *
from utilities.data_loading import *


def run_meth_level_plots(genome_name, data_dir, coverage, fig_savepath="plots"):
    """
    Run the DMR analysis for a specific genome, DMR type, and source.

    :param genome_name: Folder name of the genome.
    :type genome_name: str
    :param data_dir: Either dmr_by_gene or dmr_by_position, which is also the folder name
    :type data_dir: str
    :param fig_savepath: Folder in which to save figures
    :type fig_savepath: str
    :return: Returns the methyl_data dataframe and saves a PDF of the plot.
    :rtype: pandas.DataFrame
    """
    # Plot methylation levels per base
    methyl_data = load_combined_methyl_data_for_genome(genome_name, data_dir, common_locations=False)
    genes = get_genes(data_dir, genome_name)[['contig', 'start', 'stop']].drop_duplicates()

    if not methyl_data.empty:
        print(f"Plotting methyl data for {genome_name}")
        
        plot_methylation_levels_by_gene(methyl_data, genes, genome_name, coverage, fig_savepath=fig_savepath)

    return


if __name__ == "__main__":
    #print("Plotting methylation level at coverage 5")
    #data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../methylation_data/methylation_5")
    #for genome in os.listdir(data_dir):
        #run_meth_level_plots(genome, data_dir, "meth_5", fig_savepath="../plots/plots_5")
    
    print("Plotting methylation level for 5_agg")
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../methylation_data/methylation_5_agg")
    for genome in os.listdir(data_dir):
        run_meth_level_plots(genome, data_dir, "meth_5_agg", fig_savepath="../plots/plots_5_agg")

    print("Plotting methylation level at coverage 10")
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../methylation_data/methylation_10")
    for genome in os.listdir(data_dir):
        run_meth_level_plots(genome, data_dir, "meth_10", fig_savepath="../plots/plots_10")
