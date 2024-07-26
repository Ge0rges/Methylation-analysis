from utilities.plotting import *
from _statistics import *
from utilities.data_loading import *


def run_dmr_analysis(genome_name, dmr_type, data_dir, fig_savepath="plots"):
    """
    Run the DMR analysis for a specific genome_name, DMR type, and source.

    :param genome_name: Folder name of the genome_name.
    :type genome_name: str
    :param dmr_type: Either dmr_by_gene or dmr_by_position, which is also the folder name
    :type dmr_type: str
    :param source: Either KEGG or COG for the functional annotation source.
    :type source: str
    :return: Returns the methyl_data dataframe and saves a PDF of the plot.
    :rtype: pandas.DataFrame
    """

    # Load the data from the bed files
    bed_files = glob.glob(os.path.join(os.path.join(data_dir, genome_name, dmr_type), "*.bed"))
    bed_files = [file for file in bed_files if not file.endswith('-bedgraph.bed')]
    if len(bed_files) == 0:
        print(f"No DMR bed files found for {genome_name}")
        return

    methyl_data = get_dmr_by_sample_annotated(data_dir, genome_name, bed_files)

    # Handle empty
    if methyl_data.empty:
        print(f"No DMRs found for {genome_name}")
        return

    # Keep only statistically significant DMRs
    methyl_data['num_tests'] = methyl_data.groupby('comparison')['comparison'].transform('count')
    methyl_data['test_result'] = methyl_data.apply(lambda x: modkit_llr(x['score'], x['num_tests']), axis=1)
    methyl_data = methyl_data[methyl_data['test_result']]

    # Handle empty
    if methyl_data.empty:
        print(f"No stastistically significant DMRs found for {genome_name}")
        return

    # Plot heatmap
    methyl_data = methyl_data[methyl_data["source"].isin(["KOfam", "KEGG_Module"])]
    plot_all_sources_figure(methyl_data, genome_name, heatmap_type=dmr_type, fig_savepath=fig_savepath, plot_function=plot_heatmap)

    # # Get genomic sequence context, and for each DMR and add it to the DataFrame
    # genome_dict = get_genomic_sequence(genome_name)
    # methyl_data['sequence_context'] = methyl_data.apply(lambda x: genome_dict[x["chrom"]][x["start_x"]:x["end"]], axis=1)

    return


if __name__ == "__main__":
    print("Running DMR analysis at coverage 5 agg")
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "/Users/GeorgesKanaan/Desktop/methylation_data/methylation_5_agg")
    for genome in os.listdir(data_dir):
        run_dmr_analysis(genome, "dmr_by_gene", data_dir, fig_savepath="../plots/plots_5_agg")

    print("Running DMR analysis at coverage 5")
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "/Users/GeorgesKanaan/Desktop/methylation_data/methylation_5")
    for genome in os.listdir(data_dir):
        run_dmr_analysis(genome, "dmr_by_gene", data_dir, fig_savepath="../plots/plots_5")

    print("Running DMR analysis at coverage 10")
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "/Users/GeorgesKanaan/Desktop/methylation_data/methylation_10")
    for genome in os.listdir(data_dir):
        run_dmr_analysis(genome, "dmr_by_gene", data_dir, fig_savepath="../plots/plots_10")

    print("Running DMR analysis at coverage 10 agg")
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "/Users/GeorgesKanaan/Desktop/methylation_data/methylation_10_agg")
    for genome in os.listdir(data_dir):
        run_dmr_analysis(genome, "dmr_by_gene", data_dir, fig_savepath="../plots/plots_10_agg")
