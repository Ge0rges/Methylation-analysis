import os
import polars as pl
from _statistics import *
from utilities.utils import barcode_sample_map, add_gene_caller_id
from utilities.data_loading import load_combined_methyl_data_for_genome_polars, get_genes_polars


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
    methyl_data = load_combined_methyl_data_for_genome_polars(genome_name, data_dir, common_locations=False).collect().lazy()
    methyl_data = methyl_data.with_columns(
                contig=pl.col('name').str.split(by='|').list.get(0),
                strand=pl.col('name').str.split(by='|').list.get(1),
                start=pl.col('name').str.split(by='|').list.get(2).cast(pl.UInt32),
                end=pl.col('name').str.split(by='|').list.get(3).cast(pl.UInt32)
    )

    # Keep two samples and try to run the logistic regression
    methyl_data = methyl_data.with_columns(pl.col("sample").replace_strict(barcode_sample_map, default=pl.first()))
    sample_filter = ["top", "bottom"]
    methyl_data = methyl_data.filter(pl.col("sample").is_in(["top", "bottom"])).sort("sample").collect().lazy()
    
    genes = get_genes_polars(data_dir, genome_name)
    methyl_data = add_gene_caller_id(methyl_data, genes, True).select(pl.exclude("contig", "strand", "start", "end", "direction")).collect(streaming=True)

    # Run the Willis DMR test on each gene rows
    #groups = combined_methyl_data.filter(pl.len().over("name") == len(sample_filter)*3).group_by("name")
    groups = methyl_data.group_by("gene_callers_id")
    for name, group in groups:
        if group.get_column("sample").n_unique() == len(sample_filter):
            result = willis_dmr_test_r(group.drop("gene_callers_id"))
            if result["p"] < 0.05:
                print(name)
                print(result)


if __name__ == "__main__":
    # For each folder in the data directory
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../methylation_data/methylation_5")
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    run_analysis("Pelagibacter_r-contigs", "dmr_by_gene", data_dir, fig_savepath="../plots/plots_5")

    # for genome_name in folders:
    #     # Run the DMR analysis for the genome_name
    #     print(f"Running analysis for {genome_name}")
    #     run_analysis(genome_name, "dmr_by_gene", data_dir, fig_savepath="plots/plots_5")
