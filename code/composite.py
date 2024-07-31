from utilities.plotting import *
from _statistics import *
from utilities.data_loading import *
from utilities.utils import normalize_data_for_methylation_level, add_gene_caller_id, \
    add_functional_annotations_polars, readable_methylation_name, readable_sample_name
from scipy.stats import rankdata


def run_dmr_analysis(genome_name, dmr_type, coverage, data_dir, fig_savepath="plots"):
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
    dmr_data = get_dmrs_for_genome_polars(data_dir, genome_name, dmr_type)
    if dmr_data is None:
        print(f"No DMRs found for {genome_name}")
        return

    # Get genes and annotate the dmrs with the gene ID
    genes = get_genes_polars(data_dir, genome_name)
    dmr_data = add_gene_caller_id(dmr_data, genes, False)

    # Keep only statistically significant DMRs
    dmr_data = dmr_data.join(dmr_data.group_by('comparison').len().rename({"len": "num_tests"}), on="comparison")
    dmr_data = dmr_data.with_columns(test_result=pl.struct(['score', 'num_tests']).map_elements(
                                         lambda row: modkit_llr(row['score'], row['num_tests']), return_dtype=pl.Boolean))

    # Annotate with function
    dmr_data = add_functional_annotations_polars(dmr_data, data_dir, genome_name).collect()

    # Filter for signiciant DMR with correct source and comparison, then keep top 10
    source = "KEGG_Module"
    dmr_data = dmr_data.filter(pl.col("test_result") &
                               pl.col("source").eq(source) &
                               pl.col("comparison").is_in(["top_VS_bottom"]))
    dmr_data = dmr_data.group_by(['function', 'comparison']).agg(pl.col('score').mean(), pl.col("gene_callers_id")).top_k(10, by="score")
    dmr_data = dmr_data.explode("gene_callers_id")

    # Handle empty
    if dmr_data.is_empty():
        print(f"No stastistically significant DMRs found for {genome_name}")

    # Get methylation level data
    methylation_types = list(readable_methylation_name.keys())
    methyl_data = load_combined_methyl_data_for_genome_polars(genome_name, data_dir, common_locations=False).collect().select("name", "sample", *methylation_types)
    methyl_data = methyl_data.with_columns(
        contig=pl.col('name').str.split(by='|').list.get(0),
        strand=pl.col('name').str.split(by='|').list.get(1),
        start=pl.col('name').str.split(by='|').list.get(2).cast(pl.UInt32),
        end=pl.col('name').str.split(by='|').list.get(3).cast(pl.UInt32)
    )

    # Filter samples
    methyl_data = methyl_data.filter(pl.col("sample").is_in(["top", "middle", "bottom"])).lazy()
    
    # Create the total methylation column
    methyl_data = methyl_data.with_columns(pl.concat_list(methylation_types).list.sum().alias("total_methylation"))

    # Annonate methyl data and normalize it
    methyl_data = add_gene_caller_id(methyl_data, genes, True)
    methyl_data = normalize_data_for_methylation_level(methyl_data, genome_name, methylation_types, ("agg" in coverage)).collect()

    # Add a gene_id column, which is just a map from gene_callers_id
    all_ids = dmr_data.get_column("gene_callers_id").to_list() + methyl_data.get_column("gene_callers_id").to_list()
    ids = dict(zip(all_ids, rankdata(all_ids, method='dense')))
    dmr_data = dmr_data.with_columns(gene_id=pl.col("gene_callers_id").replace_strict(ids, default=-1))
    methyl_data = methyl_data.with_columns(gene_id=pl.col("gene_callers_id").replace_strict(ids, default=-1))

    # Create figure
    n_types = len(methylation_types)
    fig, axes = plt.subplots(3, 1, figsize=(10, 5 * n_types), sharex=False, layout="constrained")

    # Mean together all the different methylation types
    mean_data = methyl_data.select('gene_id', 'sample', 'total_methylation')
    top = pl.col(*methylation_types).filter(pl.col('sample').eq('top'))
    bot = pl.col(*methylation_types).filter(pl.col('sample').eq('bottom'))
    middle = pl.col(*methylation_types).filter(pl.col('sample').eq('middle'))
    top_bottom = methyl_data.group_by('gene_id').agg(top.mean() - bot.mean())
    top_bottom = top_bottom.unpivot(index="gene_id", on=methylation_types, variable_name="methylation_type", value_name="methylation_level")
    top_middle = methyl_data.group_by('gene_id').agg(top.mean() - middle.mean())
    top_middle = top_middle.unpivot(index="gene_id", on=methylation_types, variable_name="methylation_type", value_name="methylation_level")

    # Rename samples
    mean_data = mean_data.with_columns(pl.col('sample').replace(readable_sample_name))
    top_bottom = top_bottom.with_columns(pl.col('methylation_type').replace(readable_methylation_name))
    top_middle = top_middle.with_columns(pl.col('methylation_type').replace(readable_methylation_name))

    # Populate subplots
    plot_mean_gene_methylation_level(axes[0], mean_data)
    annotate_dmr_table_to_meth_level(axes[0], dmr_data)
    plot_gene_methylation_level_diff(axes[1], top_middle, "Top – Middle")
    plot_gene_methylation_level_diff(axes[2], top_bottom, "Top – Bottom")

    # Save the figure
    cleaned_genome_name = genome_name.title().replace("_R-Contigs", " sp.")
    fig.suptitle(f"Mean gene methylation overview for {cleaned_genome_name}", fontsize=26)
    plt.savefig(f"{fig_savepath}/{genome_name}_{coverage}_composite.svg", format='svg')

    print(f"Done plotting composite for {genome_name}")
    return


if __name__ == "__main__":
    print("Running DMR analysis at coverage 5 agg")
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../methylation_data/methylation_5_agg")
    for genome in os.listdir(data_dir):
        run_dmr_analysis(genome, "dmr_by_gene", "5_agg", data_dir, fig_savepath="../plots/plots_5_agg")
