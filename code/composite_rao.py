from utilities.plotting import *
from _statistics import add_rao_score_by_gene
from utilities.data_loading import *
from utilities.utils import normalize_data_for_methylation_level, add_gene_caller_id, \
    add_functional_annotations_polars, readable_methylation_name, readable_sample_name, barcode_sample_map, normalize_data_by_pileup
from scipy.stats import rankdata

def run_dmr_analysis(genome_name, coverage, data_dir, fig_savepath="plots"):
    """
    Run the DMR analysis for a specific genome_name, DMR type, and function_source.
    """

    print(f"Starting to generate  composite for {genome_name}")

    # Get the genes
    genes = get_genes_polars(data_dir, genome_name)

    # Get methylation level data
    methylation_types = list(readable_methylation_name.keys())
    methyl_data = load_combined_methyl_data_for_genome_polars(genome_name, data_dir).select("name", "sample", *methylation_types, "Ncanonical")
    methyl_data = methyl_data.with_columns(
        contig=pl.col('name').str.split(by='|').list.get(0),
        strand=pl.col('name').str.split(by='|').list.get(1),
        start=pl.col('name').str.split(by='|').list.get(2).cast(pl.UInt32),
        end=pl.col('name').str.split(by='|').list.get(3).cast(pl.UInt32)
    )

    # Filter samples
    #methyl_data = methyl_data.with_columns(pl.col("sample").alias("norm_sample"), pl.col("sample").replace_strict(barcode_sample_map))
    methyl_data = methyl_data.with_columns(pl.col("sample").replace_strict(barcode_sample_map, default=pl.first()))
    methyl_data = methyl_data.filter(pl.col("sample").is_in(["top", "middle", "bottom"]))

    # Add the gene_caller_id
    methyl_data = add_gene_caller_id(methyl_data, genes, True).collect(streaming=True)

    # Add rao score - Doing this first prevents row duplication issues
    methyl_data = add_rao_score_by_gene(methyl_data, ["top", "middle", "bottom"], baseline="middle")
    methyl_data = add_rao_score_by_gene(methyl_data, ["top", "middle"], baseline=False)
    methyl_data = add_rao_score_by_gene(methyl_data, ["top", "bottom"], baseline=False).lazy()

    # Create the total methylation column and normalize values
    methyl_data = normalize_data_by_pileup(methyl_data)
    methyl_data = methyl_data.with_columns(pl.concat_list(methylation_types).list.sum().alias("total_methylation")).collect(streaming=True)

    # Add a gene_id column, which is just a map from gene_callers_id
    all_ids = methyl_data.get_column("gene_callers_id").to_list()
    ids = dict(zip(all_ids, rankdata(all_ids, method='dense')))
    methyl_data = methyl_data.with_columns(gene_id=pl.col("gene_callers_id").replace_strict(ids, default=np.NAN))

    # Create figure
    n_types = len(methylation_types)
    fig, axes = plt.subplots(3, 2, figsize=(20, 5 * n_types), sharex=False, layout="constrained", gridspec_kw={'width_ratios': [5] + [5]})

    # Mean together all the different methylation types
    mean_data = methyl_data.select('gene_id', 'sample', 'total_methylation')

    top = pl.col(*methylation_types).filter(pl.col('sample').eq('top'))
    middle = pl.col(*methylation_types).filter(pl.col('sample').eq('middle'))
    bot = pl.col(*methylation_types).filter(pl.col('sample').eq('bottom'))

    top_bottom = methyl_data.group_by('gene_id').agg(top.mean() - bot.mean())
    top_middle = methyl_data.group_by('gene_id').agg(top.mean() - middle.mean())
    top_bottom = top_bottom.unpivot(index="gene_id", on=methylation_types, variable_name="methylation_type", value_name="methylation_level")
    top_middle = top_middle.unpivot(index="gene_id", on=methylation_types, variable_name="methylation_type", value_name="methylation_level")

    # Rename samples
    mean_data = mean_data.with_columns(pl.col('sample').replace(readable_sample_name))
    top_bottom = top_bottom.with_columns(pl.col('methylation_type').replace(readable_methylation_name))
    top_middle = top_middle.with_columns(pl.col('methylation_type').replace(readable_methylation_name))

    # Populate graphs
    plot_mean_gene_methylation_level(axes[0][0], mean_data)
    plot_gene_methylation_level_diff(axes[1][0], top_middle, "Top – Middle")
    plot_gene_methylation_level_diff(axes[2][0], top_bottom, "Top – Bottom")

    # Add functional annotation
    methyl_data = add_functional_annotations_polars(methyl_data.lazy(), data_dir, genome_name).collect()
    function_source = "KEGG_Module"

    # Plot functional annotations
    annotate_meth_level_with_score_function_table(axes[0][0], axes[0][1], methyl_data, function_source, "rao_score", "middle_vs_top_bottom")
    annotate_meth_level_with_score_function_table(axes[1][0], axes[1][1], methyl_data, function_source, "rao_score", "top_vs_middle")
    annotate_meth_level_with_score_function_table(axes[2][0], axes[2][1], methyl_data, function_source, "rao_score", "top_vs_bottom")

    axes[0][1].axis("off")
    axes[1][1].axis("off")
    axes[2][1].axis("off")

    # Save the figure
    cleaned_genome_name = genome_name.title().replace("_R-Contigs", " sp.")
    fig.suptitle(f"Mean gene methylation overview for {cleaned_genome_name}", fontsize=26)
    plt.savefig(f"{fig_savepath}/{genome_name}_{coverage}_composite_rao.svg", format='svg', transparent=True)

    print(f"Done plotting composite for {genome_name}")
    return


if __name__ == "__main__":
    for coverage in ["5", "5_agg"]:
        print(f"Running rao analysis at coverage {coverage}")
        data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"../../methylation_data/methylation_{coverage}")
        for genome in os.listdir(data_dir):
            if genome == ".DS_Store":
                continue

            run_dmr_analysis(genome, coverage, data_dir, fig_savepath=f"../plots/plots_{coverage}")
