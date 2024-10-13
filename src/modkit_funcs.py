from utilities.plotting import *
from _statistics import *
from utilities.data_loading import *
from utilities.utils import normalize_data_by_pileup, add_gene_caller_id, \
    add_functional_annotations_polars, readable_methylation_name, readable_sample_name, barcode_sample_map
from scipy.stats import rankdata


def run_dmr_analysis(genome_name, dmr_type, coverage, data_dir, fig_savepath="plots"):
    """
    Run the DMR analysis for a specific genome_name, DMR type, and function_source.
    """

    # Load the data from the bed files
    dmr_data = get_dmrs_for_genome_polars(data_dir, genome_name, dmr_type)
    if dmr_data is None:
        print(f"No DMRs found for {genome_name}")
        return

    # Get genes and annotate the dmrs with the gene ID
    genes = get_genes_polars(data_dir)
    dmr_data = add_gene_caller_id(dmr_data, genes)

    # Keep only statistically significant DMRs
    dmr_data = dmr_data.join(dmr_data.group_by('comparison').len().rename({"len": "num_tests"}), on="comparison")
    dmr_data = dmr_data.with_columns(test_result=pl.struct(['score', 'num_tests']).map_elements(
                                         lambda row: modkit_llr(row['score'], row['num_tests']), return_dtype=pl.Boolean))

    # Annotate with function
    dmr_data = add_functional_annotations_polars(dmr_data, data_dir).collect()

    # Filter for signiciant DMR with correct function_source and comparison, then keep top 20
    function_source = "KEGG_Module"
    dmr_data = dmr_data.filter(pl.col("test_result") &
                               pl.col("source").eq(function_source) &
                               pl.col("comparison").is_in(["top_vs_bottom", "top_vs_middle"]))
    dmr_data = dmr_data.group_by(['function', 'source', 'comparison']).agg(pl.col('score').mean(), pl.col("gene_callers_id"), pl.col("test_result").first()).top_k(20, by="score")
    dmr_data = dmr_data.explode("gene_callers_id")

    # Handle empty
    if dmr_data.is_empty():
        print(f"No stastistically significant DMRs found for {genome_name}")

    # Get methylation level data
    methylation_types = list(readable_methylation_name.keys())
    methyl_data = load_combined_methyl_data_for_genome_polars(genome_name, data_dir, coverage=5)

    # Filter samples
    methyl_data = methyl_data.with_columns(pl.col("sample").alias("norm_sample"), pl.col("sample").replace_strict(barcode_sample_map))
    methyl_data = methyl_data.filter(pl.col("sample").is_in(["top", "middle", "bottom"])).collect().lazy()

    # Create the total methylation column and normalize
    methyl_data = methyl_data.with_columns(pl.concat_list(methylation_types).list.sum().alias("total_methylation"))
    methyl_data = normalize_data_by_pileup(methyl_data)

    # Add gene caller id
    methyl_data = add_gene_caller_id(methyl_data, genes).collect(streaming=True)

    # Add a gene_id column, which is just a map from gene_callers_id
    all_ids = dmr_data.sort("strand", "chrom",  "start").get_column("gene_callers_id").to_list() + methyl_data.sort("strand", "contig",  "start").get_column("gene_callers_id").to_list()
    ids = dict(zip(all_ids, rankdata(all_ids, method='dense')))
    dmr_data = dmr_data.with_columns(gene_id=pl.col("gene_callers_id").replace_strict(ids, default=-1))
    methyl_data = methyl_data.with_columns(gene_id=pl.col("gene_callers_id").replace_strict(ids, default=-1))

    # Get mean methylation for each gene
    methyl_data = methyl_data.filter(pl.col("gene_id").is_in(dmr_data.get_column("gene_id")))
    methyl_data = methyl_data.group_by("gene_id", "sample").agg(pl.col("total_methylation").mean())
    methyl_data = add_functional_annotations_polars(methyl_data.lazy(), data_dir).collect(streaming=True)
    methyl_data = methyl_data.filter(pl.col("source").eq("KOfam"))
    methyl_data = methyl_data.pivot(on="function", index="sample", values="total_methylation")

    # Plot heatmap of the methylation difference of these DMRs
    n_types = len(methylation_types)
    fig, axes = plt.subplots(1, 1, figsize=(20, 5 * n_types), sharex=False, layout="constrained")
    sns.heatmap(methyl_data.to_pandas(), ax=axes, cmap="coolwarm", cbar_kws={'label': 'Mean methylation'})

    # Save the figure
    cleaned_genome_name = genome_name.title().replace("_R-Contigs", " sp.")
    fig.suptitle(f"Top 20 differentially methylated KOfams according to modkit {cleaned_genome_name}", fontsize=26)
    plt.savefig(f"{fig_savepath}/{genome_name}_{coverage}_modkit.pdf", format='pdf', transparent=True)

    print(f"Done plotting composite for {genome_name}")
    return


if __name__ == "__main__":
    for coverage in ["5"]:
        print(f"Running DMR analysis at coverage {coverage}")
        data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"../../methylation_data/methylation_{coverage}")
        for genome in os.listdir(data_dir):
            if genome == ".DS_Store" or "Octadecabacter" in genome or "metagenome" in genome:
                continue

            run_dmr_analysis(genome, "dmr_by_gene", coverage, data_dir, fig_savepath=f"../plots/plots_{coverage}")
