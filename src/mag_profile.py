from utilities.plotting import *
from utilities.data_loading import *
from utilities.utils import add_gene_caller_id, \
    add_functional_annotations_polars, readable_methylation_name, readable_sample_name, barcode_sample_map, normalize_data_by_pileup
from scipy.stats import rankdata
sns.set_theme(context="talk", style="white")


def run_analysis(genome_name, coverage, data_dir, fig_savepath="plots"):
    """
    Run the analysis for a specific genome_name, DMR type, and function_source.
    """

    print(f"Starting to generate  composite for {genome_name}")

    # Get the genes
    genes = get_genes_polars(data_dir)

    # Get methylation level data
    methylation_types = list(readable_methylation_name.keys())
    methyl_data = load_combined_methyl_data_for_genome_polars(genome_name, data_dir, coverage=5)

    # Filter samples
    methyl_data = methyl_data.with_columns(pl.col("sample").replace_strict(barcode_sample_map, default=pl.first()))
    methyl_data = methyl_data.filter(pl.col("sample").is_in(["top", "middle", "bottom"]))

    # Add the gene_caller_id
    methyl_data = add_gene_caller_id(methyl_data, genes, True).collect(streaming=True)

    if methyl_data.is_empty():
        print(f"No valid data for {genome_name}")
        return

    # Create the total methylation column and normalize values
    methyl_data = normalize_data_by_pileup(methyl_data)
    methyl_data = methyl_data.with_columns(pl.concat_list(methylation_types).list.sum().alias("total_methylation"))

    # Add a gene_id column, which is just a map from gene_callers_id
    all_ids = methyl_data.sort("strand", "contig",  "start").get_column("gene_callers_id").to_list()
    ids = dict(zip(all_ids, rankdata(all_ids, method='dense')))
    methyl_data = methyl_data.with_columns(gene_id=pl.col("gene_callers_id").replace_strict(ids, default=np.NAN))

    # Create figure
    n_types = len(methylation_types)
    fig, axes = plt.subplots(3, 1, figsize=(20, 5 * n_types), sharex=False, layout="constrained")

    # Mean together all the different methylation types
    total_meth_data = methyl_data.select('gene_id', 'sample', 'total_methylation')

    top = pl.col(*methylation_types).filter(pl.col('sample').eq('top'))
    middle = pl.col(*methylation_types).filter(pl.col('sample').eq('middle'))
    bot = pl.col(*methylation_types).filter(pl.col('sample').eq('bottom'))

    top_bottom = methyl_data.group_by('gene_id').agg(top.mean() - bot.mean())
    top_middle = methyl_data.group_by('gene_id').agg(top.mean() - middle.mean())
    top_bottom = top_bottom.unpivot(index="gene_id", on=methylation_types, variable_name="methylation_type", value_name="methylation_level")
    top_middle = top_middle.unpivot(index="gene_id", on=methylation_types, variable_name="methylation_type", value_name="methylation_level")

    # Rename samples
    total_meth_data = total_meth_data.with_columns(pl.col('sample').replace(readable_sample_name))
    top_middle = top_middle.with_columns(pl.col('methylation_type').replace(readable_methylation_name))
    top_bottom = top_bottom.with_columns(pl.col('methylation_type').replace(readable_methylation_name))

    # Populate graphs
    plot_mean_gene_methylation_level(axes[0], total_meth_data)
    plot_gene_methylation_level_diff(axes[1], top_middle, "Top – Middle")
    plot_gene_methylation_level_diff(axes[2], top_bottom, "Top – Bottom")

    # Save the figure
    cleaned_genome_name = genome_name.title().replace("_R-Contigs", " sp.")
    fig.suptitle(f"{cleaned_genome_name} methylome", fontsize=26)
    plt.savefig(f"{fig_savepath}/{genome_name}_{coverage}_profile.pdf", format='pdf', transparent=False)

    # Write CSV of the top methylated genes
    top_methylated_genes = total_meth_data.sort("total_methylation", descending=True)
    top_methylated_genes = add_functional_annotations_polars(top_methylated_genes.lazy(), data_dir).collect()
    top_methylated_genes.write_csv(f"../data/gene_level_data/{genome_name}_top_meth_genes.csv")

    print(f"Done plotting profile for {genome_name}")
    return


if __name__ == "__main__":
    for coverage in ["5"]:
        print(f"Running rao analysis at coverage {coverage}")
        data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"../../methylation_data/methylation_{coverage}")
        for genome in os.listdir(data_dir):
            if genome == ".DS_Store" or ".txt" in genome or "Octadecabacter" in genome or "metagenome" in genome:
                continue

            run_analysis(genome, coverage, data_dir, fig_savepath=f"../plots/plots_{coverage}")
