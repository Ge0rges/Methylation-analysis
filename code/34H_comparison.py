from utilities.plotting import *
from _statistics import *
from utilities.data_loading import *
from utilities.utils import normalize_data_for_methylation_level, add_gene_caller_id, readable_methylation_name, col34h_readable_sample_name, col34h_barcode_sample_map
from itertools import combinations


def run_34h_comparison(genome_name, data_dir, coverage, fig_savepath="plots"):
    """
    Run the DMR analysis for a specific genome_name, DMR type, and function_source.
    """

    # Get genes and annotate the dmrs with the gene ID
    genes = get_genes_polars(data_dir, genome_name)

    # Get methylation level data
    methylation_types = list(readable_methylation_name.keys())
    methyl_data = load_combined_methyl_data_for_genome_polars(genome_name, data_dir).select("name", "sample", *methylation_types)
    methyl_data = methyl_data.with_columns(
        contig=pl.col('name').str.split(by='|').list.get(0),
        strand=pl.col('name').str.split(by='|').list.get(1),
        start=pl.col('name').str.split(by='|').list.get(2).cast(pl.UInt32),
        end=pl.col('name').str.split(by='|').list.get(3).cast(pl.UInt32)
    )

    # Rename samples and make total methylation column
    methyl_data = methyl_data.with_columns(pl.col("sample").replace_strict(col34h_barcode_sample_map, default=pl.first()),
                                           pl.concat_list(methylation_types).list.sum().alias("total_methylation")).collect()

    # Calculate rao score between each group
    rows = []
    samples = methyl_data.get_column("sample").unique().to_list()
    for sampleA, sampleB in combinations(samples, 2):
        _, significant, comp_str = add_rao_score_by_sample(methyl_data, [sampleA, sampleB], baseline=False)
        rows.append([sampleA] + [None]*len(sampleA))
        rows[-1][samples.index(sampleB)] = significant

    comp_df = pl.DataFrame(rows, schema=["comparison"]+samples)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 5), sharex=False, layout="constrained", gridspec_kw={'width_ratios': [5] + [5]})

    # Mean together all the different methylation types
    mean_data = methyl_data.select('gene_id', 'sample', 'total_methylation')
    mean_data = mean_data.with_columns(pl.col('sample').replace(col34h_readable_sample_name))

    plot_mean_gene_methylation_level(axes[0], mean_data)
    sns.heatmap(comp_df, ax=axes[1], cbar=False)

    # Save the figure
    fig.suptitle(f"Comparison of different preervation treatment of 34H", fontsize=26)
    plt.savefig(f"{fig_savepath}/{genome_name}_{coverage}.svg", format='svg', transparent=True)

    return


if __name__ == "__main__":
    genome_name = "34H_methylation_10"
    coverage = "10"
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"../../methylation_data/")
    run_34h_comparison(genome_name, data_dir, coverage, fig_savepath=f"../plots/plots_34H_comparison")
