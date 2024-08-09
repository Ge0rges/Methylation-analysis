from utilities.plotting import *
from _statistics import *
from utilities.data_loading import *
from utilities.utils import normalize_data_for_methylation_level, add_gene_caller_id, col34h_barcode_sample_map
from itertools import combinations
from scipy.stats import rankdata
import multiprocess as mp


def run_34h_comparison(genome_name, data_dir, coverage, fig_savepath="plots"):
    """
    Run the DMR analysis for a specific genome_name, DMR type, and function_source.
    """

    # Get genes and annotate the dmrs with the gene ID
    genes = get_genes_polars(data_dir, genome_name)

    # Get methylation level data
    methylation_types = list(readable_methylation_name.keys())
    methyl_data = load_combined_methyl_data_for_genome_polars(genome_name, data_dir).select("name", "sample", "Ncanonical", *methylation_types)
    methyl_data = methyl_data.with_columns(
        contig=pl.col('name').str.split(by='|').list.get(0),
        strand=pl.col('name').str.split(by='|').list.get(1),
        start=pl.col('name').str.split(by='|').list.get(2).cast(pl.UInt32),
        end=pl.col('name').str.split(by='|').list.get(3).cast(pl.UInt32)
    )

    # Rename samples and make total methylation column
    methyl_data = methyl_data.with_columns(pl.col("sample").replace_strict(col34h_barcode_sample_map, default=pl.first()),
                                           pl.concat_list(methylation_types).list.sum().alias("total_methylation")).collect()

    # Calculate rao score between each group in parallel
    rows = []
    samples = methyl_data.get_column("sample").unique().to_list()
    
    def process_sample_pair(sample_tuple):
        sampleA, sampleB = sample_tuple
        _, significant, comp_str = add_rao_score_by_sample(methyl_data, [sampleA, sampleB], baseline=False)
        row = [sampleA] + [None]*len(samples)
        row[samples.index(sampleB) + 1] = significant
        return row
            
    with mp.get_context("spawn").Pool(15) as p:
        for result in p.map(process_sample_pair, combinations(samples, 2)):
            rows.append(result)
    comp_df = pl.DataFrame(rows, schema=["comparison"]+samples, orient="row")

    # Create figure
    fig, axes = plt.subplots(1, 1, figsize=(20, 5), sharex=False, layout="constrained")

    # Mean together all the different methylation types
    #genes = get_genes_polars(data_dir, genome_name)
    #methyl_data = add_gene_caller_id(methyl_data.lazy(), genes, True).collect(streaming=True)
    #all_ids = methyl_data.get_column("gene_callers_id").to_list()
    #ids = dict(zip(all_ids, rankdata(all_ids, method='dense')))
    #methyl_data = methyl_data.with_columns(gene_id=pl.col("gene_callers_id").replace_strict(ids, default=np.NAN))
    #mean_data = methyl_data.select('gene_id', 'sample', 'total_methylation')
    
    #plot_mean_gene_methylation_level(axes[0], mean_data)
    sns.heatmap(comp_df, ax=axes, cbar=False)

    # Save the figure
    fig.suptitle(f"Comparison of different preervation treatment of 34H", fontsize=26)
    plt.savefig(f"{fig_savepath}/{genome_name}_{coverage}.svg", format='svg', transparent=True)
    
    print("Done.")
    return


if __name__ == "__main__":
    genome_name = "34h_assembly"
    coverage = "10"
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"../../../colwellia_methylation/methylation_{coverage}")
    run_34h_comparison(genome_name, data_dir, coverage, fig_savepath=f"../plots/plots_34H_comparison")
