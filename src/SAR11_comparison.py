from utilities.plotting import *
from _statistics import *
from utilities.data_loading import *
from utilities.utils import add_gene_caller_id, sar11_barcode_sample_map, normalize_data_by_pileup
from itertools import combinations
from scipy.stats import rankdata
import multiprocess as mp
import pandas as pd
import math

def run_comparison(genome_name, data_dir, coverage, fig_savepath="plots"):
    """
    Run the DMR analysis for a specific genome_name, DMR type, and function_source.
    """

    # Get genes and annotate the dmrs with the gene ID
    genes = get_genes_polars(data_dir, genome_name)

    # Get methylation level data
    methylation_types = list(readable_methylation_name.keys())
    methyl_data = load_combined_methyl_data_for_genome_polars(genome_name, data_dir, coverage=5)

    # Rename samples and make total methylation column
    methyl_data = normalize_data_by_pileup(methyl_data)
    methyl_data = methyl_data.with_columns(pl.col("sample").replace_strict(sar11_barcode_sample_map),
                                           pl.concat_list(methylation_types).list.sum().alias("total_methylation")).collect(streaming=True)

    methyl_data = methyl_data.filter(pl.col("sample").str.contains("core", literal=True).not_())

    # # Calculate rao score between each group in parallel
    # samples = methyl_data.get_column("sample").unique().to_list()
    #
    # def process_sample_pair(sample_tuple):
    #     sampleA, sampleB = sample_tuple
    #     _, significant, comp_str = add_rao_score_by_sample(methyl_data, [sampleA, sampleB], baseline=False)
    #     return sampleA, sampleB, significant
    #
    # comp_df = pd.DataFrame(index=samples, columns=samples)
    #
    # with mp.get_context("spawn").Pool(15) as p:
    #     for result in p.map(process_sample_pair, combinations(samples, 2)):
    #         sampleA, sampleB, significant = result
    #         comp_df.loc[sampleA, sampleB] = significant
    #         comp_df.loc[sampleB, sampleA] = significant
    # print(comp_df)

    # Mean together all the different methylation types
    genes = get_genes_polars(data_dir, genome_name)
    methyl_data = add_gene_caller_id(methyl_data.lazy(), genes).collect(streaming=True)
    all_ids = methyl_data.sort("strand", "contig",  "start").get_column("gene_callers_id").to_list()
    ids = dict(zip(all_ids, rankdata(all_ids, method='dense')))
    #methyl_data = methyl_data.with_columns(gene_id=pl.col("gene_callers_id").replace_strict(ids, default=np.NAN))
    methyl_data = methyl_data.with_columns(pl.col("gene_callers_id").alias("gene_id"))
    mean_data = methyl_data.select('gene_id', 'sample', 'total_methylation').group_by("gene_id", "sample").mean()
    mean_data = mean_data.sort("sample")

    # Create figure
    samples = mean_data.get_column("sample").unique().to_list()
    fig, axes = plt.subplots(math.ceil(math.sqrt(len(samples))), math.ceil(math.sqrt(len(samples))), figsize=(5*len(samples), 5*len(samples)), layout="constrained")
    axes = axes.flatten()

    for i, sample in enumerate(samples):
        sns.lineplot(mean_data.filter(pl.col("sample").eq(sample)), x="gene_id", y="total_methylation", ax=axes[i], label=sample)
        axes[i].set_title(sample)
        axes[i].set_ylim(0, 0.5)
        axes[i].set_title(sample, fontsize=36)

    #sns.heatmap(comp_df, ax=axes, cbar=False)

    # Save the figure
    fig.suptitle(f"Comparison of SAR11", fontsize=38)
    plt.savefig(f"{fig_savepath}/{genome_name}_{coverage}.pdf", format='pdf', transparent=True)

    print("Done.")
    return


if __name__ == "__main__":
    genome_name = "pelagibacter_r-contigs"
    coverage = "5"
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"../../../pelagibacter_methylation/methylation_{coverage}")
    run_comparison(genome_name, data_dir, coverage, fig_savepath=f"../plots/plots_SAR11_comparison")
