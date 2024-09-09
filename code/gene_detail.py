from utilities.plotting import *
from utilities.data_loading import *
from utilities.utils import add_gene_caller_id, readable_methylation_name, readable_sample_name, barcode_sample_map, normalize_data_by_pileup
from scipy.stats import rankdata


def run_analysis(genome_name, coverage, data_dir, fig_savepath="plots"):
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
    methyl_data = methyl_data.with_columns(pl.col("sample").replace_strict(barcode_sample_map, default=pl.first()))
    methyl_data = methyl_data.filter(pl.col("sample").is_in(["top", "middle", "bottom"]))

    # Add the gene_caller_id
    methyl_data = add_gene_caller_id(methyl_data, genes, True)

    # Create the total methylation column and normalize values
    methyl_data = normalize_data_by_pileup(methyl_data)
    methyl_data = methyl_data.with_columns(pl.concat_list(methylation_types).list.sum().alias("total_methylation")).collect(streaming=True)

    # Add a gene_id column, and gene relative position
    all_ids = methyl_data.get_column("gene_callers_id").to_list()
    ids = dict(zip(all_ids, rankdata(all_ids, method='dense')))
    methyl_data = methyl_data.with_columns(gene_id=pl.col("gene_callers_id").replace_strict(ids, default=np.NAN))
    methyl_data = methyl_data.with_columns(pl.int_range(pl.len()).over("gene_callers_id").alias("gene_position"))

    # Create figure
    n_types = len(methylation_types)
    fig, axes = plt.subplots(n_types+2, 1, figsize=(20, 20 * n_types), sharex=False, layout="constrained")
    axes = axes.flatten()

    # Rename samples
    methyl_data = methyl_data.with_columns(pl.col('sample').replace(readable_sample_name))

    # Get gene length stats
    genes = methyl_data.select("gene_position", "gene_id").group_by("gene_id").max()
    min_gene_length = genes.get_column("gene_position").quantile(0.1)
    median_gene_length = genes.get_column("gene_position").median()
    max_limit = genes.get_column("gene_position").quantile(0.95)

    # Plot gene length distribution
    sns.histplot(genes.to_pandas(), x="gene_position", ax=axes[0])

    df = methyl_data.select("sample", "gene_position", "gene_id", "total_methylation").filter(pl.col("gene_position").le(max_limit))
    sns.lineplot(x='gene_position', y="total_methylation", hue="sample", data=df.to_pandas(), ax=axes[1])

    # Populate graphs
    for i, type in enumerate(methylation_types):
        ax = axes[i+2]
        df = methyl_data.select("sample", "gene_position", "gene_id", type).filter(pl.col("gene_position").le(max_limit))
        sns.lineplot(x='gene_position', y=type, hue="sample", data=df.to_pandas(), ax=ax)

        # Labels
        ax.set_title(f'Fraction methylated with {readable_methylation_name[type]} by genic position and sample up to 95% percentile length')
        ax.set_xlabel('Gene Position')
        ax.set_ylabel(f'Methylation fraction %')

        # Add vertical lines
        ax.axvline(x=62, color="blue", linestyle='--', label='Position 100')
        ax.axvline(x=min_gene_length, color="red", linestyle='--', label='10% percentile gene length')
        ax.axvline(x=median_gene_length, color="green", linestyle='--', label='Median Gene Length')

    # Save the figure
    cleaned_genome_name = genome_name.title().replace("_R-Contigs", " sp.")
    plt.savefig(f"{fig_savepath}/{genome_name}_{coverage}_gene_detail.pdf", format='pdf', transparent=True)

    print(f"Done plotting composite for {genome_name}")
    return


if __name__ == "__main__":
    for coverage in ["5", "5_agg"]:
        print(f"Running rao analysis at coverage {coverage}")
        data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"../../methylation_data/methylation_{coverage}")
        for genome in os.listdir(data_dir):
            if genome == ".DS_Store":
                continue

            run_analysis(genome, coverage, data_dir, fig_savepath=f"../plots/plots_{coverage}")
