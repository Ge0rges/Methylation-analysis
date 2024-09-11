import numpy as np

from utilities.plotting import *
from utilities.data_loading import *
from utilities.utils import add_gene_caller_id, readable_methylation_name, readable_sample_name, barcode_sample_map, normalize_data_by_pileup
from scipy.stats import rankdata


def run_analysis(genome_name, coverage, data_dir, fig_savepath="plots"):
    """
    Run the DMR analysis for a specific genome_name, DMR type, and function_source.
    """

    print(f"Starting to generate  composite for {genome_name}")

    # Get the gene_lengths
    gene_lengths = get_genes_polars(data_dir, genome_name)

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
    methyl_data = methyl_data.with_columns(pl.col("sample").replace_strict(barcode_sample_map))
    methyl_data = methyl_data.filter(pl.col("sample").is_in(["top", "middle", "bottom"]))

    # Add the gene_caller_id
    methyl_data = add_gene_caller_id(methyl_data, gene_lengths, True)

    # Create the total methylation column and normalize values
    methyl_data = normalize_data_by_pileup(methyl_data)
    methyl_data = methyl_data.with_columns(pl.concat_list(methylation_types).list.sum().alias("total_methylation")).collect(streaming=True)

    # Add a gene_id column
    all_ids = methyl_data.get_column("gene_callers_id").to_list()
    ids = dict(zip(all_ids, rankdata(all_ids, method='dense')))
    methyl_data = methyl_data.with_columns(gene_id=pl.col("gene_callers_id").replace_strict(ids, default=np.NAN))

    # Add gene relative position from start and from end
    gene_positions = methyl_data.select("gene_id", "name", "start", "strand", "end").unique()
    gene_positions = gene_positions.with_columns((pl.col("start") - pl.col("start").min()).over("gene_id").alias("gene_position"))
    gene_positions = gene_positions.with_columns((pl.col("end").max() - pl.col("end")).over("gene_id").alias("backwards_gene_position"))
    methyl_data = methyl_data.join(gene_positions, on="name", how="inner", validate="m:1")

    # Create figure
    n_types = len(methylation_types)
    fig, axes = plt.subplots(n_types+1, 6, figsize=(100, 100), sharex=False, layout="constrained")

    # Rename samples
    methyl_data = methyl_data.with_columns(pl.col('sample').replace(readable_sample_name))

    # Get gene length stats
    gene_lengths = methyl_data.select("gene_position", "gene_id").group_by("gene_id").max()
    min_gene_length = gene_lengths.get_column("gene_position").quantile(0.1)
    median_gene_length = gene_lengths.get_column("gene_position").median()
    max_percentile = gene_lengths.get_column("gene_position").quantile(0.95)

    # Plot gene length distribution
    sns.histplot(gene_lengths.to_pandas(), x="gene_position", ax=axes[0][0])

    # Plot total methylation over everything
    df = methyl_data.select("sample", "gene_position", "gene_id", "total_methylation").filter(pl.col("gene_position").le(max_percentile))
    sns.lineplot(x='gene_position', y="total_methylation", hue="sample", data=df.to_pandas(), ax=axes[1][0])
    axes[1][0].set_title(f'Total methylation by genic position and sample up to 95% percentile length')
    axes[1][0].axvline(x=62, color="blue", linestyle='--', label='Position 100')
    axes[1][0].axvline(x=min_gene_length, color="red", linestyle='--', label='10% percentile gene length')
    axes[1][0].axvline(x=median_gene_length, color="green", linestyle='--', label='Median Gene Length')

    # Plot total methylation on first 100 positions
    df = methyl_data.select("sample", "gene_position", "gene_id", "total_methylation").filter(pl.col("gene_position").le(100))
    sns.lineplot(x='gene_position', y="total_methylation", hue="sample", data=df.to_pandas(), ax=axes[2][0])
    axes[2][0].set_title(f'Total methylation by genic position and sample up to 100 positions')

    # Plot total methylation on last 100 positions
    df = methyl_data.select("sample", "backwards_gene_position", "gene_id", "total_methylation").filter(pl.col("backwards_gene_position").le(100))
    sns.lineplot(x='backwards_gene_position', y="total_methylation", hue="sample", data=df.to_pandas(), ax=axes[3][0])
    axes[3][0].set_title(f'Total methylation by genic position and on last 100 positions')

    # Populate graphs
    for j, (min_limit, max_limit) in enumerate([(0, 500), (1000, 2000), (2000, 3500), (4000, 5000), (9000, np.inf)]):
        j += 1
        # Filter out gene_lengths whose length isn't in the range
        gene_ids = gene_lengths.filter(pl.col("gene_position").ge(min_limit).le(max_limit)).get_column("gene_id").to_list()

        for i, type in enumerate(methylation_types+["total_methylation"]):
            ax = axes[i][j]
            df = methyl_data.select("sample", "gene_position", "gene_id", type).filter(pl.col("gene_id").is_in(gene_ids))
            sns.lineplot(x='gene_position', y=type, hue="sample", data=df.to_pandas(), ax=ax)

            # Labels
            if type == "total_methylation":
                ax.set_title(f'Total methylation by genic position for gene_lengths with length between {min_limit} and {max_limit}')
            else:
                ax.set_title(f'Fraction methylated with {readable_methylation_name[type]} by genic position for gene_lengths with length between {min_limit} and {max_limit}')
            ax.set_xlabel('Gene Position')
            ax.set_ylabel(f'Methylation fraction %')

            # Add vertical lines
            ax.axvline(x=62, color="blue", linestyle='--', label='Position 100')

    # Save the figure
    cleaned_genome_name = genome_name.title().replace("_R-Contigs", " sp.")
    plt.savefig(f"{fig_savepath}/{genome_name}_{coverage}_gene_detail.pdf", format='pdf', transparent=True)

    print(f"Done plotting detail gene view for {genome_name}")
    return


if __name__ == "__main__":
    for coverage in ["5", "5_agg"]:
        print(f"Running rao analysis at coverage {coverage}")
        data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"../data/methylation_data/methylation_{coverage}")
        for genome in os.listdir(data_dir):
            if genome == ".DS_Store":
                continue

            run_analysis(genome, coverage, data_dir, fig_savepath=f"../plots/plots_{coverage}")
