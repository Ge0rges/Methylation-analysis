from utilities.data_loading import *
from utilities.utils import add_gene_caller_id, readable_methylation_name, readable_sample_name, barcode_sample_map, normalize_data_by_pileup
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context="talk", style="white")


def run_analysis(genome_name, coverage, data_dir, fig_savepath="plots"):
    """
    Run the gene position analysis for a specific genome_name, DMR type, and function_source.
    """

    print(f"Starting to generate gene position plots for {genome_name}")

    # Get the gene_lengths
    gene_lengths = get_genes_polars(data_dir)

    # Get methylation level data
    methylation_types = list(readable_methylation_name.keys())
    methyl_data = load_combined_methyl_data_for_genome_polars(genome_name, data_dir, coverage=5)

    # Filter samples
    methyl_data = methyl_data.with_columns(pl.col("sample").replace_strict(barcode_sample_map))
    methyl_data = methyl_data.filter(pl.col("sample").is_in(["top", "middle", "bottom"]))

    # Add the gene_caller_id
    methyl_data = add_gene_caller_id(methyl_data, gene_lengths, True).collect(streaming=True)
    if methyl_data.is_empty():
        print(f"{genome_name} had no viable data.")
        return

    # Create the total methylation column and normalize values
    methyl_data = normalize_data_by_pileup(methyl_data)
    methyl_data = methyl_data.with_columns(pl.concat_list(methylation_types).list.sum().alias("total_methylation"))

    # Add gene relative position from start and from end
    gene_positions = methyl_data.select("gene_callers_id", "name", "start", "strand", "end").unique()
    gene_positions = gene_positions.with_columns((pl.col("start") - pl.col("start").min()).over("gene_callers_id").alias("gene_position"))
    gene_positions = gene_positions.with_columns((pl.col("end").max() - pl.col("end")).over("gene_callers_id").alias("backwards_gene_position"))
    methyl_data = methyl_data.join(gene_positions, on="name", how="inner", validate="m:1")

    # Rename samples
    methyl_data = methyl_data.with_columns(pl.col('sample').replace(readable_sample_name))
    hue_order = [readable_sample_name["top"], readable_sample_name["middle"], readable_sample_name["bottom"]]

    # Get gene length stats
    gene_lengths = methyl_data.select("gene_position", "gene_callers_id").group_by("gene_callers_id").max().rename({"gene_position": "gene_length"})

    # Create figure
    n_types = len(methylation_types)
    fig, axes = plt.subplots(2, 2, figsize=(100, 100), sharex=False, layout="constrained")
    axes = axes.flatten()

    # Plot gene length distribution
    sns.histplot(gene_lengths.to_pandas(), x="gene_length", ax=axes[0])

    # Plot total methylation over everything
    df = methyl_data.select("sample", "gene_position", "total_methylation")
    # df = df.select(pl.col("total_methylation").mean().over(pl.int_range(pl.len()) // 1000), "sample", "gene_position").with_columns(pl.col("gene_position") // 1000)
    sns.lineplot(x='gene_position', y="total_methylation", hue="sample", data=df.to_pandas(), ax=axes[1], hue_order=hue_order)
    axes[1].set_title(f'Total methylation by genic position and sample in 1000 length bins')

    # Plot total methylation on first 100 positions - rolling mean
    df = methyl_data.select("sample", "gene_position", "total_methylation").filter(pl.col("gene_position").le(100))
    df = df.group_by("sample", "gene_position").agg(pl.col("total_methylation").mean()).sort(["sample", "gene_position"]).with_columns(pl.col("total_methylation").rolling_mean(10, min_periods=1).over("sample").alias("total_methylation"))
    sns.lineplot(x='gene_position', y="total_methylation",  hue="sample", data=df.to_pandas(), ax=axes[2], hue_order=hue_order)
    axes[2].set_title(f'Rolling average of total methylation by genic position on first 100 positions of every gene')

    # Plot total methylation on last 100 positions - rolling average
    df = methyl_data.select("sample", "backwards_gene_position", "total_methylation").filter(pl.col("backwards_gene_position").le(100))
    df = df.group_by("sample", "backwards_gene_position").agg(pl.col("total_methylation").mean()).sort(["sample", "backwards_gene_position"]).with_columns(pl.col("total_methylation").rolling_mean(10, min_periods=1).over("sample").alias("total_methylation"))
    sns.lineplot(x='backwards_gene_position', y="total_methylation",  hue="sample", data=df.to_pandas(), ax=axes[3], hue_order=hue_order)
    axes[3].set_title(f'Rolling average of total methylation by genic position on last 100 positions of every gene')

    # Save the figure
    plt.savefig(f"{fig_savepath}/{genome_name}_{coverage}_gene_detail.pdf", format='pdf', transparent=False)

    # Populate graphs
    n_types = len(methylation_types)
    bins = [(0, 500), (0, 1000), (1000, 2000), (2000, 3000)]
    fig, axes = plt.subplots(n_types+1, len(bins), figsize=(100, 100), sharex=False, layout="constrained")
    axes = axes.flatten()

    for j, (min_limit, max_limit) in enumerate(bins):
        # Filter out gene_lengths whose length isn't in the range
        gene_ids = gene_lengths.filter(pl.col("gene_length").ge(min_limit) & pl.col("gene_length").le(max_limit)).get_column("gene_callers_id").to_list()
        gene_df = methyl_data.filter(pl.col("gene_callers_id").is_in(gene_ids))

        for i, meth_type, in enumerate(methylation_types + ["total_methylation"]):
            df = gene_df.select("sample", "gene_position", meth_type)

            # Get the rolling mean
            df = df.group_by("sample", "gene_position").agg(pl.col(meth_type).mean()).sort(
                ["sample", "gene_position"]).with_columns(
                pl.col(meth_type).rolling_mean(50, min_periods=1).over("sample").alias(meth_type))
            sns.lineplot(df.to_pandas(), x="gene_position", y=meth_type, hue="sample", ax=axes[i*j], hue_order=hue_order)
            axes[i*j].set_title(f"Rolling average of {meth_type} for genes in length range {min_limit}, {max_limit}")

    # Save the figure
    plt.savefig(f"{fig_savepath}/{genome_name}_{coverage}_gene_profiles.pdf", format='pdf', transparent=False)

    print(f"Done plotting detail gene view for {genome_name}")
    return


if __name__ == "__main__":
    for coverage in ["5"]:
        print(f"Running genetic position  analysis at coverage {coverage}")
        data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"../../methylation_data/methylation_{coverage}")

        for genome in os.listdir(data_dir):
            if genome == ".DS_Store" or ".txt" in genome or genome == "Octadecabacter_r-contigs" or "metagenome" in genome:
                continue

            run_analysis(genome, coverage, data_dir, fig_savepath=f"../plots/plots_{coverage}")
