from _statistics import *
from utilities.data_loading import *
from utilities.utils import normalize_data_by_pileup, add_gene_caller_id, \
    add_functional_annotations_polars, readable_methylation_name, barcode_sample_map, readable_sample_name, truncate_label
import os
import matplotlib.pyplot as plt
import seaborn as sns
import polars.selectors as cs
sns.set_theme(context="talk", style="white", font_scale=3)


def plot_genes(genome_name, data_dir, gene_ids, fig_savepath="plots"):
    """
    Run the DMR analysis for a specific genome_name, DMR type, and function_source.
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
    methyl_data = add_gene_caller_id(methyl_data, genes).filter(pl.col("gene_callers_id").is_in(gene_ids))

    # Create the total methylation column and normalize values
    methyl_data = methyl_data.with_columns(pl.concat_list(methylation_types+["Ncanonical"]).list.sum().alias("position_coverage"))
    methyl_data = normalize_data_by_pileup(methyl_data)
    methyl_data = methyl_data.with_columns(pl.concat_list(methylation_types).list.sum().alias("total_methylation"))

    # Add rao score - Doing this first prevents row duplication issues
    methyl_data = add_rao_score_by_gene(methyl_data.collect(streaming=True), ["top", "bottom"], baseline=False).lazy()

    # Add functional annotation
    methyl_data = add_functional_annotations_polars(methyl_data, data_dir).drop(cs.ends_with("_right")).unique()

    # Subtract methylation fractions: Top - Bottom
    top = pl.col("total_methylation").filter(pl.col('sample').eq('top')).mean()
    bot = pl.col("total_methylation").filter(pl.col('sample').eq('bottom')).mean()
    diff = methyl_data.select("total_methylation", "gene_callers_id", "sample").group_by('gene_callers_id').agg(top - bot)
    diff = diff.rename({"total_methylation": "total_methylation" + "_diff"})
    methyl_data = methyl_data.join(diff, on="gene_callers_id").unique()

    # Get the aboslute biggest differences
    gene_diffs = methyl_data.with_columns(pl.col("total_methylation_diff").abs().alias("abs_total_methylation_diff"))
    gene_diffs = gene_diffs.group_by("gene_callers_id").agg(pl.col("abs_total_methylation_diff").mean(), pl.col("total_methylation_diff").mean())
    methyl_data = methyl_data.join(gene_diffs, on="gene_callers_id").drop(cs.ends_with("_right")).unique()

    # Add gene relative position from start and from end
    gene_positions = methyl_data.select("gene_callers_id", "name", "start", "strand", "end").unique()
    gene_positions = gene_positions.with_columns(
        (pl.col("start") - pl.col("start").min()).over("gene_callers_id").alias("gene_position"))
    gene_positions = gene_positions.with_columns(
        (pl.col("end").max() - pl.col("end")).over("gene_callers_id").alias("backwards_gene_position"))
    gene_positions = methyl_data.join(gene_positions, on="name", how="inner", validate="m:1").drop(
        cs.ends_with("_right"))

    # Rename samples for plotting
    gene_positions = gene_positions.with_columns(pl.col('sample').replace(readable_sample_name)).collect(streaming=True)
    hue_order = [readable_sample_name["top"], readable_sample_name["middle"], readable_sample_name["bottom"]]

    # Create figure
    fig, axes = plt.subplots(len(gene_ids), 1, figsize=(40, 20*len(gene_ids)), layout="constrained")

    for i, gene_id in enumerate(gene_ids):
        ax = axes[i] if len(gene_ids) > 1 else axes

        # Get gene data
        data = gene_positions.filter(pl.col("gene_callers_id").eq(gene_id))
        meth = data.get_column("total_methylation_diff").to_list()[0]
        func = data.get_column("function").to_list()[0]
        source = data.get_column("source").to_list()[0]
        test_result = data.get_column("test_result").to_list()[0]

        # Do rolling average
        data = data.select("sample", "gene_position", "total_methylation")
        data = data.group_by("sample", "gene_position").agg(pl.col("total_methylation").mean())
        data = data.sort(["sample", "gene_position"]).with_columns(pl.col("total_methylation").rolling_mean(20, min_periods=1).over("sample").alias("total_methylation"))

        # Plot
        sns.lineplot(x='gene_position', y="total_methylation", hue="sample", data=data.to_pandas(), ax=ax, hue_order=hue_order)
        ax.set_xlabel("Nucleotide position from start")
        ax.set_title(f'Gene: {gene_id} - Methylation difference {meth:0.2f} - {source}:{func} - Significant: {test_result}')

    #  Save plot
    plt.savefig(f"{fig_savepath}/{genome_name}_{coverage}_gene_detail.pdf", format='pdf', transparent=False)

    print(f"Done writing genes for {genome_name}")
    return


if __name__ == "__main__":
    for coverage in ["5"]:
        print(f"Running gene_detail analysis at coverage {coverage}")
        data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                f"../../methylation_data/methylation_{coverage}")

        for genome in os.listdir(data_dir):
            if genome == ".DS_Store" or ".txt" in genome or genome == "Octadecabacter_r-contigs" or "metagenome" in genome:
                continue

            plot_genes(genome, data_dir, fig_savepath=f"../plots/plots_{coverage}")
