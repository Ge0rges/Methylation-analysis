from _statistics import *
from utilities.data_loading import *
from utilities.utils import normalize_data_by_pileup, add_gene_caller_id, \
    add_functional_annotations_polars, readable_methylation_name, barcode_sample_map, truncate_label
import os
import matplotlib.pyplot as plt
from pathlib import Path
import math
import seaborn as sns
sns.set_theme(context="paper", style="white")
os.environ["POLARS_TEMP_DIR"] = str(Path("./polars_temp/"))


def run_analysis(genome_names, data_dir, fig_savepath="plots"):
    """
    Run the DMR analysis for a specific genome_name, DMR type, and function_source.
    """

    print(f"Starting to generate  composite for {genome_names}")

    # Get the genes
    genes = get_genes_polars(data_dir)

    all_methyl_data = []
    for genome_name in genome_names:
        # Get methylation level data
        methylation_types = list(readable_methylation_name.keys())
        methyl_data = load_combined_methyl_data_for_genome_polars(genome_name, data_dir).select("name", "sample",
                                                                                                    *methylation_types,
                                                                                                    "Ncanonical")
        methyl_data = methyl_data.with_columns(
            contig=pl.col('name').str.split(by='|').list.get(0),
            strand=pl.col('name').str.split(by='|').list.get(1),
            start=pl.col('name').str.split(by='|').list.get(2).cast(pl.UInt32),
            end=pl.col('name').str.split(by='|').list.get(3).cast(pl.UInt32)
        )

        # Filter samples
        methyl_data = methyl_data.with_columns(pl.col("sample").replace_strict(barcode_sample_map, default=pl.first()))
        methyl_data = methyl_data.filter(pl.col("sample").is_in(["top", "middle", "bottom"]))
        methyl_data = methyl_data.filter(pl.concat_list(methylation_types).list.sum() >= 5)

        # Add the gene_caller_id
        methyl_data = add_gene_caller_id(methyl_data, genes, True)

        # Create the total methylation column and normalize values
        methyl_data = normalize_data_by_pileup(methyl_data)
        methyl_data = methyl_data.with_columns(pl.concat_list(methylation_types).list.sum().alias("total_methylation")).collect(streaming=True)

        # Add rao score - Doing this first prevents row duplication issues
        methyl_data = add_rao_score_by_gene(methyl_data, ["top", "bottom"], baseline=False)

        for type in methylation_types+["total_methylation"]:
            # Mean together all the different methylation types
            top = pl.col(type).filter(pl.col('sample').eq('top')).mean()
            bot = pl.col(type).filter(pl.col('sample').eq('bottom')).mean()
            methyl_data = methyl_data.join(methyl_data.select(type, "gene_callers_id", "sample").group_by('gene_callers_id').agg(top - bot), on="gene_callers_id").drop(type).rename({type+"_right": type}).unique()

        # Add functional annotation
        methyl_data = add_functional_annotations_polars(methyl_data.lazy(), data_dir).collect()

        # Write the dataframe to a CSV
        methyl_data = methyl_data.select("gene_callers_id", "source", "function", *methylation_types, "total_methylation", "rao_score", "test_result").unique()

        # Get the 10% biggest differences
        methyl_data = methyl_data.with_columns(pl.col("total_methylation").abs().alias("abs_total_methylation")).filter( pl.col("abs_total_methylation").gt(pl.col("abs_total_methylation").quantile(0.5))).sort("abs_total_methylation", descending=False).drop("abs_total_methylation")

        # Add to list
        methyl_data = methyl_data.with_columns(pl.lit(genome_name).alias("genome_name"))
        all_methyl_data.append(methyl_data)

    # Concat
    all_methyl_data = pl.concat(all_methyl_data)

    # Split and explode functions
    all_methyl_data = all_methyl_data.with_columns(pl.col("function").str.split("!!!")).explode("function")

    # Get functions that are in every genome
    functions = all_methyl_data.group_by("function").agg(pl.col("genome_name").n_unique().alias("n_genomes")).filter(pl.col("n_genomes").eq(len(genome_names))).get_column("function").unique().to_list()

    # Determine the number of rows and columns for subplots
    num_functions = len(functions)
    cols = 3  # You can adjust this based on how wide you want the plot grid
    rows = math.ceil(num_functions / cols)

    all_methyl_data.write_csv(f"../data/gene_level_data/{genome_name}_top_funcs_rao_filtered_common.csv")
    if functions == 0:
        print(f"No functions in common in {all_methyl_data}")

    # Create a matplotlib figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), layout="constrained")
    axes = axes.flatten()

    # Make a boxenplot for each function
    i = 0
    for i, (function, ax) in enumerate(zip(functions, axes)):
        # Filter the data for the functions of interest
        df = all_methyl_data.filter(pl.col("function").eq(function)).to_pandas()

        # Create the boxenplot
        sns.boxplot(x="genome_name", y="total_methylation", data=df, ax=ax)

        # Set plot title and labels
        ax.set_title(truncate_label(function, 50, 3))

        # Get the counts for each genome_name
        genome_counts = df.groupby("genome_name").size()

        # Annotate the number of genes above each x-tick
        for xtick, genome_name in enumerate(genome_counts.index):
            count = genome_counts[genome_name]
            ax.text(xtick, df['total_methylation'].min() - 0.05 * df['total_methylation'].ptp(),
                    # Adjust position below plot
                    f"n={count}",
                    ha='center', va='top', fontsize=10, color='black')

    # Remove any empty subplots if the number of functions doesn't fill the grid
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Save
    plt.subplots_adjust(vspace=0.5)
    plt.savefig(f"{fig_savepath}/{genome_names}_{coverage}_pathway_boxenplot.pdf", format='pdf', transparent=True)

    return


if __name__ == "__main__":
    for coverage in ["5"]:
        print(f"Running gene_detail analysis at coverage {coverage}")
        data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"../../methylation_data/methylation_{coverage}")
        run_analysis(["Pelagibacter_r-contigs", "polaribacter_r-contigs"], data_dir, fig_savepath=f"../plots/plots_{coverage}")
