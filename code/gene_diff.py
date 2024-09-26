from _statistics import *
from utilities.data_loading import *
from utilities.utils import normalize_data_by_pileup, add_gene_caller_id, \
    add_functional_annotations_polars, readable_methylation_name, barcode_sample_map
import os
import matplotlib.pyplot as plt
from pathlib import Path
os.environ["POLARS_TEMP_DIR"] = str(Path("./polars_temp/"))


def run_analysis(genome_name, data_dir, slice=None, fig_savepath="plots"):
    """
    Run the DMR analysis for a specific genome_name, DMR type, and function_source.
    """

    print(f"Starting to generate  composite for {genome_name}")

    # Get the genes
    genes = get_genes_polars(data_dir)

    # Get methylation level data
    methylation_types = list(readable_methylation_name.keys())

    methyl_data = slice
    if slice is None:
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

    # Now take only significant RAO's and order by the total methylation
    methyl_data = methyl_data.filter(pl.col("rao_score").is_not_nan()).sort("test_result", "total_methylation", descending=True)

    if slice is None:
        methyl_data.write_csv(f"../data/gene_level_data/{genome_name}_rao-filtered_gene_level.csv")
    else:
        return methyl_data

    # Get the 10% biggest differences
    methyl_data = methyl_data.with_columns(pl.col("total_methylation").abs().alias("abs_total_methylation")).filter(pl.col("test_result").eq("TRUE") & pl.col("abs_total_methylation").gt(pl.col("abs_total_methylation").quantile(0.9))).sort("abs_total_methylation", descending=False).drop("abs_total_methylation")

    # Make a figure with a table of these
    table_df = methyl_data.select("function", "total_methylation").to_pandas()
    fig, ax = plt.subplots(figsize=(10, 10), layout="constrained")

    # Hide axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    # Create the table
    table = ax.table(cellText=table_df.values, colLabels=table_df.columns, cellLoc='center', loc='center')

    # Adjust the layout for better display
    table.auto_set_font_size(True)
    table.scale(1.5, 1.5)

    # Show the plot with the table
    plt.savefig(f"{fig_savepath}/{genome_name}_{coverage}_meth_funcs.pdf", format='pdf', transparent=True)

    print(f"Done writing genes for {genome_name}")
    return


if __name__ == "__main__":
    for coverage in ["5"]:
        print(f"Running gene_detail analysis at coverage {coverage}")
        data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"../../methylation_data/methylation_{coverage}")
        for genome in os.listdir(data_dir):
            if genome == ".DS_Store" or ".txt" in genome or genome == "Octadecabacter_r-contigs":
                continue

            if genome == "metagenome_assembly":
                methylation_types = list(readable_methylation_name.keys())
                df = load_combined_methyl_data_for_genome_polars(genome, data_dir).select("name", "sample",
                                                                                                        *methylation_types,
                                                                                                        "Ncanonical")
                print("loaded metagenome")

                result_df = pl.DataFrame()
                sliced_chunks = 0
                chunk_size = 500000
                last_height = 0
                while result_df.height - last_height != 0 and result_df.height > 0:
                    last_height = result_df.height
                    print(f"Doing {sliced_chunks} which is {sliced_chunks * chunk_size}")
                    temp_df = df.slice(sliced_chunks * chunk_size, chunk_size)
                    sliced_chunks += 1

                    result_df = result_df.vstack(run_analysis(genome, data_dir, slice=temp_df, fig_savepath=f"../plots/plots_{coverage}))

                result_df.write_csv(f"../data/gene_level_data/{genome}_rao-filtered_gene_level.csv")

            else:
                continue
                run_analysis(genome, data_dir, fig_savepath=f"../plots/plots_{coverage})
