from _statistics import *
from utilities.data_loading import *
from utilities.utils import normalize_data_by_pileup, add_gene_caller_id, \
    add_functional_annotations_polars, readable_methylation_name, barcode_sample_map, readable_sample_name
import os
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
sns.set_theme(context="paper", style="white")
os.environ["POLARS_TEMP_DIR"] = str(Path("./polars_temp/"))
pl.Config.set_streaming_chunk_size(1000)

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
        methyl_data = load_combined_methyl_data_for_genome_polars(genome_name, data_dir, coverage=5)


    # Filter samples
    methyl_data = methyl_data.with_columns(pl.col("sample").replace_strict(barcode_sample_map, default=pl.first()))
    methyl_data = methyl_data.filter(pl.col("sample").is_in(["top", "middle", "bottom"]))

    # Add the gene_caller_id
    methyl_data = add_gene_caller_id(methyl_data, genes, True)

    # Create the total methylation column and normalize values
    methyl_data = normalize_data_by_pileup(methyl_data)
    methyl_data = methyl_data.with_columns(pl.concat_list(methylation_types).list.sum().alias("total_methylation")).collect(streaming=True)

    # Add gene relative position from start and from end
    gene_positions = methyl_data.select("gene_callers_id", "name", "start", "strand", "end").unique()
    gene_positions = gene_positions.with_columns((pl.col("start") - pl.col("start").min()).over("gene_callers_id").alias("gene_position"))
    gene_positions = gene_positions.with_columns((pl.col("end").max() - pl.col("end")).over("gene_callers_id").alias("backwards_gene_position"))
    gene_positions = methyl_data.join(gene_positions, on="name", how="inner", validate="m:1")

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

    # Now take only non-null RAO's and order by the total methylation, write to CSV
    methyl_data = methyl_data.filter(pl.col("rao_score").is_not_nan()).sort("test_result", "total_methylation", descending=True)

    if slice is None:
        methyl_data.write_csv(f"../data/gene_level_data/{genome_name}_rao-filtered_gene_level.csv")
    else:
        return methyl_data

    # Split and explode functions
    methyl_data = methyl_data.with_columns(pl.col("function").str.split("!!!")).explode("function")

    # Get the aboslute biggest differences
    methyl_data = methyl_data.with_columns(pl.col("total_methylation").abs().alias("abs_total_methylation"))
    methyl_data = methyl_data.group_by("function", "test_result").agg(pl.col("abs_total_methylation").mean(), pl.col("total_methylation").mean())

    # Make a figure with a table of the top 20% DMRed pathways
    table_df = methyl_data.filter(pl.col("function").eq("KEGG_BRITE") & pl.col("test_result").eq(True) & pl.col("abs_total_methylation").gt(pl.col("abs_total_methylation").quantile(0.8)))
    table_df = table_df.sort("abs_total_methylation", descending=False).drop("abs_total_methylation")
    table_df = table_df.select("function", "total_methylation").to_pandas()
    if table_df.shape[0] == 0:
        print(f"No data for table {genome_name}")
        return

    # Rename samples
    methyl_data = methyl_data.with_columns(pl.col('sample').replace(readable_sample_name))
    hue_order = [readable_sample_name["top"], readable_sample_name["middle"], readable_sample_name["bottom"]]

    # Pick the top 5 differently methylated genes
    gene_abs_meth = methyl_data.filter(pl.col("function").eq("KOfam") & pl.col("test_result").eq(True))
    gene_abs_meth = gene_abs_meth.top_k(5, by="abs_total_methylation")
    gene_ids = gene_abs_meth.get_column("gene_caller_id").to_list()
    gene_abs_meth = gene_abs_meth.get_column("total_methylation").to_list()

    # Plot table of top 20% DMRed pathways. Lineplot of top 5 DMRed genes positions.
    fig, axes = plt.subplots(1, len(gene_ids)+1, figsize=(10, 10), layout="constrained")
    axes = axes.flatten()
    table_ax = axes[0]

    # Hide axes
    table_ax.xaxis.set_visible(False)
    table_ax.yaxis.set_visible(False)
    table_ax.set_frame_on(False)

    # Create the table
    table = table_ax.table(cellText=table_df.values, colLabels=table_df.columns, cellLoc='center', loc='center')
    table.scale(1.5, 1.5)
    table_ax.set_title(f"Top 20% differentially methylated pathways for {genome_name}")

    # Plot total methylation rolling mean
    for i, gene_id in enumerate(gene_ids):
        ax = axes[i+1]
        df = gene_positions.filter(pl.col("gene_callers_id").eq(gene_id)).select("sample", "gene_position", "total_methylation")
        df = df.group_by("sample", "gene_position").agg(pl.col("total_methylation").mean()).sort(["sample", "gene_position"]).with_columns(pl.col("total_methylation").rolling_mean(10, min_periods=1).over("sample").alias("total_methylation"))
        sns.lineplot(x='gene_position', y="total_methylation", hue="sample", data=df.to_pandas(), ax=ax, hue_order=hue_order)
        ax.set_title(f'Rolling average of total methylation of gene {gene_id} - #{i} DMRed genes with {gene_abs_meth[i]} methylation difference')

    #  Save plot
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
                print("Trying to load metagenome...")
                methylation_types = list(readable_methylation_name.keys())
                methyl_data = load_combined_methyl_data_for_genome_polars(genome, data_dir, coverage=5)

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

                    result_df = result_df.vstack(run_analysis(genome, data_dir, slice=temp_df, fig_savepath=f"../plots/plots_{coverage}"))

                result_df.write_csv(f"../data/gene_level_data/{genome}_rao-filtered_gene_level.csv")

            else:
                continue
                run_analysis(genome, data_dir, fig_savepath=f"../plots/plots_{coverage}")
