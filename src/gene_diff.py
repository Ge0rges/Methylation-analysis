from _statistics import *
from utilities.data_loading import *
from utilities.utils import normalize_data_by_pileup, add_gene_caller_id, \
    add_functional_annotations_polars, readable_methylation_name, barcode_sample_map, readable_sample_name, truncate_label
import os
import matplotlib.pyplot as plt
import seaborn as sns
import polars.selectors as cs
sns.set_theme(context="talk", style="white", font_scale=3)


def run_analysis(genome_name, data_dir, fig_savepath="plots"):
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
    methyl_data = add_gene_caller_id(methyl_data, genes)

    # Create the total methylation column and normalize values
    methyl_data = methyl_data.with_columns(pl.concat_list(methylation_types+["Ncanonical"]).list.sum().alias("position_coverage"))
    methyl_data = normalize_data_by_pileup(methyl_data)
    methyl_data = methyl_data.with_columns(pl.concat_list(methylation_types).list.sum().alias("total_methylation")).collect(streaming=True)

    # Add rao score - Doing this first prevents row duplication issues
    methyl_data = add_rao_score_by_gene(methyl_data, ["top", "bottom"], baseline=False)

    # Subtract methylation fractions: Top - Bottom
    for type in methylation_types+["total_methylation"]:
        # Mean together all the different methylation types
        top = pl.col(type).filter(pl.col('sample').eq('top')).mean()
        bot = pl.col(type).filter(pl.col('sample').eq('bottom')).mean()
        methyl_data = methyl_data.join(methyl_data.select(type, "gene_callers_id", "sample").group_by('gene_callers_id').agg(top - bot), on="gene_callers_id").drop(type).rename({type+"_right": type}).unique()

    # Add functional annotation
    methyl_data = add_functional_annotations_polars(methyl_data.lazy(), data_dir).drop(cs.ends_with("_right")).unique().collect()

    # Add gene relative position from start and from end
    gene_positions = methyl_data.select("gene_callers_id", "name", "start", "strand", "end").unique()
    gene_positions = gene_positions.with_columns((pl.col("start") - pl.col("start").min()).over("gene_callers_id").alias("gene_position"))
    gene_positions = gene_positions.with_columns((pl.col("end").max() - pl.col("end")).over("gene_callers_id").alias("backwards_gene_position"))
    gene_positions = methyl_data.join(gene_positions, on="name", how="inner", validate="m:1").drop(cs.ends_with("_right"))

    # Write the dataframe to a CSV
    (methyl_data.select("gene_callers_id", "source", "function", *methylation_types, "total_methylation", "rao_score", "test_result")
               .filter(pl.col("rao_score").is_not_nan())
               .sort("test_result", "total_methylation", descending=True).unique()
               .write_csv(f"../data/gene_level_data/{genome_name}_rao-filtered_gene_level.csv"))

    # Get DFs
    table_df = make_table(methyl_data)

    genes_df = get_top_dmr_genes(methyl_data)
    genes_ids = genes_df.get_column("gene_callers_id").unique().to_list()

    promoter_df = get_top_dmr_genes_promoter(gene_positions)
    promoter_ids = promoter_df.get_column("gene_callers_id").unique().to_list()

    # Rename samples for plotting
    gene_positions = gene_positions.with_columns(pl.col('sample').replace(readable_sample_name))
    hue_order = [readable_sample_name["top"], readable_sample_name["middle"], readable_sample_name["bottom"]]

    # Plot table of top 20% DMRed pathways. Lineplot of top 5 DMRed genes positions.
    num_plots = max(len(genes_ids+promoter_ids)//2, 2)
    fig, axes = plt.subplots(num_plots, 3, figsize=(75, 25*num_plots), layout="constrained")

    # Plot table
    if table_df is not None:
        table_ax = axes[0][0]
        table_ax.xaxis.set_visible(False)
        table_ax.yaxis.set_visible(False)
        table_ax.set_frame_on(False)

        texts = []
        for text, value in table_df.values:
            texts.append([truncate_label(text, 70, 2), value])

        table = table_ax.table(cellText=texts, colLabels=table_df.columns, cellLoc='center', loc='center')
        table.scale(1.2, 1.5)
        table_ax.set_title(f"Top 20 differentially methylated pathways for {genome_name}")

    # Plot total methylation rolling mean - whole gene based
    for j, info in enumerate([genes_ids, promoter_ids]):
        data_df = genes_df if info == genes_ids else promoter_df
        for i, gene_id in enumerate(info):
            ax = axes[i][j+1]

            meth = data_df.filter(pl.col("gene_callers_id").eq(gene_id)).get_column("abs_total_methylation").to_list()[0]
            func = data_df.filter(pl.col("gene_callers_id").eq(gene_id)).get_column("function").to_list()[0]
            source = data_df.filter(pl.col("gene_callers_id").eq(gene_id)).get_column("source").to_list()[0]

            df = gene_positions.filter(pl.col("gene_callers_id").eq(gene_id)).select("sample", "gene_position", "total_methylation")
            df = df.group_by("sample", "gene_position").agg(pl.col("total_methylation").mean()).sort(["sample", "gene_position"]).with_columns(pl.col("total_methylation").rolling_mean(10, min_periods=1).over("sample").alias("total_methylation"))

            try:
                sns.lineplot(x='gene_position', y="total_methylation", hue="sample", data=df.to_pandas(), ax=ax, hue_order=hue_order)
                ax.set_xlabel("Nucleotide position from start")
            except:
                continue

            if info == promoter_ids:
                ax.set_title(f'Rolling average of total methylation  - {meth:0.2f} methylation difference - Promoter')
            else:
                ax.set_title(f'Rolling average of total methylation  - {meth:0.2f} methylation difference - {source}:{func}')

    #  Save plot
    plt.savefig(f"{fig_savepath}/{genome_name}_{coverage}_meth_funcs.pdf", format='pdf', transparent=False)

    print(f"Done writing genes for {genome_name}")
    return


def make_table(methyl_data, top=20):
    # Make table
    # Split and explode functions
    table_df = methyl_data.with_columns(pl.col("function").str.split("!!!")).explode("function")

    # Get the aboslute biggest differences
    table_df = table_df.with_columns(pl.col("total_methylation").abs().alias("abs_total_methylation"))
    table_df = table_df.group_by("source", "function", "test_result").agg(pl.col("abs_total_methylation").mean(),
                                                                          pl.col("total_methylation").mean())

    # Make a figure with a table of the top DMRed pathways
    table_df = table_df.filter(
        pl.col("source").eq("KEGG_BRITE") & pl.col("test_result").eq(True)).top_k(top, by="abs_total_methylation")
    table_df = table_df.sort("abs_total_methylation", descending=False).drop("abs_total_methylation")
    table_df = table_df.select("function", "total_methylation").to_pandas()
    if table_df.shape[0] == 0:
        return None

    return table_df

def get_top_dmr_genes(methyl_data, top=5, coverage=5):
    # Filter so that entire gene must be covered at least 5 times on every nucleotide in each sample
    cov_genes = methyl_data.group_by("gene_callers_id", "sample").agg(pl.col("position_coverage").mean()).filter(pl.col("position_coverage").gt(coverage))
    cov_genes = cov_genes.group_by("gene_callers_id").agg(pl.col("sample").n_unique()).filter(pl.col("sample").eq(3))
    methyl_data = methyl_data.filter(pl.col("gene_callers_id").is_in(cov_genes.get_column("gene_callers_id").to_list()))

    # Get the aboslute biggest differences
    genes = methyl_data.with_columns(pl.col("total_methylation").abs().alias("abs_total_methylation"))
    genes = genes.group_by("gene_callers_id", "source", "function", "test_result").agg(pl.col("abs_total_methylation").mean(),
                                                                          pl.col("total_methylation").mean())

    # Pick genes that have a KOfam, and positive DMR. And 5 coverage across all positions
    genes = genes.filter(pl.col("source").eq("KOfam") & pl.col("test_result").eq(True))

    # Pick the top 5 differently methylated genes
    genes = genes.top_k(top, by="abs_total_methylation")

    return genes


def get_top_dmr_genes_promoter(gene_positions, top=5, coverage=5):
    methyl_data = gene_positions.filter(pl.col("gene_position").le(100))

    # Filter so that entire gene must be covered at least 5 times on every nucleotide in each sample
    cov_genes = methyl_data.group_by("gene_callers_id", "sample").agg(pl.col("position_coverage").mean()).filter(
        pl.col("position_coverage").gt(coverage))
    cov_genes = cov_genes.group_by("gene_callers_id").agg(pl.col("sample").n_unique()).filter(pl.col("sample").eq(3))
    methyl_data = methyl_data.filter(pl.col("gene_callers_id").is_in(cov_genes.get_column("gene_callers_id").to_list()))

    # Get the aboslute biggest differences
    genes = methyl_data.with_columns(pl.col("total_methylation").abs().alias("abs_total_methylation"))
    genes = genes.group_by("gene_callers_id", "source", "function", "test_result").agg(
        pl.col("abs_total_methylation").mean(),
        pl.col("total_methylation").mean())

    # Pick genes that have a KOfam, and positive DMR. And 5 coverage across all positions
    genes = genes.filter(pl.col("source").eq("KOfam") & pl.col("test_result").eq(True))

    # Pick the top 5 differently methylated genes
    genes = genes.top_k(top, by="abs_total_methylation")

    return genes


if __name__ == "__main__":
    for coverage in ["5"]:
        print(f"Running gene_detail analysis at coverage {coverage}")
        data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"../../methylation_data/methylation_{coverage}")

        for genome in os.listdir(data_dir):
            if genome == ".DS_Store" or ".txt" in genome or genome == "Octadecabacter_r-contigs":
                continue

            if "metagenome" in genome:
                continue
                os.environ["POLARS_MAX_THREADS"] = "1"
            else:
                os.environ["POLARS_MAX_THREADS"] = "10"

            run_analysis(genome, data_dir, fig_savepath=f"../plots/plots_{coverage}")
