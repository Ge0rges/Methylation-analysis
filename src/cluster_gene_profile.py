from utilities.plotting import *
from utilities.data_loading import *
from utilities.utils import add_gene_caller_id, readable_methylation_name, readable_sample_name, barcode_replicate_map, normalize_data_by_pileup
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import pandas as pd




def run_analysis(genome_name, data_dir, fig_savepath="plots"):
    """
    Run the gene position analysis for a specific genome_name, DMR type, and function_source.
    """

    print(f"Starting to generate gene position plots for {genome_name}")

    # Get the genes
    genes = get_genes_polars(data_dir)

    # Get methylation level data
    methylation_types = list(readable_methylation_name.keys())
    methyl_data = load_combined_methyl_data_for_genome_polars(genome_name, data_dir, coverage=5)

    # Filter samples
    methyl_data = methyl_data.with_columns(pl.col("sample").replace_strict(barcode_replicate_map))
    methyl_data = methyl_data.filter(pl.col("sample").is_in(["top", "middle", "bottom"]))

    # Add the gene_caller_id
    methyl_data = add_gene_caller_id(methyl_data, genes).collect(streaming=True)
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
    df = methyl_data.with_columns(pl.col('sample').replace(readable_sample_name)).select("gene_position", "total_methylation", "gene_callers_id", "sample").group_by("gene_callers_id", "gene_position", "sample").agg(pl.col("total_methylation").mean()).to_pandas()

    centroid_data, df['cluster_label'] = scipy.cluster.vq.kmeans2(
        data=df[['gene_position', 'total_methylation']], k=4, seed=0,
    )
    centroids = pd.DataFrame(
        index=pd.RangeIndex(name='cluster_label', stop=len(centroid_data)),
        columns=('gene_position', 'total_methylation'),
        data=centroid_data,
    )
    print(df)
    print(centroids)

    fig, ax = plt.subplots()
    sns.scatterplot(ax=ax, data=df, x='gene_position', y='total_methylation', hue='sample', style='gene_callers_id')

    cmap = plt.cm.rainbow(np.linspace(0, 1, len(centroids)))
    for (label, cluster), color in zip(df.groupby('cluster_label'), cmap):
        ax.scatter(
            [centroids.loc[label, 'gene_position']],
            [centroids.loc[label, 'total_methylation']], s=60, color=color, marker='+',
        )
        ax.scatter(
            cluster['gene_position'], cluster['total_methylation'], s=120, color=color, marker='o', facecolors='none',
        )

    plt.show()

    print(f"Done plotting detail gene cluster for {genome_name}")
    return


if __name__ == "__main__":
    for coverage in ["5"]:
        print(f"Running genetic position  analysis at coverage {coverage}")
        data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"../data/methylation_data/methylation_{coverage}")

        for genome in os.listdir(data_dir):
            if genome == ".DS_Store" or ".txt" in genome or genome == "Octadecabacter_r-contigs":
                continue

            run_analysis(genome, data_dir, fig_savepath=f"../plots/plots_{coverage}")
