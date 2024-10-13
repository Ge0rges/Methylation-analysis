from utilities.plotting import *
from utilities.data_loading import *
from utilities.utils import add_gene_caller_id, readable_methylation_name, readable_sample_name, barcode_sample_map, normalize_data_by_pileup
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as hac
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



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
    methyl_data = methyl_data.with_columns(pl.col("sample").replace_strict(barcode_sample_map))
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

    # Step 1: Reshape the Data
    # Pivot the data to get one row per (gene_id, sample) combination with position as columns
    pivot_df = df.pivot_table(index=['sample', 'gene_callers_id'], columns='gene_position', values='total_methylation').reset_index().fillna(0)

    # Perform K-means clustering
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    profiles = pivot_df.drop(columns=['sample', 'gene_callers_id'])
    kmeans.fit(profiles)

    pivot_df['cluster'] = kmeans.labels_

    # Reduce dimensions using PCA
    pca = PCA(n_components=10)
    profiles_pca = pca.fit_transform(profiles)

    plot_df = pd.DataFrame(profiles_pca, columns=['PC1', 'PC2'])
    plot_df['cluster'] = kmeans.labels_
    plot_df['sample'] = pivot_df['sample']
    plot_df['gene_callers_id'] = pivot_df['gene_callers_id']

    # Build the plot
    plt.figure(figsize=(8, 6), layout="constrained")
    sns.scatterplot(data=plot_df, x='PC1', y='PC2', style="sample", hue="cluster", alpha=0.7)

    plt.title('KMeans Clustering of Gene Profiles with Sample Distribution (PCA Reduced)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.savefig(f"{fig_savepath}/{genome_name}_{coverage}_gene_cluster.pdf", format='pdf', transparent=True)

    sample_cluster_distribution = pivot_df.groupby(['cluster', 'sample']).size().unstack().fillna(0)
    print(sample_cluster_distribution)

    print(f"Done plotting detail gene cluster for {genome_name}")
    return


def do_elbow(X):
    # Step 2: Calculate WCSS for a range of k values
    wcss = []
    k_range = range(1, 100)  # You can adjust the range based on your data

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Step 3: Plot the WCSS values to visualize the Elbow
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.xticks(k_range)
    plt.show()


if __name__ == "__main__":
    for coverage in ["5"]:
        print(f"Running genetic position  analysis at coverage {coverage}")
        data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"../../methylation_data/methylation_{coverage}")

        for genome in os.listdir(data_dir):
            if genome == ".DS_Store" or ".txt" in genome or genome == "Octadecabacter_r-contigs":
                continue

            run_analysis(genome, data_dir, fig_savepath=f"../plots/plots_{coverage}")
