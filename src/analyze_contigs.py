import polars as pl
import matplotlib.pyplot as plt
from src.objects.contig import Contig
from src.utilities.utils import treatment_weighted_mean
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage
import numpy as np


def plot_contig_motif_heatmap(contigs: list[Contig]):
    """
    Make a heatmap where each row is a contig. Each column is a motif string. 
    Show the average methylation fraction of that motif, in that contig, across all treatments.
    This is done by having one motif occur as many columns as there are treatments. 
    Draw a line across the X axis which highlights the sample.
    Color the y-ticks (contig names) by the taxonomy("g") of the contig
    """

    # Create a dataframe with columns: contig_name treatment motif_string methylation_fraction contig_taxonomy
    data = []
    for contig in contigs:
        for motif in contig.motifs:
            motif_df = motif.data(normalize=False).collect(streaming=True)
            motif_df = motif_df.with_columns(pl.col("sample").replace_strict(contig.parent_genome.barcode_treatment_map).replace_strict(contig.parent_genome.treatment_name_map).alias("treatment"))
            motif_df = treatment_weighted_mean(motif_df)
            
            for treatment in motif_df.get_column("treatment").unique():
                methylation_fraction = motif_df.filter(pl.col("treatment") == treatment).select(motif.meth_type).mean().item()
                
                data.append({
                    "contig_name": contig.contig_name,
                    "treatment": treatment,
                    "motif_string": motif.readable_motif,
                    "methylation_fraction": methylation_fraction,
                    "contig_taxonomy": contig.taxonomy("g"),
                })
        
    df = pl.DataFrame(data)
    
    # Pivot the data with polars
    pivot_df = df.to_pandas().pivot(
        index="contig_name",
        columns=["motif_string", "treatment"],
        values="methylation_fraction"
    )
    
    # Create colors
    unique_taxonomies = df.get_column("contig_taxonomy").unique().to_list()
    pal = sns.hls_palette(len(unique_taxonomies), h=.5)
    lut = dict(zip(unique_taxonomies, pal))
    
    treatment_colors = df.with_columns(pl.col("treatment").replace_strict(contigs[0].parent_genome.treatment_color_map).alias("Treatment")).select("motif_string", "treatment", "Treatment")
    contig_colors = df.with_columns(pl.col("contig_taxonomy").replace_strict(lut).alias("Taxonomy")).select("contig_name", "Taxonomy")
    
    treatment_colors = treatment_colors.to_pandas().drop_duplicates().set_index(pivot_df.columns, drop=True).drop(columns=["motif_string", "treatment"])
    contig_colors = contig_colors.to_pandas().drop_duplicates("contig_name").set_index("contig_name")
    
    # Create a copy of pivot_df for clustering
    pivot_values = pivot_df.values
    
    # Handle NaN values by replacing them with 0 for clustering purposes
    pivot_values_filled = np.nan_to_num(pivot_values, nan=-1.0)
    
    # Compute linkage matrices for rows and columns
    row_linkage = linkage(pivot_values_filled, method='ward', metric='euclidean')
    col_linkage = linkage(pivot_values_filled.T, method='ward', metric='euclidean')
        
    # Create the clustered heatmap
    g = sns.clustermap(
        pivot_df,
        figsize=(len(df.get_column('motif_string').unique())*2, len(df.get_column("contig_name").unique())/1.5),
        row_colors=contig_colors,
        col_colors=treatment_colors,
        row_linkage=row_linkage,
        col_linkage=col_linkage,
        mask=pivot_df.isna(),
        cmap="coolwarm",
        linewidths=0.1,
        cbar_kws={"label": "Methylation fraction"},
    )
    
    # Modify x-axis labels to show only motifs, not treatments
    motifs = [label.get_text().split("-")[0] for label in g.ax_heatmap.get_xticklabels()]
    g.ax_heatmap.set_xticklabels(motifs)
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)

    # Create legends
    taxonomy_handles = [Patch(color=lut[taxa], label=taxa) for taxa in lut]    
    treatment_handles = [Patch(color=contigs[0].parent_genome.treatment_color_map[treatment], label=treatment) 
                        for treatment in df['treatment'].unique()]
    
    # Create legends with non-overlapping positions
    g.figure.legend(handles=taxonomy_handles, title="Taxonomy", loc='upper left')
    g.figure.legend(handles=treatment_handles, title="Treatment", loc='upper right')
    # Move colorbar to center left
    cbar_pos = g.ax_cbar.get_position()
    g.ax_cbar.set_position([cbar_pos.x0, 0.5 - cbar_pos.height/2, cbar_pos.width, cbar_pos.height])
    
    # Set X axis title
    g.ax_heatmap.set_xlabel("Motif")
    
    if contigs[0].is_viral:
        g.ax_heatmap.set_ylabel("Viral contig")

        plt.suptitle("Viral motif heatmap")
        g.savefig(f"{contigs[0].parent_genome.output_dir}/virus_motif_heatmap.svg", transparent=True)
    else:
        g.ax_heatmap.set_ylabel("Contig")

        plt.suptitle("Contig motif heatmap")
        g.savefig(f"{contigs[0].parent_genome.output_dir}/contig_motif_heatmap.svg", transparent=True)