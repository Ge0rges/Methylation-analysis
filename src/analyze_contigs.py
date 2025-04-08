import polars as pl
import matplotlib.pyplot as plt
from src.objects.contig import Contig
from src.objects.gene_collection import GeneCollection
from src.utilities.utils import treatment_weighted_mean
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage
import numpy as np
import scipy.stats as stats


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
            motif_df = motif.data().collect(streaming=True)
            
            # Add data with significance information
            for treatment in motif_df.get_column("treatment").unique():
                methylation_fraction = motif_df.filter(pl.col("treatment") == treatment).select(motif.meth_type).mean().item()
                
                data.append({
                    "contig_name": contig.contig_name,
                    "treatment": treatment,
                    "motif_string": motif.readable_motif,
                    "methylation_fraction": methylation_fraction,
                    "contig_taxonomy": contig.taxonomy("c" if contig.is_viral else "f"),
                })
        
    df = pl.DataFrame(data)

    if df.is_empty():
        print(f"No data to plot in {contigs[0].parent_genome.readable_name}")
        return df
    
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
        
    # Sort columns by motif_string first, then treatment
    pivot_df = pivot_df.sort_index(axis=1)

    # Sort rows by taxonomy classification
    contig_taxonomy_map = {row['contig_name']: row['contig_taxonomy'] 
                          for row in df.select('contig_name', 'contig_taxonomy').unique().to_dicts()}
    sorted_indices = sorted(pivot_df.index, key=lambda x: contig_taxonomy_map.get(x, ''))
    pivot_df = pivot_df.loc[sorted_indices]

    # Reindex row colors to match sorted rows
    contig_colors = contig_colors.reindex(pivot_df.index)
    
    # Reindex column colors to match sorted columns
    treatment_colors = treatment_colors.reindex(pivot_df.columns)
    
    # Create the heatmap without clustering to preserve sort order
    g = sns.clustermap(
        pivot_df,
        figsize=(len(df.get_column('motif_string').unique())*2, len(df.get_column("contig_name").unique())/1.5),
        row_colors=contig_colors,
        col_colors=treatment_colors,
        mask=pivot_df.isna(),
        cmap="coolwarm",
        linewidths=0.1,
        cbar_kws={"label": "Methylation fraction"},
        row_cluster=False,  # Disable row clustering
        col_cluster=False   # Disable column clustering
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
        g.savefig(f"{contigs[0].parent_genome.output_dir}/virus_motif_heatmap.pdf")
        
        # Save a CSV of the data
        pivot_df.to_csv(f"{contigs[0].parent_genome.output_dir}/virus_motif_heatmap.csv")

    else:
        g.ax_heatmap.set_ylabel("Contig")

        plt.suptitle("Contig motif heatmap")
        g.savefig(f"{contigs[0].parent_genome.output_dir}/contig_motif_heatmap.svg", transparent=True)
        g.savefig(f"{contigs[0].parent_genome.output_dir}/contig_motif_heatmap.pdf")
        
        # Save a CSV of the data
        pivot_df.to_csv(f"{contigs[0].parent_genome.output_dir}/contig_motif_heatmap.csv")

    return df


def extract_diff_methylated_genes_contigs(df, contigs: list[Contig]):
    """
    Get the DMRs that are significantly different between treatments.
    """
    # Get all the dmrs together
    all_data = []
    for contig in contigs:
        for motif in contig.motifs:
            # Let's do 1) add gene_caller_id if needed
            top_rows = motif.dmr_data.collect(streaming=True).sort("score", descending=True)
            if top_rows.is_empty():
                continue
            
            data = contig.parent_genome.add_gene_caller_id(top_rows.lazy(), include_intergenic=True).collect(streaming=True)
            
            # Add function
            gc = GeneCollection(data.get_column("gene_callers_id").unique().to_list(), contig.parent_genome)
            data = data.join(gc.get_function().collect(streaming=True), on="gene_callers_id", how="left")

            # Add nearest gene if not in gene
            data = contig.parent_genome.nearest_gene_to_positions(data)

            # Add function of nearest genes
            gc = GeneCollection(data.get_column("gene_callers_id_start").unique().to_list(), contig.parent_genome)
            data = data.join(gc.get_function().collect(streaming=True), left_on="gene_callers_id_start", right_on="gene_callers_id", how="left", suffix="_start")
            gc = GeneCollection(data.get_column("gene_callers_id_end").unique().to_list(), contig.parent_genome)
            data = data.join(gc.get_function().collect(streaming=True), left_on="gene_callers_id_end", right_on="gene_callers_id", how="left", suffix="_end")
            
            # Logic to filter down the table by creating a ranking system
            # Create ranking columns based on priority criteria
            data = data.with_columns([
                # First priority: gene_callers_id is not -1 (has gene annotation)
                (pl.col("gene_callers_id") != -1).cast(pl.Int32).alias("has_gene"),
                
                # Create a sort priority column with clear priority rules
                pl.struct([
                    # First priority: gene_callers_id is not -1 (has direct gene annotation)
                    (pl.col("gene_callers_id") != -1),
                    
                    # Second priority: KOfam > COG20_FUNCTION > others for direct annotation
                    pl.when(pl.col("source") == "KOfam").then(3)
                    .when(pl.col("source") == "COG20_FUNCTION").then(2)
                    .otherwise(1).alias("source_priority"),
                    
                    # Third priority: consistent annotations across direct and nearest genes
                    (pl.col("source") == pl.col("source_start")) & (pl.col("source") == pl.col("source_end")),
                    
                    # Fourth priority: quality of nearest gene annotations
                    pl.when(pl.col("source_start") == "KOfam").then(3)
                    .when(pl.col("source_start") == "COG20_FUNCTION").then(2)
                    .otherwise(1).alias("nearest_priority"),
                    
                    pl.when(pl.col("source_end") == "KOfam").then(3)
                    .when(pl.col("source_end") == "COG20_FUNCTION").then(2)
                    .otherwise(1).alias("nearest_priority_end")
                ]).alias("sort_priority")
            ])

            # Sort the data by priority within each position group
            data = data.sort("sort_priority", descending=True)
            
            # Group by position identifiers and take only the first row (highest priority) from each group
            data = data.group_by(["contig", "position", "strand"]).agg(
                pl.all().exclude(["contig", "position", "strand", "sort_priority", "has_gene"]).first()
            )
                
            all_data.append(data)
            
    # Write to CSV
    output_file = contig.parent_genome.output_dir / f"{contig.parent_genome.readable_name}_all_top_diff_genes.csv"
    if len(all_data) == 0:
        print(f"No data to write in {contig.parent_genome.readable_name}")
        return
    
    pl.concat(all_data).write_csv(output_file)    


def write_basic_stats_about_contigs(contigs: list[Contig], df):    
    contigs_with_motifs = [contig for contig in contigs if contig.motifs]
    motif_counts = [len(contig.motifs) for contig in contigs_with_motifs]    
    dmr_counts = [sum([motif.dmr_data.unique(subset=["contig", "position", "strand"]).collect(streaming=True).height for motif in contig.motifs]) for contig in contigs_with_motifs]    
    
    # Write those print statements to file
    with open(contigs[0].parent_genome.output_dir / "contig_stats.txt", "w") as f:
        f.write(f"Number of contigs: {len(contigs)}\n")
        f.write(f"Number of contigs with at least one motif: {len(contigs_with_motifs)}\n")
        f.write(f"Average number of motifs per contig: {sum(motif_counts) / len(motif_counts)}\n")
        f.write(f"Average number of DMRs per contig: {sum(dmr_counts) / len(dmr_counts)}\n")
    