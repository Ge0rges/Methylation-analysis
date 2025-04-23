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

sns.set_theme(context="poster", style="whitegrid")

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
            
            # For contigs, we don't care about comparing exact positions, but general means across contig so include points not in ever sample and then filter by breadth.
            motif_df = motif.data()
            if motif_df is None:
                continue
            
            motif_df = motif_df.filter(pl.col(motif.meth_type).is_not_null(), pl.col(motif.meth_type).is_not_nan()).collect()

            # Add data with significance information
            i = 0
            treatments = motif_df.get_column("treatment").unique()
            for treatment in treatments:
                # For us to take a mean, there must be at least 40% of positions
                methylation_fractions = motif_df.filter(pl.col("treatment") == treatment).select(motif.meth_type)
                if len(methylation_fractions)/motif.positions.collect().height < 0.2:
                    continue
                
                i += 1
                methylation_fraction_mean = methylation_fractions.mean().item()
                data.append({
                    "contig_name": contig.contig_name,
                    "treatment": treatment,
                    "motif_string": motif.readable_motif,
                    "methylation_fraction": methylation_fraction_mean,
                    "contig_taxonomy": contig.taxonomy("c" if contig.is_viral else "f"),
                })
            
            # Calculate the statistical difference
            if i == 2:
                motif_df = motif.genome.nearest_gene_to_positions(motif_df)
                group1_label, group2_label = treatments

                # Distribution of the means
                group1_data = motif_df.filter(pl.col("treatment").eq(group1_label)).sort("contig", "position", "strand").get_column(motif.meth_type).to_numpy()
                group2_data = motif_df.filter(pl.col("treatment").eq(group2_label)).sort("contig", "position", "strand").get_column(motif.meth_type).to_numpy()
                
                # Distribution of the means in promoters
                group1_data_pr = motif_df.filter(pl.col("treatment").eq(group1_label), pl.col("distance_to_start").le(60)).sort("contig", "position", "strand").get_column(motif.meth_type).to_numpy()
                group2_data_pr = motif_df.filter(pl.col("treatment").eq(group2_label), pl.col("distance_to_start").le(60)).sort("contig", "position", "strand").get_column(motif.meth_type).to_numpy()        
                
                # Distribution of the standard error
                motif_df = motif.data(normalize=False).filter(pl.col(motif.meth_type).is_not_null(), pl.col(motif.meth_type).is_not_nan()).collect()
                se_df = motif_df.group_by("contig", "strand", "position", "treatment").agg(pl.col(motif.meth_type).std() / pl.col(motif.meth_type).count().sqrt())
                g1_se = se_df.filter(pl.col("treatment").eq(group1_label)).sort("contig", "position", "strand").get_column(motif.meth_type).to_numpy()
                g2_se = se_df.filter(pl.col("treatment").eq(group2_label)).sort("contig", "position", "strand").get_column(motif.meth_type).to_numpy()
                
                # All
                g1_all = motif_df.filter(pl.col("treatment").eq(group1_label)).sort("contig", "position", "strand").get_column(motif.meth_type).to_numpy()
                g2_all = motif_df.filter(pl.col("treatment").eq(group2_label)).sort("contig", "position", "strand").get_column(motif.meth_type).to_numpy()
                
                # Add positions
                motif_df = motif.genome.nearest_gene_to_positions(motif_df).filter(pl.col("distance_to_start").le(60))
                
                # Distribution of the standard error promoter
                motif_df = motif.data(normalize=False).filter(pl.col(motif.meth_type).is_not_null(), pl.col(motif.meth_type).is_not_nan()).collect()
                se_df = motif_df.group_by("contig", "strand", "position", "treatment").agg(pl.col(motif.meth_type).std() / pl.col(motif.meth_type).count().sqrt())
                g1_se_pr = se_df.filter(pl.col("treatment").eq(group1_label)).sort("contig", "position", "strand").get_column(motif.meth_type).to_numpy()
                g2_se_pr = se_df.filter(pl.col("treatment").eq(group2_label)).sort("contig", "position", "strand").get_column(motif.meth_type).to_numpy()
                
                # All
                g1_all_pr = motif_df.filter(pl.col("treatment").eq(group1_label)).sort("contig", "position", "strand").get_column(motif.meth_type).to_numpy()
                g2_all_pr = motif_df.filter(pl.col("treatment").eq(group2_label)).sort("contig", "position", "strand").get_column(motif.meth_type).to_numpy()
                
                # Do kilmogorov-smirnov test
                _, means_sig = stats.ks_2samp(group1_data, group2_data)
                _, se_sig = stats.ks_2samp(g1_se, g2_se)
                _, dist_sig = stats.ks_2samp(g1_all, g2_all)
                
                # Promoter regions
                _, means_sig_pr = stats.ks_2samp(group1_data_pr, group2_data_pr)
                _, se_sig_pr = stats.ks_2samp(g1_se_pr, g2_se_pr)
                _, dist_sig_pr = stats.ks_2samp(g1_all_pr, g2_all_pr)
                
                # Build significance marker string
                val = ""
                if means_sig < 0.05:
                    val += "*"
                
                if dist_sig < 0.05:
                    val += "#"
                
                if se_sig < 0.05:
                    val += "○"
                
                if means_sig_pr < 0.05:
                    val += "!"
                
                if dist_sig_pr < 0.05:
                    val += "?"
                
                if se_sig_pr < 0.05:
                    val += "O"
                    
                data[-2]["significant"] = val
                data[-1]["significant"] = val

    # Convert data list to a Polars DataFrame first
    df_partial = pl.DataFrame(data)

    if df_partial.is_empty():
        print(f"No data to plot in {contigs[0].parent_genome.readable_name}")
        return df_partial # Return the empty dataframe early

    # Get unique combinations and treatments
    unique_contig_motifs = df_partial.select("contig_name", "motif_string", "contig_taxonomy").unique()
    unique_treatments = df_partial.select("treatment").unique()

    # Create the complete grid by cross joining
    df_complete = unique_contig_motifs.join(unique_treatments, how="cross")

    # Left join the partial data onto the complete grid
    # This ensures every (contig, motif, treatment) combination exists
    # Missing methylation_fraction values will be null
    df = df_complete.join(
        df_partial.select("contig_name", "motif_string", "treatment", "methylation_fraction", "significant"),
        on=["contig_name", "motif_string", "treatment"],
        how="left"
    )

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
    pal = sns.color_palette(palette="tab20", n_colors=len(unique_taxonomies), as_cmap=False)
    lut = dict(zip(unique_taxonomies, pal))
    
    treatment_colors = df.with_columns(pl.col("treatment").replace_strict(contigs[0].parent_genome.treatment_color_map).alias("Treatment")).select("motif_string", "treatment", "Treatment")
    contig_colors = df.with_columns(pl.col("contig_taxonomy").replace_strict(lut).alias("Taxonomy")).select("contig_name", "Taxonomy")
    
    treatment_colors = treatment_colors.to_pandas().drop_duplicates().set_index(pivot_df.columns, drop=True).drop(columns=["motif_string", "treatment"])
    contig_colors = contig_colors.to_pandas().drop_duplicates("contig_name").set_index("contig_name")
        
    # Sort columns by motif_string first, then treatment
    pivot_df = pivot_df.sort_index(axis=1)

    # Sort rows by taxonomy classification
    contig_taxonomy_map = {row['contig_name']: row['contig_taxonomy'] for row in df.select('contig_name', 'contig_taxonomy').unique().to_dicts()}
    sorted_indices = sorted(pivot_df.index, key=lambda x: (str(contig_taxonomy_map.get(x, '')) == "Unclassified", str(contig_taxonomy_map.get(x, ''))))
    pivot_df = pivot_df.loc[sorted_indices]

    # Reindex row colors to match sorted rows
    contig_colors = contig_colors.reindex(pivot_df.index)
    
    # Reindex column colors to match sorted columns
    treatment_colors = treatment_colors.reindex(pivot_df.columns)
    
    # Create the heatmap without clustering to preserve sort order
    # Calculate dynamic figure size based on the number of rows and columns
    # Aim for roughly 0.3 inches per row and 0.3 inches per column for the heatmap itself
    # Add margins for labels, colorbars, and legends
    num_rows, num_cols = pivot_df.shape
    
    # Heuristic sizing: Adjust base factors and margins as needed for aesthetics
    row_height_factor = 0.3  # inches per row
    col_width_factor = 0.3   # inches per column
    height_margin = 1        # inches for x-axis labels, title, legends etc.
    width_margin = 2         # inches for y-axis labels, colorbar, legends etc.

    # Ensure a minimum size for readability, especially with few rows/columns
    fig_height = max(6, num_rows * row_height_factor + height_margin)
    fig_width = max(8, num_cols * col_width_factor + width_margin)

    g = sns.clustermap(
        pivot_df,
        figsize=(fig_width, fig_height), # Use dynamically calculated size
        row_colors=contig_colors,
        col_colors=treatment_colors,
        mask=pivot_df.isna(),
        cmap="coolwarm",
        cbar_kws={"label": "Methylation fraction"},
        row_cluster=False,  # Disable row clustering
        col_cluster=False,   # Disable column clustering
        linewidths=0.5, # Add faint lines between cells
        linecolor='lightgrey',
        vmin=0,
        vmax=1
    )
    
    # Add asterisks to significant cells
    if 'significant' in df.columns:
        # Create a pivot table for significance data
        sig_pivot = df.to_pandas().pivot(
            index="contig_name",
            columns=["motif_string", "treatment"],
            values="significant"
        )
        
        # Match the order of rows and columns to the methylation pivot table
        sig_pivot = sig_pivot.reindex(index=pivot_df.index, columns=pivot_df.columns)
        
        # Add asterisks to significant cells
        for i, row_idx in enumerate(pivot_df.index):
            for j, col_idx in enumerate(pivot_df.columns):
                text = sig_pivot.loc[row_idx, col_idx]
                if str(text) == "nan":
                    continue
                
                # Add asterisk to the cell, centered
                g.ax_heatmap.text(j + 0.5, i + 0.5, text,
                                ha='center', va='center',
                                color='black', fontsize="small", fontweight='bold')

    # Modify x-axis labels to show only motifs, not treatments
    motifs = [label.get_text().split("-")[0] for label in g.ax_heatmap.get_xticklabels()]
    g.ax_heatmap.set_xticklabels(motifs)
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    g.ax_cbar.remove()

    # Create legends
    ordered_taxonomies = [contig_taxonomy_map[i] for i in pivot_df.index]
    ordered_taxonomies = pd.unique(ordered_taxonomies)  # preserve order of occurrence
    taxonomy_handles = [Patch(color=lut[t], label=t) for t in ordered_taxonomies]
    treatment_handles = [Patch(color=contigs[0].parent_genome.treatment_color_map[treatment], label=treatment) 
                        for treatment in df['treatment'].unique()]
    
    # Remove automatic figure legends to avoid overlap
    if g.figure.legends:
        for leg in g.figure.legends:
            leg.remove()
    
    # Get legends
    legend1 = g.fig.legend(handles=treatment_handles, title="Treatment")
    legend2 = g.fig.legend(handles=taxonomy_handles, title="Taxonomy")
    fig = g.fig
    
    # Get the figure renderer
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    
    # Get the bounding boxes of the legends in figure coordinates
    legend1_bbox = legend1.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    legend2_bbox = legend2.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    
    # Calculate the left position (centered with legends)
    plot_bbox = g.ax_heatmap.get_position()
    legend_width = legend2_bbox.width
    cbar_width = legend_width * 0.2  # Colorbar width is 1/5 of legend width
    left_pos = plot_bbox.x0 * 0.5  # Position on the left side of the plot
    
    # Position the first legend at the top
    legend1_height = legend1_bbox.height
    legend1_top = plot_bbox.y0 + plot_bbox.height  # Align with top of plot
    legend1_left = left_pos
    legend1.set_bbox_to_anchor([legend1_left, legend1_top], transform=fig.transFigure)
    
    # Position the second legend below the first
    legend2_height = legend2_bbox.height
    legend2_top = legend1_top - legend1_height
    legend2_left = left_pos
    legend2.set_bbox_to_anchor([legend2_left, legend2_top], transform=fig.transFigure)
    
    # Calculate colorbar height and position
    available_height = legend2_top - legend2_height - plot_bbox.y0  - 0.05# Space between bottom of legend2 and bottom of plot
    cbar_top = legend2_top - legend2_height
    
    # Adjust the colorbar axes
    cbar_left = left_pos - (legend_width - cbar_width)  # Center colorbar relative to legends
    cbar_height = min(available_height, 0.5 * plot_bbox.height)  # Limit height to not exceed plot bottom
    cbar_y = cbar_top - cbar_height - 0.05
    
    # Set colorbar position
    # Add color bar
    cbar_ax = g.fig.add_axes([cbar_left, cbar_y, cbar_width, cbar_height])
    cbar = plt.colorbar(g.ax_heatmap.collections[0], cax=cbar_ax, anchor = (cbar_left, cbar_y), orientation="vertical", label="Mean methylation fraction")
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar.ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
    cbar.ax.tick_params(axis='y', which='major')
    cbar.outline.set_visible(False)
        
    # Redraw the figure to apply changes
    fig.canvas.draw()

    # Set X axis title
    g.ax_heatmap.set_xlabel("Motif")
    
    if contigs[0].is_viral:
        g.ax_heatmap.set_ylabel("Viral contig")

        g.savefig(f"{contigs[0].parent_genome.output_dir}/virus_motif_heatmap.svg", transparent=True)
        g.savefig(f"{contigs[0].parent_genome.output_dir}/virus_motif_heatmap.pdf")
        
        # Save a CSV of the data
        pivot_df.to_csv(f"{contigs[0].parent_genome.output_dir}/virus_motif_heatmap.csv")

    else:
        g.ax_heatmap.set_ylabel("Bacterial contig")

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
    