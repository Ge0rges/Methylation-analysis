import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.patches import Patch

from utilities.utils import do_ks_test
from src.objects.contig import Contig
from src.objects.gene_collection import GeneCollection

sns.set_theme(context="paper", style="whitegrid")


def plot_contig_motif_heatmap_stats(contigs: list[Contig], p_value_threshold: float = 0.05):
    """
    Generates a heatmap of motif statistics, plotting only significant d-values 
    and marking non-significant ones.
    
    This function assumes that the input data function `get_contigs_data` returns
    both value columns (e.g., 'means') and corresponding p-value columns 
    (e.g., 'means_pval').
    """
    # --- 1. Data Retrieval and Initial Setup ---
    # The data source must provide values and their corresponding p-values.
    data = get_contigs_data(contigs, statistics=True, alpha=p_value_threshold)
    if len(data) == 0:
        return None
    
    df = pl.DataFrame(data).drop("treatment").unique()

    if df.is_empty():
        print(f"No data to plot in {contigs[0].parent_genome.readable_name}")
        return df

    # --- 2. Renaming and Column Definition ---
    # Define mappings for value and p-value columns for clarity.
    # IMPORTANT: Assumes input columns like 'means', 'means_pval', etc.
    rename_map = {
        "means": "Means", "se": "SE", 
        "means_pr": "Promoter means", "se_pr": "Promoter SE",
        "contig_taxonomy": "Taxonomy",
    }
    df = df.rename(rename_map)
    
    statistic_columns = ["Means", "SE", "Promoter means", "Promoter SE"]
    pvalue_columns = ["means_pval", "se_pval", "means_pr_pval", "se_pr_pval"]

    # --- 3. Data Pivoting for Values and P-values ---
    # Melt and pivot d-values for the heatmap
    df_long_values = df.melt(
        id_vars=["motif_string", "contig_name", "Taxonomy"],
        value_vars=statistic_columns,
        variable_name="statistic_key",
        value_name="d_value" 
    ).unique()
    
    pivot_df = df_long_values.to_pandas().pivot(
        index="contig_name",
        columns=["motif_string", "statistic_key"],
        values="d_value"
    )

    # Melt and pivot p-values to create the significance mask
    # First, rename p-value columns to match statistic keys for easy pivoting
    pval_rename_map = {pcol: scol for pcol, scol in zip(pvalue_columns, statistic_columns)}
    df_pvals_renamed = df.select(["motif_string", "contig_name"] + pvalue_columns).rename(pval_rename_map)

    df_long_pvals = df_pvals_renamed.melt(
        id_vars=["motif_string", "contig_name"],
        value_vars=statistic_columns,
        variable_name="statistic_key",
        value_name="p_value"
    ).unique()

    pivot_pvals_df = df_long_pvals.to_pandas().pivot(
        index="contig_name",
        columns=["motif_string", "statistic_key"],
        values="p_value"
    ).reindex_like(pivot_df) # Ensure alignment with the main data pivot table

    # --- 4. Create Mask and Annotations ---
    # Annotations will place a marker on non-significant cells
    annot_df = pivot_pvals_df.applymap(lambda p: 'X' if p >= p_value_threshold else '')

    # --- 5. Color and Legend Setup ---
    # Row colors (Taxonomy)
    unique_taxonomies = df_long_values.get_column("Taxonomy").unique().sort().to_list()
    tax_pal = sns.color_palette(palette="tab20", n_colors=len(unique_taxonomies))
    tax_lut = dict(zip(unique_taxonomies, tax_pal))
    contig_tax_map = df_long_values.select("contig_name", "Taxonomy").unique().to_pandas().set_index("contig_name")["Taxonomy"]
    contig_colors = contig_tax_map.map(tax_lut)

    # Column colors (Statistic Key)    
    stat_pal = sns.color_palette("Paired", n_colors=len(statistic_columns))
    stat_lut = dict(zip(statistic_columns, stat_pal))
    col_multi_index = pivot_df.columns
    statistic_colors_df = pd.DataFrame(index=col_multi_index)
    statistic_colors_df["Statistic"] = statistic_colors_df.index.get_level_values("statistic_key").map(stat_lut)

    # --- 6. Sorting ---
    pivot_df = pivot_df.sort_index(axis=1, level=["motif_string"])
    contig_taxonomy_map = contig_tax_map.to_dict()
    sorted_indices = sorted(
        pivot_df.index,
        key=lambda x: (str(contig_taxonomy_map.get(x, '')) == "Unclassified", str(contig_taxonomy_map.get(x, '')))
    )
    pivot_df = pivot_df.loc[sorted_indices]

    # Reindex related dataframes to match the sorted pivot table
    contig_colors = contig_colors.reindex(pivot_df.index)
    statistic_colors_df = statistic_colors_df.reindex(pivot_df.columns)
    annot_df = annot_df.reindex_like(pivot_df)
    
    # --- 7. Plotting ---
    num_rows, num_cols = pivot_df.shape
    fig_height = max(6, num_rows * 1 + 3)
    fig_width = max(8, num_cols * 1 + 3)

    g = sns.clustermap(
        pivot_df,
        figsize=(fig_width, fig_height),
        row_colors=contig_colors,
        col_colors=statistic_colors_df,
        annot=annot_df,
        fmt='s',
        annot_kws={"color": "red", "size": 24},
        cmap="viridis",
        row_cluster=False,
        col_cluster=False,
        linewidths=0.5,
        linecolor='lightgrey',
        vmin=0,
        vmax=1,
    )

    # --- 8. Final Touches and Saving ---
    motifs = [label.get_text().split("-")[0] for label in g.ax_heatmap.get_xticklabels()]
    g.ax_heatmap.set_xticklabels(motifs)
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    g.ax_cbar.remove()

    ordered_taxonomies = pd.unique([contig_taxonomy_map[i] for i in pivot_df.index])
    taxonomy_handles = [Patch(color=tax_lut[t], label=t) for t in ordered_taxonomies]
    statistic_handles = [Patch(color=stat_lut[s], label=s) for s in statistic_columns if s in stat_lut]
    
    if g.figure.legends:
        for leg in g.figure.legends:
            leg.remove()
    
    _ = g.fig.legend(handles=statistic_handles, title="Statistic", bbox_to_anchor=(1.02, 1), loc='upper left')
    _ = g.fig.legend(handles=taxonomy_handles, title="Taxonomy", bbox_to_anchor=(1.02, 0.5), loc='center left')
    
    cbar_ax = g.fig.add_axes([1.02, 0.05, 0.03, 0.4]) # Adjust position as needed
    
    cbar = plt.colorbar(g.ax_heatmap.collections[0], cax=cbar_ax, orientation="vertical", label="D-value")
    cbar.outline.set_visible(False)
    g.fig.suptitle("Contig Motif Statistics", y=0.98)
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for legends

    g.ax_heatmap.set_xlabel("Motif")
    ylabel = "Viral contig" if contigs[0].is_viral else "Bacterial contig"
    g.ax_heatmap.set_ylabel(ylabel)

    output_dir = contigs[0].parent_genome.output_dir
    
    g.savefig(f"{output_dir}/contig_motif_heatmap_dvalue_stats_significant.svg", transparent=True, bbox_inches='tight')
    g.savefig(f"{output_dir}/contig_motif_heatmap_dvalue_stats_significant.pdf", bbox_inches='tight')
    pivot_df.to_csv(f"{output_dir}/contig_motif_heatmap_dvalue_stats_significant.csv")

    return df


def get_contigs_data(contigs: list[Contig], statistics, alpha):
    # Create a dataframe with columns: contig_name treatment motif_string methylation_fraction contig_taxonomy
    data = []

    for contig in contigs:
        for motif in contig.motifs:            
            
            motif_df = motif.data()
            if motif_df is None:
                continue
            
            motif_df = motif_df.filter(pl.col(motif.meth_type).is_not_null(), pl.col(motif.meth_type).is_not_nan()).collect()

            # Add data with significance information
            i = 0
            treatments = motif_df.get_column("treatment").unique()
            for treatment in treatments:
                # For us to take a mean, there must be at least 20% of positions
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
            if i == 2 and statistics:
                adj_pvals, adj_dvals = do_ks_test(motif, treatments, alpha)
                
                data[-1].update(adj_dvals)
                data[-2].update(adj_dvals)
                data[-1].update(adj_pvals)
                data[-2].update(adj_pvals)
    
    return data


def extract_diff_methylated_genes_contigs(contigs: list[Contig]):
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
            data = data.lazy().join(gc.get_function(), on="gene_callers_id", how="left")

            # Add nearest gene if not in gene
            data = contig.parent_genome.nearest_gene_to_positions(data).collect()

            # Add function of nearest genes
            gc = GeneCollection(data.get_column("gene_callers_id_start").unique().to_list(), contig.parent_genome)
            data = data.join(gc.get_function().collect(), left_on="gene_callers_id_start", right_on="gene_callers_id", how="left", suffix="_start")
            gc = GeneCollection(data.get_column("gene_callers_id_end").unique().to_list(), contig.parent_genome)
            data = data.join(gc.get_function().collect(), left_on="gene_callers_id_end", right_on="gene_callers_id", how="left", suffix="_end")
            
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
    