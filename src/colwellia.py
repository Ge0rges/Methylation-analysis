import math 
import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from xlsxwriter import Workbook
import pandas as pd
import polars as pl
import seaborn as sns

from src.utilities.chi_squared_test import chi_squared_test
from src.objects.gene_collection import GeneCollection
from src.objects.genome import Genome
from src.objects.motif import Motif
from src.utilities.utils import readable_modification_name, get_stats_data
from src.utilities.data_loading import parse_genbank
from src.utilities.compare_methylome import *

from src.utilities.kegg_enrichment import KEGGEnrichmentAnalyzer
from src.utilities.feature_statistics import *
from src.utilities.diff_pattern import analyze_differential_expression_patterns
from adjustText import adjust_text
from sklearn.decomposition import PCA
from netgraph import Graph

sns.set_theme(context="poster", style="whitegrid")
pl.enable_string_cache()
np.random.seed(42)

def plot_number_of_positions_by_coverage_colwellia(motif: Motif, output_dir: Path) -> None:
    """
    Plot the number of positions by coverage for a given motif across all samples.
    """
    genome: Genome = motif.genome
    original_cov = genome.default_coverage
    
    # Create a list for all coverage values you want to plot
    coverage_points = list(range(1, 1001, 10))

    coverage_counts = []
    for cov in coverage_points:
        genome.default_coverage = cov  # ensure motif.data is not cached with LRU
        df = motif.data(in_every_treatment=False)
        if df is None:
            break
        df = df.collect()
        df = df.group_by("treatment").agg(pl.struct(["contig", "strand", "position"]).n_unique().alias("count"))
        df = df.with_columns(pl.lit(cov).alias("coverage"))
        coverage_counts.append(df)

    coverage_counts = pl.concat(coverage_counts)
    df_plot = coverage_counts.to_pandas()

    # Sort treatments if you want a specific legend order
    treatment_order = sorted(df_plot['treatment'].unique(), key=genome.treatment_order_map.get)

    plt.figure(figsize=(16, 10), constrained_layout=True)
    sns.lineplot(
        data=df_plot,
        x="coverage",
        y="count",
        hue="treatment",
        hue_order=treatment_order,
        style="treatment",
        markers=True
    )
    
    # Draw a vertical line at x=30
    plt.axvline(x=30, color='gray', linestyle='--')
    
    plt.title("Site Count vs Coverage by Treatment")
    plt.xlabel("Coverage")
    plt.ylabel("Site Count")
    plt.savefig(output_dir / f"{genome.readable_name}_{motif.readable_motif}_coverage_sitecount_lineplot.pdf", format="pdf")
    plt.close()
    
    genome.default_coverage = original_cov  # Reset to original coverage


def plot_whole_methylome_colwellia(motif: Motif, output_dir: Path, promoter_only: bool =False) -> None:
    """
    Plot the whole methylome (only for the motif's methylation type) across samples:
    - fraction = motif_type / (motif_type + canonical base)
    - x-axis = genome_position, color by sample
    - Creates separate subplots for each pairwise treatment comparison
    """
    genome: Genome = motif.genome
    df = motif.genome.add_genome_relative_position(motif.data()).rename({"treatment": "Treatment"}).collect()
    
    if promoter_only:
        df =  motif.genome.nearest_gene_to_positions(df).filter(pl.col("distance_to_start") < 60)

    # Get all unique treatments and create pairwise combinations
    treatments = sorted(df.get_column("Treatment").unique().to_list(), key=genome.treatment_order_map.get)
    pairwise_comparisons = list(combinations(treatments, 2))
    
    # Calculate subplot grid dimensions
    n_comparisons = len(pairwise_comparisons)
    n_cols = min(3, n_comparisons)
    n_rows = math.ceil(n_comparisons / n_cols)
    
    # Create subplots
    _, axes = plt.subplots(n_rows, n_cols, figsize=(12*n_cols, 7*n_rows), constrained_layout=True)
    
    # Handle case where we have only one subplot
    if n_comparisons == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (treat1, treat2) in enumerate(pairwise_comparisons):
        ax = axes[idx]
        
        # Filter data for current treatment pair
        pair_df = df.filter(df["Treatment"].is_in([treat1, treat2]))
        
        # If both are the same color switch and red, switch the second to purple, if both blue, switch the second to green
        palette = [genome.treatment_color_map[treat1], genome.treatment_color_map[treat2]]
        if palette[0] == palette[1]:
            if palette[0] == "red":
                palette[1] = "violet"
            elif palette[0] == "blue":
                palette[1] = "green"
        
        dark_palette = []
        for color in palette:
            if color == "red":
                dark_palette.append("darkred")
            elif color == "blue":
                dark_palette.append("darkblue")
            elif color == "green":
                dark_palette.append("darkgreen")
            elif color == "violet":
                dark_palette.append("darkviolet")
            else:
                dark_palette.append(color)
                
        # Regplots
        sns.regplot(
            data=pair_df.filter(pl.col("Treatment").eq(treat1)).to_pandas(),
            x="genome_position",
            y=motif.meth_type,
            ax=ax,
            order=4,
            label=treat1,
            scatter_kws={"s": 16, "alpha": 0.7, "color": palette[0]},
            line_kws={"color": dark_palette[0]}
        )
        
        sns.regplot(
            data=pair_df.filter(pl.col("Treatment").eq(treat2)).to_pandas(),
            x="genome_position",
            y=motif.meth_type,
            ax=ax,
            order=4,
            label=treat2,
            scatter_kws={"s": 16, "alpha": 0.7, "color": palette[1]},
            line_kws={"color": dark_palette[1]}
        )
        
        # Make a legend with two
        ax.legend(title="Treatment")
        sns.move_legend(ax, "lower left")
        
        # Axis settings
        ax.set_xlabel("Genome position (bp)")
        ax.set_ylabel(f"Methylation fraction")
        ax.tick_params(axis='y', which='both', left=True)
        ax.tick_params(axis='x', which='both', bottom=True)
        
        # Set axis limits
        ax.set_xlim(0, pair_df.select(pl.col("genome_position").max()).item())
        ax.set_ylim(0, 1)

    # Hide any unused subplots
    for idx in range(n_comparisons, len(axes)):
        axes[idx].set_visible(False)

    out_file = output_dir / f"{genome.readable_name}_whole_methylome_pairwise_{motif.readable_motif}_promoter.pdf" if promoter_only else output_dir / f"{genome.readable_name}_whole_methylome_pairwise_{motif.readable_motif}.pdf"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()


def plot_motif_methylation_distribution_colwellia(motif: Motif, output_dir: Path) -> None:
    """
    Plot the distribution (histogram) of motif's methylation fraction in each sample,
    rather than the mean fraction.

    - fraction = meth / (meth + canonical)
    - We'll do a single figure with multiple histplot calls, or subplots, showing how
    many sites fall in each fraction bin per sample.
    """
    genome: Genome = motif.genome
    df = motif.data().collect().rename({"treatment": "Treatment"})
    
    if df is None:
        return
    
    _, ax = plt.subplots(figsize=(32, 20), constrained_layout=True)

    sns.kdeplot(
        data=df.to_pandas(),
        x=motif.meth_type,
        hue="Treatment",
        ax=ax,
        hue_order=sorted(df.get_column("Treatment").unique().to_list(), key=genome.treatment_order_map.get)
    )
    
    ax.set_xlabel(f"{readable_modification_name[motif.meth_type]} methylation fraction")
    ax.set_ylabel("Sites")
    ax.set_title(f"{genome.readable_name} - Distribution of {motif.motif} Methylation")

    out_file = output_dir / f"{genome.readable_name}_{motif.readable_motif}_methylation_distribution.pdf"
    plt.savefig(out_file, format="pdf") 
    plt.close()


def plot_statistics_heatmap(
    statistic_keys: list[str],
    significance_matrix: dict[str, pd.DataFrame],
    value_matrices: dict[str, pd.DataFrame],
    motif: Motif,
    alpha: float,
    stat_name: str,
    output_path = None,
    position=None
) -> None:
    """
    Plots heatmaps and timeline charts from pre-calculated statistical data.

    Args:
        output_dir (Path): Directory to save the plot.
        statistic_keys (List[str]): List of statistic keys to plot.
        p_value_matrices (Dict[str, pd.DataFrame]): Dict of significance matrices (p-values).
        d_value_matrices (Dict[str, pd.DataFrame]): Dict of value matrices (D-values).
        timeline_df (pd.DataFrame): DataFrame for timeline plots.
        genome (Any): The genome object with metadata like treatment colors.
        alpha (float, optional): Significance threshold. Defaults to 0.05.
    """
    sns.set_theme(context="poster", style="whitegrid")

    n_stats = len(statistic_keys)
    _, axes = plt.subplots(n_stats, 3, figsize=(40, 10 * n_stats), constrained_layout=True)

    if n_stats == 1:
        axes = [axes]

    for i, stat_key in enumerate(statistic_keys):
        heatmap_ax_controls, heatmap_ax_cycling, heatmap_ax_cycling_controls = axes[i]

        pval_matrix = significance_matrix[stat_key]
        val_matrix = value_matrices[stat_key]
        
        # Round val matrix to 2
        val_matrix = (val_matrix*100).round(2) if stat_name == "1-Wasserstein" or stat_name == "Methylation percent difference" else val_matrix.round(2)

        # Sort indices and columns based on genome treatment order
        sorted_treatments = sorted(motif.genome.treatment_order_map.keys(), key=motif.genome.treatment_order_map.get)
        pval_matrix = pval_matrix.reindex(index=sorted_treatments, columns=sorted_treatments)
        val_matrix = val_matrix.reindex(index=sorted_treatments, columns=sorted_treatments)
        
        # Handle boolean pval_matrix
        if pval_matrix.dtypes.iloc[0] == object:
            condition = pval_matrix
        else:
            condition = pval_matrix < alpha
        annot_matrix = pd.DataFrame(np.where(condition, val_matrix, "X"), index=val_matrix.index, columns=val_matrix.columns)

        # Get the data we want
        controls_matrix = val_matrix.loc[
            val_matrix.index.str.contains("control"),
            val_matrix.index.str.contains("control")
        ]
        cycling_matrix = val_matrix.loc[
            val_matrix.index.str.contains("Alternating"),
            val_matrix.index.str.contains("Alternating")
        ]
        cycling_controls_matrix = val_matrix.loc[
            val_matrix.index.str.contains("Alternating"), 
            val_matrix.columns.str.contains("control")
        ]
        
        # Get the correct annotations out for each matrix
        controls_annot_matrix = annot_matrix.loc[controls_matrix.index, controls_matrix.columns]
        cycling_annot_matrix = annot_matrix.loc[cycling_matrix.index, cycling_matrix.columns]
        cycling_controls_annot_matrix = annot_matrix.loc[cycling_controls_matrix.index, cycling_controls_matrix.columns]

        # Get global min and max values
        global_min = min(controls_matrix.min().min(), cycling_matrix.min().min(), cycling_controls_matrix.min().min())
        global_max = max(controls_matrix.max().max(), cycling_matrix.max().max(), cycling_controls_matrix.max().max())

        # Make pairwise heatmap of just controls versus each other
        sns.heatmap(
            controls_matrix, ax=heatmap_ax_controls, annot=controls_annot_matrix, fmt="s", cmap="viridis",
            vmin=global_min, vmax=global_max,
            cbar_kws={'label': "D-value" if stat_name == "Kilmogorov-Smirnov" else stat_name, 'shrink': 0.8}, 
            linewidths=0.5
        )

        # Make pairwise heatmap of just cycling versus each other
        sns.heatmap(
            cycling_matrix, ax=heatmap_ax_cycling, annot=cycling_annot_matrix, fmt="s", cmap="viridis",
            vmin=global_min, vmax=global_max,
            cbar_kws={'label': "D-value" if stat_name == "Kilmogorov-Smirnov" else stat_name, 'shrink': 0.8}, 
            linewidths=0.5
        )

        # Make pairwise heatmap of just cycling versus controls
        sns.heatmap(
            cycling_controls_matrix, ax=heatmap_ax_cycling_controls, annot=cycling_controls_annot_matrix, fmt="s", cmap="viridis",
            vmin=global_min, vmax=global_max,
            cbar_kws={'label': "D-value" if stat_name == "Kilmogorov-Smirnov" else stat_name, 'shrink': 0.8}, 
            linewidths=0.5
        )

        # Decorations
        for ax in axes[i]:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

            # Color tick labels
            for tick in ax.get_xticklabels():
                treatment = tick.get_text()
                if treatment in motif.genome.treatment_color_map:
                    tick.set_color(motif.genome.treatment_color_map[treatment])
            for tick in ax.get_yticklabels():
                treatment = tick.get_text()
                if treatment in motif.genome.treatment_color_map:
                    tick.set_color(motif.genome.treatment_color_map[treatment])
        
            # Set title
            title_stat_key = stat_key.replace('_', ' ').title()
            if "no_pr" in stat_key:
                title_stat_key = "No promoters"
            elif "pr" in stat_key:
                title_stat_key = "Promoters only"
            elif position is not None:
                # Make the title the position function
                title_stat_key = f"Position {position}"
            else:
                title_stat_key = "All sites"
            ax.set_title(title_stat_key)

    output_path = motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_{stat_name}_heatmap.pdf" if output_path is None else output_path
    plt.savefig(output_path, format="pdf")
    plt.close()


def plot_statistics_graph(statistic_keys: list[str],
    significance_matrix: dict[str, pd.DataFrame],
    value_matrices: dict[str, pd.DataFrame],
    motif: Motif,
    alpha: float,
    stat_name: str
) -> None:

    if stat_name != "1-Wasserstein":
        return
    
    for i, stat_key in enumerate(statistic_keys):
        pval_matrix = significance_matrix[stat_key]
        val_matrix = value_matrices[stat_key]
        
        # Round val matrix to 2
        val_matrix = (val_matrix*100).round(2) if stat_name == "1-Wasserstein" else val_matrix.round(2)

        # Sort indices and columns based on genome treatment order
        sorted_treatments = sorted(motif.genome.treatment_order_map.keys(), key=motif.genome.treatment_order_map.get)
        pval_matrix = pval_matrix.reindex(index=sorted_treatments, columns=sorted_treatments)
        val_matrix = val_matrix.reindex(index=sorted_treatments, columns=sorted_treatments)
        
        # Create a mapping from original treatment names to cleaned names
        def clean_treatment_name(treatment):
            name = ""
            for word in treatment.split(","):
                name += word.replace("ppt control ", "").replace("Alternating ", "A") + ","
            return name[:-1]  # Remove trailing comma
  
        # Add all edges with significance information
        edge_length = {}
        for i, treatment_a in enumerate(sorted_treatments):
            for j, treatment_b in enumerate(sorted_treatments):
                if i < j:
                    pval = pval_matrix.at[treatment_a, treatment_b]
                    value = val_matrix.at[treatment_a, treatment_b]
                    
                    # Find corresponding edge for significance
                    node_a = clean_treatment_name(treatment_a)
                    node_b = clean_treatment_name(treatment_b)

                    # Add significant edges
                    if pval < alpha:
                        edge_length[(node_a, node_b)] = float(value)
                    else:
                        edge_length[(node_a, node_b)] = 0.01
        
        edges = list(edge_length.keys())
        
        # Node colors by default
        node_color = {}
        for treatment in sorted_treatments:
            node_color[clean_treatment_name(treatment)] = motif.genome.treatment_color_map[treatment]

        _, ax = plt.subplots(figsize=(20, 20), constrained_layout=True)
        Graph(edges, node_labels=True, node_layout='geometric', node_layout_kwargs=dict(edge_length=edge_length, tol=1e-6), scale=(5,5), ax=ax, node_color=node_color, node_size=25, node_label_fontdictdict={'size': 40})
        ax.set_aspect('equal')
        plt.savefig(motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_{stat_name}_{stat_key}_graph.pdf", format="pdf")
        plt.close()


def write_genbank_features_near_motifs(motif: Motif) -> None:
    features = pl.from_dict(parse_genbank("/researchdrive/gkanaan/colwellia_methylation/exp/colwellia_34h.gb")).lazy()
    features = features.with_columns(pl.lit(motif.positions.collect().get_column("contig").unique().first()).alias("contig"))
    
    # Add nearest gene
    data = motif.genome.nearest_gene_to_positions(motif.positions, genes_base=features)
    
    # Add back feature_name and feature_function for gene start
    data = data.join(features, left_on="gene_callers_id_start", right_on="gene_callers_id", suffix="_start", how="left").rename({"feature_name": "feature_name_start", "feature_function": "feature_function_start"})
    data = data.join(features, left_on="gene_callers_id_end", right_on="gene_callers_id", suffix="_end", how="left").rename({"feature_name": "feature_name_end", "feature_function": "feature_function_end"})
    data = data.select("contig", "position", "strand", "gene_callers_id_start", "gene_callers_id_end", "feature_name_start", "feature_function_start", "feature_name_end", "feature_function_end", "distance_to_start", "distance_to_end",)
    
    # Write to file
    out_file = motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_genbank_features.csv"
    data.collect().write_csv(out_file, separator=",", include_header=True)


def write_basic_stats_colwellia(genome: Genome, motifs: list[Motif]):
    text = []
    for motif in motifs:
        site_count = motif.positions.collect().height
        text.append(f"Motif: {motif.readable_motif}\n")
        text.append(f"Number of sites: {site_count}\n")

        # Compute weighted fraction using treatment_weighted_mean
        df = motif.data()
        if df is None:
            return 
        
        df = df.collect()
        
        if df.is_empty():
            continue
        
        # Compute number of sites with data for each treatment
        treatments = df.get_column("treatment").unique().to_list()
        stats_df = df.group_by("treatment").agg(pl.col(motif.meth_type).mean().alias("mean"), pl.col(motif.meth_type).std().alias("std"))
        for t in treatments:
            slice_df = df.filter(pl.col("treatment") == t)
            site_count_treatment = slice_df.unique(subset=["contig", "position", "strand"]).height
            text.append(f"Number of sites with data for {t}: {site_count_treatment}\n")

            text.append(f"Average {motif.meth_type} fraction: {stats_df.filter(pl.col('treatment') == t).select('mean').item()}\n")
            text.append(f"Standard deviation of {motif.meth_type} fraction: {stats_df.filter(pl.col('treatment') == t).select('std').item()}\n")

    text.append(f"Found motifs: {','.join([m.motif for m in motifs])}")
    
    out_file = genome.output_dir / f"{genome.readable_name}_motifs_basic_stats.txt"
    with open(out_file, "w") as f:
        f.writelines(text)
        
    print(f"Saved file: {out_file}")


def motif_distribution(motif: Motif):
    data = motif.genome.nearest_gene_to_positions(motif.positions)
    genome_length = len(motif.genome.sequence[list(motif.genome.sequence)[0]].seq)
    
    whole_genome = genome_length/motif.positions.collect().height
    
    # Delete file if it exists
    if (motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_motif_distribution_in_genome.txt").exists():
        (motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_motif_distribution_in_genome.txt").unlink()

    with open(motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_motif_distribution_in_genome.txt", "w") as f:
        for promoter_size in [35, 60, 100, 300]:
            # Number of motifs
            genic = data.filter(pl.col("gene_callers_id_start").eq(pl.col("gene_callers_id_end"))).collect().height
            intergenic = data.filter(pl.col("gene_callers_id_start").ne(pl.col("gene_callers_id_end")) & pl.col("distance_to_start").ge(promoter_size)).collect().height
            promoter = data.filter(pl.col("gene_callers_id_start").ne(pl.col("gene_callers_id_end")) & pl.col("distance_to_start").lt(promoter_size)).collect().height
            
            # Length of regions
            genic_length = motif.genome.gene_caller_df.select("start", "stop").with_columns((pl.col("stop") - pl.col("start")).alias("length")).select("length").collect().sum().item()
            distance_to_gene = (motif.genome.gene_caller_df.select("start", "stop").sort("start")
                                .with_columns((pl.col("start") - pl.col("stop").shift(1)).alias("distance_to_gene"))).collect()
            
            if not distance_to_gene.is_empty() and distance_to_gene.item(0, "distance_to_gene") == None and distance_to_gene.item(0, "start") > promoter_size:
                promoter_length = distance_to_gene.filter(pl.col("distance_to_gene").ge(promoter_size)).height * promoter_size + 1
            
            elif not distance_to_gene.is_empty():   
                promoter_length = distance_to_gene.filter(pl.col("distance_to_gene").ge(promoter_size)).height * promoter_size
            else:
                promoter_length = 0
                
            intergenic_length = genome_length - genic_length - promoter_length
            
            # Print all this to a file 
            f.write(f"Promoter size: {promoter_size} bp\n")
            f.write(f"  Number of promoter motifs: {promoter}\n")
            f.write(f"  Frequency of genic motifs: {genic_length/genic} bp\n")
            f.write(f"  Frequency of intergenic motifs: {intergenic_length/intergenic} bp\n")
            f.write(f"  Frequency of promoter motifs: {promoter_length/promoter} bp\n")

            # Print test results for each agaianst each other
            genic_intergenic_p_value = chi_squared_test(
                a=genic, 
                b=intergenic, 
                L1=genic_length, 
                L2=intergenic_length
            )
            f.write(f"  Genic vs Intergenic difference (p-value: {genic_intergenic_p_value:.4f})\n")
            
            genic_promoter_p_value = chi_squared_test(
                a=genic, 
                b=promoter, 
                L1=genic_length, 
                L2=promoter_length
            )
            
            f.write(f"  Genic vs Promoter difference (p-value: {genic_promoter_p_value:.4f})\n")
            
            intergenic_promoter_p_value = chi_squared_test(
                a=intergenic,
                b=promoter,
                L1=intergenic_length,
                L2=promoter_length
            )
            
            f.write(f"  Intergenic vs Promoter: difference (p-value: {intergenic_promoter_p_value:.4f})\n")


            # Print test results for each agaisnt whole genome
            genic_whole_p_value = chi_squared_test(
                a=genic, 
                b=whole_genome,
                L1=genic_length, 
                L2=genome_length
            )
            
            f.write(f"  Genic vs Whole Genome difference (p-value: {genic_whole_p_value:.4f})\n")

            intergenic_whole_p_value = chi_squared_test(
                a=intergenic,
                b=whole_genome,
                L1=intergenic_length,
                L2=genome_length
            )
            
            f.write(f"  Intergenic vs Whole Genome difference (p-value: {intergenic_whole_p_value:.4f})\n")
            
            promoter_whole_p_value = chi_squared_test(
                a=promoter,
                b=whole_genome,
                L1=promoter_length,
                L2=genome_length
            )
            
            f.write(f"  Promoter vs Whole Genome difference (p-value: {promoter_whole_p_value:.4f})\n")

        # Positons - Origin: 180 Terminus: 2,739,550 - from GC_SKEW
        # Around origin
        for origin_size in [100, 250, 500, 1000, 2500, 5000]:
            origin = motif.positions.filter(pl.col("position").le(180+origin_size) | pl.col("position").ge(genome_length - (origin_size-180))).collect().height
            f.write(f"Frequency of motifs within {origin_size} of origin: {origin_size*2/origin} bp\n")
            
            # Remaining genome
            non_origin = motif.positions.filter(pl.col("position").gt(180+origin_size) & pl.col("position").lt(genome_length - (origin_size-180))).collect().height
            f.write(f"Frequency of motifs in non-origin: {genome_length/non_origin} bp\n")
            
            # Test origin versus remaining genome
            origin_p_value = chi_squared_test(
                a=origin,
                b=non_origin,
                L1=origin_size*2,  # before and after origin
                L2=genome_length - origin_size*2  # Remaining genome length
            )
            f.write(f"Origin {origin_size} vs Remaining Genome difference (p-value: {origin_p_value:.4f})\n")
        
        # Whole genome
        f.write(f"Frequency of motifs in the whole genome: {whole_genome} bp\n")
        
        # Provirus
        motifs_in_provirus = motif.positions.filter(pl.col("position").ge(889352) & pl.col("position").le(902031)).collect().height
        provirus_length = 902031 - 889352 + 1  # Inclusive range
        motifs_not_in_provirus = motif.positions.filter(pl.col("position").lt(889352) | pl.col("position").gt(902031)).collect().height
        not_provirus_length = genome_length - provirus_length
        f.write(f"Frequency of motifs in the provirus: {provirus_length/motifs_in_provirus} bp\n")
        f.write(f"Frequency of motifs not in the provirus: {not_provirus_length/motifs_not_in_provirus} bp\n")
        
        motifs_in_provirus_p_value = chi_squared_test(
            a=motifs_in_provirus,
            b=whole_genome,
            L1=provirus_length,  # Length of the provirus
            L2=genome_length  # Whole genome length
        )
        
        f.write(f"Provirus vs Whole Genome difference (p-value: {motifs_in_provirus_p_value:.4f})\n")
        
        provirus_not_provirus_p_value = chi_squared_test(
            a=motifs_in_provirus,
            b=motifs_not_in_provirus,
            L1=provirus_length,  # Length of the non-provirus region
            L2=not_provirus_length  # Whole genome length
        )

        f.write(f"Provirus vs Not Provirus difference (p-value: {provirus_not_provirus_p_value:.4f})\n")


def do_whole_methylome_stats(motif: Motif, alpha: float = 0.05) -> None:
    npy_file_path = motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_stats_data.npy"
    if not npy_file_path.exists():

        unique_treatment_names = set(motif.genome.treatment_name_map[t] for t in motif.genome.default_treatments) 
        all_treatment_names = sorted(
            list(unique_treatment_names),
            key=lambda t: motif.genome.treatment_order_map.get(t, str(t))
        )
        
        # Generate pairs of treatments using the sorted, unique names
        pairs = list(tuple(sorted(x)) for x in combinations(all_treatment_names, 2))
        all_result_stats = []
        
        # Get Stats
        all_result_stats_per_site = []
        for pair_treatments in pairs:
            data = get_stats_data(motif, pair_treatments)
            beta_a = data["group1_means"]
            beta_b = data["group2_means"]
            pr_beta_a = data["group1_promoter_means"]
            pr_beta_b = data["group2_promoter_means"]
            no_pr_beta_a = data["group1_no_promoter_means"]
            no_pr_beta_b = data["group2_no_promoter_means"]
            counts_a = data["group1_counts"]
            counts_b = data["group2_counts"]

            # Perform
            results_means = compare_methylomes(beta_a, beta_b, counts_a, counts_b, motif=motif, alpha=alpha)
            results_promoters = compare_methylomes(pr_beta_a, pr_beta_b, motif=motif, alpha=alpha)
            results_no_promoters = compare_methylomes(no_pr_beta_a, no_pr_beta_b, motif=motif, alpha=alpha)

            # Add
            all_result_stats.append(results_means["global_table"].with_columns(pl.lit(pair_treatments[0]).alias("Treatment A"), pl.lit(pair_treatments[1]).alias("Treatment B"), pl.lit("means").alias("group")))
            all_result_stats.append(results_promoters["global_table"].with_columns(pl.lit(pair_treatments[0]).alias("Treatment A"), pl.lit(pair_treatments[1]).alias("Treatment B"), pl.lit("means_pr").alias("group")))
            all_result_stats.append(results_no_promoters["global_table"].with_columns(pl.lit(pair_treatments[0]).alias("Treatment A"), pl.lit(pair_treatments[1]).alias("Treatment B"), pl.lit("means_no_pr").alias("group")))
            all_result_stats_per_site.append(results_means["per_site_df"].with_columns(pl.lit(pair_treatments[0]).alias("treatment_1"), pl.lit(pair_treatments[1]).alias("treatment_2"), pl.lit("means").alias("group")))
        
        # Write per site to an excel file
        all_result_stats_per_site = pl.concat(all_result_stats_per_site, how="vertical")
        with Workbook(motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_per_site_stats.xlsx") as wb:
            all_result_stats_per_site.write_excel(wb, include_header=True)
        
        # Do multiple test correction on the whole methylomes
        all_result_stats = pl.concat(all_result_stats, how="vertical")
        pvalues = all_result_stats.get_column("p-value").to_list()
        corrected_pvalues = multipletests(pvalues, method="fdr_bh", alpha=alpha)[1]
        all_result_stats = all_result_stats.with_columns(pl.Series("p-value", corrected_pvalues))
        
        stat_groups = all_result_stats.get_column("group").unique().to_list()

        # Save all_result_stats, and value_matrices, significance_matrices, to a numpy file
        np.save(npy_file_path, {
            "all_result_stats": all_result_stats.to_pandas(),
            "stat_groups": stat_groups,
        })
        print(f"Saved numpy data file: {npy_file_path}")
    
    else:
        # Load from numpy file
        data = np.load(npy_file_path, allow_pickle=True).item()
        all_result_stats = pl.from_pandas(data["all_result_stats"])
        stat_groups = data["stat_groups"]
        
    # Plot each test
    for test in all_result_stats.get_column("Test").unique().to_list():
        # Heatmap
        value_matrices: dict[str, pd.DataFrame] = {}
        significance_matrices: dict[str, pd.DataFrame] = {}
        
        test_df = all_result_stats.filter(pl.col("Test").eq(test))

        for stat in stat_groups:
            value_matrices[stat] = test_df.filter(pl.col("group").eq(stat)).to_pandas().pivot(index="Treatment A", columns="Treatment B", values="Statistic")
            significance_matrices[stat] = test_df.filter(pl.col("group").eq(stat)).to_pandas().pivot(index="Treatment A", columns="Treatment B", values="p-value")

            # Fill in symmetric values (Treatment A vs Treatment B and vice versa)
            value_matrices[stat] = value_matrices[stat].combine_first(value_matrices[stat].T)
            significance_matrices[stat] = significance_matrices[stat].combine_first(significance_matrices[stat].T)
            
        plot_statistics_heatmap(stat_groups, significance_matrices, value_matrices, motif, alpha, test)
        plot_statistics_graph(stat_groups, significance_matrices, value_matrices, motif, alpha, test)
    
    # Print how well each test's result correlates with each other
    p_values_wide = all_result_stats.pivot(
        index=["Treatment A", "Treatment B", "group"],
        columns="Test",
        values="p-value"
    )

    statistics_wide = all_result_stats.pivot(
        index=["Treatment A", "Treatment B", "group"],
        columns="Test",
        values="Statistic"
    )
    
    test_names = all_result_stats.get_column("Test").unique().to_list()
    p_value_correlation = p_values_wide.select(test_names).corr()
    statistic_correlation = statistics_wide.select(test_names).corr()

    print("\n--- Overall p-value Correlation Matrix ---")
    print(p_value_correlation)

    print("\n--- Overall Statistic Correlation Matrix ---")
    print(statistic_correlation)
    
    # Print how well the test results correlate within groups (promoters only, non-promoters only, all)
    for test in all_result_stats.get_column("Test").unique().to_list():
        test_df = all_result_stats.filter(pl.col("Test").eq(test))
        
        p_values_wide = test_df.pivot(
            index=["Treatment A", "Treatment B"],
            columns="group",
            values="p-value"
        )
        
        statistics_wide = test_df.pivot(
            index=["Treatment A", "Treatment B"],
            columns="group",
            values="Statistic"
        )
        
        group_names = test_df.get_column("group").unique().to_list()
        print(f"\n--- Correlations for test: {test} ---")

        # P-value correlations
        print("\n--- P-value Correlations ---")
        p_values_df = p_values_wide.select(group_names).to_pandas().dropna()
        for group1, group2 in combinations(group_names, 2):
            corr, p_val = spearmanr(p_values_df[group1], p_values_df[group2])
            print(f"'{group1}' vs '{group2}': Spearman Correlation={corr:.4f}, p-value={p_val:.4f}")

        # Statistic correlations
        print("\n--- Statistic Correlations ---")
        statistics_df = statistics_wide.select(group_names).to_pandas().dropna()
        for group1, group2 in combinations(group_names, 2):
            corr, p_val = spearmanr(statistics_df[group1], statistics_df[group2])
            print(f"'{group1}' vs '{group2}': Spearman Correlation={corr:.4f}, p-value={p_val:.4f}")


def frac_investigation_with_stats(motif: Motif) -> dict[str, pl.DataFrame]:
    all_result_stats = pl.read_excel(motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_per_site_stats.xlsx")
    all_result_stats = all_result_stats.filter(pl.col("significant") == True)
    
    # Data - Restrict to promoters and big effect size
    mean_stddev = motif.data().group_by("contig", "strand", "position").agg(pl.col(motif.meth_type).std()).select(pl.col(motif.meth_type).mean()).collect().item()
    all_result_stats = motif.genome.nearest_gene_to_positions(all_result_stats.lazy()).filter(pl.col("distance_to_start") < 60, pl.col("gene_callers_id_start").eq(pl.col("gene_callers_id_end")).not_(), (pl.col("beta_A") - pl.col("beta_B")).abs() > 2*mean_stddev).collect()

    # Make a treatments dataframe
    treatment_df = (pl.from_dict({"treatment": list(set(all_result_stats.get_column("treatment_1").to_list() + all_result_stats.get_column("treatment_2").to_list()))})
                    .with_columns(pl.col("treatment").str.extract(r"(Alternating|33ppt control|55ppt control) S(\d+)", 1).alias("group"), pl.col("treatment").str.extract(r"(Alternating|33ppt control|55ppt control) S(\d+)", 2).cast(pl.Int32).alias("step"))
                    .with_columns(((pl.col("group") == "55ppt control") | ((pl.col("group") == "Alternating") & (pl.col("step") % 2 == 0))).alias("salinity"), 
                                    (pl.col("group").str.contains("control")).alias("control"))
                    )
    
    results = analyze_differential_expression_patterns(all_result_stats, treatment_df, output_file=f"{motif.genome.output_dir}/{motif.genome.readable_name}_{motif.readable_motif}_differential_meth_patterns.xlsx")
    return results.select("contig", "position", "strand", "description")


def position_stats_plots(motif: Motif, position: int):
    all_result_stats = pl.read_excel(motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_per_site_stats.xlsx")
    all_result_stats = all_result_stats.filter(pl.col("position") == position).with_columns((pl.col("beta_A") - pl.col("beta_B")).alias("beta_diff"))
    
    # Make significant true when significant is true and abs(beta_diff) >= 2*stddev
    mean_stddev = motif.data().group_by("contig", "strand", "position").agg(pl.col(motif.meth_type).std()).select(pl.col(motif.meth_type).mean()).collect().item()
    all_result_stats = all_result_stats.with_columns((pl.col("significant") & (pl.col("beta_diff").abs() >= 2*mean_stddev)).alias("significant"))
    
    # Make a heatmap where rows are treatment_1, columns are treatment_2, and values are beta_A - beta_B
    heatmap_data = all_result_stats.select("treatment_1", "treatment_2", "beta_diff").unique().to_pandas().pivot(index="treatment_1", columns="treatment_2", values="beta_diff")
    
    # Signifiance matrix
    significance_data = all_result_stats.select("treatment_1", "treatment_2", "significant").unique().to_pandas().pivot(index="treatment_1", columns="treatment_2", values="significant")
    
    # Make symmetric
    heatmap_data = heatmap_data.combine_first(-heatmap_data.T)
    significance_data = significance_data.combine_first(significance_data.T)
    
    # Print
    print(f"\n--- Position {position} statistics ---")
    print(heatmap_data)
    print(all_result_stats.select("q_BH").to_pandas().combine_first(all_result_stats.select("q_BH").to_pandas().T))
    
    # Plot heatmap 
    plot_statistics_heatmap(
        statistic_keys=["Chi2"],
        significance_matrix={"Chi2": significance_data},
        value_matrices={"Chi2": heatmap_data},
        motif=motif,
        alpha=None,
        stat_name=f"Methylation fraction difference",
        output_path=motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_position_{position}_heatmap.pdf",
        position=position
    )

    
def motif_functional_enrichment(motif: Motif):
     # Data
    all_positions = motif.genome.nearest_gene_to_positions(motif.positions).collect()
    
    # Add functions
    gene_ids = all_positions.get_column("gene_callers_id_start").to_list() + all_positions.get_column("gene_callers_id_end").to_list()
    gene_ids = list(set(gene_ids))
    
    gc = GeneCollection(gene_ids, motif.genome)
    all_positions = all_positions.join(gc.get_function().collect(streaming=True), left_on="gene_callers_id_start", right_on="gene_callers_id", how="left", suffix="_start")
    all_positions = all_positions.join(gc.get_function().collect(streaming=True), left_on="gene_callers_id_end", right_on="gene_callers_id", how="left", suffix="_end")
    
    # Promoters
    promoters = all_positions.filter(pl.col("distance_to_start") < 60, pl.col("gene_callers_id_start").eq(pl.col("gene_callers_id_end")).not_()).get_column("gene_callers_id_start").unique()
    
    # Get the function DF
    promoter_functions = gc.functional_df.filter(pl.col("gene_callers_id").is_in(promoters), pl.col("source").eq("KOfam")).collect().get_column("accession").to_list()
    other_functions = gc.functional_df.filter(pl.col("gene_callers_id").is_in(promoters).not_(), pl.col("source").eq("KOfam")).collect().get_column("accession").to_list()

    # Get number without Kofam
    no_kofam = len(promoters) - len(promoter_functions)
    
    analyzer = KEGGEnrichmentAnalyzer()
    results = analyzer.perform_enrichment_analysis(promoter_functions, other_functions, level="module")
    enriched_strings = results[results['enriched_in_set1']]

    print(enriched_strings.head())
    analyzer.save_results(results, motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_kegg_enrichment_promoters_vs_nonpromoters_pathway.csv")
    print(f"Percent of promoters without KOfam: {no_kofam / len(promoters) * 100:.2f}%")


def ensemble_significant_features(motif: Motif) -> pl.DataFrame:
    # Get motif data
    df = motif.genome.nearest_gene_to_positions(motif.data().lazy()).filter(pl.col("distance_to_start") < 60, pl.col("gene_callers_id_start").eq(pl.col("gene_callers_id_end")).not_()).collect()
    
    # Make a treatments dataframe
    unique_treatment_names = set(motif.genome.treatment_name_map[t] for t in motif.genome.default_treatments) 
    all_treatment_names = sorted(
        list(unique_treatment_names),
        key=lambda t: motif.genome.treatment_order_map.get(t, str(t))
    )
    
    treatment_df = (pl.from_dict({"treatment": all_treatment_names})
                    .with_columns(pl.col("treatment").str.extract(r"(Alternating|33ppt control|55ppt control) S(\d+)", 1).alias("group"), pl.col("treatment").str.extract(r"(Alternating|33ppt control|55ppt control) S(\d+)", 2).cast(pl.Int32).alias("step"))
                    .with_columns(((pl.col("group") == "55ppt control") | ((pl.col("group") == "Alternating") & (pl.col("step") % 2 == 0))).alias("salinity"), 
                                    (pl.col("group").str.contains("control")).alias("control"))
                    )
    
    # Make the pandas stats DF
    df = df.join(treatment_df, on="treatment", how="left")    
    pandas_df = df.rename({motif.meth_type: "value"}).to_pandas()
    pandas_df['feature'] = pandas_df.apply(lambda row: f"{row['contig']}|{row['position']}|{row['strand']}", axis=1)
    
    # Mutual information and spearman correlation 
    X = df.pivot(on=["contig", "position", "strand"], values=motif.meth_type, index="treatment")
    
    # Add a salinity column 
    y = X.with_columns(pl.col("treatment").str.extract(r"(Alternating|33ppt control|55ppt control) S(\d+)", 1).alias("group"),
                         pl.col("treatment").str.extract(r"(Alternating|33ppt control|55ppt control) S(\d+)", 2).cast(pl.Int32).alias("step"))

    y = y.with_columns(((pl.col("group") == "55ppt control") 
                        | ((pl.col("group") == "Alternating") & (pl.col("step") % 2 == 0))).alias("salinity"))
    X = X.drop("treatment").to_pandas()
            
    # Do different feature importance methods
    salinity_features = do_feature_selection(X, y.get_column("salinity").to_pandas(), alpha=0.05, top_percentile=10)
    group_features = do_feature_selection(X, y.get_column("group").to_pandas(), alpha=0.05, top_percentile=10)
    step_features = do_feature_selection(X, y.get_column("step").to_pandas(), alpha=0.05, top_percentile=10)

    feature_importance = run_pls(pandas_df)
    feature_importance = pl.from_pandas(feature_importance, include_index=True).with_columns(pl.col("feature").str.split("|").list.get(0).alias("contig"), pl.col("feature").str.split("|").list.get(1).cast(pl.Int64).alias("position"), (pl.col("feature").str.split("|").list.get(2) == "True").alias("strand"))
    
    # Make a master table by joining all on left
    master = (salinity_features.join(group_features, on=["contig", "position", "strand"], how="full", suffix="_group")
                    .join(step_features, on=["contig", "position", "strand"], how="full", suffix="_step")
                    .join(feature_importance, on=["contig", "position", "strand"], how="full", suffix="_pls"))
    
    # Write each dataframe to a sheet in an excel file
    with Workbook(motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_ensemble_significant_features.xlsx") as workbook: 
        salinity_features.write_excel(workbook=workbook, worksheet="feature_selection_salinity")
        group_features.write_excel(workbook=workbook, worksheet="feature_selection_group")
        step_features.write_excel(workbook=workbook, worksheet="feature_selection_step")

        feature_importance.write_excel(workbook=workbook, worksheet="pls_feature_importance")
        master.write_excel(workbook=workbook, worksheet="master_table")

    return master


def synthesis(motif: Motif, ensemble_df: pl.DataFrame, frac_groups_df: pl.DataFrame, pca_df: pl.DataFrame, reg_df: pl.DataFrame):

    unique_treatment_names = set(motif.genome.treatment_name_map[t] for t in motif.genome.default_treatments) 
    all_treatment_names = sorted(
        list(unique_treatment_names),
        key=lambda t: motif.genome.treatment_order_map.get(t, str(t))
    )
    
    # Merge all analyses
    merged = (pca_df.join(ensemble_df, on=["contig", "position", "strand"], how="full", suffix="_ensemble", coalesce=True)
                    .join(frac_groups_df, on=["contig", "position", "strand"], how="full", suffix="_frac_groups_df", coalesce=True)
                    .join(reg_df, on=["contig", "position", "strand"], how="full", suffix="_reg", coalesce=True))
    
    # Select the columns we want
    merged = merged.select(
        "contig", "position", "strand", "description",
        "mutual_info", "chi2", "f_classif",
        "mutual_info_group", "chi2_group", "f_classif_group",
        "mutual_info_step", "chi2_step", "f_classif_step",
        "Component_1", "Component_2",
        "PC1", "PC2", "type"
    ).rename({"mutual_info": "mutual_info_salinity",
            "chi2": "chi2_salinity",
            "f_classif": "f_classif_salinity",
            "Component_1": "PLS_Component_1",
            "Component_2": "PLS_Component_2",
            "PC1": "PCA_Component_1", "PC2": "PCA_Component_2"})
    
    # Remove any columns that are all null
    merged = merged[[s.name for s in merged if not (s.null_count() == merged.height)]]
    merged = merged.unique().sort("contig", "strand", "position")
    
    # Add the methylation fraction in each treatment
    meth_fracs_df = motif.data().collect().pivot(index=["contig", "position", "strand"], on="treatment", values=motif.meth_type)
    merged = merged.join(meth_fracs_df, on=["contig", "position", "strand"], how="left", suffix="_meth_frac")

    # Select the columns we want
    merged = merged.select(*[x for x in merged.columns if x not in all_treatment_names], *all_treatment_names).unique()

    # Add function
    merged = motif.genome.nearest_gene_to_positions(merged)
    gene_ids = list(set(merged.get_column("gene_callers_id_start").to_list() + merged.get_column("gene_callers_id_end").to_list()))

    gc = GeneCollection(gene_ids, motif.genome).get_function().collect(streaming=True)
    merged = merged.join(gc, left_on="gene_callers_id_start", right_on="gene_callers_id", how="left", suffix="_start")
    merged = merged.join(gc, left_on="gene_callers_id_end", right_on="gene_callers_id", how="left", suffix="_end")
    merged = merged.unique()
    
    # Get threshold for PLS and PCA loadings (top 10%)
    pls_threshold_1 = merged.select(pl.col("PLS_Component_1").abs()).quantile(0.9).item()
    pls_threshold_2 = merged.select(pl.col("PLS_Component_2").abs()).quantile(0.9).item()
    pca_threshold_1 = merged.select(pl.col("PCA_Component_1").abs()).quantile(0.9).item()
    pca_threshold_2 = merged.select(pl.col("PCA_Component_2").abs()).quantile(0.9).item()

    with Workbook(motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_synthesis.xlsx") as workbook:
        salinity = merged.filter(pl.col("mutual_info_salinity"), pl.col("chi2_salinity"), pl.col("f_classif_salinity")).sort("PCA_Component_1")
        salinity.write_excel(workbook, worksheet="Salinity")
        group = merged.filter(pl.col("mutual_info_group"), pl.col("chi2_group"), pl.col("f_classif_group")).sort("PCA_Component_1")
        group.write_excel(workbook, worksheet="Group")
        step = merged.filter(pl.col("mutual_info_step"), pl.col("chi2_step"), pl.col("f_classif_step")).sort("PCA_Component_1")
        step.write_excel(workbook, worksheet="Step")

        pca = merged.filter((pl.col("PCA_Component_1").abs() >= pca_threshold_1) | (pl.col("PCA_Component_2").abs() >= pca_threshold_2))
        pca.write_excel(workbook, worksheet="PCA")
        pls = merged.filter((pl.col("PLS_Component_1").abs() >= pls_threshold_1) | (pl.col("PLS_Component_2").abs() >= pls_threshold_2))
        pls.write_excel(workbook, worksheet="PLS")

        outliers = merged.filter(pl.col("type").is_not_null())
        outliers.write_excel(workbook, worksheet="Outliers")
        
        # Candidates are those which are interesting in one way
        candidates = pl.concat([salinity, group, step, pca, pls, outliers]).unique().sort("contig", "strand", "position")
        
        candidates = candidates.with_columns(
            ((pl.col("mutual_info_salinity")).cast(pl.Int32) +
             (pl.col("chi2_salinity")).cast(pl.Int32) +
             (pl.col("f_classif_salinity")).cast(pl.Int32) +
             (pl.col("mutual_info_group")).cast(pl.Int32) +
             (pl.col("chi2_group")).cast(pl.Int32) +
             (pl.col("f_classif_group")).cast(pl.Int32) +
             (pl.col("mutual_info_step")).cast(pl.Int32) +
             (pl.col("chi2_step")).cast(pl.Int32) +
             (pl.col("f_classif_step")).cast(pl.Int32) +
             ((pl.col("PCA_Component_1").abs() >= pca_threshold_1) | (pl.col("PCA_Component_2").abs() >= pca_threshold_2)).cast(pl.Int32) +
             ((pl.col("PLS_Component_1").abs() >= pls_threshold_1) | (pl.col("PLS_Component_2").abs() >= pls_threshold_2)).cast(pl.Int32) +
             (pl.col("type").is_not_null()).cast(pl.Int32) +
             (pl.col("description").is_not_null()).cast(pl.Int32)).alias("score"),
            
            # Make a text column listing why it's a candidate
            pl.concat_list([
                pl.when(pl.col("mutual_info_salinity")).then(pl.lit("MI (Salinity)")).otherwise(None),
                pl.when(pl.col("chi2_salinity")).then(pl.lit("Chi2 (Salinity)")).otherwise(None),
                pl.when(pl.col("f_classif_salinity")).then(pl.lit("ANOVA (Salinity)")).otherwise(None),
                pl.when(pl.col("mutual_info_group")).then(pl.lit("MI (Group)")).otherwise(None),
                pl.when(pl.col("chi2_group")).then(pl.lit("Chi2 (Group)")).otherwise(None),
                pl.when(pl.col("f_classif_group")).then(pl.lit("ANOVA (Group)")).otherwise(None),
                pl.when(pl.col("mutual_info_step")).then(pl.lit("MI (Step)")).otherwise(None),
                pl.when(pl.col("chi2_step")).then(pl.lit("Chi2 (Step)")).otherwise(None),
                pl.when(pl.col("f_classif_step")).then(pl.lit("ANOVA (Step)")).otherwise(None),
                pl.when((pl.col("PCA_Component_1").abs() >= pca_threshold_1) | (pl.col("PCA_Component_2").abs() >= pca_threshold_2)).then(pl.lit("PCA")).otherwise(None),
                pl.when((pl.col("PLS_Component_1").abs() >= pls_threshold_1) | (pl.col("PLS_Component_2").abs() >= pls_threshold_2)).then(pl.lit("PLS")).otherwise(None),
                pl.when(pl.col("type").is_not_null()).then(pl.col("type")).otherwise(None),
                pl.when(pl.col("description").is_not_null()).then(pl.col("description")).otherwise(None)
            ]).list.join(", ").alias("candidate_reasons")
        ).filter(pl.col("source").eq("COG20_FUNCTION"), pl.col("source_end").eq("COG20_FUNCTION"))
        
        candidates.write_excel(workbook, worksheet="Candidates")

        merged.write_excel(workbook, worksheet="All")


def annotated_pca(motif: Motif):
    df = motif.genome.nearest_gene_to_positions(motif.data()).filter(pl.col("distance_to_start") < 60, pl.col("gene_callers_id_start").eq(pl.col("gene_callers_id_end")).not_()).collect()
    
    # Pivot so that features are per row, and treatments are columns
    X = df.pivot(index=["contig", "position", "strand"], on="treatment", values=motif.meth_type)
    
    # Do PCA
    pca = PCA(n_components=2, random_state=42)
    features = X.drop("contig", "position", "strand").to_pandas().T
    pca_result = pca.fit_transform(features)
    
    # Plot PCA using seaborn, colored by clusters, add variance explained by each component
    pca_df = pd.DataFrame(pca_result, columns=["Component_1", "Component_2"])
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x="Component_1", y="Component_2", s=100, alpha=0.7)
    plt.title(f"PCA of Methylation Patterns for Motif {motif.readable_motif}")
    plt.xlabel(f"Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}% Variance)")
    plt.ylabel(f"Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}% Variance)")
    
    # Add treatment name to each point on the PCA using adjustText
    texts = []
    for i, row in pca_df.iterrows():
        texts.append(plt.text(row["Component_1"], row["Component_2"], features.index[i], fontsize=9))
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'), )
   
    # Save fig
    plt.savefig(motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_pca_clusters.pdf", format="pdf")
    plt.close()
    
    # Get scaled loadings 
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    results = pl.DataFrame(loadings, schema=["PC1", "PC2"])
    
    # Map loadings back to original features
    feature_identifiers = X.select("contig", "position", "strand")
    results = pl.concat([feature_identifiers, results], how="horizontal")

    return results


def regulatory_candidates(motif: Motif):
    df = motif.genome.nearest_gene_to_positions(motif.data()).filter(pl.col("distance_to_start") < 60, pl.col("gene_callers_id_start").eq(pl.col("gene_callers_id_end")).not_()).collect()
    
    # Pivot so that features are per row, and treatments are columns
    X = df.pivot(index=["contig", "position", "strand"], on="treatment", values=motif.meth_type).fill_null(0)
    
    # Get the list of sites whose standard deviation over the experiment is at least three times higher than the rest of the dataset
    mean_stddev = X.select(pl.concat_list(pl.exclude("contig", "strand", "position")).list.std().mean()).item()
    high_variance_sites = X.filter(pl.concat_list(pl.exclude("contig", "strand", "position")).list.std().alias("std_dev") >= 3 * mean_stddev)

    # Get the list of sites whose value is ever above or below three time the standard deviation plus the mean of any given treatment.
    all_outliers = []
    for treatment in X.columns:
        if treatment in ["contig", "position", "strand"]:
            continue
        treatment_mean = X.select(pl.col(treatment)).mean().item()
        treatment_std = X.select(pl.col(treatment)).std().item()
        outliers = X.filter((pl.col(treatment) > treatment_mean + 3 * treatment_std) | (pl.col(treatment) < treatment_mean - 3 * treatment_std))
        all_outliers.append(outliers)
    
    all_outliers_df = pl.concat(all_outliers).unique()
    
    # Save to excel file with sheets
    with Workbook(motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_regulatory_candidates.xlsx") as wb:
        high_variance_sites.write_excel(wb, worksheet="high_variance_sites", include_header=True)
        all_outliers_df.write_excel(wb, worksheet="outlier_sites", include_header=True)
    
    # Concat for return with new column ("type" = "outlier" or "high_variance")
    high_variance_sites = high_variance_sites.with_columns(pl.lit("High variance").alias("type"))
    all_outliers_df = all_outliers_df.with_columns(pl.lit("Outlier").alias("type"))
    combined = pl.concat([high_variance_sites, all_outliers_df])
    
    # Merge
    combined = combined.group_by(["contig", "position", "strand"]).agg(pl.col("type").str.concat("/"))
    return combined.select("contig", "position", "strand", "type")


def plot_stat_dists(motif: Motif) -> None:
    unique_treatment_names = set(motif.genome.treatment_name_map[t] for t in motif.genome.default_treatments) 
    all_treatment_names = sorted(
        list(unique_treatment_names),
        key=lambda t: motif.genome.treatment_order_map.get(t, str(t))
    )
    
    # Generate pairs of treatments using the sorted, unique names
    pairs = list(tuple(sorted(x)) for x in combinations(all_treatment_names, 2))
    all_result_stats = []
    
    # Get Stats
    for pair_treatments in pairs:
        data = get_stats_data(motif, pair_treatments)
        beta_a = data["group1_means"]
        beta_b = data["group2_means"]
        
        treatment_a, treatment_b = pair_treatments
        
        # Visualize
        visualize_epps_singleton_test(beta_a, beta_b, treatment_a, treatment_b, motif)
        visualize_ks_test(beta_a, beta_b, treatment_a, treatment_b, motif)
