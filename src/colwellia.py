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
from src.utilities.utils import readable_modification_name, get_stats_data, find_closest_step
from src.utilities.data_loading import parse_genbank
from src.utilities.compare_methylome import compare_methylomes
from src.utilities.kegg_enrichment import KEGGEnrichmentAnalyzer
from src.utilities.feature_statistics import *
from src.diff_pattern import analyze_differential_expression_patterns
from adjustText import adjust_text
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

sns.set_theme(context="poster", style="whitegrid")
pl.enable_string_cache()

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
        df = motif.data(in_every_treatment=False).collect()
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
        style="treatment"
    )
    
    plt.title("Site Count vs Coverage by Treatment")
    plt.xlabel("Coverage")
    plt.ylabel("Site Count")
    plt.savefig(output_dir / f"{genome.readable_name}_{motif.readable_motif}_coverage_sitecount_lineplot.pdf", format="pdf")
    plt.show()
    
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
    if df is None:
        return
    
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
        
        # Rugplot of motif instances with no data
        missing_positions = motif.genome.add_genome_relative_position(motif.positions).filter(pl.col("genome_position").is_in(pair_df.get_column("genome_position")).not_()).collect(streaming=True).to_pandas()
        sns.rugplot(data=missing_positions, x="genome_position", ax=ax, alpha=.05, clip_on=False)
        
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
    print(f"Saved PDF: {out_file}")


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
    print(f"Saved PDF: {out_file}")


def plot_statistics_heatmap_with_timeline(
    output_dir: Path,
    statistic_keys: list[str],
    significance_matrix: dict[str, pd.DataFrame],
    value_matrices: dict[str, pd.DataFrame],
    timeline_df: pd.DataFrame,
    motif: Motif,
    alpha: float,
    stat_name: str
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
    _, axes = plt.subplots(n_stats, 3, figsize=(24, 15 * n_stats), constrained_layout=True, width_ratios=[1, 0.3, 0.3])

    if n_stats == 1:
        axes = [axes]

    for i, stat_key in enumerate(statistic_keys):
        heatmap_ax, timeline_ax1, timeline_ax2 = axes[i]
        timeline_ax1.sharey(timeline_ax2)

        pval_matrix = significance_matrix[stat_key]
        val_matrix = value_matrices[stat_key]
        
        # Sort indices and columns based on genome treatment order
        sorted_treatments = sorted(motif.genome.treatment_order_map.keys(), key=motif.genome.treatment_order_map.get)
        pval_matrix = pval_matrix.reindex(index=sorted_treatments, columns=sorted_treatments)
        val_matrix = val_matrix.reindex(index=sorted_treatments, columns=sorted_treatments)

        # Make pairwise heatmap
        annot = np.where(pval_matrix < alpha, val_matrix.round(2), "X")
        sns.heatmap(
            val_matrix, ax=heatmap_ax, annot=annot, fmt="s", cmap="viridis",
            cbar_kws={'label': "D-value" if stat_name == "Kilmogorov-Smirnov" else stat_name, 'shrink': 0.8}, linewidths=0.5
        )
        heatmap_ax.set_xticklabels(heatmap_ax.get_xticklabels(), rotation=90)

        # Color tick labels
        for tick in heatmap_ax.get_xticklabels():
            treatment = tick.get_text()
            if treatment in motif.genome.treatment_color_map:
                tick.set_color(motif.genome.treatment_color_map[treatment])
        for tick in heatmap_ax.get_yticklabels():
            treatment = tick.get_text()
            if treatment in motif.genome.treatment_color_map:
                tick.set_color(motif.genome.treatment_color_map[treatment])

        title_stat_key = stat_key.replace('_', ' ').title()
        if "no_pr" in stat_key:
            title_stat_key = title_stat_key.replace("No Pr", "(No Promoter)")
        elif "pr" in stat_key:
            title_stat_key = title_stat_key.replace("Pr", "(Promoter)")
        heatmap_ax.set_title(title_stat_key)

        # Make timeline plots
        start_steps, end_steps = [1, 2], [14, 15]
        for ax, steps in [(timeline_ax1, start_steps), (timeline_ax2, end_steps)]:
            sns.set_theme(context="poster", style="whitegrid")
            sns.scatterplot(
                data=timeline_df[(timeline_df['Stat Key'] == stat_key) & (timeline_df['Cycling Step'].isin(steps))],
                x='Cycling Step', y="Statistic", hue='Control', hue_order=['35ppt control', '55ppt control'],
                palette=['blue', 'red'], style='Significant', markers={True: 'o', False: 'X'},
                s=180, alpha=0.7, ax=ax, legend="full", style_order=[True, False]
            )
            
            ax.set_ylabel("D-value" if stat_name == "Kilmogorov-Smirnov" else stat_name)
            ax.set_xlim(min(steps) - 0.5, max(steps) + 0.5)
            ax.set_xticks(steps)
            for tick in ax.get_xticklabels():
                val = int(tick.get_text())
                tick.set_color('blue' if val % 2 == 1 else 'red')
        
        # Merge legends
        handles1, labels1 = timeline_ax1.get_legend_handles_labels()
        handles2, labels2 = timeline_ax2.get_legend_handles_labels()
        combined = dict(zip(labels1 + labels2, handles1 + handles2))
        timeline_ax1.get_legend().remove()
        timeline_ax2.get_legend().remove()
        
        timeline_ax2.legend(list(combined.values()), list(combined.keys()), loc='lower right')

        # Visual cleanup for broken axis
        timeline_ax1.spines['right'].set_visible(False)
        timeline_ax2.spines['left'].set_visible(False)
        timeline_ax2.set_ylabel("")
        plt.setp(timeline_ax2.get_yticklabels(), visible=False)
        
        d = .015
        kwargs = dict(transform=timeline_ax1.transAxes, color='k', clip_on=False)
        timeline_ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs.update(transform=timeline_ax2.transAxes)
        timeline_ax2.plot((-d, +d), (-d, +d), **kwargs)

    plt.savefig(output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_{stat_name}_heatmap.pdf", format="pdf")


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

            results_means = compare_methylomes(beta_a, beta_b, counts_a, counts_b, motif=motif)
            results_promoters = compare_methylomes(pr_beta_a, pr_beta_b)
            results_no_promoters = compare_methylomes(no_pr_beta_a, no_pr_beta_b)

            # Add
            all_result_stats.append(results_means["global_table"].with_columns(pl.lit(pair_treatments[0]).alias("Treatment A"), pl.lit(pair_treatments[1]).alias("Treatment B"), pl.lit("means").alias("group")))
            all_result_stats.append(results_promoters["global_table"].with_columns(pl.lit(pair_treatments[0]).alias("Treatment A"), pl.lit(pair_treatments[1]).alias("Treatment B"), pl.lit("means_pr").alias("group")))
            all_result_stats.append(results_no_promoters["global_table"].with_columns(pl.lit(pair_treatments[0]).alias("Treatment A"), pl.lit(pair_treatments[1]).alias("Treatment B"), pl.lit("means_no_pr").alias("group")))
            all_result_stats_per_site.append(results_means["per_site_df"].with_columns(pl.lit(pair_treatments[0]).alias("treatment_1"), pl.lit(pair_treatments[1]).alias("treatment_2"), pl.lit("means").alias("group")))
        
        # Write per site to an excel file
        all_result_stats_per_site = pl.concat(all_result_stats_per_site, how="vertical")
        with Workbook(motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_per_site_stats.xlsx") as wb:
            all_result_stats_per_site.write_excel(wb, include_header=True)
            
        # For each pair and statistic, plot a heatmap
        all_result_stats = pl.concat(all_result_stats, how="vertical")
        stat_groups = all_result_stats.get_column("group").unique().to_list()

        # Find nearest matching steps between cycling and controls
        pattern = re.compile(r"(Cycling|35ppt control|55ppt control) S(\d+)")
        cycling_steps = []
        control_35ppt = []
        control_55ppt = []

        for treatment in all_treatment_names:
            match = pattern.match(treatment)
            if match:
                group, step = match.groups()
                step = int(step)
                if group == "Cycling":
                    cycling_steps.append((treatment, step))
                elif group == "35ppt control":
                    control_35ppt.append((treatment, step))
                elif group == "55ppt control":
                    control_55ppt.append((treatment, step))
        
        cycling_steps.sort(key=lambda x: x[1])
        control_35ppt.sort(key=lambda x: x[1])
        control_55ppt.sort(key=lambda x: x[1])
        
        closest_controls = {}
        for treatment, step in cycling_steps:
            closest_controls[(treatment, step)] = {
                find_closest_step(step, control_35ppt): "35ppt control",
                find_closest_step(step, control_55ppt): "55ppt control",
            }

        timeline_data = []
        for (cycling_treatment, cycling_step), correspond_controls in closest_controls.items():
            # Filter down result to the current comparison
            comparison_df = all_result_stats.filter((pl.col("Treatment A").eq(cycling_treatment) & pl.col("Treatment B").is_in(list(correspond_controls.keys()))) | (pl.col("Treatment A").is_in(list(correspond_controls.keys())) & pl.col("Treatment B").eq(cycling_treatment)))
            
            for row in comparison_df.iter_rows(named=True):            
                # Append to timeline data
                timeline_data.append({
                    'Cycling Step': cycling_step,
                    'Statistic': row['Statistic'],
                    'Control': correspond_controls.get(row['Treatment B'], correspond_controls.get(row['Treatment A'])),
                    'Significant': row['p-value'] < alpha,
                    'Stat Key': row['group'],
                    'Test': row["Test"]
                })
                                    
        timeline_df = pd.DataFrame(timeline_data)

        # Save all_result_stats, and value_matrices, significance_matrices, to a numpy file
        np.save(npy_file_path, {
            "all_result_stats": all_result_stats.to_pandas(),
            "stat_groups": stat_groups,
            "timeline_df": timeline_df
        })
        print(f"Saved numpy data file: {npy_file_path}")
    
    else:
        # Load from numpy file
        data = np.load(npy_file_path, allow_pickle=True).item()
        all_result_stats = pl.from_pandas(data["all_result_stats"])
        stat_groups = data["stat_groups"]
        timeline_df = data["timeline_df"]    
        
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
            
        plot_statistics_heatmap_with_timeline(motif.genome.output_dir, stat_groups, significance_matrices, value_matrices, timeline_df[timeline_df["Test"] == test], motif, alpha, test)
    
    # Print how well each test's result correlates with each other
    p_values_wide = all_result_stats.pivot(
        index=["Treatment A", "Treatment B", "group"],
        columns="Test",
        values="p-value"
    )

    # You can do the same for the statistics if you want to correlate them too
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
    


def frac_investigation_with_stats(motif: Motif) -> dict[str, pl.DataFrame]:
    all_result_stats = pl.read_excel(motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_per_site_stats.xlsx")
    all_result_stats = all_result_stats.filter(pl.col("significant") == True)
    
    # Data - Restrict to promoters and big effect size
    all_result_stats = motif.genome.nearest_gene_to_positions(all_result_stats.lazy()).filter(pl.col("distance_to_start") < 60, pl.col("gene_callers_id_start").eq(pl.col("gene_callers_id_end")).not_(), (pl.col("beta_A") - pl.col("beta_B")).abs() > 0.1).collect()

    # Write to an excel
    output_file = motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_promoter_differential_stats_all.xlsx"
    with Workbook(output_file) as wb:
        all_result_stats.write_excel(wb, worksheet="All Promoter Sites", include_header=True)

    # Add functions
    gene_ids = all_result_stats.get_column("gene_callers_id_start").to_list() + all_result_stats.get_column("gene_callers_id_end").to_list()
    gene_ids = list(set(gene_ids))
    
    gc = GeneCollection(gene_ids, motif.genome)
    all_result_stats = all_result_stats.join(gc.get_function().collect(streaming=True), left_on="gene_callers_id_start", right_on="gene_callers_id", how="left", suffix="_start")
    all_result_stats = all_result_stats.join(gc.get_function().collect(streaming=True), left_on="gene_callers_id_end", right_on="gene_callers_id", how="left", suffix="_end")
    
    # Make a treatments dataframe
    treatment_df = (pl.from_dict({"treatment": list(set(all_result_stats.get_column("treatment_1").to_list() + all_result_stats.get_column("treatment_2").to_list()))})
                    .with_columns(pl.col("treatment").str.extract(r"(Cycling|35ppt control|55ppt control) S(\d+)", 1).alias("group"), pl.col("treatment").str.extract(r"(Cycling|35ppt control|55ppt control) S(\d+)", 2).cast(pl.Int32).alias("step"))
                    .with_columns(((pl.col("group") == "55ppt control") | ((pl.col("group") == "Cycling") & (pl.col("step") % 2 != 0))).alias("salinity"), 
                                    (pl.col("group").str.contains("control")).alias("control"))
                    )
    
    results = analyze_differential_expression_patterns(all_result_stats.to_pandas(), treatment_df.to_pandas(), output_file=f"{motif.genome.output_dir}/{motif.genome.readable_name}_{motif.readable_motif}_differential_meth_patterns.xlsx")
    return results


def position_stats_heatmap(motif: Motif, position: int):
    all_result_stats = pl.read_excel(motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_per_site_stats.xlsx")
    all_result_stats = all_result_stats.filter(pl.col("significant") == True)
    
    all_result_stats = all_result_stats.filter(pl.col("position") == position).with_columns((pl.col("beta_A") - pl.col("beta_B")).alias("beta_diff"))
    
    # Make a heatmap where rows are treatment_1, columns are treatment_2, and values are beta_A - beta_B
    heatmap_data = all_result_stats.select("treatment_1", "treatment_2", "beta_diff").unique().to_pandas().pivot(index="treatment_1", columns="treatment_2", values="beta_diff")
    heatmap_data = heatmap_data.combine_first(heatmap_data.T)  # Make symmetric
    
    # Annotate the heatmap with the value and an asterik if significant is True
    annot_data = all_result_stats.with_columns(pl.when(pl.col("significant")).then(pl.col("beta_diff").cast(pl.Utf8) + "*").otherwise(pl.col("beta_diff").cast(pl.Utf8)).alias("annot")).select("treatment_1", "treatment_2", "annot").unique().to_pandas().pivot(index="treatment_1", columns="treatment_2", values="annot")
    annot_data = annot_data.combine_first(annot_data.T)

    # Sort annot_data and heatmap_data so rows and columns are in the same order
    sorted_treatments = sorted(motif.genome.treatment_order_map.keys(), key=motif.genome.treatment_order_map.get)
    heatmap_data = heatmap_data.reindex(index=sorted_treatments, columns=sorted_treatments)
    annot_data = annot_data.reindex(index=sorted_treatments, columns=sorted_treatments)
    
    sns.heatmap(heatmap_data, annot=annot_data, cmap="coolwarm", center=0, fmt="")
    plt.title(f"Methylation Difference at Position {position} for Motif {motif.readable_motif}")
    plt.xlabel("Treatment 2 (Negative)")
    plt.ylabel("Treatment 1 (Positive)")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.savefig(motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_position_{position}_heatmap.pdf", format="pdf")


def write_frac_sequence_with_stats(motif: Motif) -> pl.DataFrame:
    all_result_stats = pl.read_excel(motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_per_site_stats.xlsx")

    all_result_stats = motif.genome.nearest_gene_to_positions(all_result_stats)

    # Data - Restrict to promoters and big effect size
    all_result_stats = all_result_stats.filter(pl.col("distance_to_start") < 60, pl.col("gene_callers_id_start").eq(pl.col("gene_callers_id_end")).not_(),(pl.col("beta_A") - pl.col("beta_B")).abs() > 0.1)

    # Get the motif data
    motif_data = motif.data().filter(pl.col("position").is_in(all_result_stats.get_column("position"))).collect().pivot(
        index=["contig", "position", "strand"],
        on="treatment",
        values=motif.meth_type
    )
    
    # Add columns on the end saying if significant
    result = all_result_stats.pivot(
        index=["contig", "position", "strand"],
        on=["treatment_1", "treatment_2"],
        values="significant"
    ).join(motif_data, on=["contig", "position", "strand"], how="left")
    
    # Reorder columns
    # , '{"35ppt control S2","35ppt control S23"}'
    result = result.select("contig", "position", "strand", "35ppt control S1", "35ppt control S2", "35ppt control S23", "35ppt control S24", "55ppt control S1", "55ppt control S2", "55ppt control S12", "Cycling S1", "Cycling S2", "Cycling S14", "Cycling S15", '{"35ppt control S1","35ppt control S2"}', '{"35ppt control S23","35ppt control S24"}', '{"55ppt control S1","55ppt control S2"}', '{"55ppt control S12","55ppt control S2"}', '{"Cycling S1","Cycling S2"}', '{"Cycling S14","Cycling S2"}', '{"Cycling S14","Cycling S15"}')

    # Add functions
    result = motif.genome.nearest_gene_to_positions(result)
    gene_ids = result.get_column("gene_callers_id_start").to_list() + result.get_column("gene_callers_id_end").to_list()
    gene_ids = list(set(gene_ids))
    
    gc = GeneCollection(gene_ids, motif.genome)
    result = result.join(gc.get_function().collect(streaming=True), left_on="gene_callers_id_start", right_on="gene_callers_id", how="left", suffix="_start")
    result = result.join(gc.get_function().collect(streaming=True), left_on="gene_callers_id_end", right_on="gene_callers_id", how="left", suffix="_end")
    
    
    # Write to excel
    excel_file_path = motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_stats_sequence.xlsx"
    with Workbook(excel_file_path) as wb:
        result.write_excel(wb, worksheet="Seq")

    return result

    
def motif_functional_enrichment(motif: Motif):
     # Data
    all_positions = motif.genome.nearest_gene_to_positions(motif.positions.lazy()).collect()
    
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
                    .with_columns(pl.col("treatment").str.extract(r"(Cycling|35ppt control|55ppt control) S(\d+)", 1).alias("group"), pl.col("treatment").str.extract(r"(Cycling|35ppt control|55ppt control) S(\d+)", 2).cast(pl.Int32).alias("step"))
                    .with_columns(((pl.col("group") == "55ppt control") | ((pl.col("group") == "Cycling") & (pl.col("step") % 2 != 0))).alias("salinity"), 
                                    (pl.col("group").str.contains("control")).alias("control"))
                    )
    
    # Make the pandas stats DF
    df = df.join(treatment_df, on="treatment", how="left")    
    pandas_df = df.rename({motif.meth_type: "value"}).to_pandas()
    pandas_df['feature'] = pandas_df.apply(lambda row: f"{row['contig']}|{row['position']}|{row['strand']}", axis=1)
    
    # Mutual information and Welch t-test    
    X = df.pivot(on=["contig", "position", "strand"], values=motif.meth_type, index="treatment")
    
    # Add a salinity column 
    y = X.with_columns(pl.col("treatment").str.extract(r"(Cycling|35ppt control|55ppt control) S(\d+)", 1).alias("group"),
                         pl.col("treatment").str.extract(r"(Cycling|35ppt control|55ppt control) S(\d+)", 2).cast(pl.Int32).alias("step"))

    y = y.with_columns(((pl.col("group") == "55ppt control") 
                        | ((pl.col("group") == "Cycling") & (pl.col("step") % 2 != 0))).alias("salinity")).get_column("salinity").to_pandas()
    X = X.drop("treatment").to_pandas()
            
    
    # Do different feature importance methods
    mi_df = do_mutual_information(X, y)
    t_test_df = do_t_test(X, y)
    correlation_df = do_spearmanr(pandas_df)
    feature_importance, feature_importance_ci = bootstrap_pls(pandas_df)
    
    # Get feature columns back
    mi_df = pl.from_pandas(mi_df)
    t_test_df = pl.from_pandas(t_test_df).with_columns(pl.col("feature").str.split("|").list.get(0).alias("contig"), pl.col("feature").str.split("|").list.get(1).cast(pl.Int64).alias("position"), (pl.col("feature").str.split("|").list.get(2) == "True").alias("strand"))
    correlation_df = pl.from_pandas(correlation_df).with_columns(pl.col("feature").str.split("|").list.get(0).alias("contig"), pl.col("feature").str.split("|").list.get(1).cast(pl.Int64).alias("position"), (pl.col("feature").str.split("|").list.get(2) == "True").alias("strand"))
    feature_importance = pl.from_pandas(feature_importance, include_index=True).with_columns(pl.col("feature").str.split("|").list.get(0).alias("contig"), pl.col("feature").str.split("|").list.get(1).cast(pl.Int64).alias("position"), (pl.col("feature").str.split("|").list.get(2) == "True").alias("strand"))
    
    # Make a master table by joining all on left
    master = (mi_df.join(t_test_df, on=["contig", "position", "strand"], how="outer", suffix="_t_test")
                    .join(correlation_df, on=["contig", "position", "strand"], how="outer", suffix="_correlation")
                    .join(feature_importance, on=["contig", "position", "strand"], how="outer", suffix="_pls"))
    
    # Add functions onto master
    master = motif.genome.nearest_gene_to_positions(master.lazy()).collect()
    gene_ids = list(set(master.get_column("gene_callers_id_start").to_list() + master.get_column("gene_callers_id_end").to_list()))
    
    gc = GeneCollection(gene_ids, motif.genome)
    master = master.join(gc.get_function().collect(streaming=True), left_on="gene_callers_id_start", right_on="gene_callers_id", how="left", suffix="_start")
    master = master.join(gc.get_function().collect(streaming=True), left_on="gene_callers_id_end", right_on="gene_callers_id", how="left", suffix="_end")
    
    # Write each dataframe to a sheet in an excel file
    with Workbook(motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_ensemble_significant_features.xlsx") as workbook: 
        mi_df.write_excel(workbook=workbook, worksheet="mutual_information")
        t_test_df.write_excel(workbook=workbook, worksheet="t_test")
        correlation_df.write_excel(workbook=workbook, worksheet="spearman_correlation")
        feature_importance.write_excel(workbook=workbook, worksheet="pls_feature_importance")
        master.write_excel(workbook=workbook, worksheet="master_table")

    return master


def synthesis(motif: Motif, ensemble_df: pl.DataFrame, frac_groups_df: dict[str, pl.DataFrame], seq_df: pl.DataFrame):

    # For group in frac_groups, get the list of unique features, and merge with results from ensemble and seq_df
    with Workbook(motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_synthesis.xlsx") as workbook:

         # For each group
        for key, df in frac_groups_df.items():
            if df.is_empty():
                continue
            
            merged = df.join(ensemble_df, on=["contig", "position", "strand"], how="left", suffix="_ensemble").join(seq_df, on=["contig", "position", "strand"], how="left", suffix="_seq")
            
            # Select and filter
            try:
                merged = merged.select('contig', 'strand',	'position', 'description', 'mi_score',	'Component_1',	'Component_2',	'35ppt control S1',	'35ppt control S2',	'35ppt control S23',	'35ppt control S24',	'55ppt control S1',	'55ppt control S2',	'55ppt control S12',	'Cycling S1',	'Cycling S2',	'Cycling S14',	'Cycling S15',	'{"35ppt control S1","35ppt control S2"}',	'{"35ppt control S23","35ppt control S24"}',	'{"55ppt control S1","55ppt control S2"}',	'{"55ppt control S12","55ppt control S2"}',	'{"Cycling S1","Cycling S2"}',	'{"Cycling S14","Cycling S2"}',	'{"Cycling S14","Cycling S15"}',	'gene_callers_id_start_seq',	'gene_callers_id_end_seq',	'distance_to_start_seq',	'distance_to_end_seq',	'function_seq',	'source_seq',	'function_end_seq',	'source_end_seq')
            except Exception as e:
                merged = merged.select('contig', 'strand',	'position', 'shifting_pairs', 'num_pairs_shifted', 'mi_score',	'Component_1',	'Component_2',	'35ppt control S1',	'35ppt control S2',	'35ppt control S23',	'35ppt control S24',	'55ppt control S1',	'55ppt control S2',	'55ppt control S12',	'Cycling S1',	'Cycling S2',	'Cycling S14',	'Cycling S15',	'{"35ppt control S1","35ppt control S2"}',	'{"35ppt control S23","35ppt control S24"}',	'{"55ppt control S1","55ppt control S2"}',	'{"55ppt control S12","55ppt control S2"}',	'{"Cycling S1","Cycling S2"}',	'{"Cycling S14","Cycling S2"}',	'{"Cycling S14","Cycling S15"}',	'gene_callers_id_start_seq',	'gene_callers_id_end_seq',	'distance_to_start_seq',	'distance_to_end_seq',	'function_seq',	'source_seq',	'function_end_seq',	'source_end_seq')

            merged = merged.filter(pl.col("source_seq").eq("COG20_FUNCTION"), pl.col("source_end_seq").eq("COG20_FUNCTION")).unique()
            merged.write_excel(workbook, worksheet=key[:31])
            

def average_change_between_treatments(motif: Motif) -> float:
    # Get the motif data
    motif_data = motif.genome.nearest_gene_to_positions(motif.data().lazy()).filter(pl.col("distance_to_start") < 60, pl.col("gene_callers_id_start").eq(pl.col("gene_callers_id_end")).not_()).collect().pivot(
        index=["contig", "position", "strand"],
        on="treatment",
        values=motif.meth_type
    )

    # Calculate the average change between every treatment pair seperately for those that change positively, and those that change negatively
    unique_treatment_names = set(motif.genome.treatment_name_map[t] for t in motif.genome.default_treatments) 
    all_treatment_names = sorted(
        list(unique_treatment_names),
        key=lambda t: motif.genome.treatment_order_map.get(t, str(t))
    )
    
    # Generate pairs of treatments using the sorted, unique names
    pairs = list(tuple(sorted(x)) for x in combinations(all_treatment_names, 2))
    
    results = []
    for pair in pairs:
       df = motif_data.select("contig", "position", "strand", pair[0], pair[1]).with_columns((pl.col(pair[1]) - pl.col(pair[0])).alias(f"change_{pair[0]}_to_{pair[1]}"))
       
       # Average all positive/negative changes together and get STD Val
       positive_changes = df.filter(pl.col(f"change_{pair[0]}_to_{pair[1]}") > 0)
       negative_changes = df.filter(pl.col(f"change_{pair[0]}_to_{pair[1]}") < 0)

       average_positive = positive_changes.select(pl.mean(f"change_{pair[0]}_to_{pair[1]}")).item()
       average_negative = negative_changes.select(pl.mean(f"change_{pair[0]}_to_{pair[1]}")).item()
       std_positive = positive_changes.select(pl.std(f"change_{pair[0]}_to_{pair[1]}")).item()
       std_negative = negative_changes.select(pl.std(f"change_{pair[0]}_to_{pair[1]}")).item()
       
       # Make a new df with the results
       result = pl.DataFrame({
            "treatment_pair": [f"{pair[0]}_to_{pair[1]}"],
            "num_positive_changes": [positive_changes.height],
            "num_negative_changes": [negative_changes.height],
            "average_positive_change": [average_positive],
            "average_negative_change": [average_negative],
            "std_positive_change": [std_positive],
            "std_negative_change": [std_negative]
        })
       results.append(result)
    
    # Concat and save
    results_df = pl.concat(results)
    results_df.write_csv(motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_average_changes_between_treatments.csv")
    return results_df


def annotated_pca(motif: Motif):
    df = motif.genome.nearest_gene_to_positions(motif.data()).filter(pl.col("distance_to_start") < 60, pl.col("gene_callers_id_start").eq(pl.col("gene_callers_id_end")).not_()).collect()
    
    # Pivot so that features are per row, and treatments are columns
    X = df.pivot(index=["contig", "position", "strand"], on="treatment", values=motif.meth_type)
    
    # Do PCA
    pca = PCA(n_components=2)
    features = X.drop("contig", "position", "strand").to_pandas()
    pca_result = pca.fit_transform(features)
    
    # Do clustering
    kmeans = KMeans(n_clusters=2, random_state=0)
    clusters = kmeans.fit_predict(features)
    
    # Plot PCA using seaborn, colored by clusters, add variance explained by each component
    pca_df = pd.DataFrame(pca_result, columns=["Component_1", "Component_2"])
    pca_df["contig"] = X.get_column("contig").to_list()
    pca_df["position"] = X.get_column("position").to_list()
    pca_df["strand"] = X.get_column("strand").to_list()
    pca_df["cluster"] = clusters
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x="Component_1", y="Component_2", hue="cluster", palette="Set1", s=100, alpha=0.7)
    plt.title(f"PCA of Methylation Patterns for Motif {motif.readable_motif}")
    plt.xlabel(f"Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}% Variance)")
    plt.ylabel(f"Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}% Variance)")
   
    # Add arrows using adjustText for outlier points designating the position
    texts = []
    for i, row in pca_df.iterrows():
        if (abs(row["Component_1"]) > 2 * pca_df["Component_1"].std()) or (abs(row["Component_2"]) > 2 * pca_df["Component_2"].std()):
            texts.append(plt.text(row["Component_1"], row["Component_2"], f"{row['position']}", fontsize=6))

    adjust_text(texts, arrowprops=dict(arrowstyle="->", color="red"))
    
    # Save fig
    plt.savefig(motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_pca_clusters.pdf", format="pdf")
    plt.close()