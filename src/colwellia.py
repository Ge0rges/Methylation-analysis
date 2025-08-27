import math 
import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import openpyxl

from chi_squared_test import chi_squared_test
from utilities.utils import do_ks_test, find_closest_step
from src.objects.gene_collection import GeneCollection
from src.objects.genome import Genome
from src.objects.motif import Motif
from src.utilities.utils import readable_modification_name, get_stats_data, identify_predictive_features
from src.utilities.data_loading import parse_genbank
from utilities.compare_methylome import compare_methylomes
from src.utilities.kegg_enrichment import KEGGEnrichmentAnalyzer


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
    n_stats = len(statistic_keys)
    _, axes = plt.subplots(n_stats, 3, figsize=(30, 10 * n_stats), constrained_layout=True, width_ratios=[1, 0.5, 0.5])

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
            cbar_kws={'label': stat_name, 'shrink': 0.8}, vmin=0, linewidths=0.5
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
            sns.scatterplot(
                data=timeline_df[(timeline_df['Stat Key'] == stat_key) & (timeline_df['Cycling Step'].isin(steps))],
                x='Cycling Step', y="Statistic", hue='Control', hue_order=['35ppt control', '55ppt control'],
                palette=['blue', 'red'], style='Significant', markers={True: 'o', False: 'X'},
                s=150, alpha=0.7, ax=ax, legend="full", style_order=[True, False]
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
        
        timeline_ax2.legend(list(combined.values()), list(combined.keys()), title='Legend')

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


def plot_dmr_scores_heatmap_colwellia(motif: Motif, output_dir: Path) -> None:
    """
    Plot a heatmap of DMR scores for the specified motif across sample comparisons:
      - rows = sample_a
      - columns = sample_b
      - values = mean_dmr_score
    Each motif -> separate PDF.
    """
    genome: Genome = motif.genome
    dmr = motif.dmr_data.collect(streaming=True)
    if dmr.is_empty():
        print(f"No DMR data for motif {motif.motif}")
        return
    
    # Compute mean score per (sample_a, sample_b)
    pdf = (dmr.group_by(["treatment_a", "treatment_b"]).agg(pl.col("score").mean().alias("mean_dmr_score"))).to_pandas()

    # Make sure to unify sample names if needed, e.g. barcode -> treatment
    # If you want to rename sample_a, sample_b via genome.barcode_treatment_map:
    pdf["treatment_a"] = pdf["treatment_a"].replace(genome.barcode_treatment_map).replace(genome.treatment_name_map)
    pdf["treatment_b"] = pdf["treatment_b"].replace(genome.barcode_treatment_map).replace(genome.treatment_name_map)
    
    # Fill in values for missing pairs
    # This ensures that if (A, B) exists, (B, A) will also be present with the same score
    pdf_swapped = pdf.rename(columns={'treatment_a': 'treatment_b', 'treatment_b': 'treatment_a'})
    pdf = pd.concat([pdf, pdf_swapped], ignore_index=True).drop_duplicates(subset=['treatment_a', 'treatment_b'])

    # Get all unique treatments from both columns
    all_treatments_set = set(pdf["treatment_a"].unique()) | set(pdf["treatment_b"].unique())
    
    # Sort treatments based on genome.treatment_order_map
    # Fallback for sorting: if treatment not in map, use treatment name itself (converted to str for safety)
    sorted_treatments = sorted(
        list(all_treatments_set),
        key=lambda t: genome.treatment_order_map.get(t, str(t)) 
    )

    # Create pivot with sample_a as rows, sample_b as columns
    pivot_df = pdf.pivot(index="treatment_a", columns="treatment_b", values="mean_dmr_score")

    # Reindex to ensure the desired order and include all treatments, filling missing with NaN
    pivot_df = pivot_df.reindex(index=sorted_treatments, columns=sorted_treatments)

    _, ax = plt.subplots(figsize=(len(genome.default_treatments)*2, len(genome.default_treatments)*2), constrained_layout=True)
    sns.heatmap(pivot_df, cmap="viridis", annot=True, fmt=".2f", ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    ax.set_title(f"DMR Score Heatmap - {genome.readable_name} ({motif.motif})")
    ax.set_xlabel("Treatment B")
    ax.set_ylabel("Treatment A")

    out_file = output_dir / f"{genome.readable_name}_{motif.readable_motif}_dmr_heatmap.pdf"
    plt.savefig(out_file, format="pdf")
    plt.close()
    print(f"Saved PDF: {out_file}")
    

def extract_diff_methylated_genes_colwellia(
    motif: Motif,
    top_n: int = None
) -> pl.DataFrame:
    """
    Extract the top differentially methylated locations (by DMR 'score') for a motif,
    attach nearest/overlapping gene, and include the gene function in the final CSV.

    Returns a Polars DataFrame.
    """
    genome: Genome = motif.genome
    genome.use_balanced = True
    top_rows = motif.dmr_data.filter(pl.col("balanced_map_pvalue") > 0, pl.col("balanced_map_pvalue") < 0.05).collect(streaming=True).sort("score", descending=True)
    if top_rows.is_empty():
        print(f"No DMR data for motif {motif.motif}")
        return

    # Add gene_caller_id if needed
    data = genome.add_gene_caller_id(top_rows.lazy(), include_intergenic=True).collect(streaming=True)

    # Add function
    gc = GeneCollection(data.get_column("gene_callers_id").unique().to_list(), genome)
    data = data.join(gc.get_function().collect(streaming=True), on="gene_callers_id", how="left")

    # Add nearest gene if not in gene
    data = genome.nearest_gene_to_positions(data.lazy()).collect(streaming=True)

    # Add function of nearest genes
    gc = GeneCollection(data.get_column("gene_callers_id_start").unique().to_list(), genome)
    data = data.join(gc.get_function().collect(streaming=True), left_on="gene_callers_id_start", right_on="gene_callers_id", how="left", suffix="_start")
    gc = GeneCollection(data.get_column("gene_callers_id_end").unique().to_list(), genome)
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
        
    # Take top_n rows by score if requested
    if top_n is not None and top_n > 0 and data.height > top_n:
        data = data.sort("score", descending=True).head(top_n)
    
    # Write to excel
    output_file = genome.output_dir / f"{genome.readable_name}_{motif.readable_motif}_diff_genes.xlsx"
    if not {'treatment_a', 'treatment_b'}.issubset(set(data.columns)):
            raise ValueError("Data does not contain 'treatment_a' and 'treatment_b' columns.")

    df_pd = data.to_pandas()
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for (a, b), group in df_pd.groupby(["treatment_a", "treatment_b"]):
            sheet = f"{str(a)}_{str(b)}"[:31].replace("/", "-").replace("\\", "-").replace("?", "").replace("*", "")
            group.to_excel(writer, sheet_name=sheet, index=False)
            group.to_excel(writer, sheet_name="Main", index=False)

    output_file = genome.output_dir / f"{genome.readable_name}_{motif.readable_motif}_diff_genes_promoter.xlsx"
    df_pd = data.filter(pl.col("distance_to_start") < 60).to_pandas()
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_pd.to_excel(writer, sheet_name="Main", index=False)

            
    # Write group stats to a text file
    effect_sizes = {}
    effect_sizes_promoters = {}
    group_stats_file = genome.output_dir / f"{genome.readable_name}_{motif.readable_motif}_diff_genes_group_stats.txt"
    with open(group_stats_file, "w") as f:
        for (t_a, t_b), group in df_pd.groupby(["treatment_a", "treatment_b"]):
            f.writelines(f"Summary for ({t_a}, {t_b}):\n")
            f.write(f"  Number of rows: {group.shape[0]}\n")
            f.write(f"  Average effect size: {group['balanced_effect_size'].mean()}\n")
        
            effect_sizes[(t_a, t_b)] = group["balanced_effect_size"].abs().mean()
            effect_sizes_promoters[(t_a, t_b)] = group[(group["distance_to_start"] <= 60) & (group["gene_callers_id"] == -1)]["balanced_effect_size"].abs().mean()

            gene_id_neg1 = group["gene_callers_id"] == -1
            f.write(f"  Rows where gene_callers_id == -1: {gene_id_neg1.shape[0]}\n")

            dist_start_le60 = group[(group["distance_to_start"] <= 60) & (group["gene_callers_id"] == -1)]
            f.write(f"  Rows where in promoter: {dist_start_le60.shape[0]}\n")
    
    # Plot a heatmap of effect sizes
    unique_treatment_names = set(genome.treatment_name_map[t] for t in genome.default_treatments) 
    all_treatment_names = sorted(
        list(unique_treatment_names),
        key=lambda t: genome.treatment_order_map.get(t, str(t))
    )
    
    effect_size_matrix = pd.DataFrame(np.nan, index=all_treatment_names, columns=all_treatment_names, dtype=float)
    effect_size_promoter_matrix = pd.DataFrame(np.nan, index=all_treatment_names, columns=all_treatment_names, dtype=float)
    for (treat_a, treat_b), effect_size in effect_sizes.items():
        treat_a = genome.treatment_name_map.get(treat_a)
        treat_b = genome.treatment_name_map.get(treat_b)
        
        effect_size_matrix.loc[treat_a, treat_b] = effect_size
        effect_size_matrix.loc[treat_b, treat_a] = effect_size

    for (treat_a, treat_b), effect_size_promoter in effect_sizes_promoters.items():
        treat_a = genome.treatment_name_map.get(treat_a)
        treat_b = genome.treatment_name_map.get(treat_b)
        
        effect_size_promoter_matrix.loc[treat_a, treat_b] = effect_size_promoter
        effect_size_promoter_matrix.loc[treat_b, treat_a] = effect_size_promoter
    
    _, axes = plt.subplots(1, 2, figsize=(30, 30), constrained_layout=True)
    
    for ax in axes:
        sns.heatmap(
            effect_size_matrix if ax == axes[0] else effect_size_promoter_matrix,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            cbar_kws={'label': 'Effect size', 'shrink': 0.8},
            linewidths=0.5,
            square=True
        )
        
        # Color tick labels based on treatment
        for tick in ax.get_xticklabels():
            treatment = tick.get_text()
            if treatment in genome.treatment_color_map:
                tick.set_color(genome.treatment_color_map[treatment])
        
        for tick in ax.get_yticklabels():
            treatment = tick.get_text()
            if treatment in genome.treatment_color_map:
                tick.set_color(genome.treatment_color_map[treatment])
    
        ax.set_title("Effect sizes of all DMRs" if ax == axes[0] else "Effect sizes of DMRs in promoters")
    output_file = genome.output_dir / f"{genome.readable_name}_{motif.readable_motif}_diff_genes.xlsx"
    plt.savefig(output_file.with_suffix(".pdf"), format="pdf")

    return data


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


def do_stats(motif: Motif, alpha: float = 0.05) -> None:
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
            positions = data["counts_positions"]

            results_means = compare_methylomes(beta_a, beta_b, counts_a, counts_b)
            results_promoters = compare_methylomes(pr_beta_a, pr_beta_b)
            results_no_promoters = compare_methylomes(no_pr_beta_a, no_pr_beta_b)

            # Add
            all_result_stats.append(results_means["global_table"].with_columns(pl.lit(pair_treatments[0]).alias("Treatment A"), pl.lit(pair_treatments[1]).alias("Treatment B"), pl.lit("means").alias("group")))
            all_result_stats.append(results_promoters["global_table"].with_columns(pl.lit(pair_treatments[0]).alias("Treatment A"), pl.lit(pair_treatments[1]).alias("Treatment B"), pl.lit("means_pr").alias("group")))
            all_result_stats.append(results_no_promoters["global_table"].with_columns(pl.lit(pair_treatments[0]).alias("Treatment A"), pl.lit(pair_treatments[1]).alias("Treatment B"), pl.lit("means_no_pr").alias("group")))

            # Replace index with positions
            results_means["per_site_df"] = pl.concat([positions, pl.from_pandas(results_means["per_site_df"])], how="horizontal") if results_means["per_site_df"] is not None else None
            
            output_dir = motif.genome.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            file_path = output_dir / f"stats_test_{motif.readable_motif}_{pair_treatments[0]}_vs_{pair_treatments[1]}.txt"

            significant_sites = results_means["per_site_df"].filter(pl.col("significant") == True).drop("index") if results_means["per_site_df"] is not None else None
            with open(file_path, "w") as f:
                f.write(f"Comparison: {pair_treatments}\n\n")
                f.write("Global Table Means:\n")
                f.write(results_means["global_table_str"])
                f.write("\n\n")
                f.write("Global Table Promoters:\n")
                f.write(results_promoters["global_table_str"])
                f.write("\n\n")
                f.write("Global Table No Promoters:\n")
                f.write(results_no_promoters["global_table_str"])
                f.write("\n\n")
                f.write("Significant Per-Site Results:\n")
                f.write(str(significant_sites))
            
            # Write significant_sites to excel
            if significant_sites is not None:
                excel_file_path = output_dir / f"stats_test_{motif.readable_motif}_{pair_treatments[0]}_vs_{pair_treatments[1]}.xlsx"
                with pd.ExcelWriter(excel_file_path, engine="openpyxl") as writer:
                    significant_sites_pd = significant_sites.to_pandas()
                    significant_sites_pd.to_excel(writer, sheet_name="Significant Sites", index=False)
                    results_means["per_site_df"].to_pandas().to_excel(writer, sheet_name="All Sites", index=False)            
            
            print(f"Wrote stats to {file_path}")            
        
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
    


def fraction_investigation(motif: Motif):
    
    unique_treatment_names = set(motif.genome.treatment_name_map[t] for t in motif.genome.default_treatments) 
    all_treatment_names = sorted(
        list(unique_treatment_names),
        key=lambda t: motif.genome.treatment_order_map.get(t, str(t))
    )
    
    # Generate pairs of treatments using the sorted, unique names
    low_methylation = []
    
    # Data - Restrict to promoters
    data = motif.data()
    data = motif.genome.nearest_gene_to_positions(data).filter(pl.col("distance_to_start") < 60, pl.col("gene_callers_id_start").eq(pl.col("gene_callers_id_end")).not_()).collect()
    
    # Add functions
    gene_ids = data.get_column("gene_callers_id_start").to_list() + data.get_column("gene_callers_id_end").to_list()
    gene_ids = list(set(gene_ids))
    
    gc = GeneCollection(gene_ids, motif.genome)
    data = data.join(gc.get_function().collect(streaming=True), left_on="gene_callers_id_start", right_on="gene_callers_id", how="left", suffix="_start")
    data = data.join(gc.get_function().collect(streaming=True), left_on="gene_callers_id_end", right_on="gene_callers_id", how="left", suffix="_end")

    # Get sites that are lowly methylated
    with pd.ExcelWriter(motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.motif}_low_methylation.xlsx", engine="openpyxl") as writer:
        for treatment in all_treatment_names:
            # Get all positions where the methylation fraction is less than 2*sigma of the mean
            low_threshold = data.select(pl.col(motif.meth_type).mean() - 2 * pl.col(motif.meth_type).std()).item()
            low_positions = data.filter(pl.col(motif.meth_type) < low_threshold, pl.col("treatment") == treatment)
            
            # Add to overall results
            low_methylation.append(low_positions)
            
            # Print to an excel file
            low_positions.to_pandas().to_excel(writer, sheet_name=f"{treatment}_{low_threshold:.2f}", index=False)

        # Add a sheet to the low methylation excel file with positions that are low in all conditions
        low_methylation_df = pl.concat(low_methylation, how="vertical")
        low_methylation_positions_df = low_methylation_df.group_by(["contig", "position", "strand"]).agg(pl.col("treatment").n_unique().alias("count")).filter(pl.col("count") == len(all_treatment_names))
        low_methylation_df_always = low_methylation_df.join(low_methylation_positions_df, on=["contig", "position", "strand"], how="right").unique()
        
        low_methylation_df_always.to_pandas().to_excel(writer, sheet_name="Always", index=False)
        
        # Add a sheet to the low methylation excel file with positions that are low in 1 conditions
        low_methylation_positions_df = low_methylation_df.group_by(["contig", "position", "strand"]).agg(pl.col("treatment").n_unique().alias("count")).filter(pl.col("count") == 1)
        low_methylation_df_once = low_methylation_df.join(low_methylation_positions_df, on=["contig", "position", "strand"], how="right").unique()
        
        low_methylation_df_once.to_pandas().to_excel(writer, sheet_name="Once", index=False)

    # Excel file for big changes
    with pd.ExcelWriter(motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.motif}_big_change.xlsx", engine="openpyxl") as writer:
        for big_change_threshold in [0.05, 0.1, 0.15, 0.2]:# 
            big_change_df = data.filter(((pl.col(motif.meth_type).max() - pl.col(motif.meth_type).min()) > big_change_threshold).over("contig", "position", "strand"))
            big_change_df = big_change_df.with_columns(pl.col("treatment").str.extract(r"(Cycling|35ppt control|55ppt control) S(\d+)", 1).alias("treatment_group"), pl.col("treatment").str.extract(r"(Cycling|35ppt control|55ppt control) S(\d+)", 2).cast(pl.Int32).alias("treatment_step"))
            
            # Make a dataframe that shows where big changes happen across cycles
            change_df = big_change_df.select("contig", "position", "strand", "treatment", motif.meth_type, "treatment_group", "treatment_step").unique().sort(["contig", "strand", "position", "treatment_group", "treatment_step"])
            change_df = change_df.with_columns(pl.when((pl.col(motif.meth_type) - pl.col(motif.meth_type).shift(1)).abs() > big_change_threshold).then(True).otherwise(False).over("contig", "strand", "position", "treatment_group").alias("changed"))
            change_df = change_df.join(data, on=["contig", "position", "strand"], how="left").select("contig", "position", "strand", "function", "function_end", "source", "source_end", "gene_callers_id_start", "gene_callers_id_end", "distance_to_start", "distance_to_end", "treatment", motif.meth_type, "changed")
            change_df = change_df.pivot(index=["contig", "position", "strand", "function", "function_end", "source", "source_end", "gene_callers_id_start", "gene_callers_id_end", "distance_to_start", "distance_to_end"], on="treatment", values=[motif.meth_type, "changed"], aggregate_function="first")            
            
            # Print to the excel file in a sheet
            change_df.to_pandas().to_excel(writer, sheet_name=f"Threshold {big_change_threshold}", index=False)
            
            # Identify positions that are always different between the controls
            always_different = big_change_df.group_by(["contig", "position", "strand", "treatment_group"]).agg(pl.col(motif.meth_type))
            always_different = always_different.pivot(index=["contig", "position", "strand"], columns="treatment_group", values=motif.meth_type)

            always_different = always_different.filter(pl.struct(["35ppt control", "55ppt control"]).map_elements(lambda row: all(abs(a - b) >= big_change_threshold for a in row["35ppt control"] for b in row["55ppt control"]), return_dtype=pl.Boolean))

            always_different.to_pandas().to_excel(writer, sheet_name=f"Always Diff Cntrl {big_change_threshold}", index=False)


def frac_investigation_with_stats(motif: Motif):
    npy_file_path = motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_stats_data.npy"
    if not npy_file_path.exists():
        assert False, f"Expected stats data file {npy_file_path} to exist. Run do_stats() first."
    
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
        excel_file_path = motif.genome.output_dir / f"stats_test_{motif.readable_motif}_{pair_treatments[0]}_vs_{pair_treatments[1]}.xlsx"
        # Load into a dataframe
        df = pl.from_pandas(pd.read_excel(excel_file_path, sheet_name="Significant Sites"), schema_overrides={"contig": pl.Utf8,
                                                                    "position": pl.Int64,
                                                                    "strand": pl.Boolean,
                                                                    "beta_A": pl.Float64,
                                                                    "beta_B": pl.Float64,
                                                                    "n_replicates_A": pl.Int64,
                                                                    "n_replicates_B": pl.Int64,
                                                                    "test_method": pl.Utf8,
                                                                    "p_raw": pl.Float64,
                                                                    "q_BH": pl.Float64,
                                                                    "significant": pl.Boolean,
                                                                    "treatment_1": pl.Utf8,
                                                                    "treatment_2": pl.Utf8,
                                                                })
        df = df.with_columns(pl.lit(pair_treatments[0]).alias("treatment_1"), pl.lit(pair_treatments[1]).alias("treatment_2"))
        all_result_stats.append(df)
    
    all_result_stats = pl.concat(all_result_stats)
    
    # Data - Restrict to promoters
    all_result_stats = motif.genome.nearest_gene_to_positions(all_result_stats.lazy()).filter(pl.col("distance_to_start") < 60, pl.col("gene_callers_id_start").eq(pl.col("gene_callers_id_end")).not_()).collect()
    
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
    
    from temp import analyze_differential_expression_patterns,comprehensive_advanced_analysis
    analyze_differential_expression_patterns(all_result_stats.to_pandas(), treatment_df.to_pandas(), output_file=f"{motif.genome.output_dir}/{motif.genome.readable_name}_{motif.readable_motif}_differential_meth_patterns.xlsx")
    comprehensive_advanced_analysis(all_result_stats.to_pandas(), treatment_df.to_pandas(), output_file=f"{motif.genome.output_dir}/{motif.genome.readable_name}_{motif.readable_motif}_comprehensive_advanced_analysis.xlsx")


def write_promoter_functions(motif:Motif):
    # Data - Restrict to promoters
    all_positions = motif.genome.nearest_gene_to_positions(motif.positions.lazy()).filter(pl.col("distance_to_start") < 60, pl.col("gene_callers_id_start").eq(pl.col("gene_callers_id_end")).not_()).collect()
    
    # Add functions
    gene_ids = all_positions.get_column("gene_callers_id_start").to_list() + all_positions.get_column("gene_callers_id_end").to_list()
    gene_ids = list(set(gene_ids))
    
    gc = GeneCollection(gene_ids, motif.genome)
    all_positions = all_positions.join(gc.get_function().collect(streaming=True), left_on="gene_callers_id_start", right_on="gene_callers_id", how="left", suffix="_start")
    all_positions = all_positions.join(gc.get_function().collect(streaming=True), left_on="gene_callers_id_end", right_on="gene_callers_id", how="left", suffix="_end")
    
    # Write to excel
    all_positions.write_excel(motif.genome.output_dir / f"promoter_functions{motif.readable_motif}.xlsx")
    
    
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
    
    
def find_important_features(motif: Motif, promoter_only: bool):
    df = motif.data().collect()
    
    if promoter_only:
        df = motif.genome.nearest_gene_to_positions(motif.data().lazy()).filter(pl.col("distance_to_start") < 60, pl.col("gene_callers_id_start").eq(pl.col("gene_callers_id_end")).not_()).collect()

    X = df.pivot(on=["contig", "position", "strand"], values=motif.meth_type, index="treatment")
    
    # Add a salinity column 
    y = X.with_columns(pl.col("treatment").str.extract(r"(Cycling|35ppt control|55ppt control) S(\d+)", 1).alias("group"),
                         pl.col("treatment").str.extract(r"(Cycling|35ppt control|55ppt control) S(\d+)", 2).cast(pl.Int32).alias("step"))

    y = y.with_columns(
    (
        (pl.col("group") == "55ppt control") |
        (
            (pl.col("group") == "Cycling") &
            (pl.col("step") % 2 != 0)
        )
    ).alias("salinity")).get_column("salinity").to_pandas()
    X = X.drop("treatment").to_pandas()
    
    from src.utilities.utils import identify_predictive_features
    
    result = identify_predictive_features(X, y, n_top_features=100)

    # Get positions from consensus
    consensus_positions = result['consensus']["feature"].tolist()
    
    pattern = r'\{"(.*?)",(\d+),(true|false)\}'
    parsed = [re.findall(pattern, s)[0] for s in consensus_positions]

    df = pl.DataFrame(
        {
            'contig': [x[0] for x in parsed],
            'position': [int(x[1]) for x in parsed],
            'strand': [x[2] == 'true' for x in parsed],
        }
    )
    
    # Join functions
    df = motif.genome.nearest_gene_to_positions(df.lazy()).collect()
    gene_ids = df.get_column("gene_callers_id_start").to_list() + df.get_column("gene_callers_id_end").to_list()
    gene_ids = list(set(gene_ids))
    
    gc = GeneCollection(gene_ids, motif.genome)
    df = df.join(gc.get_function().collect(streaming=True), left_on="gene_callers_id_start", right_on="gene_callers_id", how="left", suffix="_start", maintain_order="left")
    df = df.join(gc.get_function().collect(streaming=True), left_on="gene_callers_id_end", right_on="gene_callers_id", how="left", suffix="_end", maintain_order="left")

    # Save to csv
    if promoter_only:
        df.write_csv(motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_important_features_prom_only.csv")
    else:
        df.write_csv(motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_important_features.csv")


def find_hypomethylated_sites(motif: Motif):
    from src.utilities.compare_methylome import is_site_hypomethylated
    
    df = motif.data(normalize=False).collect()
    mean_meth = motif.data().select(pl.col(motif.meth_type).mean()).collect().item()
    result_df = pl.DataFrame([], schema=df.schema)
        
    for site in df.iter_rows(named=True):
        if is_site_hypomethylated(site[motif.meth_type], site[motif.canonical_base], background_rate=mean_meth, alpha=0.05):
            result_df = pl.concat([result_df, pl.DataFrame([site])])
            
    # Add a column with the percent
    result_df = result_df.with_columns(
        (pl.col(motif.meth_type) / (pl.col(motif.meth_type) + pl.col(motif.canonical_base)) * 100).alias("percent_hypomethylation")
    )
    
    # Aggregate by percentage range for every 10$
    result_df = result_df.with_columns(
        (pl.col("percent_hypomethylation") // 10 * 10).cast(pl.Int32).alias("percent_range")
    )
    
    # Make a frequency plot of the values and save it
    sns.histplot(result_df.get_column("percent_hypomethylation").to_list(), bins=50)
    plt.savefig(motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_hypomethylated_sites.png")
    
    # Number of times each site is hypoemethylated
    result_df_sheet_two = result_df.group_by("contig", "strand", "position").agg(pl.count("percent_hypomethylation").alias("num_times_hypomethylated"))

    # Write to excel result_df and result_df_sheet_two on different sheets
    with pd.ExcelWriter(motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_hypomethylated_sites.xlsx") as writer:
        result_df.to_excel(writer, sheet_name="Sheet1", index=False)
        result_df_sheet_two.to_excel(writer, sheet_name="Sheet2", index=False)
