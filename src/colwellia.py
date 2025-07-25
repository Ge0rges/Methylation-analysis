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

from utilities.utils import do_ks_test
from src.objects.gene_collection import GeneCollection
from src.objects.genome import Genome
from src.objects.motif import Motif
from src.utilities.utils import readable_modification_name

sns.set_theme(context="paper", style="whitegrid")
pl.enable_string_cache()

def plot_number_of_positions_by_coverage_colwellia(motif: Motif, output_dir: Path) -> None:
    """
    Plot the number of positions by coverage for a given motif across all samples.
    """
    genome: Genome = motif.genome
    original_cov = genome.default_coverage
    
    # Get data for different coverages
    coverage_counts = []
    for cov in [10, 30, 100, 150, 300, 500, 1000]:
        genome.default_coverage = cov  # This requires the data method to not be LRU cached
        df = motif.data(in_every_treatment=False).collect()
        df = df.group_by("treatment").agg(pl.struct(["contig", "strand", "position"]).n_unique().alias("count"))
        df = df.with_columns(pl.lit(cov).alias("coverage"))
        coverage_counts.append(df)
    
    coverage_counts = pl.concat(coverage_counts)
    
    # Plot barplot using seaborn
    _, ax = plt.subplots(figsize=(16, 16), constrained_layout=True)
    sns.barplot(
        data=coverage_counts.to_pandas(),
        x="treatment",
        y="count",
        hue="coverage",
        legend='full',
        ax=ax,
        palette="Set1",
        order=sorted(coverage_counts.get_column("treatment").unique().to_list(), key=genome.treatment_order_map.get)
    )
    
    plt.title("Not in every treatment")    
    plt.xticks(rotation=90, ha='right')
    
    plt.savefig(output_dir / f"{genome.readable_name}_{motif.readable_motif}_coverage_counts_noteverytreatment.pdf", format="pdf")
    
    genome.default_coverage = original_cov  # Reset to original coverage


def plot_whole_methylome_colwellia(motif: Motif, output_dir: Path) -> None:
    """
    Plot the whole methylome (only for the motif's methylation type) across samples:
    - fraction = motif_type / (motif_type + canonical base)
    - x-axis = genome_position, color by sample
    - Creates separate subplots for each pairwise treatment comparison
    """
    genome: Genome = motif.genome
    df = motif.data()
    if df is None:
        return
    
    df = df.collect()
    df = genome.add_genome_relative_position(df).rename({"treatment": "Treatment"})
    
    if df.is_empty():
        return
    
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
                palette[1] = "purple"
            elif palette[0] == "blue":
                palette[1] = "green"
                
        # Create scatterplot for this pair
        sns.scatterplot(
            data=pair_df.to_pandas(),
            x="genome_position",
            y=motif.meth_type,
            hue="Treatment",
            style="Treatment",
            ax=ax,
            s=16,
            alpha=0.7,
            hue_order=[treat1, treat2],
            palette=palette    
        )
        
        sns.move_legend(ax, "lower left")
        
        # Add polynomial regression lines for each treatment in the pair
        for j, treatment in enumerate([treat1, treat2]):
            treatment_df = pair_df.filter(pair_df["Treatment"] == treatment).to_pandas()
            
            if len(treatment_df) > 4:
                avg_df = treatment_df.groupby("genome_position")[motif.meth_type].mean().reset_index()
                
                if len(avg_df) > 4:
                    # Fit polynomial regression
                    x = avg_df["genome_position"].values
                    y = avg_df[motif.meth_type].values
                    z = np.polyfit(x, y, 4)
                    p = np.poly1d(z)
                    
                    # Generate smooth curve
                    x_smooth = np.linspace(x.min(), x.max(), 300)
                    y_smooth = p(x_smooth)
                    
                    # Plot the smoothed average line
                    ax.plot(x_smooth, y_smooth, color=genome.treatment_color_map[treatment], linewidth=2)
        
        ax.set_xlabel("Genome position (bp)")
        ax.set_ylabel(f"Methylation fraction")
        
        ax.tick_params(axis='y', which='both', left=True)
        ax.tick_params(axis='x', which='both', bottom=True)

        ax.legend_.set_title(None)

    # Hide any unused subplots
    for idx in range(n_comparisons, len(axes)):
        axes[idx].set_visible(False)

    out_file = output_dir / f"{genome.readable_name}_whole_methylome_pairwise_{motif.readable_motif}.pdf"
    plt.savefig(out_file, bbox_inches='tight')
    print(f"Saved PDF: {out_file}")


def plot_whole_methylome_kde_colwellia(motif: Motif, output_dir: Path) -> None:
    """
    Plot the whole methylome (only for the motif's methylation type) across samples:
    - fraction = motif_type / (motif_type + canonical base)
    - x-axis = genome_position, color by sample
    - Creates separate subplots for each pairwise treatment comparison
    """
    genome: Genome = motif.genome
    df = motif.data()
    if df is None:
        return
    
    df = df.collect()
    df = genome.add_genome_relative_position(df).rename({"treatment": "Treatment"})
    
    if df.is_empty():
        return
    
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
        
        # If both are the same color switch and red, switch the second to purple, if both blue, switch the second to green
        palette = [genome.treatment_color_map[treat1], genome.treatment_color_map[treat2]]
        if palette[0] == palette[1]:
            if palette[0] == "red":
                palette[1] = "purple"
            elif palette[0] == "blue":
                palette[1] = "green"
        
        # Filter data for current treatment pair
        pair_df = df.filter(df["Treatment"].is_in([treat1, treat2]))
        
        # Create KDE plot for this pair
        sns.kdeplot(
            data=pair_df.to_pandas(),
            x="genome_position",
            y=motif.meth_type,
            hue="Treatment",
            ax=ax,
            alpha=0.4,
            fill=True,
            hue_order=[treat1, treat2],
            palette=palette            
        )
        
        # Turn on ticks for Y and X axis
        ax.tick_params(axis='y', which='both', left=True)
        ax.tick_params(axis='x', which='both', bottom=True)
        
        sns.move_legend(ax, "lower left")
        
        ax.set_xlabel("Genome position (bp)")
        ax.set_ylabel(f"Methylation fraction")

        ax.legend_.set_title(None)

    # Hide any unused subplots
    for idx in range(n_comparisons, len(axes)):
        axes[idx].set_visible(False)

    out_file = output_dir / f"{genome.readable_name}_whole_methylome_pairwise_KDE_{motif.readable_motif}.pdf"
    plt.savefig(out_file, bbox_inches='tight')
    print(f"Saved PDF: {out_file}")

##############################################################################
# 2) METHYLATION DISTRIBUTION PER SAMPLE
##############################################################################

def plot_motif_methylation_distribution_colwellia(motif: Motif, output_dir: Path) -> None:
    """
    Plot the distribution (histogram) of motif's methylation fraction in each sample,
    rather than the mean fraction.

    - fraction = meth / (meth + canonical)
    - We'll do a single figure with multiple histplot calls, or subplots, showing how
    many sites fall in each fraction bin per sample.
    """
    genome: Genome = motif.genome
    df = motif.data()
    
    if df is None:
        return
    
    df = df.collect().rename({"treatment": "Treatment"})
    if df.is_empty():
        return
    
    # One approach: single figure, color by treatment. Another approach: subplots per treatment.
    # Example: single figure, multiple histplot calls with "multiple='dodge'"
    _, ax = plt.subplots(figsize=(32, 20), constrained_layout=True)

    sns.histplot(
        data=df.to_pandas(),
        x=motif.meth_type,
        hue="Treatment",
        bins=20,
        multiple="dodge",
        edgecolor="white",
        ax=ax,
        kde=True,
        hue_order=sorted(df.get_column("Treatment").unique().to_list(), key=genome.treatment_order_map.get)
    )
    ax.set_xlabel(f"{readable_modification_name[motif.meth_type]} methylation fraction")
    ax.set_ylabel("Number of sites")
    ax.set_title(f"{genome.readable_name} - Distribution of {motif.motif} Methylation")

    out_file = output_dir / f"{genome.readable_name}_{motif.readable_motif}_methylation_distribution.pdf"
    plt.savefig(out_file, format="pdf") 
    plt.close()
    print(f"Saved PDF: {out_file}")


def plot_ks_heatmap_colwellia(motif: Motif, output_dir: Path) -> None:
    """
    Plot the distribution statistics (mean, median, std) of motif's methylation fraction
    in each sample.
    """
    alpha = 0.05
    genome: Genome = motif.genome

    unique_treatment_names = set(genome.treatment_name_map[t] for t in genome.default_treatments) 
    all_treatment_names = sorted(
        list(unique_treatment_names),
        key=lambda t: genome.treatment_order_map.get(t, str(t))
    )

    # Extract cycling steps and control steps from treatment names
    cycling_steps = []
    control_35ppt = []
    control_55ppt = []

    pattern = re.compile(r"(Cycling|35ppt control|55ppt control) S(\d+)")

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

    # Sort by step number
    cycling_steps.sort(key=lambda x: x[1])
    control_35ppt.sort(key=lambda x: x[1])
    control_55ppt.sort(key=lambda x: x[1])
    
    # For each cycling step, find closest steps in each control
    closest_controls = {}
    for treatment, step in cycling_steps:
        closest_controls[treatment] = {
            "35ppt control": find_closest_step(step, control_35ppt),
            "55ppt control": find_closest_step(step, control_55ppt),
        }

    # Generate pairs of treatments using the sorted, unique names
    pairs = list(tuple(sorted(x)) for x in combinations(all_treatment_names, 2))
    results = {}

    # Get Stats
    for pair_treatments in pairs:
        adj_pvals, dvals = do_ks_test(motif, pair_treatments, alpha)
        results[pair_treatments] = adj_pvals, dvals
        
    # Create DataFrame for Timeline
    timeline_data = []
    for cycling_treatment, controls in closest_controls.items():
        step_match = re.search(r'S(\d+)', cycling_treatment)
        if not step_match:
            continue
        step_num = int(step_match.group(1))
        
        for control_name, control_treatment in controls.items():
            if control_treatment is not None:
                pair = tuple(sorted([cycling_treatment, control_treatment]))
                if pair in results:
                    pvals, dvals = results[pair]
                    for stat_key in ['means', 'means_pr']:
                        dval = dvals.get(stat_key)
                        pval = pvals.get(stat_key+"_pval")
                        if pval is None:
                            continue
                        
                        timeline_data.append({
                            'Cycling Step': step_num,
                            'D-value': dval,
                            'Control': control_name,
                            'Significant': pval < alpha,
                            'Stat Key': stat_key
                        })

    timeline_df = pd.DataFrame(timeline_data)

    # Make figure
    _, axes = plt.subplots(2, 3, figsize=(30, 30), constrained_layout=True, width_ratios=[1, 0.5, 0.5])

    for i, stat_key in enumerate(["means", "means_pr"]):
        # Get axes
        dval_heatmap_ax = axes[i, 0]
        dval_timeline_ax1 = axes[i, 1]
        dval_timeline_ax2 = axes[i, 2]
        
        # Share Ys
        dval_timeline_ax1.sharey(dval_timeline_ax2)

        # Get matrices for heatmap
        pval_matrix = pd.DataFrame(np.nan, index=all_treatment_names, columns=all_treatment_names, dtype=float)
        dval_matrix = pd.DataFrame(np.nan, index=all_treatment_names, columns=all_treatment_names, dtype=float)

        for (treat1, treat2), (pval_dict, dval_dict) in results.items():            
            pval_matrix.loc[treat1, treat2] = pval_dict[stat_key+"_pval"]
            pval_matrix.loc[treat2, treat1] = pval_dict[stat_key+"_pval"]
            dval_matrix.loc[treat1, treat2] = dval_dict[stat_key]
            dval_matrix.loc[treat2, treat1] = dval_dict[stat_key]
        
        # Make pairwise heatmap
        annot = np.where(pval_matrix < alpha, dval_matrix.round(2), "X")
        
        sns.heatmap(
            dval_matrix,
            ax=dval_heatmap_ax,
            annot=annot,
            fmt="s",
            cmap="viridis",
            cbar_kws={'label': 'D-value', 'shrink': 0.8},
            vmin=0,
            # vmax=1,
            linewidths=0.5,
        )
        
        # Color tick labels based on treatment
        for tick in dval_heatmap_ax.get_xticklabels():
            treatment = tick.get_text()
            if treatment in genome.treatment_color_map:
                tick.set_color(genome.treatment_color_map[treatment])
        
        for tick in dval_heatmap_ax.get_yticklabels():
            treatment = tick.get_text()
            if treatment in genome.treatment_color_map:
                tick.set_color(genome.treatment_color_map[treatment])
        
        title_stat_key = stat_key.replace('_', ' ').title()
        if "_pr" in stat_key: # Make "Pr" -> "Promoter"
            title_stat_key = title_stat_key.replace(" Pr", " (Promoter)")
        
        dval_heatmap_ax.set_title(title_stat_key)
        
        # Make timeline
        sns.scatterplot(
            data=timeline_df[(timeline_df['Stat Key'] == stat_key) & (timeline_df['Cycling Step'].isin([1, 2]))],
            x='Cycling Step', 
            y='D-value',
            hue='Control', 
            hue_order=['35ppt control', '55ppt control'],
            palette=['blue', 'red'],
            style='Significant',
            markers={True: 'o', False: 'X'},
            s=250, alpha=0.5, ax=dval_timeline_ax1,
            legend="full"
        )
        dval_timeline_ax1.set_xlim(0.5, 2.5) # Set x-axis limits for clarity
        dval_timeline_ax1.set_xticks([1, 2]) # Define specific x-ticks
        
        # Color ticks on the left side: blue for odd, red for even
        for tick in dval_timeline_ax1.get_xticklabels():
            val = int(tick.get_text())
            if val % 2 == 1:
                tick.set_color('blue')
            else:
                tick.set_color('red')

        # Plot data for the right side (Cycling Steps 14, 15)
        sns.scatterplot(
            data=timeline_df[(timeline_df['Stat Key'] == stat_key) & (timeline_df['Cycling Step'].isin([14, 15]))],
            x='Cycling Step', 
            y='D-value',
            hue='Control', 
            hue_order=['35ppt control', '55ppt control'],
            palette=['blue', 'red'],
            style='Significant',
            markers={True: 'o', False: 'X'},
            s=250, alpha=0.5, ax=dval_timeline_ax2,
            legend="full"
        )
        dval_timeline_ax2.set_xlim(13.5, 15.5) # Set x-axis limits for clarity
        dval_timeline_ax2.set_xticks([14, 15]) # Define specific x-ticks
        
        # Turn on ticks for Y and X axis
        dval_timeline_ax1.tick_params(axis='y', which='both', left=True)
        dval_timeline_ax2.tick_params(axis='y', which='both', left=True)
        dval_timeline_ax1.tick_params(axis='x', which='both', bottom=True)
        dval_timeline_ax2.tick_params(axis='x', which='both', bottom=True)

        # Merge legends
        handles1, labels1 = dval_timeline_ax1.get_legend_handles_labels()
        handles2, labels2 = dval_timeline_ax2.get_legend_handles_labels()

        combined = dict(zip(labels1 + labels2, handles1 + handles2))
        combined_labels = list(combined.keys())
        combined_handles = list(combined.values())

        dval_timeline_ax1.get_legend().remove()
        dval_timeline_ax2.get_legend().remove()

        dval_timeline_ax2.legend(combined_handles, combined_labels, title='Legend')
        
        # Color ticks on the right side: blue for odd, red for even
        for tick in dval_timeline_ax2.get_xticklabels():
            val = int(tick.get_text())
            if val % 2 == 1:
                tick.set_color('blue')
            else:
                tick.set_color('red')

        # Hide the spines between the two plots to create a visual break
        dval_timeline_ax1.spines['right'].set_visible(False)
        dval_timeline_ax2.spines['left'].set_visible(False)
        dval_timeline_ax1.spines['top'].set_visible(False)
        dval_timeline_ax2.spines['top'].set_visible(False)
        dval_timeline_ax2.set_ylabel('') # Remove y-label for the right subplot to avoid redundancy
        dval_timeline_ax2.yaxis.set_visible(False)
        
        # Add diagonal lines to indicate the break in the axis
        d = .015  # size of diagonal lines in axes coordinates
        kwargs = dict(transform=dval_timeline_ax1.transAxes, color='k', clip_on=False)
        dval_timeline_ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)

        kwargs.update(transform=dval_timeline_ax2.transAxes)  # Switch to the right axes
        dval_timeline_ax2.plot((-d, +d), (-d, +d), **kwargs)
        
    # Save to PDF
    plt.savefig(output_dir / "motif_methylation_pvalue_heatmap.pdf", format="pdf")
    
    
# Function to find closest control step for a given cycling step
def find_closest_step(cycling_step, control_steps):
    control_step_nums = [step for _, step in control_steps]
    if not control_step_nums:
        return None
    closest_step_num = min(control_step_nums, key=lambda x: abs(x - cycling_step))
    # Find the treatment name for this step
    for treatment, step in control_steps:
        if step == closest_step_num:
            return treatment
    return None


##############################################################################
# 3) DMR HEATMAP (ROWS = sample_a, COLUMNS = sample_b, VALUES = SCORE)
##############################################################################

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


def parse_genbank(file_path="/researchdrive/gkanaan/colwellia_methylation/exp/colwellia_34h.gb"):
    """
    Parses a GenBank file and extracts features into a list of dictionaries.

    Args:
        file_path (str): The path to the GenBank file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a feature
            and contains 'contig', 'position', 'strand', 'feature_name',
            and 'feature_function'.
    """
    from Bio import SeqIO

    records = list(SeqIO.parse(file_path, "genbank"))
    features_dict = {
        'contig': [],
        'start': [],
        'stop': [],
        'strand': [],
        'gene_callers_id': [],
        'feature_name': [],
        'feature_function': []
    }
    
    i = 0
    for record in records:
        contig = record.id
        for feature in record.features:
            if feature.type != "source":  # Ignore the 'source' feature type
                start = feature.location.start.real
                end = feature.location.end.real
                strand = feature.location.strand
                feature_name = feature.qualifiers.get('gene', [''])  # Get gene name, default to empty string if not found
                feature_function = feature.qualifiers.get('product', [''])  # Get product function, default to empty string if not found

                features_dict['contig'].append(contig)
                features_dict['start'].append(start)  # INCLUSIVE
                features_dict['stop'].append(end + 1)  # INCLUSIVE + 1 = EXCLUSIVE
                features_dict['strand'].append((strand == 1))
                features_dict['gene_callers_id'].append(i)
                features_dict['feature_name'].append(feature_name[0] if feature_name else '')
                features_dict['feature_function'].append(feature_function[0] if feature_function else '')

                i += 1
    return features_dict


def write_features_with_genbank(motif: Motif) -> None:
    features = pl.from_dict(parse_genbank()).lazy()
    features = features.with_columns(pl.lit(motif.positions.collect().get_column("contig").unique().first()).alias("contig"))
    
    # Add nearest gene
    data = motif.genome.nearest_gene_to_positions(motif.positions, genes_base=features)
    
    # Add back feature_name and feature_function for gene start
    data = data.join(features, left_on="gene_callers_id_start", right_on="gene_callers_id", suffix="_start", how="left").rename({"feature_name": "feature_name_start", "feature_function": "feature_function_start"})
    data = data.join(features, left_on="gene_callers_id_end", right_on="gene_callers_id", suffix="_end", how="left").rename({"feature_name": "feature_name_end", "feature_function": "feature_function_end"})
    data = data.select("contig", "position", "strand", "gene_callers_id_start", "gene_callers_id_end",
                       "feature_name_start", "feature_function_start", "feature_name_end", "feature_function_end", "distance_to_start", "distance_to_end",)
    
    # Write to file
    out_file = motif.genome.output_dir / f"{motif.genome.readable_name}_{motif.readable_motif}_genbank_features.csv"
    data.collect().write_csv(out_file, separator=",", include_header=True)


def write_basic_stats_colwellia(genome: Genome, motifs: list[Motif]):
    text = []
    for motif in motifs:
        site_count = motif.positions.collect().height
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
        for t in treatments:
            slice_df = df.filter(pl.col("treatment") == t)
            site_count_treatment = slice_df.unique(subset=["contig", "position", "strand"]).height
            text.append(f"Number of sites with data for {t}: {site_count_treatment}\n")
        
        # Compute average fraction for the motif's methylation type
        avg_fraction = df.select(pl.col(motif.meth_type).mean()).item()
        text.append(f"Average {motif.meth_type} fraction: {avg_fraction}\n")

    text.append(f"Found motifs: {','.join([m.motif for m in motifs])}")
    
    out_file = genome.output_dir / f"{genome.readable_name}_motifs_basic_stats.txt"
    with open(out_file, "w") as f:
        f.writelines(text)
        
    print(f"Saved file: {out_file}")


def motif_distribution(motif: Motif):
    data = motif.genome.nearest_gene_to_positions(motif.positions)
    genome_length = len(motif.genome.sequence[list(motif.genome.sequence)[0]].seq)
    
    whole_genome = genome_length/motif.positions.collect().height
    
    for promoter_size in [35, 60, 100]:
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
        
        # Print
        print(f"Promoter size: {promoter_size} bp")
        print(f"  Frequency of genic motifs: {genic_length/genic} bp")
        print(f"  Frequency of intergenic motifs: {intergenic_length/intergenic} bp")
        print(f"  Frequency of promoter motifs: {promoter_length/promoter} bp")
        
        # Print test results for each agaianst each other
        genic_intergenic_p_value = chi_squared_test(
            a=genic, 
            b=intergenic, 
            L1=genic_length, 
            L2=intergenic_length
        )
        if genic_intergenic_p_value < 0.05:
            print(f"  Genic vs Intergenic: Significant difference (p-value: {genic_intergenic_p_value:.4f})")
        
        genic_promoter_p_value = chi_squared_test(
            a=genic, 
            b=promoter, 
            L1=genic_length, 
            L2=promoter_length
        )
        if genic_promoter_p_value < 0.05:
            print(f"  Genic vs Promoter: Significant difference (p-value: {genic_promoter_p_value:.4f})")
        
        intergenic_promoter_p_value = chi_squared_test(
            a=intergenic,
            b=promoter,
            L1=intergenic_length,
            L2=promoter_length
        )
        if intergenic_promoter_p_value < 0.05:
            print(f"  Intergenic vs Promoter: Significant difference (p-value: {intergenic_promoter_p_value:.4f})")
        
        
        # Print test results for each agaisnt whole genome
        genic_whole_p_value = chi_squared_test(
            a=genic, 
            b=whole_genome,
            L1=genic_length, 
            L2=genome_length
        )
        if genic_whole_p_value < 0.05:
            print(f"  Genic vs Whole Genome: Significant difference (p-value: {genic_whole_p_value:.4f})")
        
        intergenic_whole_p_value = chi_squared_test(
            a=intergenic,
            b=whole_genome,
            L1=intergenic_length,
            L2=genome_length
        )
        if intergenic_whole_p_value < 0.05:
            print(f"  Intergenic vs Whole Genome: Significant difference (p-value: {intergenic_whole_p_value:.4f})")
        
        promoter_whole_p_value = chi_squared_test(
            a=promoter,
            b=whole_genome,
            L1=promoter_length,
            L2=genome_length
        )
        if promoter_whole_p_value < 0.05:
            print(f"  Promoter vs Whole Genome: Significant difference (p-value: {promoter_whole_p_value:.4f})")

    # Around origin
    origin = motif.positions.filter(pl.col("position").le(5000) | pl.col("position").ge(genome_length - 5000)).collect().height
    print(f"Frequency of motifs within 5000 of origin: {10000/origin} bp")
    
    # Remaining genome
    non_origin = motif.positions.filter(pl.col("position").gt(5000) & pl.col("position").lt(genome_length - 5000)).collect().height
    print(f"Frequency of motifs in non-origin: {genome_length/non_origin} bp")
    
    # Test origin versus remaining genome
    origin_p_value = chi_squared_test(
        a=origin,
        b=non_origin,
        L1=10000,  # 5000 before and after origin
        L2=genome_length - 10000  # Remaining genome length
    )
    if origin_p_value < 0.05:
        print(f"Origin vs Remaining Genome: Significant difference (p-value: {origin_p_value:.4f})")
    
    
    # Test origin versus whole genome
    origin_whole_p_value = chi_squared_test(
        a=origin,
        b=whole_genome,
        L1=10000,  # 5000 before and after origin
        L2=genome_length  # Whole genome length
    )
    if origin_whole_p_value < 0.05:
        print(f"Origin vs Whole Genome: Significant difference (p-value: {origin_whole_p_value:.4f})")
    
    # Whole genome
    print(f"Frequency of motifs in the whole genome: {whole_genome} bp")
    
    # Provirus
    motifs_in_provirus = motif.positions.filter(pl.col("position").ge(889352) & pl.col("position").le(902031)).collect().height
    provirus_length = 902031 - 889352 + 1  # Inclusive range
    motifs_not_in_provirus = motif.positions.filter(pl.col("position").lt(889352) | pl.col("position").gt(902031)).collect().height
    not_provirus_length = genome_length - provirus_length
    print(f"Frequency of motifs in the provirus: {provirus_length/motifs_in_provirus} bp")
    print(f"Frequency of motifs not in the provirus: {not_provirus_length/motifs_not_in_provirus} bp")
    
    motifs_in_provirus_p_value = chi_squared_test(
        a=motifs_in_provirus,
        b=whole_genome,
        L1=provirus_length,  # Length of the provirus
        L2=genome_length  # Whole genome length
    )
    
    if motifs_in_provirus_p_value < 0.05:
        print(f"Provirus vs Whole Genome: Significant difference (p-value: {motifs_in_provirus_p_value:.4f})")
    
    provirus_not_provirus_p_value = chi_squared_test(
        a=motifs_in_provirus,
        b=motifs_not_in_provirus,
        L1=provirus_length,  # Length of the non-provirus region
        L2=not_provirus_length  # Whole genome length
    )

    if provirus_not_provirus_p_value < 0.05:
        print(f"Provirus vs Not Provirus: Significant difference (p-value: {provirus_not_provirus_p_value:.4f})")


def chi_squared_test(a, b, L1, L2):
    from scipy.stats import chi2_contingency

    observed = np.array([
        [a,       b      ],   # motif counts
        [L1 - a,  L2 - b ]    # lengths
    ])

    chi2, p_value, dof, expected = chi2_contingency(observed)

    # print(f"Chi² statistic:           {chi2:.4f}")
    # print(f"Degrees of freedom:      {dof}")
    # print(f"p‑value:                 {p_value}")
    # print("Expected frequencies:\n", expected)
    
    return p_value