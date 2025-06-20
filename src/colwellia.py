import math 
import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

from utilities.utils import do_ks_test
from src.objects.gene_collection import GeneCollection
from src.objects.genome import Genome
from src.objects.motif import Motif
from src.utilities.utils import readable_modification_name

sns.set_theme(context="paper", style="whitegrid")
pl.enable_string_cache()

def plot_number_of_positions_by_coverage_colwellia(genome: Genome, motif: Motif, output_dir: Path) -> None:
    """
    Plot the number of positions by coverage for a given motif across all samples.
    """
    original_cov = genome.default_coverage
    
    # Get data for different coverages
    coverage_counts = []
    for cov in [1, 5, 10, 20, 30, 50, 100]:
        genome.default_coverage = cov
        df = motif.data(in_every_treatment=False).collect()
        df = df.group_by("treatment").agg(pl.struct(["contig", "strand", "position"]).unique().count().alias("count"))
        df = df.with_columns(pl.lit(cov).alias("coverage"))
        coverage_counts.append(df)
    
    coverage_counts = pl.concat(coverage_counts)
    
    # Plot barplot using seaborn
    _, ax = plt.subplots(figsize=(16, 16))
    sns.barplot(
        data=coverage_counts.to_pandas(),
        x="treatment",
        y="count",
        hue="coverage",
        legend='full',
        ax=ax,
        order=sorted(coverage_counts.get_column("treatment").unique().to_list(), key=genome.treatment_order_map.get)
    )
    
    plt.title("Not in every treatment")    
    plt.xticks(rotation=90, ha='right')
    
    plt.savefig(output_dir / f"{genome.readable_name}_{motif.readable_motif}_coverage_counts_noteverytreatment.pdf", format="pdf")
    
    genome.default_coverage = original_cov  # Reset to original coverage


def plot_whole_methylome_colwellia(genome: Genome, motif: Motif, output_dir: Path) -> None:
    """
    Plot the whole methylome (only for the motif's methylation type) across samples:
    - fraction = motif_type / (motif_type + canonical base)
    - x-axis = genome_position, color by sample
    - Creates separate subplots for each pairwise treatment comparison
    """
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
            palette=[genome.treatment_color_map[treat1], genome.treatment_color_map[treat2]]            
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

        ax.legend_.set_title(None)

    # Hide any unused subplots
    for idx in range(n_comparisons, len(axes)):
        axes[idx].set_visible(False)

    out_file = output_dir / f"{genome.readable_name}_whole_methylome_pairwise_{motif.readable_motif}.pdf"
    plt.savefig(out_file, bbox_inches='tight')
    print(f"Saved PDF: {out_file}")


##############################################################################
# 2) METHYLATION DISTRIBUTION PER SAMPLE
##############################################################################

def plot_motif_methylation_distribution_colwellia(genome: Genome, motif: Motif, output_dir: Path) -> None:
    """
    Plot the distribution (histogram) of motif's methylation fraction in each sample,
    rather than the mean fraction.

    - fraction = meth / (meth + canonical)
    - We'll do a single figure with multiple histplot calls, or subplots, showing how
    many sites fall in each fraction bin per sample.
    """
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


def plot_motif_distribution_stats_colwellia(genome: Genome, motif: Motif, output_dir: Path) -> None:
    """
    Plot the distribution statistics (mean, median, std) of motif's methylation fraction
    in each sample.
    """
    alpha = 0.05

    unique_treatment_names = set(genome.treatment_name_map.values())
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
            vmax=1,
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
            legend=False
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
            s=250, alpha=0.5, ax=dval_timeline_ax2
        )
        dval_timeline_ax2.set_xlim(13.5, 15.5) # Set x-axis limits for clarity
        dval_timeline_ax2.set_xticks([14, 15]) # Define specific x-ticks

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

def plot_dmr_scores_heatmap_colwellia(genome: Genome, motif: Motif, output_dir: Path) -> None:
    """
    Plot a heatmap of DMR scores for the specified motif across sample comparisons:
      - rows = sample_a
      - columns = sample_b
      - values = mean_dmr_score
    Each motif -> separate PDF.
    """
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
    genome: Genome,
    motif: Motif,
    top_n: int = 10
) -> pl.DataFrame:
    """
    Extract the top differentially methylated locations (by DMR 'score') for a motif,
    attach nearest/overlapping gene, and include the gene function in the final CSV.

    Returns a Polars DataFrame.
    """
    top_rows = motif.dmr_data.collect(streaming=True).sort("score", descending=True)
    if top_rows.is_empty():
        print(f"No DMR data for motif {motif.motif}")
        return

    # Now, we want to add the gene function from some table. Typically:
    # 1) Add gene_caller_id for direct hits
    # 2) For start/end, do the same and join gene function columns

    # Let's do 1) add gene_caller_id if needed
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
    if top_n > 0 and data.height > top_n:
        data = data.sort("score", descending=True).head(top_n)
    
    # Write to CSV
    output_file = genome.output_dir / f"{genome.readable_name}_{motif.readable_motif}_top_diff_genes.csv"
    data.write_csv(output_file)
    
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
    features_list = []
    for record in records:
        contig = record.id
        for feature in record.features:
            if feature.type != "source":  # Ignore the 'source' feature type
                start = feature.location.start.position
                end = feature.location.end.position
                strand = feature.location.strand
                feature_name = feature.qualifiers.get('gene', [''])  # Get gene name, default to empty string if not found
                feature_function = feature.qualifiers.get('product', [''])  # Get product function, default to empty string if not found
                features_list.append({
                    'contig': contig,
                    'start': start,
                    'end': end,  # INCLUSIVE
                    'strand': strand,
                    'feature_name': feature_name[0] if feature_name else '', # Access first element or empty
                    'feature_function': feature_function[0] if feature_function else '' # Access first element or empty
                })
    return features_list


def write_basic_stats_colwellia(genome: Genome, motifs: list[Motif]):
    text = []
    for motif in motifs:
        site_count = motif.positions.unique(subset=["contig", "position", "strand"]).collect().height
        text.append(f"Number of sites: {site_count}\n")

        # Compute weighted fraction using treatment_weighted_mean
        df = motif.data()
        if df is None:
            return 
        
        df = df.collect()
        
        if df.is_empty():
            continue
        
        avg_fraction = df.select(pl.col(motif.meth_type).mean()).item()
        treatments = df.get_column("treatment").unique().to_list()
        for t in treatments:
            slice_df = df.filter(pl.col("treatment") == t)
            site_count_treatment = slice_df.unique(subset=["contig", "position", "strand"]).height
            text.append(f"Number of sites with data for {t}: {site_count_treatment}\n")
        text.append(f"Average {motif.meth_type} fraction: {avg_fraction}\n")

    text.append(f"Found motifs: {','.join([m.motif for m in motifs])}")
    
    out_file = genome.output_dir / f"{genome.readable_name}_motifs_basic_stats.txt"
    with open(out_file, "w") as f:
        f.writelines(text)
        
    print(f"Saved file: {out_file}")
