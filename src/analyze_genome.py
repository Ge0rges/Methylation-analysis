import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path

from src.objects.genome import Genome
from src.objects.gene_collection import GeneCollection
from src.objects.motif import Motif
from src.utilities.utils import treatment_weighted_mean, readable_modification_name, create_methylation_bins
import numpy as np

##############################################################################
# 1) WHOLE METHYLOME (unchanged logic)
##############################################################################

def plot_whole_methylome(
    genome: Genome,
    motif: Motif,
    output_dir: Path,
) -> None:
    """
    Plot the whole methylome (only for the motif’s methylation type) across samples:
      - fraction = motif_type / (motif_type + canonical base)
      - x-axis = genome_position, color by sample
    """
    df = motif.data().collect(streaming=True)
    
    # Order the contigs by their mean methylation
    ordered_contigs = None
    if "0_9_3" in genome.genome_path.stem:
        ordered_contigs = df.group_by("contig").agg(pl.col(motif.meth_type).mean().alias("mean_meth")).sort("mean_meth").get_column("contig").to_list()
        ordered_contigs = [
        "c_000000069174", "c_000000073274", "c_000000091796", "c_000000032055",
        "c_000000070176", "c_000000071756", "c_000000077392", "c_000000091831",
        "c_000000028731", "c_000000071011", "c_000000070925", "c_000000070918",
        "c_000000070876", "c_000000070869", "c_000000070843", "c_000000070810",
        "c_000000070403", "c_000000070398", "c_000000071322", "c_000000077581",
        "c_000000077510", "c_000000070967", "c_000000091805"]

    df = genome.add_genome_relative_position(df, order=ordered_contigs).rename({"treatment": "Treatment"})

    if df.is_empty():
        return
    
    _, axes = plt.subplots(2, 1, figsize=(24, 12), constrained_layout=True)
    hue_order = sorted(df.get_column("Treatment").unique().to_list(), key=genome.treatment_order_map.get)

    for i, ax in enumerate(axes):
        sns.scatterplot(
            data=df.filter(pl.col("strand").eq(i==0)).to_pandas(),
            x="genome_position",
            y=motif.meth_type,
            hue="Treatment",
            ax=ax,
            s=6,
            alpha=1,
            hue_order=hue_order,
            palette=[genome.treatment_color_map[treatment] for treatment in hue_order]
        )
        
        # Add vertical lines marking ori and term
        # Ori:537,590 Ter:152,210 if genome is g__pelagibacter_0_9_3
        if "0_9_3" in genome.genome_path.stem:
            ax.axvline(537590, color="green", linestyle="--", linewidth=5)
            ax.axvline(152210, color="red", linestyle="--", linewidth=5)
            
        for j, treatment in enumerate(hue_order):
            # Filter data for this treatment
            treatment_df = df.filter(df["Treatment"] == treatment).to_pandas()
            
            # Group by genome position and calculate mean for this treatment
            avg_df = treatment_df.groupby("genome_position")[motif.meth_type].mean().reset_index()
            
            if len(avg_df) > 4:  # Need at least 5 points for a quartic fit
                # Fit polynomial regression
                x = avg_df["genome_position"].values
                y = avg_df[motif.meth_type].values
                z = np.polyfit(x, y, 4)
                p = np.poly1d(z)
                
                # Add R-squared value and p value of fit
                r2 = np.corrcoef(x, p(x))[0, 1] ** 2
                ax.text(0.05, 0.95-j*0.1, f"R^2: {r2:.2f}", transform=ax.transAxes, fontsize=10) 
                ax.text(0.05, 0.90-j*0.1, f"p: {z}", transform=ax.transAxes, fontsize=10)
                
                # Generate smooth curve with more points
                x_smooth = np.linspace(x.min(), x.max(), 300)
                y_smooth = p(x_smooth)
                
                # Plot the polynomial regression line using the treatment's color
                ax.plot(x_smooth, y_smooth, color=genome.treatment_color_map[treatment], linewidth=2) 

        ax.set_xlabel("Genome position (bp)")
        ax.set_ylabel(f"Fraction of {readable_modification_name[motif.meth_type]} methylation")
        ax.set_title(f"{genome.readable_name} - {motif.motif} Methylome - Strand: {['+', '-'][i]}")

    out_file = output_dir / f"{genome.readable_name}_whole_methylome_{motif.readable_motif}.pdf"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved PDF: {out_file}")


##############################################################################
# 2) METHYLATION DISTRIBUTION PER SAMPLE
##############################################################################

def plot_motif_methylation_distribution(
    genome: Genome,
    motif: Motif,
    output_dir: Path,
) -> None:
    """
    Plot the distribution (histogram) of motif's methylation fraction in each sample,
    rather than the mean fraction.

    - fraction = meth / (meth + canonical)
    - We'll do a single figure with multiple histplot calls, or subplots, showing how
      many sites fall in each fraction bin per sample.
    """
    df = motif.data().collect(streaming=True).rename({"treatment": "Treatment"})
    
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
    ax.set_ylabel("Count of sites")
    ax.set_title(f"{genome.readable_name} - Distribution of {motif.motif} Methylation")

    out_file = output_dir / f"{genome.readable_name}_{motif.readable_motif}_methylation_distribution.pdf"
    plt.savefig(out_file, format="pdf")
    plt.close()
    print(f"Saved PDF: {out_file}")


##############################################################################
# 3) DMR HEATMAP (ROWS = sample_a, COLUMNS = sample_b, VALUES = SCORE)
##############################################################################

def plot_dmr_scores_heatmap(
    genome: Genome,
    motif: Motif,
    output_dir: Path
) -> None:
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
    
    # Make a distribution plot of DMR scores
    score_df = dmr.to_pandas()
    _, ax = plt.subplots(figsize=(16, 10), constrained_layout=True)
    sns.histplot(data=score_df, x="score", bins=20, kde=True, ax=ax)
    ax.set_yscale("log")
    ax.set_ylim(1, 1e3)
    ax.set_title(f"Distribution of DMR Scores - {genome.readable_name} ({motif.motif})")
    ax.set_xlabel("DMR Score")
    ax.set_ylabel("Count")
    out_file_dist = output_dir / f"{genome.readable_name}_{motif.motif}_dmr_distribution.pdf"
    plt.savefig(out_file_dist, format="pdf")
    plt.close()
    print(f"Saved PDF: {out_file_dist}")
    
    # Compute mean score per (sample_a, sample_b)
    pdf = (dmr.group_by(["treatment_a", "treatment_b"]).agg(pl.col("score").mean().alias("mean_dmr_score"))).to_pandas()

    # Make sure to unify sample names if needed, e.g. barcode -> treatment
    # If you want to rename sample_a, sample_b via genome.barcode_treatment_map:
    pdf["treatment_a"] = pdf["treatment_a"].replace(genome.barcode_treatment_map).replace(genome.treatment_name_map)
    pdf["treatment_b"] = pdf["treatment_b"].replace(genome.barcode_treatment_map).replace(genome.treatment_name_map)

    # Create pivot with sample_a as rows, sample_b as columns
    pivot = pdf.pivot(index="treatment_a", columns="treatment_b", values="mean_dmr_score")

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    sns.heatmap(pivot, cmap="viridis", annot=True, fmt=".2f", ax=ax)

    ax.set_title(f"DMR Score Heatmap - {genome.readable_name} ({motif.motif})")
    ax.set_xlabel("Treatment B")
    ax.set_ylabel("Treatment A")

    out_file = output_dir / f"{genome.readable_name}_{motif.readable_motif}_dmr_heatmap.pdf"
    plt.savefig(out_file, format="pdf")
    plt.close()
    print(f"Saved PDF: {out_file}")


##############################################################################
# 4) PARALLEL CATEGORIES WITH CLEARER BIN LABELS & COLOR
##############################################################################

def plot_parallel_categories_methylation(
    genome: Genome,
    motif: Motif,
    output_dir: Path,
    bins: int = 3
) -> None:
    """
    Create a Plotly parallel categories plot with clearer bin labels ("Low", "Medium", "High") 
    and color. For instance, we color by the first dimension's value.
    """
    df = motif.data().collect(streaming=True)
    treatments = df.get_column("treatment").unique().to_list()
        
    if df.is_empty():
        return
    
    # Pivot by (contig, position, strand), columns = treatment, values = fraction_meth
    pivoted = df.pivot(
        index=["contig", "position", "strand"],
        on="treatment",
        values=motif.meth_type,
    )

    # Bin data
    cut_points, bin_labels = create_methylation_bins(bins)
    pivoted = pivoted.with_columns(pl.col(treatment).cut(cut_points, labels=bin_labels) for treatment in treatments).to_pandas()

    cat_to_num = {key: i for i, key in enumerate(bin_labels)}
    pivoted["meth_numeric"] = pivoted[treatments[0]].map(cat_to_num)

    sorted_treatments = sorted(treatments, key=genome.treatment_order_map.get)
    fig = px.parallel_categories(
        pivoted,
        dimensions=sorted_treatments,
        color="meth_numeric",
        color_continuous_scale=[
            (0.0, "blue"),
            (0.5, "orange"),
            (1.0, "red"),
        ],
        range_color=[0, 2]
    )
    
    fig.update_layout(title=f"Methylation state transitions in {genome.readable_name} across conditions", coloraxis_showscale=False)
    
    out_file = output_dir / f"{genome.readable_name}_{motif.readable_motif}_parallel_categories.html"
    fig.write_html(str(out_file))
    print(f"Saved HTML: {out_file}")

     

def extract_motif_data_all_transitions(
    genome: Genome,
    motif: Motif,
    bins: int = 3,
) -> None:
    """
    Extract all motif data for each unique methylation
    transition across treatments. For each unique transition found in the data, this function
    outputs a separate CSV file (with gene annotations).

    The binning of methylation values is performed using a shared helper function to ensure
    consistency with the parallel categories plot.

    Parameters:
      genome: Genome object containing barcode/treatment maps and output info.
      motif: Motif object that provides the full motif data.
      bins: The number of bins used for methylation values (default: 3).

    Returns:
      None. (CSV files are written to genome.output_dir.)
    """
    # 1. Collect the full motif data and map samples to treatments.
    df = motif.data().collect(streaming=True)
    if df.is_empty():
        return

    # Add a composite key for later filtering.
    df = df.with_columns(
        pl.concat_str(
            [pl.col("contig").cast(str),
             pl.col("position").cast(str),
             pl.col("strand").cast(str)],
            separator="_"
        ).alias("composite_key")
    )

    # 2. Pivot the data so that each row (identified by contig, position, strand)
    #    has one column per treatment with methylation values given by motif.meth_type.
    pivoted = df.pivot(
        index=["contig", "position", "strand"],
        on="treatment",
        values=motif.meth_type,
    )

    # 3. Determine the unique treatments and sort them using genome.treatment_order_map.
    treatments = df["treatment"].unique().to_list()
    sorted_treatments = sorted(treatments, key=genome.treatment_order_map.get)

    # 4. Create bins for the methylation values using the helper function.
    cut_points, bin_labels = create_methylation_bins(bins)

    # 5. Bin the methylation values in each treatment column.
    pivoted = pivoted.with_columns(
        [pl.col(treatment).cut(cut_points, labels=bin_labels) for treatment in treatments]
    )

    # 6. Identify every unique transition present in the data.
    unique_transitions_df = pivoted.select(sorted_treatments).unique()

    # 7. For each unique transition, filter the data and output a CSV.
    result = []
    for transition_row in unique_transitions_df.iter_rows(named=True):
        # Build the transition tuple in the sorted treatment order.
        transitions = tuple(transition_row[col] for col in sorted_treatments)
        
        # Skip this for loop if the transitions has a None element
        if None in transitions:
            continue
        
        # Build the filter condition.
        condition = pl.lit(True)
        for col, val in zip(sorted_treatments, transitions):
            if val is None:
                condition &= pl.col(col).is_null()
            else:
                condition &= (pl.col(col) == val)
        
        filtered = pivoted.filter(condition)
        if filtered.is_empty():
            continue

        # 10. Annotate with gene information.
        data = genome.add_gene_caller_id(filtered.lazy(), include_intergenic=True).collect(streaming=True)
        gc = GeneCollection(data.get_column("gene_callers_id").unique().to_list(), genome)
        data = data.join(gc.get_function().collect(streaming=True), on="gene_callers_id", how="left")
        
        # Check if we can continue
        out_file = genome.output_dir / f"{genome.readable_name}_{motif.readable_motif}_motif_transition_{'_'.join(transitions)}.csv"
        if data.is_empty():
            # Print "No data for this transition" to output file
            with open(out_file, "w") as f:
                f.write(f"No data for transition {transitions}")
            print(f"Saved CSV for transition {transitions}: {out_file}")
            continue
        
        data = genome.nearest_gene_to_positions(data)
        gc = GeneCollection(data.get_column("gene_callers_id_start").unique().to_list(), genome)
        data = data.join(
            gc.get_function().collect(streaming=True),
            left_on="gene_callers_id_start",
            right_on="gene_callers_id",
            how="left",
            suffix="_start"
        )
        gc = GeneCollection(data.get_column("gene_callers_id_end").unique().to_list(), genome)
        data = data.join(
            gc.get_function().collect(streaming=True),
            left_on="gene_callers_id_end",
            right_on="gene_callers_id",
            how="left",
            suffix="_end"
        )

        # 11. Write the annotated data to CSV. The filename encodes the transition pattern.
        data.write_csv(out_file)
        print(f"Saved CSV for transition {transitions}: {out_file}")
        result.append(data)
    
    return pl.concat(result)


##############################################################################
# 5) EXTRACT DIFF METHYLATED GENES (WITH GENE FUNCTION)
##############################################################################

def extract_diff_methylated_genes(
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
    data = genome.nearest_gene_to_positions(data)

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


def write_basic_stats(genome: Genome, motifs: list[Motif]):
    text = []
    for motif in motifs:
        site_count = motif.positions.unique(subset=["contig", "position", "strand"]).collect().height
        text.append(f"Number of sites: {site_count}\n")

        # Compute weighted fraction using treatment_weighted_mean
        df = motif.data().collect(streaming=True)
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


def extract_consensus_genes(genome: Genome, trans: pl.DataFrame, dmrs: pl.DataFrame, motif: Motif):
    # Get the intersection
    dmrs = dmrs.select("contig", "position", "strand", "score", "balanced_map_pvalue", "balanced_effect_size", "treatment_a", "treatment_b")
    consensus = trans.join(dmrs, how="inner", on=["contig", "position", "strand"])
    consensus.write_csv(genome.output_dir / f"{genome.readable_name}_{motif.readable_motif}_innerjoin_dmr_trans_genes.csv")