from src.objects import Genome, GeneCollection
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from platform import system
from src.objects.motif import Motif
import plotly.express as px

sns.set_theme(context="poster", style="white")


def motif_methylated_frequency(genome: Genome, motif: Motif):
    # Mean the methylation fraction for each motif
    data = (motif.data
            .filter(pl.col("Treatment").is_not_null() & (pl.col(motif.meth_type).is_not_null() | pl.col(motif.canonical_base).is_not_null()))
            .filter(pl.col("Treatment").n_unique().eq(3).over("contig", "strand", "position"))
            .group_by("contig", "strand", "position", "Treatment")
            .agg(pl.col(motif.meth_type).mean()))

    data = (data.unpivot(on=motif.meth_type, index=["Treatment", "contig", "strand", "position"],
                         variable_name="Methylation type",
                         value_name="Normalized methylation fraction")
            .filter(pl.col("Normalized methylation fraction").is_not_null() & pl.col("Normalized methylation fraction").is_not_nan()))

    # For each one get counts methylated and unmethylated
    data = data.with_columns(pl.col("Normalized methylation fraction").lt(0.5).alias("Methylated"),
                             pl.col("Treatment").replace(genome._barcode_treatment_map))

    hue_order = [genome._barcode_treatment_map[x] for x in genome._default_treatments]
    plt.subplots(figsize=(16, 12))
    sns.histplot(data.to_pandas(), x="Methylated", hue="Treatment", hue_order=hue_order, stat="count", multiple="dodge", element="bars")
    plt.suptitle(f"{genome.name} methylation motifs")

    plt.tight_layout()
    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig(genome.plot_dir / "motif_methylated_frequency.pdf", format="pdf")
    plt.close()




def number_of_positions_switched(genome: Genome, motif: Motif, treatments: list[str]):
    """
    Calculates the methylation states (low, middle, high) for each position and
    plots a Parallel Categories diagram showing transitions across multiple conditions.

    Parameters
    ----------
    genome : Genome
        A Genome object with plot_dir and readable_name attributes, and other
        metadata/annotation methods as needed.
    motif : Motif
        A Motif object with `data` (Polars DataFrame) and `meth_type`.
    treatments : list[str]
        The list of  treatments to compare. For example:
        ["bottom", "top", "control"] or ["CTL", "LN2"], etc.
    """
    # 1. Filter raw data to valid rows and only the specified treatments
    #    We assume that each position must have all treatments present
    data = (
        motif.data.filter(
            pl.col("Treatment").is_in(treatments),
            (pl.col(motif.meth_type).is_not_null() | pl.col(motif.canonical_base).is_not_null())
        )
        # Over each (contig, strand, position), we want to ensure that we have
        # all of the given treatments present. So, check for n_unique == len(treatments).
        .filter(
            pl.col("Treatment").n_unique().eq(len(treatments)).over("contig", "strand", "position")
        )
        .group_by("contig", "strand", "position", "Treatment")
        .agg(pl.col(motif.meth_type).mean())
    )

    # 2. Binarize or categorize the methylation values:
    #    - "low": < 0.25
    #    - "middle": [0.25, 0.75]
    #    - "high": > 0.75
    data = data.with_columns(
        pl.when(pl.col(motif.meth_type) < 0.25)
        .then("low")
        .otherwise(
            pl.when(pl.col(motif.meth_type) > 0.75)
            .then("high")
            .otherwise("middle")
        )
        .alias("binarized_meth_type")
    )

    # 3. Pivot the data "wide" so that each condition has its own column.
    #    - index = [contig, strand, position]
    #    - columns = "Treatment"
    #    - values = "binarized_meth_type"
    data_wide = data.pivot(
        index=["contig", "strand", "position"],
        on="Treatment",
        values="binarized_meth_type"
    )

    # Convert to a pandas DataFrame for plotting
    data_wide_pd = data_wide.to_pandas().reset_index(drop=True)

    # If no positions survived, bail out
    if data_wide_pd.empty:
        print(f"No positions found with all treatments {treatments} in {motif.motif}")
        return

    # 4. Build a Parallel Categories plot with Plotly
    #    Each condition is one "axis" in the parallel categories plot.
    fig = px.parallel_categories(
        data_wide_pd,
        dimensions=treatments,  # each condition becomes one axis
        color=treatments[0],  # color by the first condition, or pick any
        color_continuous_scale=px.colors.sequential.Inferno
    )

    fig.update_layout(
        title=f"Methylation state transitions in {genome.readable_name} across conditions"
    )

    # 5. Show or save the figure depending on the OS
    if system() == "Darwin":
        fig.show()
    else:
        out_html = genome.plot_dir / f"{motif.motif}_parallel_categories.html"
        fig.write_html(str(out_html))
        print(f"Saved parallel categories plot to {out_html}")



def annotate_switched_positions(genome: Genome, motif: Motif):
    # Filter down to data that has switched
    data = (motif.data
            .filter(pl.col("Treatment").is_not_null() & (
                pl.col(motif.meth_type).is_not_null() | pl.col(motif.canonical_base).is_not_null()))
            .filter(pl.col("Treatment").n_unique().eq(3).over("contig", "strand", "position"))
            .group_by("contig", "strand", "position", "Treatment")
            .agg(pl.col(motif.meth_type).mean()))

    # Categorize values into low <25%), middle (20-80%), and high (>75%) into column binarized_meth_type
    data = data.with_columns(pl.when(pl.col(motif.meth_type).lt(0.25))
                             .then(pl.lit("low"))
                             .otherwise(pl.when(pl.col(motif.meth_type).gt(0.75))
                                        .then(pl.lit("high"))
                                        .otherwise(pl.lit("middle")))
                             .alias("binarized_meth_type"))

    # Assuming `data` is the DataFrame with the required columns
    # Filter data for "bottom" and "top" treatments
    bottom_data = data.filter(pl.col("Treatment").eq("bottom"))
    top_data = data.filter(pl.col("Treatment").eq("top"))

    # Ensure the data is aligned by some key, e.g., sample_id, to compare "bottom" and "top"
    # Here, we assume there's a common column "sample_id" to align both treatments
    aligned_data = bottom_data.join(top_data, on=["contig", "strand", "position"], suffix="_top")

    # Filter down to switched positions
    switched_positions = aligned_data.filter(pl.col("binarized_meth_type").ne(pl.col("binarized_meth_type_top")))

    if switched_positions.height == 0:
        print(f"No switched positions for {motif.motif}")
        return

    # Add function
    data = genome.add_gene_caller_id(switched_positions.lazy(), include_intergenic=True).collect(streaming=True)
    gc = GeneCollection(data.get_column("gene_callers_id").unique().to_list(), genome)
    data = data.join(gc.get_function().collect(streaming=True), on="gene_callers_id", how="left")

    # Add nearest gene if not in gene
    data = genome.nearest_gene_to_positions(data)

    # Add function of nearest genes
    gc = GeneCollection(data.get_column("gene_callers_id_start").unique().to_list(), genome)
    data = data.join(gc.get_function().collect(streaming=True), left_on="gene_callers_id_start", right_on="gene_callers_id", how="left", suffix="_start")
    gc = GeneCollection(data.get_column("gene_callers_id_end").unique().to_list(), genome)
    data = data.join(gc.get_function().collect(streaming=True), left_on="gene_callers_id_end", right_on="gene_callers_id", how="left", suffix="_end")

    # Write to CSV
    data.write_csv(genome.plot_dir / f"{motif.motif}_motif_view.csv")


if __name__ == "__main__":
    import os
    from pathlib import Path

    data_path = Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/methylation_data/"))
    for methylation_path in data_path.iterdir():
        if methylation_path.is_dir():
            Genome._Genome__methylation_data_dir = methylation_path

            for name in Genome.valid_genome_names():
                if "metagenome" in name or "brevundimonas" in name or "microbemod" in name:
                    continue

                genome = Genome(name)
                print(f"Processing {genome.name}")

                # Motif
                motifs = genome.motifs

                if len(motifs) == 0:
                    print(f"No motifs for {genome.name}")
                    continue

                for motif in motifs:
                    # Print percent of positions with no data
                    data = motif.data.get_column(motif.meth_type)
                    if len(data) == 0:
                        print(f"No data for {motif.motif} in any treatment")
                        continue

                    for treatment in motif.data.get_column("Treatment").unique().to_list():
                        if treatment is None:
                            d = motif.data.filter(pl.col("Treatment").is_null())
                        else:
                            d = motif.data.filter(pl.col("Treatment").eq(treatment))

                        if d.height == 0:
                            print(f"No data for {motif.motif} in {treatment}")
                            continue
                        print(f"Motif {motif.motif} in {treatment} has {d.null_count().get_column(motif.meth_type).item() / d.height * 100:.2f}% of positions with no data")

                    motif_methylated_frequency(genome, motif)
                    number_of_positions_switched(genome, motif, genome._default_treatments)
                    #annotate_switched_positions(genome, motif)
