from src.objects import Genome, GeneCollection
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from platform import system
from src.utilities.utils import (readable_modification_name, normalize_data_by_pileup, barcode_replicate_map,
                                 readable_sample_name, readable_methylation_name, base_methylation_map)

sns.set_theme(context="poster", style="white")


def plot_methylation_dist_by_sample_violin(genome):

    data = genome.load_all_methylation_data()

    # Preprocess the data. Sort, rename, filter, and make position absolute to genome.
    data = data.sort("strand", "contig", "position", descending=False)
    data = data.with_columns(pl.col('sample').replace(barcode_replicate_map))
    data = data.filter(pl.col("sample").is_in(["top", "middle", "bottom"]))

    # Make it presentable
    data = data.with_columns(pl.col('sample').replace(readable_sample_name))
    data = data.rename(readable_methylation_name).rename({"sample": "Sample", "strand": "Strand"})

    # Get contigs cumsum
    contigs = data.select("contig").unique().collect(streaming=True).get_column("contig")
    cum_sum = 0
    offsets = {}
    for key in contigs:
        offsets[key] = cum_sum
        cum_sum += len(genome.sequence[key])

    # Convert positiont to absolute
    data = data.with_columns(pl.col("position").add(pl.col("contig").replace_strict(offsets)).alias("Position"))

    # Long form the data
    data = (data.unpivot(on=list(readable_methylation_name.values()),
                         index=["Sample", "Position", "Strand"],
                         variable_name="Methylation type",
                         value_name="Normalized methylation fraction")
                .filter(pl.col("Normalized methylation fraction").gt(0))).collect(streaming=True).to_pandas()

    # Plot the strand in two seperate columns, one row per methylation type
    hue_order = [readable_sample_name["top"], readable_sample_name["middle"], readable_sample_name["bottom"]]
    # g = sns.relplot(data, x="Position", y="Normalized methylation fraction", col="Methylation type", row="Strand",
    #             hue="Sample", height=8, aspect=2, row_order=[True, False], hue_order=hue_order)

    # # Add vertical lines marking contigs
    # for ax in g.axes.flatten():
    #     for contig in offsets.keys():
    #         ax.axvline(x=offsets[contig], color='black', linestyle='--', alpha=0.7)


    g = sns.catplot(data, x="Sample", y="Normalized methylation fraction", col="Methylation type", height=8, aspect=2, row_order=[True, False], order=hue_order, hue="Sample", kind="violin")
    g.fig.suptitle(f"{genome.readable_name} methylome violin")

    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig(genome.plot_dir / "violin_methylome.pdf", format="pdf")

    g = sns.displot(data, x="Position", y="Normalized methylation fraction", col="Methylation type", row="Strand", height=8, hue="Sample", aspect=2, row_order=[True, False], hue_order=hue_order, kind="kde")
    g.fig.suptitle(f"{genome.readable_name} methylome KDE")

    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig(genome.plot_dir / "kde_methylome.pdf", format="pdf")


def plot_methylation_by_coverage(genome):
    data = genome.load_all_methylation_data(normalize=False, coverage=0)

    # Filter to sample we want
    data = data.with_columns(pl.col('sample').replace(barcode_replicate_map).alias("Sample"))
    data = data.filter(pl.col("Sample").is_in(["top", "middle", "bottom"]))
    data = data.with_columns(pl.col('Sample').replace(readable_sample_name))

    # Coverage column
    data = data.with_columns(pl.concat_list(list(readable_modification_name.keys())).list.sum().alias("Coverage"))

    # Now normalize to fraction
    data = normalize_data_by_pileup(data)
    data = data.drop("Ncanonical_A", "Ncanonical_C")

    # Long form it
    data = data.unpivot(on=list(readable_methylation_name.keys()),
                        index=["Sample", "Coverage"],
                        variable_name="Methylation type",
                        value_name="Fraction methylated").collect(streaming=True)

    # Show only coverage that is in the 90% percentile (filter outliers)
    data = data.filter(pl.col("Coverage").lt(data.get_column("Coverage").quantile(0.9)))

    # Scatter plot, by methylation type of coverage over methylation
    hue_order = [readable_sample_name["top"], readable_sample_name["middle"], readable_sample_name["bottom"]]

    for meth_type in readable_methylation_name.keys():
        df = data.filter(pl.col("Methylation type").eq(meth_type)).to_pandas()
        g = sns.jointplot(df, x="Fraction methylated", y="Coverage", hue="Sample", hue_order=hue_order, height=16, kind="scatter")
        g.fig.suptitle(f"{readable_methylation_name[meth_type]} coverage vs. methylation in {genome.readable_name}")

        if system() == "Darwin":
            plt.show()
        else:
            plt.savefig(genome.plot_dir / f"coverage_{readable_methylation_name[meth_type]}.pdf", format="pdf")


def plot_methylation_genic_intergenic(genome: Genome):
    data = genome.gene_caller_df.select("start", "stop", "strand", "contig").sort("start", descending=False).collect(streaming=True)
    if data.height == 0:
        print(f"No data for {genome.name}")
        return

    data = data.group_by("contig", "strand", maintain_order=True)

    ranges = {"filter_contig": [], "filter_strand": [], "filter_start": [], "filter_end": []}
    for group in data:
        group_name = group[0]
        group = group[1]
        group_ranges = []

        position = 0

        for row in group.iter_rows():
            if row[0] <= position:
                position = row[1]
                continue
            group_ranges.append((position, row[0] - 1))
            position = row[1]

        contig_length = len(genome.sequence[group_name[0]])
        if position < contig_length:
            group_ranges.append((position, contig_length))

        ranges["filter_contig"].extend([group_name[0]] * len(group_ranges))
        ranges["filter_strand"].extend([group_name[1]] * len(group_ranges))
        ranges["filter_start"].extend([r[0] for r in group_ranges])
        ranges["filter_end"].extend([r[1] for r in group_ranges])

    # Handle no gene on contig
    for contig in genome.sequence.keys():
        if not contig in ranges["filter_contig"]:
            # Include whole contig positive strand
            ranges["filter_contig"].append(contig)
            ranges["filter_strand"].append(True)
            ranges["filter_start"].append(0)
            ranges["filter_end"].append(len(genome.sequence[contig]))

            # Include whole contig negative strand
            ranges["filter_contig"].append(contig)
            ranges["filter_strand"].append(False)
            ranges["filter_start"].append(0)
            ranges["filter_end"].append(len(genome.sequence[contig]))

    ranges_df = pl.from_dict(ranges).lazy()

    # Get proportion of contig that is genic
    intergenic_prop = (ranges_df.group_by("filter_contig", "filter_strand")
                       .agg((pl.col("filter_end") - pl.col("filter_start")).sum().alias("length"))
                       .select("filter_contig", "filter_strand", "length")
                       .with_columns(pl.lit("Intra-genic").alias("Region"))
                       .rename({"filter_strand": "strand", "filter_contig": "contig"})
                       .collect(streaming=True))

    genic_prop = (data.agg((pl.col("stop") - pl.col("start")).sum().alias("length"))
                  .select("contig", "strand", "length")
                  .with_columns(pl.lit("Genic").alias("Region")))

    prop_df = pl.concat([intergenic_prop, genic_prop])
    prop_df = prop_df.group_by("Region").agg(pl.col("length").sum()).sort("Region", descending=False)
    ratio = prop_df.get_column("length").to_list()
    genic_ratio = ratio[0] / (ratio[0] + ratio[1]) * 100

    # Get corresponding methylation data
    intergenic_data = genome.load_region_methylation_data(region_filter=ranges_df)
    genic_data = GeneCollection(genome.gene_ids, genome).methylation_data

    intergenic_data = intergenic_data.with_columns(pl.lit("Intra-genic").alias("Region")).select(*readable_methylation_name.keys(), "sample", "Region")
    genic_data = genic_data.with_columns(pl.lit("Genic").alias("Region")).select(*readable_methylation_name.keys(), "sample", "Region")

    data = pl.concat([intergenic_data, genic_data])

    # Wrangle dataframe
    data = data.with_columns(pl.col('sample').replace(barcode_replicate_map).alias("Sample"))
    data = data.with_columns(pl.col('Sample').replace(readable_sample_name))

    # Long form it
    data = data.unpivot(on=list(readable_methylation_name.keys()),
                        index=["Sample", "Region"],
                        variable_name="Methylation type",
                        value_name="Fraction methylated").collect(streaming=True)
    data = data.with_columns(pl.col('Methylation type').replace(readable_methylation_name))

    # Plot
    hue_order = [readable_sample_name["top"], readable_sample_name["middle"], readable_sample_name["bottom"]]
    sns.catplot(data.to_pandas(), x="Region", y="Fraction methylated", col="Methylation type", hue="Sample", kind="bar", height=8, aspect=2, hue_order=hue_order)

    # Show genic ration in title
    plt.suptitle(f"{genome.readable_name} genic ratio: {genic_ratio:.2f}")

    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig(genome.plot_dir / "genic_intergenic.pdf", format="pdf")


def uniquely_methylated_positions(genome: Genome):
    data = genome.load_all_methylation_data(triplicates_only=True)

    # Per type
    df = []
    for meth_type in readable_methylation_name.keys():
        unique = (data.select(meth_type, "contig", "strand", "position", "sample")
                  .filter(pl.col(meth_type) > 0.95)
                  .group_by("contig", "strand", "position")
                  .agg(pl.col("sample").n_unique().alias("unique_samples"), pl.col("sample"))
                  .filter(pl.col("unique_samples").eq(1))
                  .explode(pl.col("sample"))
                  .with_columns(pl.lit(meth_type).alias("meth_type")))
        df.append(unique)

    df = pl.concat(df).collect(streaming=True)
    df = df.with_columns(pl.col("meth_type").replace(readable_methylation_name), pl.col("sample").replace(readable_sample_name))

    plt.subplots(figsize=(12, 8), layout="constrained")
    hue_order = ["S2-1", "S3-1", "S4-1", "S2-2", "S3-2", "S4-2", "S2-3", "S3-3", "S4-3"]
    custom_palette = [
        "#e63946", "#d62839", "#c2182c",  # Shades of red
        "#457b9d", "#3a6c89", "#2e5c75",  # Shades of blue
        "#2a9d8f", "#228779", "#1a7064"  # Shades of teal-green
    ]
    sns.histplot(df.to_pandas(), x="meth_type", stat="count", hue="sample", multiple="stack", hue_order=hue_order, palette=custom_palette)
    plt.title(f"Number of positions 95% methylated in only 1 treatment in {genome.readable_name}")

    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig(genome.plot_dir / "unique_methylation.pdf", format="pdf")


def always_methylated_positions(genome: Genome):
    data = genome.load_all_methylation_data(triplicates_only=True)

    # Per type
    df = []
    for meth_type in readable_methylation_name.keys():
        unique = (data.select(meth_type, "contig", "strand", "position", "sample")
                  .filter(pl.col(meth_type) > 0.95)
                  .with_columns(pl.lit(meth_type).alias("meth_type"))).select("meth_type", "sample")
        df.append(unique)

    df = pl.concat(df).collect(streaming=True)
    df = df.with_columns(pl.col("meth_type").replace(readable_methylation_name), pl.col("sample").replace(readable_sample_name))

    plt.subplots(figsize=(12, 8), layout="constrained")
    hue_order = ["S2-1", "S3-1", "S4-1", "S2-2", "S3-2", "S4-2", "S2-3", "S3-3", "S4-3"]
    custom_palette = [
        "#e63946", "#d62839", "#c2182c",  # Shades of red
        "#457b9d", "#3a6c89", "#2e5c75",  # Shades of blue
        "#2a9d8f", "#228779", "#1a7064"  # Shades of teal-green
    ]
    sns.histplot(df.to_pandas(), x="meth_type", stat="count", hue="sample", multiple="stack", hue_order=hue_order, palette=custom_palette)

    plt.title(f"Number of positions 95% methylated in all 3 treatments in {genome.readable_name}")

    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig(genome.plot_dir / "always_methylated.pdf", format="pdf")


def methylation_counts(genome: Genome):
    data = genome.load_all_methylation_data(normalize=False)
    data = data.with_columns(pl.col("sample").replace_strict(readable_sample_name))

    # Longform it
    data = data.unpivot(on=list(readable_modification_name.keys()),
                        index="sample",
                        variable_name="methylation_type",
                        value_name="methylation_count")

    data = data.group_by("sample", "methylation_type").agg(pl.col("methylation_count").sum())
    data = data.with_columns(pl.col("methylation_type").replace_strict(readable_modification_name)).collect(streaming=True)

    hue_order = ["S2-1", "S3-1", "S4-1", "S2-2", "S3-2", "S4-2", "S2-3", "S3-3", "S4-3"]
    custom_palette = [
        "#e63946", "#d62839", "#c2182c",  # Shades of red
        "#457b9d", "#3a6c89", "#2e5c75",  # Shades of blue
        "#2a9d8f", "#228779", "#1a7064"  # Shades of teal-green
    ]
    fig, ax = plt.subplots(figsize=(12, 8), layout="constrained")
    sns.barplot(data.to_pandas(), x="methylation_type", y="methylation_count", hue="sample", hue_order=hue_order, ax=ax,
                order=readable_modification_name.values(), palette=custom_palette)
    ax.set_yscale("log")
    plt.title(f"Number of methylation counts in {genome.readable_name}")

    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig(genome.plot_dir / "methylome_counts.pdf", format="pdf")


def positions_by_threshold(genome: Genome):
    df = []
    for coverage in [5, 10, 20, 30, 50, 100]:
        data = genome.load_all_methylation_data(coverage=coverage, triplicates_only=False)
        data = data.group_by("sample").agg(pl.len().alias("count")).with_columns(pl.lit(coverage).alias("coverage"))
        df.append(data)

    df = pl.concat(df).collect(streaming=True)
    df = df.with_columns(pl.col("sample").replace_strict(readable_sample_name))

    fig, ax = plt.subplots(figsize=(12, 8), layout="constrained")
    hue_order = ["S2-1", "S3-1", "S4-1", "S2-2", "S3-2", "S4-2", "S2-3", "S3-3", "S4-3"]
    custom_palette = [
        "#e63946", "#d62839", "#c2182c",  # Shades of red
        "#457b9d", "#3a6c89", "#2e5c75",  # Shades of blue
        "#2a9d8f", "#228779", "#1a7064"  # Shades of teal-green
    ]
    sns.barplot(df.to_pandas(), x="coverage", y="count", hue="sample", ax=ax, hue_order=hue_order, palette=custom_palette)
    ax.set_yscale("log")

    plt.title(f"Number of positions with coverage above threshold in {genome.readable_name}")

    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig(genome.plot_dir / "positions_by_coverage.pdf", format="pdf")

def positions_by_threshold_triplicates(genome: Genome):
    df = []
    for coverage in [5, 10, 20, 30, 50, 100]:
        data = genome.load_all_methylation_data(coverage=coverage, triplicates_only=True)
        data = data.group_by("sample").agg(pl.len().alias("count")).with_columns(pl.lit(coverage).alias("coverage"))
        df.append(data)

    df = pl.concat(df).collect(streaming=True)
    df = df.with_columns(pl.col("sample").replace_strict(readable_sample_name))

    fig, ax = plt.subplots(figsize=(12, 8), layout="constrained")
    hue_order = ["S2-1", "S3-1", "S4-1", "S2-2", "S3-2", "S4-2", "S2-3", "S3-3", "S4-3"]
    custom_palette = [
        "#e63946", "#d62839", "#c2182c",  # Shades of red
        "#457b9d", "#3a6c89", "#2e5c75",  # Shades of blue
        "#2a9d8f", "#228779", "#1a7064"  # Shades of teal-green
    ]
    sns.barplot(df.to_pandas(), x="coverage", y="count", hue="sample", ax=ax, hue_order=hue_order, palette=custom_palette)
    ax.set_yscale("log")

    plt.title(f"Number of triplicate positions with coverage above threshold in {genome.readable_name}")

    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig(genome.plot_dir / "triplicate_positions_by_coverage.pdf", format="pdf")


def positions_by_threshold_common(genome: Genome):
    df = []
    for coverage in range(5, 50, 5):
        data = genome.load_all_methylation_data(coverage=coverage, common_only=True)
        data = data.group_by("sample").agg(pl.len().alias("count")).with_columns(pl.lit(coverage).alias("coverage"))
        df.append(data)

    df = pl.concat(df).collect(streaming=True)
    df = df.with_columns(pl.col("sample").replace_strict(readable_sample_name))

    fig, ax = plt.subplots(figsize=(12, 12), layout="constrained")
    hue_order = ["S2-1", "S3-1", "S4-1", "S2-2", "S3-2", "S4-2", "S2-3", "S3-3", "S4-3"]
    custom_palette = [
        "#e63946", "#d62839", "#c2182c",  # Shades of red
        "#457b9d", "#3a6c89", "#2e5c75",  # Shades of blue
        "#2a9d8f", "#228779", "#1a7064"  # Shades of teal-green
    ]
    sns.barplot(df.to_pandas(), x="coverage", y="count", hue="sample", ax=ax, hue_order=hue_order, palette=custom_palette)
    ax.set_yscale("log")

    plt.title(f"Number of common positions with coverage above threshold in {genome.readable_name}")

    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig(genome.plot_dir / "common_positions_by_coverage.pdf", format="pdf")


def number_of_positions_switched(genome: Genome):
    data = genome.load_all_methylation_data(normalize=True, common_only=True, treatments=["top", "bottom"]).collect(streaming=True)
    if data.height == 0:
        print(f"No data for {genome.name}")
        return

    # Make a binary decision on methylation state at a positon
    data = data.with_columns(pl.col("sample").replace(barcode_replicate_map).alias("treatment"))
    data = data.group_by("contig", "strand", "position", "treatment").agg(pl.col(*readable_methylation_name.keys()))

    # If no triplicates for this type remove
    # for meth in readable_methylation_name.keys():
    #     data = data.with_columns(pl.when(pl.col(meth).list.drop_nulls().list.len() < 3).then(None).otherwise(pl.col(meth)).alias(meth))

    # Binarize
    data = data.with_columns(pl.col(*readable_methylation_name.keys()).list.mean() > 0.5)

    for key, meth_group in base_methylation_map.items():
        data = data.with_columns(pl.any_horizontal(*meth_group).alias(key))

    # Count the number of transitions between each possibility in a treatment
    data = data.sort("treatment").group_by("contig", "strand", "position", maintain_order=True).agg(
                pl.col("A").alias("A_bottom"),
                pl.col("C").alias("C_bottom"),
                pl.col("A").shift(-1).alias("A_top"),
                pl.col("C").shift(-1).alias("C_top")
    ).explode(pl.col("A_bottom"), pl.col("C_bottom"), pl.col("A_top"), pl.col("C_top"))

    # Count transitions
    Atransition_counts = data.group_by(["A_bottom", "A_top"]).len().rename({"len": "transition_count"})
    Ctransition_counts = data.group_by(["C_bottom", "C_top"]).len().rename({"len": "transition_count"})

    # Convert to pandas and then pivot
    Atransition_counts = Atransition_counts.to_pandas().pivot(index="A_bottom", columns="A_top", values="transition_count")
    Ctransition_counts = Ctransition_counts.to_pandas().pivot(index="C_bottom", columns="C_top", values="transition_count")

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(Atransition_counts, annot=True, fmt="g", cmap="coolwarm", cbar_kws={'label': 'Transition Count'})
    plt.title(f"Transition Heatmap (A_bottom vs. A_top) in {genome.readable_name}")
    plt.xlabel("A_top")
    plt.ylabel("A_bottom")

    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig(genome.plot_dir / "A_positions_switched.pdf", format="pdf")

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(Ctransition_counts, annot=True, fmt="g", cmap="coolwarm", cbar_kws={'label': 'Transition Count'})
    plt.title(f"Transition Heatmap (C_bottom vs. C_top) in {genome.readable_name}")
    plt.xlabel("C_top")
    plt.ylabel("C_bottom")
    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig(genome.plot_dir / "C_positions_switched.pdf", format="pdf")


def positions_by_methylation(genome: Genome):
    data = genome.load_all_methylation_data(normalize=True, common_only=True).collect(streaming=True)
    data = data.with_columns(pl.col("sample").replace(barcode_replicate_map).alias("treatment"))
    data = data.with_columns(pl.col("treatment").replace(readable_sample_name).alias("treatment"))
    data = data.rename(readable_methylation_name)

    if data.height == 0:
        print(f"No data for {genome.name}")
        return

    data = data.unpivot(
        index=["contig", "strand", "position", "treatment"],
        on=readable_methylation_name.values(),
        variable_name="methylation_type",
        value_name="methylation_value"

    ).filter(pl.col("methylation_value").is_not_null())

    # Set up the seaborn plot
    plt.figure(figsize=(12, 8), layout="constrained")
    hue_order = [readable_sample_name["top"], readable_sample_name["middle"], readable_sample_name["bottom"]]
    g = sns.displot(data.to_pandas(), x="methylation_value", hue="treatment", row="methylation_type", kind="hist", kde=True, stat="count", hue_order=hue_order)

    # Add titles and labels
    g.set_axis_labels("Methylation value", "Count")
    g.set_titles("{row_name} distribution")

    # Make axis log
    for ax in g.axes.flatten():
        ax.set_yscale("log")
        ax.set_ylim(1, 1e4)

    g.fig.suptitle(
        f"Methylation value distribution of common positions in {genome.readable_name}",
        y=1.02,  # Adjust the vertical position of the figure title
    )

    g.fig.subplots_adjust(top=0.9)  # Adjust subplot layout to avoid overlap

    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig(genome.plot_dir / "positions_by_methylation.pdf", format="pdf")


if __name__ == "__main__":
    import os
    from pathlib import Path

    data_path = Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/methylation_data/"))
    for methylation_path in data_path.iterdir():
        if methylation_path.is_dir():
            Genome._Genome__methylation_data_dir = methylation_path

            for name in Genome.valid_genome_names():
                if "metagenome" in name:
                    continue

                genome = Genome(name)
                print(f"Plotting methylome of {name}")
                plot_methylation_dist_by_sample_violin(genome)
                plot_methylation_by_coverage(genome)
                plot_methylation_genic_intergenic(genome)
                uniquely_methylated_positions(genome)
                always_methylated_positions(genome)
                methylation_counts(genome)
                positions_by_threshold(genome)
                positions_by_threshold_triplicates(genome)
                positions_by_threshold_common(genome)
                number_of_positions_switched(genome)
                positions_by_methylation(genome)
                print(f"Done plotting methylome of {name}")
