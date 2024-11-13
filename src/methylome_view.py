from src.objects import Genome, GeneCollection
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from platform import system
from src.utilities.utils import readable_modification_name, normalize_data_by_pileup
from utilities.utils import readable_methylation_name, barcode_replicate_map, readable_sample_name


sns.set_theme(context="poster", style="white")


def plot_methylome(genome):

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

    print(f"Data collected for methylome distribution plot  for {genome.name}")

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
    data = genome.load_all_methylation_data(normalize=False)

    # Filter to sample we want
    data = data.with_columns(pl.col('sample').replace(barcode_replicate_map).alias("Sample"))
    data = data.filter(pl.col("Sample").is_in(["top", "middle", "bottom"]))
    data = data.with_columns(pl.col('Sample').replace(readable_sample_name))

    # Coverage column
    data = data.with_columns(pl.concat_list(list(readable_modification_name.keys())).list.sum().alias("Coverage"))

    # Now normalize to fraction
    data = normalize_data_by_pileup(data)
    data = data.drop("Ncanonical")

    # Long form it
    data = data.unpivot(on=list(readable_methylation_name.keys()),
                        index=["Sample", "Coverage"],
                        variable_name="Methylation type",
                        value_name="Fraction methylated").collect(streaming=True)

    print(f"Data collected for methylome by coverage  for {genome.name}")

    # Show only coverage that is in the 90% percentile (filter outliers)
    data = data.filter(pl.col("Coverage").lt(data.get_column("Coverage").quantile(0.9)))

    # Scatter plot, by methylation type of coverage over methylation
    hue_order = [readable_sample_name["top"], readable_sample_name["middle"], readable_sample_name["bottom"]]

    for meth_type in readable_methylation_name.keys():
        df = data.filter(pl.col("Methylation type").eq(meth_type)).to_pandas()
        g = sns.jointplot(df, x="Fraction methylated", y="Coverage", hue="Sample", hue_order=hue_order, height=16, kind="scatter")
        g.fig.suptitle(f"{readable_methylation_name[meth_type]}")

        if system() == "Darwin":
            plt.show()
        else:
            plt.savefig(genome.plot_dir / f"coverage_{readable_methylation_name[meth_type]}.pdf", format="pdf")


def plot_methylation_genic_intergenic(genome: Genome):
    data = genome.gene_caller_df.select("start", "stop", "strand", "contig").sort("start", descending=False).collect(streaming=True).group_by("contig", "strand", maintain_order=True)

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
    intragenic_prop = (ranges_df.group_by("filter_contig", "filter_strand")
                       .agg((pl.col("filter_end") - pl.col("filter_start")).sum().alias("length"))
                       .select("filter_contig", "filter_strand", "length")
                       .with_columns(pl.lit("Intra-genic").alias("Region"))
                       .rename({"filter_strand": "strand", "filter_contig": "contig"})
                       .collect(streaming=True))
    genic_prop = (data.agg((pl.col("stop") - pl.col("start")).sum().alias("length"))
                  .select("contig", "strand", "length")
                  .with_columns(pl.lit("Genic").alias("Region")))
    prop_df = pl.concat([intragenic_prop, genic_prop])
    prop_df = prop_df.group_by("Region").agg(pl.col("length").sum()).sort("Region", descending=False)
    ratio = prop_df.get_column("length").to_list()
    genic_ratio = ratio[0] / (ratio[0] + ratio[1]) * 100

    # Get corresponding methylation data
    intragenic_data = genome.load_region_methylation_data(region_filter=ranges_df)
    genic_data = GeneCollection(genome.gene_ids, genome).methylation_data

    intragenic_data = intragenic_data.with_columns(pl.lit("Intra-genic").alias("Region"))
    genic_data = genic_data.with_columns(pl.lit("Genic").alias("Region"))

    data = pl.concat([intragenic_data, genic_data])

    # Wrangle dataframe
    data = data.with_columns(pl.col('sample').replace(barcode_replicate_map).alias("Sample"))
    data = data.filter(pl.col("Sample").is_in(["top", "middle", "bottom"]))
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
        plt.savefig(genome.plot_dir / "genic_intragenic.pdf", format="pdf")


if __name__ == "__main__":
    for name in Genome.valid_genome_names():
        if "metagenome" in name:
            continue

        genome = Genome(name)
        # print(f"Plotting methylome of {name}")
        # plot_methylome(genome)
        # plot_methylation_by_coverage(genome)
        plot_methylation_genic_intergenic(genome)
