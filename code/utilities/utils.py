import textwrap
import polars as pl
import utilities.data_loading as dl


readable_methylation_name = {"21839": "4mC", "a": "6mA", "m": "5mC"}

readable_sample_name = {"barcode01": "S2-1",
                        "barcode02": "S2-2",
                        "barcode03": "S2-3",
                        "barcode04": "Control",
                        "barcode05": "S3-1",
                        "barcode06": "S3-2",
                        "barcode07": "S3-3",
                        "barcode08": "S4-1",
                        "barcode09": "S4-2",
                        "barcode10": "S4-3",
                        "barcode11": "IC3-1 (30 cm)",
                        "barcode12": "IC3-2 (160 cm)",
                        "barcode13": "IC3-3 (205 cm)",
                        "barcode14": "IC3-4 (70 cm)",
                        "top": "Sackhole Top (40 cm)",
                        "middle": "Sackhole Middle (70 cm)",
                        "bottom": "Sackhole Bottom (160 cm)",
                        "control": "Control",
                        "core-40": "Ice core 40 cm",
                        "core-160": "Ice core 160 cm",
                        "core-205": "Ice core 205 cm",
                        'core-70': "Ice core 70 cm"
}

barcode_sample_map = {"barcode01": "top",
                      "barcode02": "middle",
                      "barcode03": "bottom",
                      "barcode04": "control",
                      "barcode05": "top",
                      "barcode06": "middle",
                      "barcode07": "bottom",
                      "barcode08": "top",
                      "barcode09": "middle",
                      "barcode10": "bottom",
                      "barcode11": "core-40",
                      "barcode12": "core-160",
                      "barcode13": "core-205",
                      "barcode14": "core-70",
                      "top": "top",
                      "middle": "middle",
                      "bottom": "bottom"
}


read_counts = {
    "barcode01": 1093788,
    "barcode02": 296042,
    "barcode03": 5812056,
    "barcode04": 57626,
    "barcode05": 344880,
    "barcode06": 180208,
    "barcode07": 1056185,
    "barcode08": 178883,
    "barcode09": 1776313,
    "barcode10": 1163651,
    "barcode11": 41324,
    "barcode12": 591165,
    "barcode13": 39685,
    "barcode14": 96793,
}

col34h_barcode_sample_map = {
        'barcode01': 'CTL_1',
        'barcode02': 'CTL_2',
        'barcode03': 'CTL_3',
        'barcode04': 'LN2_1',
        'barcode05': 'LN2_2',
        'barcode06': 'LN2_3',
        'barcode07': 'FREEZER_1',
        'barcode08': 'FREEZER_2',
        'barcode09': 'FREEZER_3',
        'barcode10': 'RNA_Later_1',
        'barcode11': 'RNA_Later_2',
        'barcode12': 'RNA_Later_3',
        'barcode13': 'CTL_Pellet_1',
        'barcode14': 'CTL_Pellet_2',
        'barcode15': 'CTL_Pellet_3',
        'barcode16': 'RNA_Pellet_1',
        'barcode17': 'RNA_Pellet_2',
        'barcode18': 'RNA_Pellet_3'
}


def truncate_label(label, max_length, max_lines):
    """Truncate labels to a maximum length and line count, adding an ellipsis if truncated."""

    # Hide extra alternatives
    i = 0
    result = label.split("!!!")[i]
    while i+1 < len(label.split("!!!")) and len(result + "!!!" + label.split("!!!")[i+1]) < max_length * max_lines:
        i += 1
        result += "!!!" + label.split("!!!")[i]

    result += " !!!..." if len(label.split("!!!")) > i+1 else ""

    # Wrap the text
    lines = textwrap.wrap(result, max_length, break_long_words=False)
    result = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        result += "..."
    return result


def reshape_pileup_to_matrix_polars(methyl_data) -> pl.LazyFrame:
    methyl_data = methyl_data.with_columns((pl.col('chrom') + '|' + pl.col('strand') + '|' + pl.col(
        'inclusive start position').cast(pl.Utf8) + '|' + pl.col('exclusive end position').cast(pl.Utf8)).alias('name'))

    methyl_data = methyl_data.filter(pl.col('Ndiff') < pl.col('Nvalid_cov'))

    mod_base_map = {"a": "A", "m": "C", "21839": "C"}
    methyl_data = methyl_data.with_columns(
        pl.col('modified base code and motif').replace(mod_base_map).alias('mod_group'))

    grouped = methyl_data.group_by(['name', 'mod_group']).agg(pl.max('Nvalid_cov').alias('max_valid_cov'))
    methyl_data = methyl_data.join(grouped, on=['name', 'mod_group'], how='inner').filter(
        pl.col('Nvalid_cov') == pl.col('max_valid_cov'))

    pivot_df = methyl_data.collect(streaming=True).pivot(index='name', columns='modified base code and motif',
                                                         values='Nmod', aggregate_function='first').lazy()

    pivot_df = pivot_df.join(methyl_data.select(['name', 'Ncanonical']), on='name', how='left').unique().fill_null(0)

    return pivot_df.select('name', '21839', 'a', 'm', 'Ncanonical')


def add_gene_caller_id(df: pl.LazyFrame, genes: pl.LazyFrame, strand_aware) -> pl.LazyFrame:
    """
    Aggregate methylation data by genes.

    :param df: The methylation data as a count table
    :type df: pd.Dataframe
    :param genes: The genes as a list of ranges
    :type genes: pd.Dataframe
    :param aggregate: A list of aggregation functions to apply to the columns e.g. [pl.min, pl.max, pl.mean, pl.sum]
    :type aggregate: list
    :return: The aggregated methylation data.
    :rtype: pd. Dataframe
    """

    a = df.collect().select('contig').unique().get_column('contig').to_list()
    b = genes.collect().select('contig').unique().get_column('contig').to_list()

    assert all(g1 in b for g1 in a), "Not all contigs are in this genome_name."
    del a, b

    # Get rid of any contigs not in df to limit join size
    genes = genes.filter(pl.col("contig").is_in(df.select("contig").unique().collect().get_column("contig").to_list()))

    # Merge merged_df with ranges based on conditions
    og_columns = df.collect_schema().names()
    df = df.join(genes, on='contig')

    # Filter rows where merged_df start and end values are within range start and end.
    # Gene range is inclusive of end, modkit bed is not.
    df = df.filter((pl.col('start') >= pl.col('start_right')) & (pl.col('end') <= pl.col('stop')))

    if strand_aware:
        df = df.filter(pl.col('direction') == pl.col('strand'))
    else:
        print("WARNING: Not filtering by strand. This may result in multiple gene_callers_id for the same name. Picking the first gene_callers_id.")

    # If there are still multiple gene_callers_id for the same name, pick the first one
    df = df.unique(subset=og_columns, keep="first")

    # Clean
    df = df.drop(['start_right', 'stop'])

    return df


def normalize_data_for_methylation_level(df: pl.LazyFrame, genome_name, aggregate=False) -> pl.LazyFrame:
    # Normalize to coverage
    coverages = dl.get_coverage("../data/", genome_name, agg=aggregate).drop("Genome").collect().to_dict(as_series=False)

    for key, value in coverages.items():
        coverages[key] = value[0]
        if value == 0 and key in df.select("norm_sample").unique():
            print(f"Coverage for {key} is 0")

    methylation_types = list(readable_methylation_name.keys())
    if "total_methylation" in df.collect_schema().names():
        df = df.with_columns(pl.col("total_methylation") / (pl.col('norm_sample').replace_strict(coverages).mul(len(methylation_types))))

    df = df.with_columns(pl.col(methylation_types) / pl.col('norm_sample').replace_strict(coverages))

    return df


def add_functional_annotations_polars(df: pl.LazyFrame, data_dir: str, genome_name: str) -> pl.LazyFrame:
    """
    Add functional annotations to DMRs (Differentially Methylated Regions) by matching DMR positions with
    genomic annotations to find overlaps. Drops the "partial" column from the merged DataFrame.
    """
    # Load functional annotations for the specified genome_name from a data directory
    functions = dl.get_coordinated_functions_polars(data_dir, genome_name)

    # Merge DMR data with functional annotations based on contig name
    merged_df = df.join(functions, on="gene_callers_id", how="left")

    # Fill missing values in the merged DataFrame
    merged_df = merged_df.with_columns(pl.col("function").fill_null("Unknown"), pl.col("source").fill_null("Unannotated"))

    # Remove duplicate entries from the merged DataFrame
    merged_df = merged_df.unique()

    # Ensure that all unique names from the DMRs are still present after merging
    merged_df = merged_df.collect()
    df = df.collect()
    assert set(merged_df.get_column("name").unique()) == set(df.get_column("name").unique())

    # Clean up by dropping columns that are no longer needed
    merged_df = merged_df.sort(by=['contig', 'start_right']).lazy()
    return merged_df.drop("contig", "start_right", "stop", "e_value")
