import itertools
import textwrap
import random
import polars as pl
import src.utilities.data_loading as dl

readable_modification_name = {"21839": "4mC", "a": "6mA", "m": "5mC", "Ncanonical": "Canonical"}
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

barcode_replicate_map = {"barcode01": "top",
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
    'barcode01': 'CTL',
    'barcode02': 'CTL',
    'barcode03': 'CTL',
    'barcode04': 'LN2',
    'barcode05': 'LN2',
    'barcode06': 'LN2',
    'barcode07': 'FREEZER',
    'barcode08': 'FREEZER',
    'barcode09': 'FREEZER',
    'barcode10': 'RNA_Later',
    'barcode11': 'RNA_Later',
    'barcode12': 'RNA_Later',
    'barcode13': 'CTL_Pellet',
    'barcode14': 'CTL_Pellet',
    'barcode15': 'CTL_Pellet',
    'barcode16': 'RNA_Pellet',
    'barcode17': 'RNA_Pellet',
    'barcode18': 'RNA_Pellet'
}

sar11_barcode_sample_map = {"barcode01": "top",
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
                            "uisw_101": "uisw_101",
                            "uisw_104": "uisw_104",
                            "uisw_106": "uisw_106",
                            "uisw_113": "uisw_113",
                            "uisw_114": "uisw_114",
                            "uisw_116": "uisw_116",
                            "uisw_121": "uisw_121",
                            "uisw_127": "uisw_127",
                            "uisw_130": "uisw_130",
                            "uisw_134": "uisw_134",
                            "uisw_136": "uisw_136",
                            "uisw_137": "uisw_137",
                            "uisw_90": "uisw_90",
                            "uisw_92": "uisw_92",
                            "uisw_94": "uisw_94"
                            }


def truncate_label(label, max_length, max_lines):
    """Truncate labels to a maximum length and line count, adding an ellipsis if truncated."""

    # Hide extra alternatives
    i = 0
    result = label.split("!!!")[i]
    while i + 1 < len(label.split("!!!")) and len(result + "!!!" + label.split("!!!")[i + 1]) < max_length * max_lines:
        i += 1
        result += "!!!" + label.split("!!!")[i]

    result += " !!!..." if len(label.split("!!!")) > i + 1 else ""

    # Wrap the text
    lines = textwrap.wrap(result, max_length, break_long_words=False)
    result = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        result += "..."
    return result


def reshape_pileup_to_matrix_polars(methyl_data) -> pl.LazyFrame | None:
    # Add a name column
    methyl_data = methyl_data.with_columns((pl.col('chrom') + '|' + pl.col('strand') + '|' + pl.col(
        'inclusive start position').cast(pl.Utf8) + '|' + pl.col('exclusive end position').cast(pl.Utf8)).alias('name'))

    # Keep only what we need
    methyl_data = methyl_data.select(
        ['name', 'modified base src and motif', 'Nvalid_cov', "Ndiff", "Nmod", "Ncanonical"])

    # Ndiff is reads with a base other than the canonical base for this modification
    methyl_data = methyl_data.filter(pl.col('Ndiff') < pl.col('Nvalid_cov'))

    # mod_base_map = {"a": "A", "m": "C", "21839": "C"}
    # methyl_data = methyl_data.with_columns(
    #     pl.col('modified base src and motif').replace(mod_base_map).alias('mod_group'))
    #
    # grouped = methyl_data.group_by(['name', 'mod_group']).agg(pl.max('Nvalid_cov'))
    #
    # methyl_data = methyl_data.join(grouped, on=['name', 'mod_group', 'Nvalid_cov'], how='inner')

    pivot_df = methyl_data.collect(streaming=True)
    if pivot_df.height == 0:
        return None

    pivot_df = pivot_df.pivot(index='name', columns='modified base src and motif', values='Nmod',
                              aggregate_function='first').lazy()
    pivot_df = pivot_df.join(methyl_data.select(['name', 'Ncanonical']), on='name', how='left').fill_null(0)

    # If there was no methylation of one type add 0s
    for meth_type in readable_modification_name.keys():
        if meth_type not in pivot_df.collect_schema().names():
            pivot_df = pivot_df.with_columns(pl.lit(0).cast(pl.Int64).alias(meth_type))

    return pivot_df.select("name", *readable_modification_name.keys())  # Select is needed to ensure order for vstack


def add_gene_caller_id(df: pl.LazyFrame, genes: pl.LazyFrame) -> pl.LazyFrame:
    """
    Add the gene caller id.
    """
    # Merge merged_df with ranges based on conditions
    # Filter rows where merged_df start and end values are within sequence_range start and end.
    # Gene sequence_range is inclusive of end, modkit bed is not.
    og_columns = df.collect_schema().names()
    result = df.join_where(genes,
                           pl.col('position').ge(pl.col('start')),
                           pl.col('position').lt(pl.col('stop')),
                           pl.col("contig").eq(pl.col("contig_right")),
                           pl.col('strand').eq(pl.col('strand_right')))

    # If there are still multiple gene_callers_id for the same name, pick the first one
    result = result.unique(subset=og_columns, keep="first")

    # Toss superfluous columns
    result = result.select(*og_columns, "gene_callers_id")

    return result


def normalize_data_by_genome_coverage(df: pl.LazyFrame, genome_name, aggregate=False) -> pl.LazyFrame:
    # Normalize to coverage
    coverages = dl.get_coverage("../data/", genome_name, agg=aggregate).drop("Genome").collect().to_dict(
        as_series=False)

    for key, value in coverages.items():
        coverages[key] = value[0]
        if value == 0 and key in df.select("norm_sample").unique():
            print(f"Coverage for {key} is 0")

    methylation_types = list(readable_methylation_name.keys())
    if "total_methylation" in df.collect_schema().names():
        df = df.with_columns(
            pl.col("total_methylation") / (pl.col('norm_sample').replace_strict(coverages).mul(len(methylation_types))))

    df = df.with_columns(pl.col(methylation_types) / pl.col('norm_sample').replace_strict(coverages))

    return df


def normalize_data_by_pileup(df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame | pl.DataFrame:
    methylation_types = list(readable_modification_name.keys())
    df = df.with_columns(pl.col(methylation_types) / pl.concat_list(methylation_types).list.sum())

    return df


def add_functional_annotations_polars(df: pl.LazyFrame, data_dir: str) -> pl.LazyFrame:
    """
    Add functional annotations to DMRs (Differentially Methylated Regions) by matching DMR positions with
    genomic annotations to find overlaps. Drops the "partial" column from the merged DataFrame.
    """
    # Load functional annotations for the specified genome_name from a data directory
    functions = dl.get_coordinated_functions_polars(data_dir).select("gene_callers_id", "function", "source")

    # Merge DMR data with functional annotations based on contig name
    merged_df = df.join(functions, on="gene_callers_id", how="left")

    # Fill missing values in the merged DataFrame
    merged_df = merged_df.with_columns(pl.col("function").fill_null("Unknown"),
                                       pl.col("source").fill_null("Unannotated"))

    # Remove duplicate entries from the merged DataFrame
    return merged_df.unique()


def generate_cross_validation_sets(df: pl.DataFrame, unique_col: str, treatmeant_col: str, sample_col: str,
                                   boot_id: int) -> pl.DataFrame:
    # Get all possible combinations of replicate_labels and treatments
    all_permutations = list(itertools.product(
        *[df.filter(pl.col(treatmeant_col).eq(group)).get_column(sample_col).unique().to_list() for group in
          df.get_column(treatmeant_col).unique().to_list()]))
    if boot_id >= len(all_permutations):
        print(f"Max bootstraps is {len(all_permutations)}")
        boot_id = random.randint(0, len(all_permutations) - 1)

    # Keep only names (positions) that are in all samples
    labels_in_all_groups = df.group_by(unique_col).agg(pl.col(sample_col).n_unique().alias("unique_groups")).filter(
        pl.col("unique_groups") == df.get_column(sample_col).n_unique()).get_column(unique_col).to_list()
    df = df.filter(pl.col(unique_col).is_in(labels_in_all_groups))

    # Get the combination of samples for this bootstrap
    combination = all_permutations[boot_id]
    df = df.filter(pl.col(sample_col).is_in(combination))
    return df
