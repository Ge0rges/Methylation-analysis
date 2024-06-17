import os
import glob
import polars as pl


def get_pileup_polars(path) -> pl.LazyFrame:
    """
    Read pileup data from a file.

    :param path: Path to .bed file.
    :type path: str
    :return: Dataframe of file data
    :rtype: pandas.DataFrame
    """
    pileup = pl.scan_csv(path, separator="\t", has_header=False, new_columns=["chrom", "inclusive start position", "exclusive end position", "modified base code and motif", "score", "strand", "start position2", "end position2", "color", "Nvalid_cov", "fraction modified", "Nmod", "Ncanonical", "Nother_mod", "Ndelete", "Nfail", "Ndiff", "Nnocall"])

    # Drop redundant columns
    pileup = pileup.drop(columns=["score", "start position2", "end position2", "color"])

    return pileup


def load_combined_methyl_data_for_genome_polars(genome_name, data_dir, common_locations) -> pl.LazyFrame:
    """
    Load the methyl data from every sample into a matrix.

    :param genome_name: Folder name of the genome.
    :type genome_name: str
    :param data_dir: Path to the data directory.
    :type data_dir: str
    :param common_locations: Exclude locations that are not common to all samples
    :type common_locations: bool
    :return: Dataframe of the combined methyl data.
    :rtype: pd.DataFrame
    """
    # Load the methyl_dfs from the bed files
    bed_files = glob.glob(os.path.join(os.path.join(data_dir, genome_name), "*.bed"))

    if len(bed_files) == 0:
        print(f"No pileup bed files found for {data_dir}/{genome_name}")
        return pl.LazyFrame()

    combined_methyl_data = pl.LazyFrame()
    for i, bed_file in enumerate(bed_files):
        methyl_data = get_pileup_polars(bed_file)
        methyl_data = reshape_pileup_to_matrix_polars(methyl_data, genome_name)

        # Add sample column
        sample_name = os.path.basename(bed_file).split(".")[0]
        methyl_data = methyl_data.with_columns(pl.lit(sample_name).alias('sample'))

        # Keep only common locations
        if common_locations and i > 0:
            combined_methyl_data = pl.join(combined_methyl_data, methyl_data, on="name", how="inner")
        else:
            combined_methyl_data = pl.concat([combined_methyl_data, methyl_data])

    return combined_methyl_data


def reshape_pileup_to_matrix_polars(methyl_data, genome_name) -> pl.LazyFrame:
    methyl_data = methyl_data.with_columns((pl.col('chrom') + '|' + pl.col('strand') + '|' + pl.col(
        'inclusive start position').cast(pl.Utf8) + '|' + pl.col('exclusive end position').cast(pl.Utf8)).alias('name'))

    methyl_data = methyl_data.filter(pl.col('Ndiff') < pl.col('Nvalid_cov'))

    mod_base_map = {"a": "A", "m": "C", "21839": "C"}
    methyl_data = methyl_data.with_columns(pl.col('modified base code and motif').map_dict(mod_base_map).alias('mod_group'))

    grouped = methyl_data.groupby(['name', 'mod_group']).agg(pl.max('Nvalid_cov').alias('max_valid_cov'))
    methyl_data = methyl_data.join(grouped, on=['name', 'mod_group'], how='inner').filter(pl.col('Nvalid_cov') == pl.col('max_valid_cov'))

    pivot_df = methyl_data.collect(streaming=True).pivot(index='name', columns='modified base code and motif', values='Nmod', aggregate_function='first').lazy()

    pivot_df = pivot_df.join(methyl_data.select(['name', 'Ncanonical']), on='name', how='left').unique().fill_null(0)

    return pivot_df.select('name', '21839', 'a', 'm', 'Ncanonical')
