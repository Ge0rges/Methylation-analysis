import glob
from plotting import *
from statistics import *
from data_loading import *


def get_dmr_by_sample_annotated(data_dir, genome_name, bed_files):
    # Get all the methylation data from the bed files
    dmrs = []
    for bed_file in bed_files:
        dmrs.append(get_dmrs(bed_file))

    # Concatenate the list of dmrs for this sample into a single dataframe
    dmrs = pd.concat(dmrs, ignore_index=True)

    # Add functional annotation
    df = add_functional_annotations(dmrs, data_dir, genome_name)

    # Dropping all other columns except the ones in columns_to_keep
    df = df.drop(columns=["name", "gene_callers_id", "direction", "partial", "call_type", "rbs_spacer", "gc_cont",
                          "start_codon", "rbs_motif", "accession", "e_value"])
    return df


def add_functional_annotations(dmrs, data_dir, genome_name):
    """
    Add functional annotations to DMRs (Differentially Methylated Regions) by matching DMR positions with
    genomic annotations to find overlaps.
    """
    # Load functional annotations for the specified genome from a data directory
    functions = get_coordinated_functions(data_dir, genome_name)

    # Rename 'start' column to 'start_y' to avoid name clash and remove 'gene_callers_id' from function columns
    func_cols = [col if col != "start" else "start_y" for col in functions.columns]
    func_cols.remove("gene_callers_id")

    # Merge DMR data with functional annotations based on contig name
    merged_df = pd.merge(dmrs, functions, how='left', left_on='chrom', right_on='contig')

    # Define a condition for DMRs that actually overlap the annotated regions
    condition = (merged_df['start_x'] >= merged_df['start_y']) & (merged_df['end'] <= merged_df['stop'])

    # For rows not meeting the condition, clear out irrelevant function columns and set default values
    merged_df.loc[~condition, func_cols] = np.nan
    merged_df.loc[~condition, "gene_callers_id"] = -1
    merged_df.loc[~condition, "source"] = "Unannotated"
    merged_df.loc[~condition, "function"] = "Unknown"

    # Remove duplicate entries from the merged DataFrame
    merged_df.drop_duplicates(inplace=True)

    # Convert the 'accession' column to string format for processing
    merged_df['accession'] = merged_df['accession'].astype(str)

    # Apply the helper function on groups defined by unique columns, and remove the temporary column afterwards
    unique_cols = ['name', 'gene_callers_id', 'source']
    merged_df = (merged_df.groupby(unique_cols, as_index=False).apply(select_best_annotation_row)
                 .drop('accession_split_len', axis=1))

    # Ensure that all unique names from the DMRs are still present after merging and that there are no duplicate groups
    assert set(merged_df["name"].unique()) == set(dmrs["name"].unique())
    assert merged_df.shape[0] == merged_df.groupby(unique_cols).ngroups, "There are duplicate groups in the result."

    # Clean up by dropping columns that are no longer needed
    merged_df.drop(columns=["contig", "start_y", "stop", "version"], inplace=True)

    return merged_df


def run_dmr_analysis(genome_name, dmr_type):
    """
    Run the DMR analysis for a specific genome, DMR type, and source.

    :param genome: Folder name of the genome.
    :type genome: str
    :param dmr_type: Either dmr_by_gene or dmr_by_position, which is also the folder name
    :type dmr_type: str
    :param source: Either KEGG or COG for the functional annotation source.
    :type source: str
    :return: Returns the methyl_data dataframe and saves a PDF of the plot.
    :rtype: pandas.DataFrame
    """
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/methylation")

    # Load the data from the bed files
    bed_files = glob.glob(os.path.join(os.path.join(data_dir, genome_name, dmr_type), "*.bed"))
    if len(bed_files) == 0:
        return

    methyl_data = get_dmr_by_sample_annotated(data_dir, genome_name, bed_files)

    # Keep only statistically significant DMRs
    methyl_data['num_tests'] = methyl_data.groupby('comparison')['comparison'].transform('count')
    methyl_data['test_result'] = methyl_data.apply(lambda x: likelihood_ratio_test(x['score'], x['num_tests']), axis=1)
    methyl_data = methyl_data[methyl_data['test_result']]

    # Plot
    plot_all_sources_heatmaps(methyl_data, genome_name, heatmap_type=dmr_type)


if __name__ == "__main__":
    # For each folder in the data directory
    for genome in os.listdir("data/methylation"):
        # Run the DMR analysis for the genome
        run_dmr_analysis(genome, "dmr_by_gene")
