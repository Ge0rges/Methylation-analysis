from utilities.plotting import *
from _statistics import *
from utilities.data_loading import *
from utilities.data_loading_polars import *
from utilities.utils import group_and_normalize_data_for_methylation_level


def get_dmr_by_sample_annotated(data_dir, genome_name, bed_files):
    # Get all the methylation data from the bed files
    dmrs = []
    for bed_file in bed_files:
        dmrs.append(get_dmrs(bed_file))

    # Concatenate the list of dmrs for this sample into a single dataframe
    dmrs = pd.concat(dmrs, ignore_index=True)

    # Handle empty
    if dmrs.empty:
        return dmrs

    # Add functional annotation
    df = add_functional_annotations(dmrs, data_dir, genome_name)

    # Dropping all other columns except the ones in columns_to_keep
    #df = df.drop(columns=["name", "gene_callers_id", "direction", "call_type", "rbs_spacer", "gc_cont",
    df = df.drop(columns=["name", "gene_callers_id", "accession"])
    return df


def add_functional_annotations(dmrs, data_dir, genome_name):
    """
    Add functional annotations to DMRs (Differentially Methylated Regions) by matching DMR positions with
    genomic annotations to find overlaps. Drops the "partial" column from the merged DataFrame.
    """
    # Load functional annotations for the specified genome_name from a data directory
    functions = get_coordinated_functions(data_dir, genome_name)

    # Rename 'start' column to 'start_y' to avoid name clash and remove 'gene_callers_id' from function columns
    func_cols = [col if col != "start" else "start_y" for col in functions.columns]
    func_cols.remove("gene_callers_id")

    # Merge DMR data with functional annotations based on contig name
    merged_df = pd.merge(dmrs, functions, how='left', left_on='chrom', right_on='contig')

    # Define a condition for DMRs that actually overlap the annotated regions
    condition = (merged_df['start_x'] >= merged_df['start_y']) & (merged_df['end'] <= merged_df['stop'])

    # For rows not meeting the condition, clear out irrelevant function columns and set default values
    merged_df.loc[~condition, func_cols] = pd.NA
    merged_df.loc[~condition, "gene_callers_id"] = -1
    merged_df.loc[~condition, "source"] = "Unannotated"
    merged_df.loc[~condition, "function"] = "Unknown"

    # Remove duplicate entries from the merged DataFrame
    merged_df.drop_duplicates(inplace=True)

    # Convert the 'accession' column to string format for processing l
    merged_df['accession'] = merged_df['accession'].astype(str)

    # Apply the helper function on groups defined by unique columns, and remove the temporary column afterwards
    unique_cols = ['name', 'gene_callers_id', 'source']
    merged_df = (merged_df.groupby(unique_cols, as_index=False)
                 .apply(select_best_annotation_row, include_groups=False)
                 .drop('accession_split_len', axis=1))

    # Ensure that all unique names from the DMRs are still present after merging and that there are no duplicate groups
    assert set(merged_df["name"].unique()) == set(dmrs["name"].unique())
    assert merged_df.shape[0] == merged_df.groupby(unique_cols).ngroups, "There are duplicate groups in the result."

    # Clean up by dropping columns that are no longer needed
    merged_df.drop(columns=["contig", "start_y", "stop", "e_value"], inplace=True)

    return merged_df


def run_dmr_analysis(genome_name, dmr_type, coverage, data_dir, fig_savepath="plots"):
    """
    Run the DMR analysis for a specific genome_name, DMR type, and source.

    :param genome_name: Folder name of the genome_name.
    :type genome_name: str
    :param dmr_type: Either dmr_by_gene or dmr_by_position, which is also the folder name
    :type dmr_type: str
    :param source: Either KEGG or COG for the functional annotation source.
    :type source: str
    :return: Returns the methyl_data dataframe and saves a PDF of the plot.
    :rtype: pandas.DataFrame
    """

    # Load the data from the bed files
    bed_files = glob.glob(os.path.join(os.path.join(data_dir, genome_name, dmr_type), "*.bed"))
    bed_files = [file for file in bed_files if not file.endswith('-bedgraph.bed')]
    if len(bed_files) == 0:
        print(f"No DMR bed files found for {genome_name}")
        return

    dmr_data = get_dmr_by_sample_annotated(data_dir, genome_name, bed_files)

    # Handle empty
    if dmr_data.empty:
        print(f"No DMRs found for {genome_name}")
        return

    # Keep only statistically significant DMRs
    dmr_data['num_tests'] = dmr_data.groupby('comparison')['comparison'].transform('count')
    dmr_data['test_result'] = dmr_data.apply(lambda x: modkit_llr(x['score'], x['num_tests']), axis=1)
    dmr_data = dmr_data[dmr_data['test_result']]

    # Handle empty
    if dmr_data.empty:
        print(f"No stastistically significant DMRs found for {genome_name}")
        return

    # Get ax_heatmap data
    dmr_data = dmr_data[dmr_data["source"] == "KEGG_Module"]

    # Get methylation level data
    methyl_data = load_combined_methyl_data_for_genome_polars(genome_name, data_dir, common_locations=False)
    genes = get_genes(data_dir, genome_name)[['contig', 'start', 'stop']].drop_duplicates()
    methyl_data = group_and_normalize_data_for_methylation_level(methyl_data, genes, genome_name, ("agg" in coverage))

    # Filter samples
    methyl_data = methyl_data.filter(pl.col("sample").is_in(["top", "middle", "bottom"]))
    dmr_data = dmr_data[dmr_data["comparison"].isin(["top_VS_bottom"])]

    # Create figure and subplots
    methylation_types = methyl_data.columns[1:4]
    n_types = len(methylation_types)
    fig, axes = plt.subplots(n_types * 3, 1, figsize=(20, 10 * n_types), sharex=False, layout="constrained",
                             gridspec_kw={'height_ratios': [1, 2, 7] * n_types})

    for i, methylation_type in enumerate(methylation_types):
        ax_heatmap = axes[i*3]
        ax_top = axes[i*3+1]
        ax_bottom = axes[i*3+2]

        # Merge DMR and methyl data
        methyl_data = methyl_data.with_columns(
            chrom=pl.col('name').str.split(by='|').list.get(0),
            start=pl.col('name').str.split(by='|').list.get(2).cast(pl.UInt32),
            stop=pl.col('name').str.split(by='|').list.get(3).cast(pl.UInt32)
        )
        composite_data = methyl_data.join(pl.from_pandas(dmr_data), on='chrom')

        composite_data = composite_data.filter(
            (pl.col('start') >= pl.col('start_x')) & (pl.col('stop') <= pl.col('end')))

        # Keep the dmr_data rows with top 10 score
        composite_data = composite_data.unique(subset=["function", "gene_id", "sample"]).select(pl.all().top_k_by("score", 10).over("sample", mapping_strategy="explode"))

        plot_gene_methylation_level(ax_top, ax_bottom, methyl_data, methylation_type, composite=True)
        plot_heatmap(composite_data.to_pandas(), ax_heatmap, "KEGG_Module", fig=fig, composite=True)
        annotate_heatmap_to_meth_level(fig, ax_bottom, ax_heatmap, composite_data, methylation_type)

    # Save the figure
    cleaned_genome_name = genome_name.title().replace("_R-Contigs", " sp.")
    fig.suptitle(f"Mean gene methylation and DMR score by type for {cleaned_genome_name}", fontsize=26)
    plt.savefig(f"{fig_savepath}/{genome_name}_{coverage}_composite.svg", format='svg')

    print(f"Done plotting composite for {genome_name}")
    return


if __name__ == "__main__":
    print("Running DMR analysis at coverage 5 agg")
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/methylation_data/methylation_5_agg")
    for genome in os.listdir(data_dir):
        run_dmr_analysis(genome, "dmr_by_gene", "5_agg", data_dir, fig_savepath="../plots/plots_5_agg")
