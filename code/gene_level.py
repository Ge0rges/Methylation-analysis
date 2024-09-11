from utilities.plotting import *
from _statistics import *
from utilities.data_loading import *
from utilities.utils import normalize_data_for_methylation_level, add_gene_caller_id, \
    add_functional_annotations_polars, readable_methylation_name, readable_sample_name, barcode_sample_map
from scipy.stats import rankdata


def run_analysis(genome_name, data_dir, fig_savepath="plots"):
    """
    Run the DMR analysis for a specific genome_name, DMR type, and function_source.
    """

    print(f"Starting to generate  composite for {genome_name}")

    # Get the genes
    genes = get_genes_polars(data_dir, genome_name)

    # Get methylation level data
    methylation_types = list(readable_methylation_name.keys())
    methyl_data = load_combined_methyl_data_for_genome_polars(genome_name, data_dir).select("name", "sample",
                                                                                            *methylation_types,
                                                                                            "Ncanonical")
    methyl_data = methyl_data.with_columns(
        contig=pl.col('name').str.split(by='|').list.get(0),
        strand=pl.col('name').str.split(by='|').list.get(1),
        start=pl.col('name').str.split(by='|').list.get(2).cast(pl.UInt32),
        end=pl.col('name').str.split(by='|').list.get(3).cast(pl.UInt32)
    )

    # Filter samples
    methyl_data = methyl_data.with_columns(pl.col("sample").replace_strict(barcode_sample_map, default=pl.first()))
    methyl_data = methyl_data.filter(pl.col("sample").is_in(["top", "middle", "bottom"]))

    # Create the total methylation column and normalize values
    methyl_data = normalize_data_for_methylation_level(methyl_data)
    methyl_data = methyl_data.with_columns(pl.concat_list(methylation_types).list.sum().alias("total_methylation"))

    # Add the gene_caller_id
    methyl_data = add_gene_caller_id(methyl_data, genes, True)

    # Add functional annotation
    methyl_data = add_functional_annotations_polars(methyl_data, data_dir, genome_name).collect(streaming=True)

    # Add rao score - Doing this first prevents row duplication issues
    methyl_data = add_rao_score_by_gene(methyl_data, ["top", "bottom"], baseline=False)

    for type in methylation_types+["total_methylation"]:
        # Mean together all the different methylation types
        top = pl.col(type).filter(pl.col('sample').eq('top'))
        bot = pl.col(type).filter(pl.col('sample').eq('bottom'))
        methyl_data = methyl_data.replace_column(methyl_data.get_column_index(type), methyl_data.group_by('gene_id').agg(top.mean() - bot.mean()).get_column(type))


    # Write the dataframe to a CSV
    methyl_data = methyl_data.select("gene_callers_id", "source", "function", *methylation_types, "total_methylation", "rao_score", "test_result")
    methyl_data.write_csv(f"../data/gene_level_data/{genome_name}_all_gene_level.csv")

    # Now take only significant RAO's and order by the total methylation
    methyl_data = methyl_data.filter(pl.col("test_result") == True).sort("total_methylation", descending=True)
    methyl_data.write_csv(f"../data/gene_level_data/{genome_name}_rao-filtered_gene_level.csv")

    # Now only those who have a function of interest
    # methyl_data = methyl_data.filter(pl.col("accession").is_in([]))
    # methyl_data.write_csv(f"../data/gene_level_data/{genome_name}_function-filtered_gene_level.csv")

    print(f"Done writing genes for {genome_name}")
    return


if __name__ == "__main__":
    for coverage in ["5"]:
        print(f"Running gene_detail analysis at coverage {coverage}")
        data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"../data/methylation_data/methylation_{coverage}")
        for genome in os.listdir(data_dir):
            if genome == ".DS_Store":
                continue

            run_analysis(genome, data_dir, fig_savepath=f"../plots/plots_{coverage}")
