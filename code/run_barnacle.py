from utilities.data_loading import *
from utilities.utils import readable_methylation_name, barcode_sample_map, add_gene_caller_id, normalize_data_by_pileup
from barnacle_grid_search import barnacle_grid_search


replicate_map = {"barcode01": "A",
                 "barcode02": "A",
                 "barcode03": "A",
                 "barcode04": "control",
                 "barcode05": "B",
                 "barcode06": "B",
                 "barcode07": "B",
                 "barcode08": "C",
                 "barcode09": "C",
                 "barcode10": "C",
                 "barcode11": "D",
                 "barcode12": "D",
                 "barcode13": "D",
                 "barcode14": "D"}


def run_dmr_analysis(genome_name, data_dir):
    """
    Run the DMR analysis for a specific genome_name, DMR type, and function_source.
    """

    print(f"Starting to generate  composite for {genome_name}")

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

    # Add gene caller id
    #genes = get_genes_polars(data_dir, genome_name)
    #methyl_data = add_gene_caller_id(methyl_data, genes, True)

    # Filter samples
    methyl_data = methyl_data.with_columns(
        treatment=pl.col("sample").replace_strict(barcode_sample_map, default=pl.first()))
    methyl_data = methyl_data.filter(pl.col("treatment").is_in(["top", "middle", "bottom"]))
    # methyl_data = methyl_data.with_columns(
    #     replicate=pl.col("sample").replace_strict(replicate_map, default=pl.first()))

    # Sort by contig, strand, start, end.
    methyl_data = methyl_data.sort("strand", "contig", "start")

    # Add gene position column and use it as a dimension
    #methyl_data = methyl_data.with_columns(pl.int_range(pl.len()).over("gene_callers_id").alias("gene_position"))
    methyl_data = methyl_data.collect(streaming=True)

    # Add absolute position column
    name_map = methyl_data.select("name").unique().with_row_index("position").to_dict(as_series=False)
    name_map = dict(zip(name_map["name"], name_map["position"]))
    methyl_data = methyl_data.with_columns(position=pl.col("name").replace_strict(name_map, default=pl.first()))

    # Normalize
    # methyl_data_norm = normalize_data_by_pileup(methyl_data.lazy()).collect(streaming=True)

    # Pivot the dataframe
    methyl_data = methyl_data.unpivot(index=["position", "treatment", "sample"], #, "gene_callers_id", "gene_position"],
                                      on=methylation_types + ["Ncanonical"],
                                      variable_name="methylation_type")

    # Filter out NaNs
    methyl_data = methyl_data.filter(pl.col("value").is_not_nan())
    
    methyl_data = methyl_data.sample(100000)

    # Generate cross-validation datasets
    methyl_cv_params = [methyl_data, "treatment", "sample"]

    # Call barnacle grid search on it
    barnacle_grid_search(methyl_cv_params, ["A", "B", "C"], ["position", "treatment", "methylation_type", "value"],  '../data/models/abs/')

    return


if __name__ == "__main__":
    for coverage in ["5", "5_agg"]:
        print(f"Running barnacle analysis at coverage {coverage}")
        data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                f"../../methylation_data/methylation_{coverage}")
        for genome in os.listdir(data_dir):
            if genome == ".DS_Store":
                continue

            run_dmr_analysis(genome, data_dir)
