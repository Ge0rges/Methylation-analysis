from code.Utilities.data_loading_polars import *
from code.Utilities.data_loading import *

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/methylation_5")
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    genome_name = "polaribacter_r-contigs"

    df = load_combined_methyl_data_for_genome(genome_name, data_dir, common_locations=False).sort_values(['name', 'sample'], ignore_index=True)
    polars_df = load_combined_methyl_data_for_genome_polars(genome_name, data_dir, common_locations=False)
    polars_df = polars_df.collect().to_pandas().sort_values(['name', 'sample'], ignore_index=True)

    # # Check they are the same
    # for (idx1, row1), (idx2, row2) in zip(df.iterrows(), polars_df.iterrows()):
    #     if not row1.equals(row2):
    #         print(f"Row {idx1} is not equal in both DataFrames: {row1} and {row2}")

    df = load_combined_methyl_data_for_genome(genome_name, data_dir, common_locations=True).sort_values(['name', 'sample'], ignore_index=True)
    polars_df = load_combined_methyl_data_for_genome_polars(genome_name, data_dir, common_locations=True)
    polars_df = polars_df.collect().sort_values(['name', 'sample'], ignore_index=True)

    # Check they are the same
    for (idx1, row1), (idx2, row2) in zip(df.iterrows(), polars_df.iterrows()):
        if not row1.equals(row2):
            print(f"Row {idx1} is not equal in both DataFrames: {row1} and {row2}")
