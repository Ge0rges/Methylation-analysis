import glob
import pickle
from plotting import *
from statistics import *
from data_loading import *


def get_dict_of_all_dmrs(data_dir, all_folders, dmr_dirs):
    processed_data = {}
    for dmr_dir in dmr_dirs:
        for genome_name in all_folders:
            dmr_folder = os.path.join(data_dir, genome_name, dmr_dir)
            bed_files = glob.glob(os.path.join(dmr_folder, "*.bed"))

            genome_data = get_dmr_by_sample_annotated(data_dir, genome_name, bed_files)

            if not genome_data.empty:
                key = (genome_name, dmr_dir)
                processed_data[key] = genome_data

    return processed_data


def get_dmr_by_sample_annotated(data_dir, genome_name, bed_files):
    dmrs = []

    # Get all the methylation data into one dataframe
    for bed_file in bed_files:
        dmrs_df = get_sample_from_dmr(bed_file)
        print(bed_file)
        for df in dmrs_df:
            df.reset_index(drop=True, inplace=True)

            dmrs.append(df)

    # Concatenate and remove duplicates
    dmrs = pd.concat(dmrs, ignore_index=True).drop_duplicates()

    dmrs["genome"] = genome_name

    for key in [col for col in dmrs.columns if "counts_" in col or "fractions_" in col]:
        dmrs[key].fillna(0, inplace=True)

    key_columns = ['chrom', 'start', 'end', 'sample']
    d = set(dmrs.columns)
    aggregation_dict = {col: 'first' for col in d-set(key_columns)}  # Replace 'mean' with your preferred aggregation method

    dmrs_aggregated = dmrs.groupby(key_columns).agg(aggregation_dict).reset_index()

    # Add functional annotation
    df = add_functional_annotations(dmrs_aggregated, data_dir, genome_name)

    return df


def add_functional_annotations(dmrs, data_dir, genome_name):
    functions = get_coordinated_functions(os.path.join(data_dir, genome_name))

    func_cols = [col if col != "start" else "start_y" for col in functions.columns]
    func_cols.remove("gene_callers_id")

    merged_df = pd.merge(dmrs, functions, how='left', left_on='chrom', right_on='contig')
    condition = (merged_df['start_x'] >= merged_df['start_y']) & (merged_df['end'] <= merged_df['stop'])

    merged_df.loc[~condition, func_cols] = np.nan
    merged_df.loc[~condition, "gene_callers_id"] = -1
    merged_df.loc[~condition, "source"] = "Unannotated"
    merged_df.loc[~condition, "function"] = "Unknown"

    merged_df.drop_duplicates(inplace=True)

    assert set(merged_df["name"].unique()) == set(dmrs["name"].unique())

    merged_df.drop(columns=["contig", "start_y", "stop", "version"], inplace=True)

    return merged_df



if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
    dmr_dirs = ["dmr_by_gene"] #, "dmr_by_position"]
    sources = ["KEGG", "COG"]

    # Filepath for the serialized data
    processed_data_file = os.path.join(data_dir, "methylation_df.pkl")

    try:
        # Attempt to read from the file
        with open(processed_data_file, 'rb') as file:
            methyl_data = pickle.load(file)

    except (FileNotFoundError, IOError, pickle.PickleError):
        # If reading fails, preprocess and store the results
        all_folders = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
        all_folders = ["Polaribacter_r-contigs"]
        methyl_data = get_dict_of_all_dmrs(data_dir, all_folders, dmr_dirs)

        # Save the processed data to a file
        with open(processed_data_file, 'wb') as file:
            pickle.dump(methyl_data, file)

