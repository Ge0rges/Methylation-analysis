from src.Barnacle.barnacle_manager import BarnacleManager
from src.utilities.data_loading import *
from src.utilities.utils import readable_methylation_name, barcode_replicate_map, normalize_data_by_pileup
from src.Barnacle.barnacle_grid_search import barnacle_grid_search
import pickle
from src.Objects.genome import Genome

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


def run_barnacle(genome_name, data_dir):
    """
    Run the barnacle analysis for a specific genome_name, DMR type, and function_source.
    """

    print(f"Starting to generate  composite for {genome_name}")

    genome = Genome(genome_name)
    methyl_data = BarnacleManager(genome).get_genome_barnacle_format_by_position().collect(streaming=True)

    # Generate cross-validation datasets
    methyl_cv_params = [methyl_data, "position", "treatment", "sample"]

    # Call barnacle grid search on it
    out = f'../data/models/{genome_name}/'
    result = barnacle_grid_search(methyl_cv_params, ["A", "B", "C"], ["position", "treatment", "methylation_type", "value"], out)

    print(result)
    with open(f"{out}/result.pickle", 'wb') as file:
            pickle.dump(result, file)

    return


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            f"../../methylation_data/methylation_5")
    run_barnacle("Pelagibacter_r-contigs", data_dir)
