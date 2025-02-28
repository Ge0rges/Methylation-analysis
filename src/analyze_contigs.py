import polars as pl
import matplotlib.pyplot as plt
from src.objects.contig import Contig
from src.objects.motif  import Motif
from src.utilities.utils import treatment_weighted_mean, readable_modification_name


def plot_contig_motif_heatmap(contigs: list[Contig]):
    """
    Make a heatmap where each row is a contig. Each column is a motif string. 
    Show the average methylation fraction of that motif, in that contig, across all treatments.
    This is done by having one motif occur as many columns as there are treatments. 
    Draw a line across the X axis which highlights the sample.
    Color the y-ticks (contig names) by the taxonomy("g") of the contig
    """

    # Create a dataframe with columns: contig_name treatment motif_string methylation_fraction contig_taxonomy
    data = []
    for contig in contigs:
        for motif in contig.motifs:
            motif_df = motif.data(normalize=False).collect(streaming=True)
            motif_df = motif_df.with_columns(pl.col("sample").replace_strict(contig.parent_genome.barcode_treatment_map).replace_strict(contig.parent_genome.treatment_name_map).alias("treatment"))
            motif_df = treatment_weighted_mean(motif_df).rename({"treatment": "Treatment"})
            
            for treatment in motif_df.get_column("treatment").unique():
                methylation_fraction = motif_df.filter(pl.col("treatment") == treatment).select(pl.col(motif.meth_type)).mean()
                
                data.append({
                    "contig_name": contig.contig_name,
                    "treatment": treatment,
                    "motif_string": motif.motif,
                    "methylation_fraction": methylation_fraction,
                    "contig_taxonomy": contig.taxonomy("g)")
                })
        
    df = pl.DataFrame(data)
    
    
    