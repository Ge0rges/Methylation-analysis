from __future__ import annotations
from typing import TYPE_CHECKING
from functools import cached_property, lru_cache
import itertools
import polars as pl
import Bio.Data.IUPACData as bd

from src.utilities.utils import methylation_base_map, readable_methylation_name
from src.utilities.data_loading import load_methylation_data

if TYPE_CHECKING:  # Only for type hints
    from genome import Genome
    from contig import Contig


class Motif(object):

    def __init__(self, genome: Genome, contig: Contig, motif: str):
        self.genome: Genome = genome
        self.motif: str = motif
        self.meth_type: str = ""
        self.offset: int = -1
        self.frac_mod: float = -1
        self.high_count: int = -1
        self.low_count: int = -1
        self.mid_count: int = -1
        self.contig: Contig = contig
        self.readable_motif: str = None
        
        # Set the path
        if contig is None:
            self.motif_data_path = genome.methylation_data_dir / "motifs" / motif
        else:
            self.motif_data_path = genome.methylation_data_dir / contig.contig_name / motif

        if not self.motif_data_path.exists():
            raise FileNotFoundError(f"Motif data directory not found: {self.motif_data_path}")

        self.strings: list[str] = generate_possible_sequences(motif)
        
    @cached_property
    def readable_motif(self):
        return self.motif[:self.offset] + f"[{readable_methylation_name[self.meth_type]}]" + self.motif[self.offset+1]

    @cached_property
    def canonical_base(self):
        return f"Ncanonical_{methylation_base_map[self.meth_type]}"

    @classmethod
    def load_from_modkit(cls, genome: Genome, contig: Contig) -> list[Motif]:
        motifs_path = genome.methylation_data_dir / "motifs" / f"{genome.default_coverage}_motifs.tsv"
        if contig is not None:
            motifs_path = genome.methylation_data_dir / "motifs" / f"{contig.default_coverage}_{contig.contig_name}_motifs.tsv"
        
        if not motifs_path.exists():
            raise FileNotFoundError(f"Motif list file not found at {motifs_path}, check coverage parameter")
        
        try:
            motifs = pl.read_csv(str(motifs_path), separator="\t", has_header=True)
        except pl.exceptions.NoDataError as e:
            return []

        # Create a Motif object for each row
        motif_objs = []
        for row in motifs.iter_rows(named=True):
            motif = Motif(genome, contig, row["motif"])

            # Header: mod_code	motif	offset	frac_mod	high_count	low_count	mid_count
            motif.meth_type = str(row["mod_code"])
            motif.motif = row["motif"]
            motif.offset = row["offset"]
            motif.frac_mod = row["frac_mod"]
            motif.high_count = row["high_count"]
            motif.low_count = row["low_count"]
            motif.mid_count = row["mid_count"]

            motif_objs.append(motif)

        return motif_objs


    @cached_property
    def positions(self) -> pl.LazyFrame:
        # Load the positions of the methylated nucleotide of this motif
        positions_bed = self.motif_data_path / "location.bed"
        if not positions_bed.is_file():
            raise FileNotFoundError(f"Motif positions BED file not found: {positions_bed}")
        
        positions = pl.scan_csv(str(positions_bed), separator="\t", has_header=False, new_columns=["contig", "start", "end", "name", "score", "strand"]).with_columns((pl.col("strand") == "+").alias("strand"), pl.col("start").alias("position"))
        return positions

    @lru_cache
    def data(self, normalize=True) -> pl.LazyFrame:
        # Get all the data for the motif by loading all bed files except "location.bed" in the data directory
        
        # Gather every .bed file except 'location.bed'
        bed_files = [f for f in self.motif_data_path.glob("*.bed") if f.name != "location.bed"]
        if len(bed_files) == 0:
            raise FileNotFoundError(f"No data files found for motif {self.motif}")

        df = load_methylation_data(
            genome=self.genome,
            bed_files=bed_files,
            in_every_treatment=True,
            triplicates_only=False,
            treatments=self.genome.default_treatments,
            normalize=normalize
        )

        if df is None:
            print(f"No data found for motif {self.motif}")
            return None
        
        return df

    @cached_property
    def dmr_data(self) -> pl.LazyFrame:
        # Check folder exists
        dmr_dir = self.motif_data_path / "dmr"
        if not dmr_dir.is_dir():
            raise FileNotFoundError(f"DMR directory not found for motif {self.motif}")

        # Load DMR from each bed file formatted SAMPLE1_SAMPLE2.bed
        all_data = []
        for bed_file in dmr_dir.glob("*.bed"):
            sample_a = bed_file.stem.split("_")[0]
            sample_b = bed_file.stem.split("_")[-1]
            assert len(bed_file.stem.split("_")) == 3, f"DMR bed file name should be formatted as SAMPLE1_vs_SAMPLE2.bed, found {bed_file.stem}"

            # Load DMR data
            try:
                dmr_data = (pl.scan_csv(str(bed_file), separator="\t", has_header=False, 
                                        new_columns=["contig", "position", "end", "name", "score", "strand", 
                                                     "sample_a counts", "sample_a total",  
                                                    "sample_b counts", "sample_b total", "sample_a percents", 
                                                    "sample_b percents", "sample_a fraction modified", "sample_b fraction modified"], 
                                        schema_overrides={"contig": pl.String, "position": pl.Int64, "end": pl.Int64, 
                                                          "name": pl.String, "score": pl.Float64, "strand": pl.String})
                            .with_columns((pl.col("strand") == "+").alias("strand"),
                                           pl.lit(sample_a).alias("treatment_a"), 
                                           pl.lit(sample_b).alias("treatment_b"))
                            .select("contig", "position", "strand", "score", "treatment_a", "treatment_b"))
                
                # Filter such that both treatments are in the requested ones
                dmr_data = dmr_data.filter(pl.col("treatment_a").is_in(self.genome.default_treatments), 
                                           pl.col("treatment_b").is_in(self.genome.default_treatments))
                all_data.append(dmr_data)

            except Exception as e:
                # If the file reads "Error! not enough datapoints, got" then ignore the error
                with open(bed_file, "r") as f:
                    if "Error! not enough datapoints, got" in f.read():
                        continue
                
                # Unexpected issue happened
                print(f"Encountered an error reading the CSV: {e} file was {bed_file}")
                continue
            
        if len(all_data) == 0:
            print(f"No DMR data found for motif {self.motif}")
            return pl.DataFrame(schema={"contig": pl.String, "position": pl.Int64, "strand": pl.Boolean, "score": pl.Float64, "treatment_a": pl.String, "treatment_b": pl.String}).lazy()

        dmr_data = pl.concat(all_data)
    
        return dmr_data


def generate_possible_sequences(seq):
    """return list of all possible sequences given an ambiguous DNA input"""
    d = bd.ambiguous_dna_values
    return list(map("".join, itertools.product(*map(d.get, seq))))
