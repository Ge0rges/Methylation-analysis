import os
from pathlib import Path
from functools import cached_property
import glob

import numpy as np
import polars as pl
from Bio import SeqIO, SeqRecord, SeqUtils

# Remove or adjust as needed to point to your local utilities
from src.utilities.utils import *
from src.utilities.data_loading import get_dataset_genes, load_methylation_data, get_pileup
from src.objects.gene_collection import GeneCollection
from src.objects.motif import Motif

import csv


class Genome(object):
    """
    A class that represents a Genome
    """

    def __init__(
        self,
        genome_path: Path,
        methylation_data_dir: Path,
        gene_calls_path: Path,
        functions_path: Path,
        barcode_treatment_sample_file: Path,
        output_dir: Path,
        default_treatments: list[str],
        coverage: int,
        treatment_info: Path,
    ):
        self.genome_path: Path = genome_path
        if not self.genome_path.is_file():
            raise FileNotFoundError(f"Genome FASTA file {self.genome_path} does not exist.")
        
        self.gene_calls_path: Path = gene_calls_path
        if not self.gene_calls_path.is_file():
            raise FileNotFoundError(f"Gene calls file {self.gene_calls_path} does not exist.")
        
        self.function_path: Path = functions_path
        if not self.function_path.is_file():
            raise FileNotFoundError(f"Function file {self.function_path} does not exist.")
        
        self.methylation_data_dir: Path = methylation_data_dir
        if not self.methylation_data_dir.is_dir():
            raise FileNotFoundError(f"Directory {self.methylation_data_dir} does not exist.")
        
        self.output_dir: Path = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        if not self.output_dir.is_dir():
            raise FileNotFoundError(f"Output directory {self.output_dir} does not exist and could not be created.")

        self.default_coverage: int = coverage
        if coverage < 1:
            raise ValueError(f"Coverage must be at least 1, not {coverage}.")
        
        self.default_treatments: list[str] = default_treatments
        if default_treatments is None:
            raise ValueError("No default treatments provided.")
        
        elif len(default_treatments) < 2:
            raise ValueError("At least two treatments must be provided.")
        
        # Load treatment information mappings
        treatment_name_map = {}
        treatment_color_map = {}
        treatment_order_map = {}
        
        with open(treatment_info, mode='r') as file:
            reader = csv.reader(file, delimiter='\t')
            for i, row in enumerate(reader):
                if row == []:
                    print(f"Warning: Empty row found in treatment info file at line {i}. Skipping.")
                    continue
                
                if i==0:
                    if row == ["treatment", "readable_treatment", "color", "order"]:
                        continue
                    else:
                        raise ValueError("Treatment info TSV file must have header 'treatment', 'readable_treatment', 'color', and 'order' at {treatment_info}.")
                
                treatment_name_map[row[0]] = row[1]
                treatment_color_map[row[1]] = row[2].replace(" ", "")
                treatment_order_map[row[1]] = row[3]
        
        self.treatment_name_map: dict[str, str] = treatment_name_map
        self.treatment_color_map: dict[str, str] = treatment_color_map
        self.treatment_order_map: dict[str, str] = treatment_order_map
        
        # Load barcode→treatment mappings
        barcode_replicate_map = {}
        barcode_treatment_map = {}

        with open(barcode_treatment_sample_file, mode='r') as file:
            reader = csv.reader(file, delimiter='\t')
            for i, row in enumerate(reader):
                if i==0:
                    if row == []:
                        print(f"Warning: Empty row found in treatment info file at line {i}. Skipping.")
                        continue
                
                    if row == ['barcode', 'treatment', 'sample']:
                        continue
                    else:
                        raise ValueError("Barcode treatment sample TSV file must have header 'barcode', 'treatment', and 'sample' at {barcode_treatment_sample_file}.")
                
                barcode_treatment_map[row[0]] = row[1]
                barcode_replicate_map[row[0]] = row[2]

        self.barcode_treatment_map: dict[str, str] = barcode_treatment_map                
        self.barcode_replicate_map: dict[str, str] = barcode_replicate_map

        # Create a "readable_name" for display
        self.readable_name: str = (genome_path.stem.capitalize().split("_r-contigs")[0] + " sp.")

    
    @cached_property
    def sequence(self) -> dict[str, SeqRecord.SeqRecord]:
        """
        Lazily load the FASTA into a dictionary of SeqRecords by contig name.
        """
        return SeqIO.index(str(self.genome_path), "fasta")


    @cached_property
    def gc_content(self):
        gcs = np.asarray([SeqUtils.gc_fraction(x.seq, ambiguous="weighted") for x in self.sequence.values()])
        return gcs.mean()
    
    
    @cached_property
    def motifs(self) -> list[Motif]:
        """
        Load Motif objects.
        """
        return Motif.load_from_modkit(genome=self, contig=None)


    @cached_property
    def gene_caller_df(self) -> pl.LazyFrame:
        return get_dataset_genes(self).filter(pl.col("gene_callers_id").is_in(self.gene_ids))


    @cached_property
    def gene_ids(self) -> list[int]:
        # This works because modkit takes a reference and then does pileup one area within that reference only.
        bed_files = [Path(f) for f in glob.glob(str(self.methylation_data_dir / "*.bed")) if '-bedgraph' not in os.path.basename(f)]

        all_data = []
        for bed_file in bed_files:
            # Load the data for the positions that overlap only
            methyl_data = get_pileup(bed_file).select("contig", "inclusive start position", "exclusive end position", "strand")
            methyl_data = methyl_data.rename({"inclusive start position": "position", "exclusive end position": "end"})

            all_data.append(methyl_data)

        all_data = pl.concat(all_data).unique()

        all_genes = get_dataset_genes(self)
        gene_ids = add_gene_caller_id(all_data, all_genes, include_intergenic=False).select("gene_callers_id").unique().collect(
            streaming=True).get_column("gene_callers_id").to_list()
        return gene_ids


    def load_methylation_data(
        self,
        in_every_treatment: bool = True,
        triplicates_only: bool = False,
        treatments: list[str] | None = None,
        region_filter: pl.Expr | pl.LazyFrame | None = None,
        normalize: bool = True,
    ) -> pl.LazyFrame | None:
        # Get bed_files
        bed_files = [
            Path(f) for f in glob.glob(str(self.methylation_data_dir / "*.bed"))
            if '-bedgraph' not in os.path.basename(f)
        ]

        df = load_methylation_data(
            self,
            bed_files=bed_files,
            in_every_treatment=in_every_treatment,
            triplicates_only=triplicates_only,
            treatments=treatments or self.default_treatments,
            region_filter=region_filter,
            normalize=normalize
        )

        if df is None:
            print(f"No data found for {self.name}")
            return None
        
        return df


    def add_genome_relative_position(self, df: pl.LazyFrame) -> pl.LazyFrame:
        # Get contigs cumsum
        contigs = list(self.sequence.keys())
        contigs.sort()
        cum_sum = 0
        offsets = {}
        for key in contigs:
            offsets[key] = cum_sum
            cum_sum += len(self.sequence[key])

        # Convert position to absolute
        return df.with_columns(pl.col("position").add(pl.col("contig").replace_strict(offsets, return_dtype=pl.UInt64)).alias("genome_position"))


    def add_sequence_around_position(self, df: pl.LazyFrame, before: int, after: int) -> pl.LazyFrame:
        # Make dataframe of the sequence of each contig
        sequences = {}
        com_sequences = {}
        for key, value in self.sequence.items():
            sequences[key] = str(value.seq)
            com_sequences[key] = str(value.seq.complement())
        sequences = {"contig": sequences.keys(), "sequence": sequences.values(),
                     "complement_sequence": com_sequences.values()}
        sequences = pl.from_dict(sequences, schema=["contig", "sequence", "complement_sequence"]).lazy()

        # Merge the data with the sequences, get the sequence of the 4 nucelotides around
        data = df.join(sequences, on="contig")
        data = data.with_columns(pl.when(pl.col("strand"))
                                 .then(pl.col("sequence").str.slice(pl.col("position") - before, before+after+1))
                                 .otherwise(pl.col("complement_sequence").str.slice(pl.col("position") - before, before+after+1))
                                 .alias("Sequence")).drop("sequence", "complement_sequence")

        return data


    def add_gene_caller_id(self, df: pl.LazyFrame, include_intergenic: bool = False) -> pl.LazyFrame:
        genes = get_dataset_genes(self)
        return add_gene_caller_id(df, genes, include_intergenic=include_intergenic)


    @cached_property
    def gene_cassettes(self) -> list[GeneCollection]:
        # Get genes with a pribnow box
        gene_collection = GeneCollection(self.gene_ids, self)
        all_genes = gene_collection.gene_caller_df.select("gene_callers_id", "contig", "start", "strand")

        # Get genes with promoters
        promoter_genes = self.genes_with_promoter.pribnow_box_position_and_sequence.select("pribnow_box_position", "gene_callers_id")
        promoter_genes = promoter_genes.with_columns(pl.col("gene_callers_id").alias("operon_id"))

        # Merge to know which gene has a promoter
        df = all_genes.join(promoter_genes, on="gene_callers_id", how="left")

        # Group by strand, and contig, then sort by start position and foward the operon ID
        df = df.sort("strand", "contig", "start", descending=False)
        df = (df.group_by("strand", "contig", maintain_order=True)
              .agg(pl.col("operon_id").forward_fill(), pl.col("gene_callers_id"))
              .explode("gene_callers_id", "operon_id"))
        df = (df.filter(pl.col("operon_id").is_not_null())  # Aggregation happens on null also
              .group_by("operon_id")
              .agg(pl.col("gene_callers_id"))
              .filter(pl.col("gene_callers_id").list.len().gt(1)))

        gcs = [GeneCollection(ids, self) for ids in df.collect(streaming=True).get_column("gene_callers_id").to_list()]

        return gcs


    @cached_property
    def genes_with_promoter(self) -> GeneCollection:
        gene_collection = GeneCollection(self.gene_ids, self)
        return GeneCollection(gene_collection.pribnow_box_position_and_sequence
                              .filter(pl.col("pribnow_box_position").is_not_null())
                              .collect(streaming=True).get_column("gene_callers_id").to_list(), self)


    def nearest_gene_to_positions(self, positions_df: pl.DataFrame) -> pl.DataFrame:
        results = []
        g = self.gene_caller_df.collect(streaming=True)

        for row in positions_df.iter_rows(named=True):  # Iterate over rows
            contig = row["contig"]
            position = row["position"]
            strand = row["strand"]


            # Compute distances
            genes = g.filter(pl.col("contig") == contig, pl.col("strand") == strand).with_columns(
                (pl.col("start") - position).abs().alias("distance_to_start"),
                (pl.col("stop") - position).abs().alias("distance_to_end"),
            )

            # Get nearest gene for start and end
            nearest_start = genes.sort("distance_to_start").head(1).rename({"gene_callers_id": "gene_callers_id_start"}).select("gene_callers_id_start", "distance_to_start")
            nearest_end = genes.sort("distance_to_end").head(1).rename({"gene_callers_id": "gene_callers_id_end"}).select("gene_callers_id_end", "distance_to_end")

            # Combine results and include original query information
            nearest_combined = pl.concat([nearest_start, nearest_end], how="horizontal")

            results.append(nearest_combined)

        # Concatenate all results into a single DataFrame
        results = pl.concat(results, how="vertical")
        
        return pl.concat([positions_df, results], how="horizontal")

