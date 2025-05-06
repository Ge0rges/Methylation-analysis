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
        treatment_name_map, treatment_color_map, treatment_order_map = parse_treatment_tsv(treatment_info)
        
        self.treatment_name_map: dict[str, str] = treatment_name_map
        self.treatment_color_map: dict[str, str] = treatment_color_map
        self.treatment_order_map: dict[str, str] = treatment_order_map
        
        # Load barcode mappings
        barcode_treatment_map, barcode_sample_map = parse_barcode_tsv(barcode_treatment_sample_file)

        self.barcode_treatment_map: dict[str, str] = barcode_treatment_map                
        self.barcode_sample_map: dict[str, str] = barcode_sample_map

        # Create a "readable_name" for display
        self.readable_name: str = ". ".join([s.capitalize() for s in genome_path.stem.split("__")[0:2]])

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


    def add_genome_relative_position(self, df: pl.LazyFrame, order=None) -> pl.LazyFrame:
        # Get contigs cumsum
        contigs = list(self.sequence.keys())
        contigs.sort()
        cum_sum = 0
        offsets = {}
        
        if order: # Concatenate the contigs as indicated
            for key in order:
                offsets[key] = cum_sum
                cum_sum += len(self.sequence[key])
        else:   
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
    

    def nearest_gene_to_positions(self, positions_df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Finds the nearest gene (by start and end) for each position using join_asof
        for improved memory efficiency.

        Args:
            positions_df: LazyFrame with columns 'contig', 'position', 'strand'.
                          Must contain at least these columns.

        Returns:
            LazyFrame with original position info plus nearest gene IDs and distances.
            Columns added: 'gene_callers_id_start', 'distance_to_start',
                           'gene_callers_id_end', 'distance_to_end'.
            Note: If no matching gene is found on the contig/strand, the corresponding
                  gene ID and distance columns will contain nulls.
        """
        # --- Input Validation and Preparation ---
        original_pos_cols = positions_df.collect_schema().names() # Capture original columns

        # Handle potential existing gene_callers_id column in positions_df
        temp_gene_id_col = "gene_callers_id"
        if "gene_callers_id" in original_pos_cols:
            # Use a more unique temporary name to avoid potential clashes later
            temp_gene_id_col = "_input_gene_callers_id_"
            positions_df = positions_df.rename({"gene_callers_id": temp_gene_id_col})
            # Update original_cols list to reflect the rename
            original_pos_cols = [temp_gene_id_col if c == "gene_callers_id" else c for c in original_pos_cols]


        # Add a unique ID to restore original order at the end if needed
        # Also sort positions for join_asof
        positions_prep = positions_df.sort("contig", "strand", "position")

        genes_base = self.gene_caller_df.select(
            "contig", "strand", "gene_callers_id", "start", "stop"
        )

        # --- Perform join_asof (Nearest Start) ---
        # Sort genes by 'start' for the first join
        genes_sorted_by_start = genes_base.sort("contig", "strand", "start").rename({
                "gene_callers_id": "gene_callers_id_start",
                "start": "_gene_start_", # Keep gene start/stop for distance calc
                "stop": "_gene_stop_for_start_" # Not strictly needed unless used later
        })

        # Join to find the gene with the start position nearest to the query position
        joined_start = positions_prep.join_asof(
            # Rename columns from the gene table *before* the join to avoid conflicts
            genes_sorted_by_start,
            left_on="position",         # The column to match proximity on
            right_on="_gene_start_",
            by=["contig", "strand"], # Exact match required on these columns
            strategy="nearest",    # Find the single nearest row
        )

        # --- Perform join_asof (Nearest End/Stop) ---
        # Sort genes by 'stop' for the second join
        genes_sorted_by_stop = genes_base.sort("contig", "strand", "stop").rename({
                "gene_callers_id": "gene_callers_id_end",
                "start": "_gene_start_for_stop_", # Not strictly needed
                "stop": "_gene_stop_" # Keep gene stop for distance calc
        })

        # Join the intermediate result to find the gene with the 'stop' position nearest
        # to the query position.
        joined_both = joined_start.join_asof(
            genes_sorted_by_stop,
            left_on="position",         # The column to match proximity on
            right_on="_gene_stop_",            
            by=["contig", "strand"],
            strategy="nearest",
        )

        # --- Calculate Distances and Finalize ---
        result = joined_both.with_columns(
            # Calculate distance to the start of the gene found nearest by *start*
            (pl.col("_gene_start_") - pl.col("position")).abs().alias("distance_to_start"),
            # Calculate distance to the end of the gene found nearest by *stop*
            (pl.col("_gene_stop_") - pl.col("position")).abs().alias("distance_to_end"),
        )

        # Define final columns to select (original + new ones)
        # Use the potentially renamed original columns list
        final_output_cols = list(original_pos_cols) + [
            "gene_callers_id_start",
            "gene_callers_id_end",
            "distance_to_start",
            "distance_to_end"
        ]

        # Select, cast, and reorder columns
        result_final = result.select(*final_output_cols).with_columns([
                # Cast to desired final types. strict=False allows nulls from joins.
                pl.col("gene_callers_id_start").cast(pl.Int32, strict=False),
                pl.col("gene_callers_id_end").cast(pl.Int32, strict=False),
                pl.col("distance_to_start").cast(pl.Int32, strict=False),
                pl.col("distance_to_end").cast(pl.Int32, strict=False)
            ])

        # Rename the original gene_callers_id back if it was temporarily renamed
        if temp_gene_id_col != "gene_callers_id":
            result_final = result_final.rename({temp_gene_id_col: "gene_callers_id"})

        return result_final
