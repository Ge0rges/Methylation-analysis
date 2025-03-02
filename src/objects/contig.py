from pathlib import Path
from functools import cached_property

import polars as pl
from Bio import SeqUtils

from src.objects.genome import Genome
from src.objects.motif import Motif
from src.objects.gene_collection import GeneCollection
from src.utilities.data_loading import get_dataset_genes
from src.utilities.utils import add_gene_caller_id


class Contig:
    """
    A class that represents a single contig within a Genome
    """
    def __init__(self, parent_genome: Genome, contig_name: str, taxonomy_tsv: Path, is_viral: bool):
        # Check if contig exists in the parent genome
        if contig_name not in parent_genome.sequence.keys():
            raise ValueError(f"Contig '{contig_name}' not found in genome")
        
        # Store reference to the parent genome and contig name
        self.parent_genome = parent_genome
        self.contig_name = contig_name
        
        # Reference parent genome properties for convenience
        self.genome_path = parent_genome.genome_path
        self.methylation_data_dir = parent_genome.methylation_data_dir
        self.default_treatments = parent_genome.default_treatments
        self.default_coverage = parent_genome.default_coverage
        
        # Create a readable name for this contig
        self.readable_name = f"{parent_genome.readable_name} - Contig {contig_name}"
        
        # Load taxonomy information
        self.taxonomy_tsv = taxonomy_tsv
        
        self.is_viral = is_viral

    @cached_property
    def sequence(self):
        """
        Return just the specified contig's sequence
        """
        return self.parent_genome.sequence[self.contig_name]
    
    @cached_property
    def length(self) -> int:
        """
        Get the length of this contig
        """
        return len(self.sequence)
    
    @cached_property
    def gc_content(self) -> float:
        """
        Calculate GC content for just this contig
        """
        return SeqUtils.gc_fraction(self.sequence.seq, ambiguous="weighted")
    
    @cached_property
    def gene_ids(self) -> list[int]:
        """
        Get gene IDs only for genes on this contig
        """
        all_genes = get_dataset_genes(self.parent_genome)
        genes_in_contig = all_genes.filter(pl.col("contig") == self.contig_name)
        gene_ids = genes_in_contig.select("gene_callers_id").unique().collect(
            streaming=True).get_column("gene_callers_id").to_list()
        return gene_ids
    
    @cached_property
    def gene_caller_df(self) -> pl.LazyFrame:
        """
        Get gene caller dataframe filtered for this contig
        """
        return get_dataset_genes(self.parent_genome).filter(
            (pl.col("gene_callers_id").is_in(self.gene_ids)) & 
            (pl.col("contig") == self.contig_name)
        )
    
    
    def taxonomy(self, rank: str) -> str:
        """
        Get the taxonomy of the contig
        """
        if self.is_viral:
            # Read genomad virus summary TSV
            df = pl.read_csv(
                self.taxonomy_tsv,
                has_header=True,
                columns=[
                    "seq_name", "length", "topology", "coordinates", "n_genes", 
                    "genetic_code", "virus_score", "fdr", "n_hallmarks", 
                    "marker_enrichment", "taxonomy"
                ],
                separator="\t"
            )
            
            # GeNomad is: root;realm;kingdom;phylum;class;family;subfamily;genus;subgenus;species;
            # Taxonomy is a string like "Viruses;Duplodnaviria;Heunggongvirae;Uroviricota;Caudoviricetes;;"
            taxonomy = df.filter(pl.col("seq_name") == self.contig_name).select("taxonomy").item()
            rank_i = 9 if rank == "s" else 8 if rank == "sg" else 7 if rank == "g" else 6 if rank == "sf" else 5 if rank == "f" else 4 if rank == "c" else 3 if rank == "p" else 2 if rank == "k" else 1 if rank == "r" else 0
            
            try:
                taxonomy = taxonomy.split(";")
                if "Unclassified" in taxonomy:
                    return "Unclassified"
                else:
                    taxonomy = taxonomy[rank_i]
                    if taxonomy == "":
                        taxonomy = "Unknown at this rank"
                    return taxonomy
            except:
                print(f"No rank {rank} for contig {self.contig_name}, printing last available.")
                return ""
            
        else:
            # Parse Kaiju. Different lines have different column numbers, so go line by line.
            # Kaiju is: superkingdom,phylum,class,order,family,genus,species
            with open(self.taxonomy_tsv, "r") as f:
                for line in f:
                    line_cols = line.split("\t")
                    if line_cols[1] == self.contig_name:
                        if line_cols[0] == "U":
                            return "Unclassified"
                        else:
                            taxonomy = line_cols[7].split(";")
                            rank_i = 6 if rank == "s" else 5 if rank == "g" else 4 if rank == "f" else 3 if rank == "o" else 2 if rank == "c" else 1 if rank == "p" else 0
                            return taxonomy[rank].strip()
    
        
    def add_gene_caller_id(self, df: pl.LazyFrame, include_intergenic: bool = False) -> pl.LazyFrame:
        """
        Add gene caller IDs for positions in this contig
        """
        genes = self.gene_caller_df
        return add_gene_caller_id(df, genes, include_intergenic=include_intergenic)
    
    
    @cached_property
    def motifs(self) -> list[Motif]:
        """
        Load Motif objects.
        """
        return Motif.load_from_modkit(genome=self.parent_genome, contig=self)