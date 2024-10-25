from copy import deepcopy
import polars as pl
from functools import lru_cache, cached_property
from src.Objects.genome import Genome
from src.Objects.gene_collection import GeneCollection

try:
    from _statistics import get_rao_score
except:
    pass


class Gene(object):

    def __init__(self, id: int, gene_collection: GeneCollection):
        assert id in gene_collection.ids, f"Gene ID {id} not in GeneCollection"
        self.id: int = id
        self.gene_collection: GeneCollection = deepcopy(gene_collection)
        self.gene_collection.ids = [id]
        self.gene_collection._load_data()  # Reload data for the specific gene


    @classmethod
    def from_id(cls, id: int, genome: Genome):
        collection = GeneCollection([id], genome)
        return cls(id, collection)


    @cached_property
    def methylation_data(self) -> pl.LazyFrame:
        return self.gene_collection.methylation_data.drop("gene_callers_id")


    @cached_property
    def contig(self) -> str:
        return self.gene_collection.contig.select("contig").collect(streaming=True).item()


    @cached_property
    def start(self) -> int:
        return self.gene_collection.start.select("start").collect(streaming=True).item()


    @cached_property
    def stop(self) -> int:
        return self.gene_collection.stop.select("stop").collect(streaming=True).item()


    @cached_property
    def strand(self) -> bool:
        return self.gene_collection.strand.select("strand").collect(streaming=True).item()


    @cached_property
    def source(self) -> list[str]:
        return self.gene_collection.source.select("source").collect(streaming=True).item()


    @cached_property
    def sequence(self) -> str:
        return self.gene_collection.sequence.select("sequence").collect(streaming=True).item()


    @cached_property
    def length(self) -> int:
        return self.gene_collection.length.select("length").collect(streaming=True).item()


    @cached_property
    def is_start_missing(self) -> bool:
        return self.gene_collection.is_start_missing.select("partial_begin").collect(streaming=True).item()


    @cached_property
    def is_end_missing(self) -> bool:
        return self.gene_collection.is_end_missing.select("partial_end").collect(streaming=True).item()


    @cached_property
    def start_codon_sequence(self) -> str | None:
        return self.gene_collection.start_codon_sequence.select("start_type").collect(
            streaming=True).item()


    @cached_property
    def start_codon_position(self) -> int | None:
        return self.gene_collection.start_codon_sequence.select("start_codon_position").collect(
            streaming=True).item()


    @cached_property
    def stop_codon_sequence(self) -> str | None:
        return self.gene_collection.stop_codon_sequence.select("stop_codon_sequence").collect(
            streaming=True).item()

    @cached_property
    def stop_codon_position(self) -> int | None:
        return self.gene_collection.stop_codon_position.select("stop_codon_position").collect(
            streaming=True).item()


    @cached_property
    def candidate_rbs_motifs(self) -> list[str] | None:
        return self.gene_collection.candidate_rbs_motifs.select("candidate_rbs_motifs").collect(streaming=True).item()


    @cached_property
    def rbs_motif_position(self) -> int | None:
        return self.gene_collection.rbs_motif_and_relative_position.select("rbs_motif_position").item()


    @cached_property
    def rbs_motif(self) -> str | None:
        return self.gene_collection.rbs_motif_and_relative_position.select("rbs_motif").item()


    @cached_property
    def rbs_spacer_length(self) -> tuple | None:
        return self.gene_collection.rbs_spacer_length.select("rbs_spacer_length").collect(streaming=True).item()


    @cached_property
    def gc_content(self) -> float:
        return self.gene_collection.gc_content.select("gc_cont").collect(streaming=True).item()


    @lru_cache
    def is_significantly_different_between_samples(self, samples: list[str] = None,
                                                   baseline: str | bool = False) -> bool:
        # More efficient than calling the gene collection level function
        if samples is None:
            samples = ["top", "bottom"]
        _, is_diff, _ = get_rao_score(self.methylation_data, samples, baseline)
        return is_diff


    @lru_cache
    def get_function(self, source: str | list[str] | None = None) -> str | list[str]:
        return self.gene_collection.get_function(source).select("function").collect(streaming=True).get_column("function").to_list()


    @lru_cache
    def get_flanking_sequence(self, relative_position: int, seq_range: (int, int)) -> str:
        return self.gene_collection.get_flanking_sequence(relative_position, seq_range).select("sequence").collect(streaming=True).item()


    @lru_cache
    def load_flanking_methylation_data(self, relative_position: int, meth_range: (int, int)) -> pl.LazyFrame:
        return self.gene_collection.load_flanking_methylation_data(relative_position, meth_range).drop("gene_callers_id")
