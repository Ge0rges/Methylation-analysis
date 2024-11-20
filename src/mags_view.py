import os

import matplotlib.pyplot as plt

from src.objects.genome import Genome
from src.objects.gene_collection import GeneCollection
import seaborn as sns
import pandas as pd
import polars as pl
from platform import system

sns.set_theme(context="poster", style="white")


def plot_mags_by_gc_content(genomes: list[Genome]):
    # Get GC content for each genome
    dataframe = {"MAG": [g.name for g in genomes],
                 "GC content": [g.gc_content for g in genomes]}

    # Plot
    plt.subplots(1, 1, figsize=(10, 10), layout="constrained")

    df = pd.DataFrame(dataframe)
    sns.scatterplot(data=df, x="MAG", y="GC content")

    plt.xticks(rotation=90)
    plt.title("GC content of MAGs")

    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig(genomes[0].plot_dir / ".." / "mags_gc_content.pdf", format="pdf")


def plot_start_codon_dist(genomes: list[Genome]):
    gene_collections = [GeneCollection(g.gene_ids, g) for g in genomes]
    starts = [gc.start_codon_sequence.with_columns(pl.lit(gc.genome.name).alias("MAG")).group_by("MAG", "start_codon_sequence").agg(pl.len()) for gc in gene_collections]
    starts = pl.concat(starts).collect(streaming=True)

    plt.subplots(1, 1, figsize=(10, 10), layout="constrained")

    sns.barplot(data=starts.to_pandas(), x="start_codon_sequence", y="len", hue="MAG")

    plt.title("Start codon distribution in each MAG")

    if system() == "Darwin":
        plt.show()
    else:
        plt.savefig(genomes[0].plot_dir / ".." / "mags_start_dist.pdf", format="pdf")



if __name__ == "__main__":
    from pathlib import Path

    data_path = Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/methylation_data/"))
    for methylation_path in data_path.iterdir():
        if methylation_path.is_dir():
            Genome._Genome__methylation_data_dir = methylation_path

            genomes = [Genome(n) for n in Genome.valid_genome_names() if "metagenome" not in n]

            plot_mags_by_gc_content(genomes)
            plot_start_codon_dist(genomes)
