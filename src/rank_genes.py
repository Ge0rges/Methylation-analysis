from data_manager import *
import seaborn as sns
import matplotlib.pyplot as plt


def run_analysis():
    genome = Genome("Pelagibacter_r-contigs")
    gene = Gene(2195033, genome)

    data = gene.methylation_data
    sns.lineplot(data.collect().to_pandas(), x="position", y="total_methylation", hue="sample")
    fig_savepath = f"../plots/plots_5
    plt.savefig(f"{fig_savepath}/{genome.name}_5_test.pdf", format='pdf', transparent=False)


if __name__ == "__main__":
    run_analysis()
