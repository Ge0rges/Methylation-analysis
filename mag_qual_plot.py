import seaborn as sns
import pandas as pd
import matplotlib.pylab as plt

def plot_mag_eval():
    """
    Plot the MAG evaluation metrics.
    :return: Saves a file
    :rtype: None
    """

    # Load data
    checkm2_quality = pd.read_csv("data/mag_eval/checkm2.tsv", sep="\t", header=0)
    anvio_quality = pd.read_csv("data/mag_eval/anvio.tsv", sep="\t", header=0)
    coverage = pd.read_csv("data/mag_eval/coverm.tsv", sep="\t", header=0)

    # Merge the quality dataframes
    checkm2_quality['Name'] = checkm2_quality['Name'].str.replace("_r-contigs", "_r")
    quality = pd.merge(checkm2_quality, anvio_quality, left_on="Name", right_on="bins", how="inner")

    # Clean the mag names
    coverage['Genome'] = coverage['Genome'].str.title()
    coverage.columns = coverage.columns.str.title()
    coverage['Genome'] = coverage['Genome'].str.replace("_R-Contigs", " sp.")
    quality['Name'] = quality['Name'].str.title()
    quality['Name'] = quality['Name'].str.replace("_R", " sp.")
    quality.rename(inplace=True, columns={"percent_completion": "Anvi'o Completeness",
                                          "percent_redundancy": "Anvi'o Redundancy",
                                          "Completeness": "CheckM2 Completeness",
                                          "Contamination": "CheckM2 Redundancy"})

    # Replace the coverage column names based on dictionnary mapping
    coverage.columns = coverage.columns.str.replace(".Fastq Mean", "")
    coverage.rename(inplace=True, columns={"Barcode01": "S2-1", "Barcode02": "S2-2", "Barcode03": "S2-3",
                                           "Barcode04": "Control", "Barcode05": "S3-1", "Barcode06": "S3-2",
                                           "Barcode07": "S3-3", "Barcode08": "S4-1", "Barcode09": "S4-2",
                                           "Barcode10": "S4-3", "Barcode11": "IC3-1", "Barcode12": "IC3-2",
                                           "Barcode13": "IC3-3", "Barcode14": "IC3-4"})

    # Format it
    bar_chart_data = pd.melt(quality[["Name", "CheckM2 Completeness", "CheckM2 Redundancy",
                                      "Anvi'o Completeness", "Anvi'o Redundancy"]],
                             id_vars="Name", var_name="Metric", value_name="percent")
    coverage.set_index(coverage['Genome'], inplace=True)
    coverage.drop(columns=['Genome'], inplace=True)

    # Make a faceted plot with coverage, quality, and taxonomy
    fig, axes = plt.subplots(2, 1, figsize=(20, 20))

    # Heatmap with MAG name as rows, samples as columns, and coverage as numbers
    sns.heatmap(coverage, annot=True, fmt=".2f",  cmap="viridis", ax=axes[0])

    axes[0].set_title("Coverage Heatmap", fontsize=24)
    axes[0].set_xlabel("Samples", fontsize=20)
    axes[0].set_ylabel("MAG Names", fontsize=20)

    # Stacked bar chart with Completeness and Contamination percentages
    sns.barplot(data=bar_chart_data, x='Name', y='percent', hue="Metric", palette="pastel", ax=axes[1])

    for i in range(0, len(axes[1].containers)):
        axes[1].bar_label(axes[1].containers[i], fontsize=10, fmt="%.2f")

    axes[1].axhline(y=10, color='red', linestyle='--', linewidth=1)  # 10% line
    axes[1].axhline(y=90, color='green', linestyle='--', linewidth=1)  # 90% line

    axes[1].set_title("CheckM 2 Quality Assessment", fontsize=24)
    axes[1].set_xlabel("MAG Names")
    axes[1].set_ylabel("Percentage")
    axes[1].yaxis.grid(False)
    axes[1].legend(loc='center right')

    # # Table with the MAG name and Taxonomy classification
    # taxonomy = pd.read_csv("data/mag_eval/gtdbtk.tsv", sep="\t", header=0)
    # table_data = taxonomy[['user_genome', 'classification']]
    # table = axes[2].table(cellText=table_data.values, colLabels=table_data.columns, loc='center')
    # table.set_fontsize(10)
    # table.scale(1.2, 1.2)
    # axes[2].axis('off')  # Hide the axes for the table

    # Display the figure
    plt.savefig("plots/mag_eval.pdf", format='pdf', bbox_inches='tight')


if __name__ == "__main__":
    plt.style.use('ggplot')
    sns.set(style="whitegrid")

    plot_mag_eval()
