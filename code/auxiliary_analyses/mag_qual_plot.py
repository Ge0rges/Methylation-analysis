import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from code.Utilities.utils import barcode_sample_map, read_counts


def plot_coverage():
    """
    Plot the coverage metrics.
    :return: Saves a file
    :rtype: None
    """

    # Load data
    coverage = pd.read_csv("../../data/mag_eval/coverm.tsv", sep="\t", header=0)

    # Clean the mag names
    coverage['Genome'] = coverage['Genome'].str.title()
    coverage.columns = coverage.columns.str.title()
    coverage['Genome'] = coverage['Genome'].str.replace("_R-Contigs", " sp.")

    # Replace the coverage column names based on dictionnary mapping
    coverage.columns = coverage.columns.str.replace(".Fastq Mean", "")
    coverage.rename(inplace=True, columns=barcode_sample_map)

    # Format it
    coverage.set_index(coverage['Genome'], inplace=True)
    coverage.drop(columns=['Genome'], inplace=True)

    # Makea plot
    fig, axes = plt.subplots(1, 1, figsize=(15, 15))

    # Heatmap with MAG name as rows, samples as columns, and coverage as numbers
    sns.heatmap(coverage, annot=True, fmt=".2f",  cmap="viridis", ax=axes, square=True, cbar_kws={"shrink": 0.5})

    axes.set_title("Coverage Heatmap", fontsize=20)
    axes.set_xlabel("Samples", fontsize=16)
    axes.set_ylabel("MAG names", fontsize=16)

    plt.savefig("plots/mag_coverage.pdf", format='pdf', bbox_inches='tight')


def plot_mag_eval():
    """
    Plot MAG contamiantion and redundancy.
    :return: Saves a file
    """
    # Load data
    checkm2_quality = pd.read_csv("../../data/mag_eval/checkm2.tsv", sep="\t", header=0)
    anvio_quality = pd.read_csv("../../data/mag_eval/anvio.tsv", sep="\t", header=0)

    # Merge the quality dataframes
    checkm2_quality['Name'] = checkm2_quality['Name'].str.replace("_r-contigs", "_r")
    quality = pd.merge(checkm2_quality, anvio_quality, left_on="Name", right_on="bins", how="inner")

    # Clean the mag names
    quality['Name'] = quality['Name'].str.title()
    quality['Name'] = quality['Name'].str.replace("_R", " sp.")
    quality.rename(inplace=True, columns={"percent_completion": "Anvi'o Completeness",
                                          "percent_redundancy": "Anvi'o Redundancy",
                                          "Completeness": "CheckM2 Completeness",
                                          "Contamination": "CheckM2 Redundancy"})


    # Format it
    bar_chart_data = pd.melt(quality[["Name", "CheckM2 Completeness", "CheckM2 Redundancy",
                                      "Anvi'o Completeness", "Anvi'o Redundancy"]],
                             id_vars="Name", var_name="Metric", value_name="percent")

    # Define colors and hatches
    colors = sns.color_palette(["lightgreen", "lightcoral", "lightgreen", "lightcoral"])

    # Makea plot
    fig, axes = plt.subplots(1, 1, figsize=(20, 10))

    #  Bar chart with Completeness and Contamination percentages
    sns.barplot(data=bar_chart_data, x='Name', y='percent', hue="Metric", palette=colors, ax=axes)

    # Apply hatch patterns to each group of four bars
    for i, bar in enumerate(axes.patches):
        group = i // len(bar_chart_data['Name'].unique())
        if 4 > group > 1:
            bar.set_hatch("//")

        elif group == 4 and len(bar_chart_data['Metric'].unique()) > i - len(bar_chart_data['Metric'].unique()) * len(bar_chart_data['Name'].unique()) >= len(bar_chart_data['Metric'].unique()) - 2:
            bar.set_hatch("//")

    # Set the format for the bar labels
    for i in range(0, len(axes.containers)):
        axes.bar_label(axes.containers[i], fontsize=10, fmt="%.2f")

    axes.axhline(y=10, color='red', linestyle='--', linewidth=1)  # 10% line
    axes.axhline(y=90, color='green', linestyle='--', linewidth=1)  # 90% line

    axes.set_title("MAG quality assessment", fontsize=20)
    axes.set_xlabel("MAG names", fontsize=16)
    axes.set_ylabel("Percentage", fontsize=16)
    axes.yaxis.grid(False)
    axes.legend(loc='center right')

    # # Table with the MAG name and Taxonomy classification
    # taxonomy = pd.read_csv("data/mag_eval/gtdbtk.tsv", sep="\t", header=0)
    # table_data = taxonomy[['user_genome', 'classification']]
    # table = axes[2].table(cellText=table_data.values, colLabels=table_data.columns, loc='center')
    # table.set_fontsize(10)
    # table.scale(1.2, 1.2)
    # axes[2].axis('off')  # Hide the axes for the table

    # Display the figure
    plt.savefig("plots/mag_eval.svg", format='svg', bbox_inches='tight')


def read_count_plot():
    """
    Barplot of read counts
    """

    # Replace the key in read_counts with the value in barcode_sample_map
    read_counts_renamed = {barcode_sample_map[key]: value for key, value in read_counts.items()}

    df = pd.DataFrame(read_counts_renamed.items(), columns=["Sample", "Read Count"])
    sns.catplot(data=df, x="Sample", y="Read Count", kind="bar", height=5, aspect=3.5)
    plt.title("Read counts per sample")

    plt.savefig("plots/read_counts.svg", format='svg', bbox_inches='tight')


if __name__ == "__main__":
    plt.style.use('ggplot')
    sns.set(style="whitegrid")

    plot_mag_eval()
    plot_coverage()
    read_count_plot()
