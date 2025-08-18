
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.objects.motif import Motif
from sklearn.metrics import r2_score

sns.set_theme(context="paper", style="whitegrid")
##############################################################################
# 1) WHOLE METHYLOME (unchanged logic)
##############################################################################

def plot_whole_methylome_pelagibacter(motif: Motif,) -> None:
    """
    Plot the whole methylome (only for the motif’s methylation type) across samples:
      - fraction = motif_type / (motif_type + canonical base)
      - x-axis = genome_position, color by sample
    """
    df = motif.data()
    if df is None:
        return
    
    df = df.collect()
    df = motif.genome.add_genome_relative_position(df).rename({"treatment": "Treatment"})
    
    if df.is_empty():
        return
    
    _, ax = plt.subplots(1, 1, figsize=(12, 7), constrained_layout=True)
    hue_order = sorted(df.get_column("Treatment").unique().to_list(), key=motif.genome.treatment_order_map.get)
    
    sns.scatterplot(
        data=df.to_pandas(),
        x="genome_position",
        y=motif.meth_type,
        hue="Treatment",
        ax=ax,
        s=16,
        alpha=1,
        hue_order=hue_order,
        palette=[motif.genome.treatment_color_map[treatment] for treatment in hue_order]
    )
    
    for j, treatment in enumerate(hue_order):
        # Filter data for this treatment
        treatment_df = df.filter(df["Treatment"] == treatment).to_pandas()
        
        # Group by genome position and calculate mean for this treatment
        avg_df = treatment_df.groupby("genome_position")[motif.meth_type].mean().reset_index()
        
        if len(avg_df) > 4:  # Need at least 5 points for a fit
            # Fit polynomial regression
            x = avg_df["genome_position"].values
            y = avg_df[motif.meth_type].values
            z = np.polyfit(x, y, 4)
            p = np.poly1d(z)
                            
            # Generate smooth curve with more points
            x_smooth = np.linspace(x.min(), x.max(), 300)
            y_smooth = p(x_smooth)
            
            # Plot the smoothed average line using the treatment's color
            ax.plot(x_smooth, y_smooth, color=motif.genome.treatment_color_map[treatment], linewidth=2)

            # Calculate R-squared value
            y_pred = p(x)  # Predicted values at original x points
            r_squared = r2_score(y, y_pred)

            # Add R-squared as text to the plot
            ax.text(0.15, 0.05 - j*0.035, "${R}$²" + f" = {r_squared:.2f}",
                transform=ax.transAxes, ha='right',
                color=motif.genome.treatment_color_map[treatment])

            # Plot the polynomial regression line using the treatment's color
            ax.plot(x_smooth, y_smooth, color=motif.genome.treatment_color_map[treatment], linewidth=2) 

        ax.set_xlabel("Genome position (bp)")
        ax.set_ylabel(f"Methylation fraction")

    # Add lines showing ori and ter and contigs
    if "0_9_3" in str(motif.genome.genome_path):
        # Origin: 520,200 Terminus: 119,890
        ax.axvline(x=520200, color="blue", linestyle="--", label="Ori")
        ax.axvline(x=119890, color="red", linestyle="--", label="Ter")

    out_file = motif.genome.output_dir / f"{motif.genome.readable_name}_whole_methylome_{motif.readable_motif}.pdf"
    plt.savefig(out_file)
    plt.savefig(str(out_file)[:-3] + "svg", format="svg")
    plt.close()
    print(f"Saved PDF: {out_file}")