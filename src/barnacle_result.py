import os
import glob
import pickle
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

def analyze_result(result):
    cv_results = []
    for vals in result.values():
        cv_results.extend(vals[1])

    results_df = pd.DataFrame(cv_results)

    # add features
    results_df['comparison'] = (results_df['modeled_replicate'] == results_df['comparison_replicate']).map({
        True: 'fitting', False: 'cross-validation'
    })
    results_df = results_df.sort_values(
        ['bootstrap_id', 'rank', 'lambda', 'replicate_pair']).reset_index(drop=True)
    results_df['sparsity coefficient'] = results_df['lambda'].astype(str)

    plot_df = results_df[results_df['comparison'] == 'cross-validation']
    # plot_df = plot_df[plot_df['bootstrap_id'].isin(np.arange(10))]
    # plot_df = plot_df[plot_df['rank'].isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]
    # plot_df = plot_df[plot_df['lambda'].isin([0.0, 0.05, 0.1, 0.2, 0.5, 1.0])]

    # plot figure
    fig, axis = plt.subplots(figsize=(10, 4), layout="constrained")

    sns.lineplot(
        x='rank',
        y='relative_sse',
        hue='sparsity coefficient',
        #         style='comparison',
        errorbar='se',
        err_style='bars',
        data=plot_df,
        ax=axis,
        alpha=0.5
        #     label=lamb,
    )

    # plt.ylim([0.33, 0.52])
    plt.title('model fit vs. parameterization')
    plt.xlabel('R')
    plt.ylabel('CV SSE')
    plt.legend(title='λ', loc='center left', bbox_to_anchor=[1, 0.5])
    plt.show()

    # look at SSE and FMS vs lambda

    # define data
    rank = 3
    plot_df = results_df[results_df['comparison'] == 'cross-validation']
    plot_df = plot_df[plot_df['rank'] == rank]

    # plot SSE
    mpl.rcParams['axes.spines.left'] = True
    mpl.rcParams['axes.spines.right'] = False
    color = sns.color_palette()[0]
    fig, axis = plt.subplots(figsize=(6, 6), layout="constrained")
    sns.lineplot(
        x='lambda',
        y='relative_sse',
        color=color,
        #     style='rank',
        errorbar='se',
        err_style='bars',
        data=plot_df,
        ax=axis,
        legend=False
    )
    axis.set(ylim=[-0.05, 1.02], xlabel='λ', ylabel='CV SSE')
    axis.yaxis.label.set_color(color)
    axis.tick_params(axis='y', colors=color)
    axis.spines['left'].set_color(color)

    # plot FMS
    color = sns.color_palette()[1]
    mpl.rcParams['axes.spines.right'] = True
    mpl.rcParams['axes.spines.left'] = False
    axis2 = plt.twinx(axis)
    sns.lineplot(
        x='lambda',
        y='fms_cv',
        color=color,
        #     style='rank',
        errorbar='se',
        err_style='bars',
        data=plot_df,
        ax=axis2,
        legend=False
    )
    axis2.set(ylim=[-.05, 1.05], xlabel='λ', ylabel='CV FMS')
    axis2.yaxis.label.set_color(color)
    axis2.tick_params(axis='y', colors=color)
    axis2.spines['right'].set_color(color)

    print(plot_df.groupby("lambda")['relative_sse'].mean())

    # put defaults back where you found them
    mpl.rcParams['axes.spines.right'] = True
    mpl.rcParams['axes.spines.left'] = True

    # fix x axis stuff
    plt.xscale('log')
    plt.title('model scores vs. sparsity (R={})'.format(rank))
    plt.show()

    # Max FMS
    summary_df = results_df[results_df['comparison'] == 'cross-validation']
    summary_df = summary_df[summary_df['rank'].isin([3])]
    summary_df = summary_df.groupby(['rank', 'lambda'])[['mean_gene_sparsity', 'relative_sse', 'fms_cv']].agg(
        mean_gene_sparsity=('mean_gene_sparsity', 'mean'),
        relative_sse=('relative_sse', 'mean'),
        fms_cv=('fms_cv', 'mean'),
        fms_sem=('fms_cv', 'sem'),
        bootstraps=('fms_cv', 'count')
    ).reset_index()

    best_FMS = summary_df.loc[summary_df.fms_cv.idxmax(), :]
    print('max CV FMS: \n\n{}\n'.format(best_FMS))
    se_fms = best_FMS['fms_cv'] - best_FMS['fms_sem']
    print('max CV FMS - 1SE: {}\n'.format(se_fms))

    # show all models with at least the minimum FMS, sorted from sparsest to least sparse
    print(summary_df[summary_df.fms_cv.ge(se_fms)].sort_values('lambda', ascending=False))


if __name__ == "__main__":
    # For each folder in ../data/models/*/, load result.pickle
    for model_dir in glob.glob("../data/models/*/*"):
        with open(os.path.join(model_dir, "result.pickle"), "rb") as f:
            result = pickle.load(f)

        analyze_result(result)
