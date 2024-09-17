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
    fig, axis = plt.subplots(figsize=(6, 4))

    sns.lineplot(
        x='rank',
        y='relative_sse',
        hue='sparsity coefficient',
        #         style='comparison',
        errorbar='se',
        err_style='bars',
        data=plot_df,
        ax=axis,
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
    rank = 4
    plot_df = results_df[results_df['comparison'] == 'cross-validation']
    plot_df = plot_df[plot_df['rank'] == rank]

    # plot SSE
    mpl.rcParams['axes.spines.left'] = True
    mpl.rcParams['axes.spines.right'] = False
    color = sns.color_palette()[0]
    fig, axis = plt.subplots(figsize=(6, 4))
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

    # put defaults back where you found them
    mpl.rcParams['axes.spines.right'] = True
    mpl.rcParams['axes.spines.left'] = True

    # fix x axis stuff
    plt.xscale('log')
    plt.title('model scores vs. sparsity (R={})'.format(rank))
    plt.show()


if __name__ == "__main__":
    # For each folder in ../data/models/*/, load result.pickle
    for model_dir in glob.glob("../data/models/*/*"):
        with open(os.path.join(model_dir, "result.pickle"), "rb") as f:
            result = pickle.load(f)

        analyze_result(result)
