import math
from src.utils.load_data import Database, add_LI_members_to_missing_ranks
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
def plot_deviation_specific_queries(
    df,
    max_k=200,
    max_missing_ratio=None,
    column_name = 'gender',
    save_path='deviation_lines.png'
):
    """
    Line plot of deviation = p_actual - (count_in_top_k / k) for each gender,
    one line per query, with Query numbers in the legend. Handles varying ranking lengths.

    Args:
        df: DataFrame with ['query','date','rank','gender'].
        max_k: maximum rank to plot (caps longer rankings).
        max_missing_ratio: if set, drops queries whose (max_rank - count)/max_rank > max_missing_ratio.
        save_path: file path to save the figure.
    """
    # 1) Earliest-date filter
    df['date'] = pd.to_datetime(df['date'])
    earliest = df.groupby('query')['date'].min()
    df_e = df.merge(earliest, on=['query','date'])
    original_queries = sorted(df_e['query'].unique())
    # 1b) Filter queries by missing ratio using their own max_rank
    if max_missing_ratio is not None:
        counts    = df_e.groupby('query').size()
        max_ranks = df_e.groupby('query')['rank'].max()
        missing_ratio = (max_ranks - counts) / max_ranks
        keep_q = missing_ratio[missing_ratio <= max_missing_ratio].index
        df_e = df_e[df_e['query'].isin(keep_q)]
        print(f"num queries remaining: {len(keep_q)} from {len(original_queries)}")
        if df_e.empty:
            raise RuntimeError("No queries remain after missing-ratio filter")

    queries = sorted(df_e['query'].unique())
    if column_name == "gender":
        groups = ['F', 'M']
    elif column_name == 'race':
        groups = ["nh_white", "other"]

    # 2) Empirical overall proportion p_actual per (query, gender)
    total_per_query  = df_e.groupby('query').size()
    count_per_group  = df_e.groupby(['query',column_name]).size()
    overall_prop     = (count_per_group / total_per_query).rename('p_actual')

    dev_matrices = {}
    for g in groups:
        mat = np.zeros((len(queries), max_k))
        for i, q in enumerate(queries):
            max_rank = df_e[df_e['query'] == q]['rank'].max()
            p = overall_prop.get((q, g), 0.0)
            sub = df_e[df_e['query'] == q].dropna(subset=['rank']).sort_values('rank')
            for k in range(1, max_k + 1):
                if k <= max_rank:
                    t = ((sub[column_name] == g) & (sub['rank'] <= k)).sum()
                    mat[i, k - 1] = p - (t / k)
                else:
                    mat[i, k - 1] = np.nan
        dev_matrices[g] = mat

    # Determine common color limits
    allv = np.concatenate([m.flatten() for m in dev_matrices.values()])
    allv = allv[~np.isnan(allv)]
    mm = np.max(np.abs(allv))

    # Plot a row of heatmaps (one per gender)
    fig = plt.figure(
        figsize=(5 * len(groups) + 1, max(4, len(queries) * 0.3))
    )
    # make a GridSpec: len(genders) columns for plots, 1 for the colorbar
    gs = fig.add_gridspec(
        1, len(groups) + 1,
        width_ratios=[1] * len(groups) + [0.08],
        wspace=0.02
    )
    # create the heatmap axes
    axes = []
    for i in range(len(groups)):
        if i == 0:
            ax = fig.add_subplot(gs[0, i])
        else:
            ax = fig.add_subplot(gs[0, i], sharey=axes[0])
            ax.tick_params(labelleft=False, left=False)
            ax.set_yticklabels([])  # hide redundant labels
        axes.append(ax)
    # create the colorbar axes in the final column
    cax = fig.add_subplot(gs[0, len(groups)])


    cmap = plt.get_cmap('RdBu_r')
    cmap.set_bad('darkgray')
    for ax, g in zip(axes, groups):
        masked_mat = np.ma.masked_invalid(dev_matrices[g])
        im = ax.imshow(masked_mat, aspect='auto', cmap=cmap, vmin=-mm, vmax=+mm)
        ax.set_title(f'{column_name.capitalize()}: {g}')
        ax.set_xlabel('Rank k')
    labels = [f"Query {original_queries.index(q)+ 1}" for q in queries]
    axes[0].set_yticks(np.arange(len(queries)))
    axes[0].set_yticklabels(labels)
    axes[0].set_ylabel('Query Number')
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label('Overall Prop − Top‐k Prop')
    fig.subplots_adjust(left=0.21)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400,format='pdf')
    plt.close(fig)

def plot_deviation(df, p_star_method = 'actual value', max_missing_ratio=None, column_name = 'gender'):
    """
    For each query, uses only the earliest date's ranking to compute,
    for each gender and rank k (1..200), the deviation:

        deviation = (overall_prop) - (count_in_top_k / k)

    Then plots one heat‐map per column, with:
      - y axis: query
      - x axis: rank (1..200)
      - color: deviation (same vmin/vmax across all column values)
    """
    # 1) Keep only earliest date per query
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['query'] != "Materials engineer"]
    earliest = df.groupby('query')['date'].min().rename('min_date')
    df_earliest = df.merge(earliest, how='inner', left_on=['query', 'date'], right_on=['query', 'min_date'])
    if max_missing_ratio is not None:
        counts = df_earliest.groupby('query').size()
        missing_ratio = (200 - counts) / 200
        keep_queries = missing_ratio[missing_ratio <= max_missing_ratio].index
        df_earliest = df_earliest[df_earliest['query'].isin(keep_queries)]
        print(f"num queries remaining: {len(keep_queries)}")
        if df_earliest.empty:
            raise RuntimeError("No queries remain after applying max_missing_ratio filter")

    if p_star_method == 'actual value':
        total_per_query = df_earliest.groupby('query').size()
        count_per_group = df_earliest.groupby(['query', column_name]).size()
        overall_prop = (count_per_group / total_per_query).rename('overall_prop')
    elif p_star_method == 'BLS':
        # overal_prop = database.get_BLS_data().set_index('Occupation')[['Gender_M','Gender_F']]/100.0
        bls_df = database.get_BLS_data()
        if column_name == 'gender':
            bls_long = bls_df.melt(
                id_vars=['Occupation'],
                value_vars=['Gender_M', 'Gender_F'],
                var_name='gender_code',
                value_name='prop'
            )
            # strip trailing 's' so occupations match queries
            bls_long['Occupation'] = bls_long['Occupation'].str.rstrip('s')
            # filter to only occupations present in our queries
            bls_long = bls_long[bls_long['Occupation'].isin(df_earliest['query'].unique())]
            # Convert percentages to fractions and extract 'M'/'F'
            bls_long['prop'] /= 100.0
            bls_long['gender'] = bls_long['gender_code'].str[-1]
            # Set index to (Occupation, gender)
            overall_prop = bls_long.set_index(['Occupation', 'gender'])['prop']
        elif column_name == 'race':
            # Keep only Occupation + the nh_white percentage
            race_df = bls_df[['Occupation', 'race_nh_white']].copy()
            # Strip trailing 's' to match your query names
            race_df['Occupation'] = race_df['Occupation'].str.rstrip('s')
            # Filter to only the queries you actually have
            race_df = race_df[race_df['Occupation'].isin(df_earliest['query'].unique())]
            # Convert to [0,1] fractions
            race_df['prop_white'] = race_df['race_nh_white'] / 100.0
            race_df['prop_other'] = 1.0 - race_df['prop_white']
            # Melt to long form so we have one row per (Occupation, race_category)
            race_long = race_df.melt(
                id_vars=['Occupation'],
                value_vars=['prop_white', 'prop_other'],
                var_name='race_code',
                value_name='prop'
            )
            # Map to your two levels
            race_long['race'] = race_long['race_code'].map({
                'prop_white': 'nh_white',
                'prop_other': 'other'
            })
            overall_prop = race_long.set_index(['Occupation', 'race'])['prop']


    # 3) Setup axes indices
    queries = sorted(df_earliest['query'].unique())
    if column_name == "gender":
        groups = ['F','M']
    elif column_name == 'race':
        groups = ["nh_white","other"]
    # genders = ['F','M']
    ranks = np.arange(1, 201)

    # 4) Build deviation matrices
    dev_matrices = {}
    # dev_matrices_ = {}
    for g in groups:
        mat = np.zeros((len(queries), len(ranks)))
        # mat_ = np.zeros((len(queries), len(ranks)))
        for i, q in enumerate(queries):
            sub = df_earliest[df_earliest['query'] == q].dropna(subset=['rank']).sort_values('rank')
            # overall proportion for this (q, g)
            if p_star_method == 'BLS':
                p = overall_prop.get((q, g), 0.0)
                # p = overal_prop.at[q+'s', f'Gender_{g[0].upper()}']
            elif p_star_method == 'actual value':
                p = overall_prop.get((q, g), 0.0)
            # elif p_star_method == 'optimized':
            # # Compute p* at each k by minimizing the worst‐case deviation:
            # # 1) Precompute cumulative counts for all genders
            #     all_props = []
            #     for g2 in groups:
            #         for k in ranks:
            #             t2 = ((sub[column_name] == g2) & (sub['rank'] <= k)).sum()
            #             all_props.append(t2 / k)
            #     p = np.median(all_props)
            # subset and sort

            for j, k in enumerate(ranks):
                t = ((sub[column_name] == g) & (sub['rank'] <= k)).sum()
                mat[i, j] = p - (t / k)
                # mat_[i, j] = p - (t / k)
        dev_matrices[g] = mat
        # dev_matrices_[g] = mat_

    # 5) Determine a common color scale
    all_vals = np.concatenate([m.flatten() for m in dev_matrices.values()])
    max_dev = np.max(np.abs(all_vals))
    vmin, vmax = -max_dev, max_dev

    # 6) Plot one subplot per gender
    # 6) Plot one subplot per gender, +1 extra column for a clean colorbar
    fig = plt.figure(
            figsize = (5 * len(groups) + 1, max(4, len(queries) * 0.3))
                           )
  # make a GridSpec: len(genders) columns for plots, 1 for the colorbar
    gs = fig.add_gridspec(
            1, len(groups) + 1,
            width_ratios = [1] * len(groups) + [0.08],
        wspace = 0.02
                      )
  # create the heatmap axes
    axes = []
    for i in range(len(groups)):
        if i == 0:
            ax = fig.add_subplot(gs[0, i])
        else:
            ax = fig.add_subplot(gs[0, i], sharey=axes[0])
            ax.tick_params(labelleft=False, left=False)
            ax.set_yticklabels([])  # hide redundant labels
        axes.append(ax)
  # create the colorbar axes in the final column
    cax = fig.add_subplot(gs[0, len(groups)])


    for ax, g in zip(axes, groups):
        im = ax.imshow(
                dev_matrices[g],
                aspect = 'auto',
            cmap = 'RdBu_r',
            vmin = vmin,
            vmax = vmax
                            )
        ax.set_title(f'{column_name.capitalize()}: {g}')
        ax.set_xlabel('Rank k')
    axes[0].set_yticks(np.arange(len(queries)))
    axes[0].set_yticklabels(queries)
    # axes[0].set_yticklabels(queries)
    axes[0].set_ylabel('Query')

          # draw a standalone colorbar in the reserved cax
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label('Overall Prop − Top‐k Prop')
    fig.subplots_adjust(left=0.21)
    # plt.tight_layout()
    plt.savefig(f'./plots/overall/pdf_versions/{column_name}_deviation_{p_star_method}_{max_missing_ratio}.pdf', dpi=400,format='pdf')   # write to file instead of popping up
    plt.close()


def plot_minskewk(
    df,
    max_missing_ratio=0.01,
    k_intervals=None,
    save_path=None,
    query_order=None,
    normalize_against_best_possible_skew=False,
    plot_only_min=False
):
    """
    Plots skew@k for each query, optionally showing only the minimum skew across groups.

    Args:
        df: DataFrame with ['query','date','rank','gender']
        max_missing_ratio: filter threshold
        k_intervals: iterable of k values; defaults to 1..200 or provided list
        save_path: file path to save the plot
        query_order: optional list to order queries
        normalize_against_best_possible_skew: adjust skew by best possible
        plot_only_min: if True, plot only the minimum skew across genders
    """
    # 1) Prepare and filter
    df['date'] = pd.to_datetime(df['date'])
    earliest = df.groupby('query')['date'].min()
    df_e = df.merge(earliest, on=['query','date'])
    if query_order is None:
        query_order = sorted(df_e['query'].unique())
    if max_missing_ratio is not None:
        counts = df_e.groupby('query').size()
        max_ranks = df_e.groupby('query')['rank'].max()
        missing_ratio = (max_ranks - counts) / max_ranks
        keep_q = missing_ratio[missing_ratio <= max_missing_ratio].index
        df_e = df_e[df_e['query'].isin(keep_q)]
        if df_e.empty:
            raise RuntimeError("No queries remain after missing-ratio filter")

    # 2) Set queries and ks
    queries = [q for q in query_order if q in df_e['query'].unique()]
    if k_intervals is None:
        ks = np.arange(1, int(df_e['rank'].max()) + 1)
    else:
        ks = np.array(k_intervals, dtype=int)

    # 3) Compute overall proportions
    total_per_query = df_e.groupby('query').size()
    count_per_group = df_e.groupby(['query','gender']).size()
    overall_prop = (count_per_group / total_per_query).rename('p_q')

    # 4) Compute skew matrices manually
    genders = ['F','M']
    skew_matrices = {g: np.zeros((len(queries), len(ks))) for g in genders}
    for i, q in enumerate(queries):
        sub = df_e[df_e['query'] == q].dropna(subset=['rank']).sort_values('rank')
        p_qg = {g: overall_prop.get((q,g), np.nan) for g in genders}
        max_rank = sub['rank'].max()
        for j, k in enumerate(ks):
            for g in genders:
                if k <= max_rank:
                    t = ((sub['gender'] == g) & (sub['rank'] <= k)).sum()
                    p_rk = t / k
                    p_star = p_qg.get(g, np.nan)
                    if p_rk > 0 and p_star > 0:
                        val = math.log(p_rk / p_star)
                        if normalize_against_best_possible_skew:
                            opts = []
                            for cnt in (math.floor(p_star*k), math.ceil(p_star*k)):
                                if cnt > 0:
                                    opts.append(math.log((cnt/k) / p_star))
                            best = min(opts) if opts else np.nan
                            val -= best
                        skew_matrices[g][i, j] = val
                    else:
                        skew_matrices[g][i, j] = np.nan
                else:
                    skew_matrices[g][i, j] = np.nan

    # 5) Compute min matrix if requested
    if plot_only_min:
        min_mat = np.minimum(skew_matrices['F'], skew_matrices['M'])
        plot_genders = ['min']
    else:
        min_mat = None
        plot_genders = genders

    # 6) Plotting
    n_rows = 1 if plot_only_min else len(plot_genders)
    fig, axes = plt.subplots(nrows=n_rows, ncols=1, sharex=True, figsize=(8, 4*n_rows))
    if n_rows == 1:
        axes = [axes]
    cmap = plt.get_cmap('tab20', len(queries)) if len(queries) <= 20 else plt.get_cmap('hsv', len(queries))
    tick_locs = np.arange(0, ks.max(), 25)

    for ax, g in zip(axes, plot_genders):
        ax.set_xticks(tick_locs)
        data = min_mat if plot_only_min else skew_matrices[g]
        title = 'Min Skew Across Genders' if plot_only_min else f'Gender: {g}'
        for i, q in enumerate(queries):
            idx = query_order.index(q) + 1
            ax.plot(ks, data[i], label=f"Query {idx}", color=cmap(i))
        ax.set_title(title)
        ax.set_ylabel('Skew @ k')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.axhline(-0.011, color='black', linestyle='-', linewidth=1)
        ax.legend(title='Query', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(True, linestyle='--', linewidth=0.5)

    axes[-1].set_xlabel('Rank k')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, format='pdf')
    plt.close(fig)


def plot_skewk(df, max_missing_ratio = 0.01, k_intervals = None, save_path = None, query_order=None, normalize_against_best_possible_skew = False):
    df['date'] = pd.to_datetime(df['date'])
    earliest = df.groupby('query')['date'].min()
    df_e = df.merge(earliest, on=['query', 'date'])
    if query_order is None:
        query_order = sorted(df_e['query'].unique())
    # 1b) Filter queries by missing ratio using their own max_rank
    if max_missing_ratio is not None:
        counts = df_e.groupby('query').size()
        max_ranks = df_e.groupby('query')['rank'].max()
        missing_ratio = (max_ranks - counts) / max_ranks
        keep_q = missing_ratio[missing_ratio <= max_missing_ratio].index
        df_e = df_e[df_e['query'].isin(keep_q)]
        print(f"num queries remaining: {len(keep_q)} from {len(query_order)}")
        if df_e.empty:
            raise RuntimeError("No queries remain after missing-ratio filter")

    queries = [q for q in query_order if q in df_e['query'].unique()]
    genders = ['F','M']

    # 3) Compute overall target proportions p_q for each (query, gender)
    total_per_query = df_e.groupby('query').size()
    count_per_group = df_e.groupby(['query', 'gender']).size()
    overall_prop = (count_per_group / total_per_query).rename('p_q')

    # 4) Determine k values
    if k_intervals is None:
        global_max = 200
        # global_max = int(df_e['rank'].max())
        ks = np.arange(1, global_max + 1)
    else:
        pass

    # 5) Build skew matrices: shape (n_queries, len(ks))
    skew_matrices = {g: np.zeros((len(queries), len(ks))) for g in genders}

    for i, q in enumerate(queries):
        sub = (
            df_e[df_e['query'] == q]
            .dropna(subset=['rank'])
            .sort_values('rank')
        )
        # target p for this query/gender
        p_qg = {g: overall_prop.get((q, g), np.nan) for g in genders}
        query_max_rank = df_e[df_e['query'] == q]['rank'].max()

        for j, k in enumerate(ks):
            for g in genders:
                if k <= query_max_rank:
                    t = ((sub['gender'] == g) & (sub['rank'] <= k)).sum()
                    p_rk = t / k
                    p_star = p_qg[g]
                    # log ratio, handle zero/NaN
                    if p_rk > 0 and p_star > 0:
                        if normalize_against_best_possible_skew:
                            possible_skew_values = []
                            for cnt in (math.floor(p_star * k), math.ceil(p_star * k)):
                                # only compute log if cnt>0 (to avoid log(0))
                                if cnt > 0:
                                    possible_skew_values.append(np.abs(np.log((cnt / k) / p_star)))
                            best_skew_value = min(possible_skew_values) if possible_skew_values else np.nan
                            if best_skew_value == np.nan:
                                print(f"best skew value nan for k-{k} and {g}")
                                if (np.abs(np.log(p_rk / p_star)) - best_skew_value) <0:
                                    print("negative")
                            skew_matrices[g][i, j] = np.sign(np.log(p_rk / p_star))* max(np.abs(np.log(p_rk / p_star)) - best_skew_value,0)
                        else:
                            skew_matrices[g][i, j] = np.log(p_rk / p_star)
                    else:
                        skew_matrices[g][i, j] = np.nan
                else:
                    skew_matrices[g][i, j] = np.nan

    # 6) Plotting
    if len(queries) <= 20:
        cmap = plt.get_cmap('tab20', len(queries))
    else:
        cmap = plt.get_cmap('hsv', len(queries))
    fig, axes = plt.subplots(
        nrows=len(genders),
        ncols=1,
        sharex=True,
        figsize=(8, 4 * len(genders))
    )
    if len(genders) == 1:
        axes = [axes]
    tick_locations = np.arange(0, ks.max(), 25)
    # color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for ax, g in zip(axes, genders):
        ax.set_xticks(tick_locations)
        for i, q in enumerate(queries):
            orig_idx = query_order.index(q) + 1
            ax.plot(
                ks,
                skew_matrices[g][i],
                label=f"Query {orig_idx}",
                color=cmap(i)
            )
        ax.set_title(f'Gender: {g}')
        ax.set_ylabel('Skew @ k')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.axhline(-0.011, color='black', linestyle='-', linewidth=1)
        ax.legend(title='Query', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(True, linestyle='--', linewidth=0.5)

    axes[-1].set_xlabel('Rank k')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, format = 'pdf')
    plt.close(fig)


def plot_normalized_vs_non(df, query, k_intervals=None, save_path=None):
    """
    For a given query, plots normalized vs non-normalized skew@k for each gender.
    Produces two subplots: one for F and one for M.
    """
    # --- Filter for the selected query ---
    df['date'] = pd.to_datetime(df['date'])
    earliest = df.groupby('query')['date'].min()
    df_e = df.merge(earliest, on=['query', 'date'])
    df_q = df_e[df_e['query'] == query].copy()
    if df_q.empty:
        raise ValueError(f"No data found for query '{query}'")

    df_q = df_q.dropna(subset=['rank']).sort_values('rank')
    genders = ['F', 'M']

    # --- Determine k range ---
    if k_intervals is None:
        ks = np.arange(1, 201)
    else:
        ks = np.arange(1, k_intervals + 1)

    # --- Compute baseline group proportions (p_star) ---
    total_count = len(df_q)
    count_per_group = df_q.groupby('gender').size()
    overall_prop = (count_per_group / total_count).rename('p_q')

    # --- Prepare plot ---
    fig, axes = plt.subplots(nrows=len(genders), ncols=1, figsize=(8, 6), sharex=True)
    if len(genders) == 1:
        axes = [axes]

    for ax, g in zip(axes, genders):
        skew_non = []
        skew_norm = []

        for k in ks:
            t = ((df_q['gender'] == g) & (df_q['rank'] <= k)).sum()
            p_rk = t / k
            p_star = overall_prop.get(g, np.nan)

            if np.isnan(p_star) or p_star <= 0 or p_rk <= 0:
                skew_non.append(np.nan)
                skew_norm.append(np.nan)
                continue

            # --- Non-normalized skew ---
            log_skew = np.log(p_rk / p_star)
            skew_non.append(log_skew)

            # --- Normalized skew ---
            possible_skew_values = []
            for cnt in (math.floor(p_star * k), math.ceil(p_star * k)):
                if cnt > 0:
                    possible_skew_values.append(abs(np.log((cnt / k) / p_star)))
            best_skew_value = min(possible_skew_values) if possible_skew_values else np.nan
            if np.isnan(best_skew_value):
                skew_norm.append(np.nan)
            else:
                norm_skew = np.sign(log_skew) * max(abs(log_skew) - best_skew_value, 0)
                skew_norm.append(norm_skew)

        # --- Plot both versions ---
        ax.plot(ks, skew_non, label='Non-normalized', lw=2, color='tab:blue')
        ax.plot(ks, skew_norm, label='Normalized', lw=2, linestyle='--', color='tab:orange')

        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_title(f"Gender: {g}")
        ax.set_ylabel("Skew@k")
        ax.legend()
        ax.grid(True, linestyle='--', linewidth=0.5)

    axes[-1].set_xlabel("Rank k")
    fig.suptitle(f"Normalized vs Non-Normalized Skew@k for Query: {query}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, format='pdf')

    plt.show()

def plot_sankey_rank_transitions(
        df,
        group_column,
        output_path_prefix='./plots/overall/sankey',
        bucket_size=25
):
    """
    Creates and saves sankey diagrams per group showing transitions
    across rank buckets over 5 consecutive days, aggregated across queries
    with max(rank)=200.

    Args:
        df: DataFrame with columns ['query','date','candidate_id','rank',group_column]
        group_column: e.g. 'gender' or 'race'
        output_path_prefix: filename prefix for saved diagrams (one per group)
        bucket_size: size of each rank bucket (default 25)
    """
    # Ensure date dtype and sort
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['query', 'date', 'candidate_id', 'rank'])

    # Drop any subsequent duplicates, keeping only the first row per group
    df = df.drop_duplicates(
        subset=['query', 'date', 'candidate_id'],
        keep='first'
    )

    # Identify eligible queries (max rank==200 and >=5 days)
    eligible = []
    for q, sub in df.groupby('query'):
        if sub['rank'].max() == 200 and sub['date'].nunique() >= 5:
            eligible.append(q)
    df_e = df[df['query'].isin(eligible)]
    print(f"total queries: {df['query'].nunique()}, after selection: {df_e['query'].nunique()}")
    # Precompute for each query its first 5 sorted distinct dates
    query_dates = {
        q: sorted(sub['date'].unique())[:5]
        for q, sub in df_e.groupby('query')
    }
    num_buckets = 200 // bucket_size
    buckets = [f"{i * bucket_size + 1}-{(i + 1) * bucket_size}" for i in range(num_buckets)]
    buckets.append('unranked')
    # Define bucket function
    def bucketize(r):
        if pd.isna(r):
            return 'unranked'
        b = int((r - 1) // bucket_size)
        low = b * bucket_size + 1
        high = min((b + 1) * bucket_size, bucket_size * (bucket_size))
        return f"{low}-{high}"


    # Process per group
    for group_value in df_e[group_column].unique():
        # Initialize transition counts dict
        link_counts = {}
        # Node labels: for each day 1..5 and each bucket label discovered
        # all_buckets = set()

        # Accumulate transitions across queries
        for q in eligible:
            dates = query_dates[q]
            # subset to those dates
            df_q = df_e[(df_e['query'] == q) & (df_e['date'].isin(dates))]
            # pivot: index candidate, columns dates
            pivot = df_q.pivot(index='candidate_id', columns='date', values='rank')
            # filter to this group_value
            ids = df_q[df_q[group_column] == group_value]['candidate_id'].unique()
            pivot = pivot.loc[pivot.index.isin(ids)]
            # bucketize for each day
            buckets_df = pivot.applymap(bucketize)
            # all_buckets.update(buckets.values.flatten())
            # transitions day i → i+1
            for i in range(4):
                src = buckets_df.iloc[:, i]
                tgt = buckets_df.iloc[:, i + 1]
                for s, t in zip(src, tgt):
                    key = (f"Day{i + 1}:{s}", f"Day{i + 2}:{t}")
                    link_counts[key] = link_counts.get(key, 0) + 1

        # Build unique node list
        nodes = [f"Day{d + 1}:{b}" for d in range(5) for b in buckets]
        node_idx = {n: i for i, n in enumerate(nodes)}

        # Build sankey inputs
        sources, targets, values = [], [], []
        for (s, t), v in link_counts.items():
            if s in node_idx and t in node_idx:
                sources.append(node_idx[s])
                targets.append(node_idx[t])
                values.append(v)

        # Create Sankey figure
        xs, ys = [], []
        for d in range(5):
            x = d / 4
            for i in range(len(buckets)):
                y = (len(buckets) - 1 - i) / (len(buckets) - 1) if len(buckets)>1 else 0
                xs.append(x)
                ys.append(y)
        fig = go.Figure(go.Sankey(
            arrangement='fixed',
            node=dict(
                label=nodes,
                x=xs,y=ys,
                pad=15,
                thickness=20,
                line=dict(color='black', width=0.5)
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values
            )
        ))
        fig.update_layout(
            title_text=f"Rank Transitions for {group_column} = {group_value}",
            font_size=10
        )
        for d in range(5):
            fig.add_annotation(
                x=d / 4, y=-0.05, text=f"Day {d + 1}",
                showarrow=False, xref='paper', yref='paper'
            )
        # Save to file (requires kaleido)
        output_path = f"{output_path_prefix}_{group_value}.png"
        fig.write_image(output_path, scale=2)
        print(f"Saved Sankey diagram for {group_column}={group_value} to {output_path}")

def churn_heatmap_daypairs(df,column_title, bucket_size = 25, max_missing_ratio = 0.15, save_path = "./plots/overall/pdf_versions/", query_order = None):
    df['date'] = pd.to_datetime(df['date'])
    earliest = df.groupby('query')['date'].min()
    df_e = df.merge(earliest, on=['query', 'date'])
    if max_missing_ratio is not None:
        counts = df_e.groupby('query').size()
        max_ranks = df_e.groupby('query')['rank'].max()
        missing_ratio = (max_ranks - counts) / max_ranks
        keep_q = missing_ratio[missing_ratio <= max_missing_ratio].index
        df = df[df['query'].isin(keep_q)]
        # print(f"num queries remaining: {len(keep_q)} from {len(query_order)}")
        if df_e.empty:
            raise RuntimeError("No queries remain after missing-ratio filter")
    eligible = []
    for q, sub in df.groupby('query'):
        if sub['rank'].max() >= 200 and sub['date'].nunique() >= 5:
            eligible.append(q)
    df_filtered = df[df['query'].isin(eligible)]
    print(f"total queries {df['query'].nunique()},after missing {df_e['query'].nunique()} ,remaining {df_filtered['query'].nunique()}")
    churn_dfs = df_filtered.groupby('query').apply(lambda g: compute_churn_for_query(g,bucket_size, column_title, max_rank=200)).reset_index()
    days = sorted(churn_dfs['end_day'].unique())
    if query_order is None:
        queries = list(dict.fromkeys(churn_dfs['query'].tolist()))
    else:
        queries = [q for q in query_order if q in churn_dfs['query'].unique()]
    thresholds = sorted(churn_dfs['threshold'].unique())
    if column_title == 'gender':
        attributes = ['F', 'M']
    elif column_title == 'race':
        attributes = ['nh_white', 'other']
    col_labels = [f"{thr},{g}" for thr in thresholds for g in attributes]
    all_rates = churn_dfs['churn_rate'].values
    vmin, vmax = np.nanmin(all_rates), np.nanmax(all_rates)

    for day in days:
        # Filter for this day pair
        sub = churn_dfs[churn_dfs['end_day'] == day]
        # Build a full matrix initialized to NaN
        mat = np.full((len(queries), len(col_labels)), np.nan)
        # Create a lookup Series for fast access
        lookup = sub.set_index(['query', 'threshold', column_title])['churn_rate']
        # Populate the matrix
        for i, q in enumerate(queries):
            for j, (thr, g) in enumerate((pair.split(',') for pair in col_labels)):
                key = (q, type(thresholds[0])(thr), g)  # cast thr to original type
                mat[i, j] = lookup.get(key, np.nan)

        # Plot heatmap
        fig, ax = plt.subplots(
            figsize=(0.4 * len(col_labels) + 3, 0.3 * len(queries) + 2)
        )
        im = ax.imshow(mat, aspect='auto', vmin=vmin, vmax=vmax, cmap='Reds')
        gcount = len(attributes)  # e.g. 2
        for t_idx in range(1, len(thresholds)):
            sep_x = t_idx * gcount - 0.5
            ax.axvline(sep_x, color='black', linewidth=1)
        ax.set_title(f'Day 1 → {day}')
        ax.set_yticks(np.arange(len(queries)))
        if query_order:
            # labels = [f"Query {query_order.index(q) + 1}" for q in queries]
            ax.set_yticklabels(queries)
        else:
            ax.set_yticklabels(queries)
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=90)
        ax.set_xlabel(f'Threshold,{column_title}')
        ax.set_ylabel('Query')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Churn Rate')
        plt.tight_layout()

        # Save figure
        out_path = f"{save_path}day1to{day}_{column_title}_{max_missing_ratio}.pdf"
        fig.savefig(out_path, dpi=300, format='pdf')
        plt.close(fig)
        print(f"Saved heatmap: {out_path}")



def plot_skewk_optimal_vs_actual(df, max_missing_ratio = 0.01, k_intervals = None, save_path = None, query_order=None):
    df['date'] = pd.to_datetime(df['date'])
    earliest = df.groupby('query')['date'].min()
    df_e = df.merge(earliest, on=['query', 'date'])
    if query_order is None:
        query_order = sorted(df_e['query'].unique())
    # 1b) Filter queries by missing ratio using their own max_rank
    if max_missing_ratio is not None:
        counts = df_e.groupby('query').size()
        max_ranks = df_e.groupby('query')['rank'].max()
        missing_ratio = (max_ranks - counts) / max_ranks
        keep_q = missing_ratio[missing_ratio <= max_missing_ratio].index
        df_e = df_e[df_e['query'].isin(keep_q)]
        print(f"num queries remaining: {len(keep_q)} from {len(query_order)}")
        if df_e.empty:
            raise RuntimeError("No queries remain after missing-ratio filter")

    queries = [q for q in query_order if q in df_e['query'].unique()]
    genders = ['F','M']

    # 3) Compute overall target proportions p_q for each (query, gender)
    total_per_query = df_e.groupby('query').size()
    count_per_group = df_e.groupby(['query', 'gender']).size()
    overall_prop = (count_per_group / total_per_query).rename('p_q')

    # 4) Determine k values
    if k_intervals is None:
        global_max = 200
        # global_max = int(df_e['rank'].max())
        ks = np.arange(1, global_max + 1)
    else:
        pass

    # 5) Build skew matrices: shape (n_queries, len(ks))
    skew_matrices = {g: np.zeros((len(queries), len(ks))) for g in genders}
    optimal_min_skew = np.zeros((len(queries), len(ks)))

    for i, q in enumerate(queries):
        sub = (
            df_e[df_e['query'] == q]
            .dropna(subset=['rank'])
            .sort_values('rank')
        )
        # target p for this query/gender
        p_qg = {g: overall_prop.get((q, g), np.nan) for g in genders}
        query_max_rank = df_e[df_e['query'] == q]['rank'].max()

        for j, k in enumerate(ks):
            if k <= query_max_rank:
                possible_skew_values = [np.log((math.floor(p_qg[genders[0]]*k)/k)/p_qg[genders[0]]),
                                        np.log((math.ceil(p_qg[genders[0]]*k)/k)/p_qg[genders[0]]),
                                        np.log((math.floor(p_qg[genders[1]]*k)/k)/p_qg[genders[1]]),
                                        np.log((math.ceil(p_qg[genders[1]]*k)/k)/p_qg[genders[1]])]

                optimal_min_skew[i,j] = min(possible_skew_values)

            for g in genders:
                if k <= query_max_rank:
                    t = ((sub['gender'] == g) & (sub['rank'] <= k)).sum()
                    p_rk = t / k
                    p_star = p_qg[g]
                    # log ratio, handle zero/NaN
                    if p_rk > 0 and p_star > 0:
                        skew_matrices[g][i, j] = np.log(p_rk / p_star)
                    else:
                        skew_matrices[g][i, j] = np.nan
                else:
                    skew_matrices[g][i, j] = np.nan


    actual_min_skew = [
    [
        min(skew_matrices[k][i][j] for k in skew_matrices)
        for j in range(len(ks))
    ]
    for i in range(len(queries))
    ]



def churn_heatmap_groups(df,column_title, bucket_size = 25, max_missing_ratio = 0.15, max_rank_in_plot = 75,save_path = "./plots/overall/pdf_versions/", query_order = None):
    df['date'] = pd.to_datetime(df['date'])
    earliest = df.groupby('query')['date'].min()
    df_e = df.merge(earliest, on=['query', 'date'])
    if max_missing_ratio is not None:
        counts = df_e.groupby('query').size()
        max_ranks = df_e.groupby('query')['rank'].max()
        missing_ratio = (max_ranks - counts) / max_ranks
        keep_q = missing_ratio[missing_ratio <= max_missing_ratio].index
        df = df[df['query'].isin(keep_q)]
        # print(f"num queries remaining: {len(keep_q)} from {len(query_order)}")
        if df_e.empty:
            raise RuntimeError("No queries remain after missing-ratio filter")
    eligible = []
    for q, sub in df.groupby('query'):
        if sub['rank'].max() >= max_rank_in_plot and sub['date'].nunique() >= 5:
            eligible.append(q)
    df_filtered = df[df['query'].isin(eligible)]
    print(
        f"total queries {df['query'].nunique()},after missing {df_e['query'].nunique()} ,remaining {df_filtered['query'].nunique()}")
    churn_dfs = df_filtered.groupby('query').apply(
        lambda g: compute_churn_for_query_all_days(g, bucket_size, column_title, max_rank=max_rank_in_plot)).reset_index()
    return churn_dfs
    days = sorted(churn_dfs['end_day'].unique())
    if query_order is None:
        queries = list(dict.fromkeys(churn_dfs['query'].tolist()))
    else:
        queries = [q for q in query_order if q in churn_dfs['query'].unique()]
    thresholds = sorted(churn_dfs['threshold'].unique())
    if column_title == 'gender':
        attributes = ['F', 'M']
    elif column_title == 'race':
        attributes = ['nh_white', 'other']
    col_labels = [f"{d},{thr}" for thr in thresholds for d in days]
    all_rates = churn_dfs['churn_rate'].values
    vmin, vmax = np.nanmin(all_rates), np.nanmax(all_rates)

    for g in attributes:
        # Filter for this day pair
        mat = np.full((len(queries), len(col_labels)), np.nan)
        # lookup table
        lookup = churn_dfs[churn_dfs[column_title] == g].set_index(['query', 'threshold', 'end_day'])['churn_rate']
        # populate
        for i, q in enumerate(queries):
            for j, label in enumerate(col_labels):
                d, thr = label.split(',')
                mat[i, j] = lookup.get((q, int(thr), int(d)), np.nan)

        fig, ax = plt.subplots(
            figsize=(0.4 * len(col_labels) + 2, 0.3 * len(queries) + 2)
        )
        im = ax.imshow(mat, aspect='auto', vmin=vmin, vmax=vmax, cmap='Reds')
        ax.set_title(f'Race: {g}')
        ax.set_yticks(np.arange(len(queries)))
        dcount = len(days)  # e.g. 2
        for t_idx in range(1, len(thresholds)):
            sep_x = t_idx * dcount - 0.5
            ax.axvline(sep_x, color='black', linewidth=1)
        ax.set_yticks(np.arange(len(queries)))
        if query_order:
            labels = [f"Query {query_order.index(q) + 1}" for q in queries]
            ax.set_yticklabels(labels)
        else:
            ax.set_yticklabels(queries)
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=90)
        ax.set_xlabel('Day,Threshold')
        ax.set_ylabel('Query')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Churn Rate')
        plt.tight_layout()

        # Save figure
        out_path = f"{save_path}_{g}_{column_title}_{max_missing_ratio}_specific_queries.pdf"
        fig.savefig(out_path, dpi=300, format='pdf')
        plt.close(fig)
        print(f"Saved heatmap: {out_path}")


def plot_skew_vs_prop_scatter(
    df,
    column_name='gender',
    k_values=[25, 50, 75, 100],
    max_missing_ratio=0.01,
    normalize_against_best_possible_skew=False,
    query_order=None
):
    """
    Uses manual skew computation to generate scatter plots for each group:
      - x-axis: skew@k
      - y-axis: overall group proportion (based on earliest date per query)
    for k in k_values.
    """

    # 1) Prepare data and filter by missing ratio
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    earliest = df.groupby('query')['date'].min().rename('min_date')
    df_e = df.merge(earliest, left_on=['query', 'date'], right_on=['query', 'min_date'])
    if max_missing_ratio is not None:
        counts = df_e.groupby('query').size()
        max_ranks = df_e.groupby('query')['rank'].max()
        missing_ratio = (max_ranks - counts) / max_ranks
        keep_q = missing_ratio[missing_ratio <= max_missing_ratio].index
        df_e = df_e[df_e['query'].isin(keep_q)]

    # determine queries order
    if query_order is None:
        queries = sorted(df_e['query'].unique())
    else:
        queries = [q for q in query_order if q in df_e['query'].unique()]

    # 2) Compute overall group proportions p_q per query
    total_per_query = df_e.groupby('query').size()
    count_per_group = df_e.groupby(['query', column_name]).size()
    overall_prop = (count_per_group / total_per_query).rename('p_q')

    # 3) Compute skew matrices manually
    genders = sorted(df_e[column_name].dropna().unique())
    ks = np.array(k_values, dtype=int)
    skew_records = []

    for q in queries:
        sub = df_e[df_e['query'] == q].dropna(subset=['rank']).sort_values('rank')
        p_qg = {g: overall_prop.get((q, g), np.nan) for g in genders}
        max_rank = sub['rank'].max()

        for g in genders:
            for k in ks:
                if k <= max_rank:
                    t = ((sub[column_name] == g) & (sub['rank'] <= k)).sum()
                    p_rk = t / k
                    p_star = p_qg.get(g, np.nan)
                    if p_rk > 0 and p_star > 0:
                        actual_skew = math.log(p_rk / p_star)
                        if normalize_against_best_possible_skew:
                            opts = []
                            for cnt in (math.floor(p_star*k), math.ceil(p_star*k)):
                                if cnt > 0:
                                    opts.append(math.log((cnt/k) / p_star))
                            best_skew = min(opts) if opts else np.nan
                            skew_val = actual_skew - best_skew
                        else:
                            skew_val = actual_skew
                    else:
                        skew_val = np.nan
                else:
                    skew_val = np.nan
                skew_records.append({
                    'query': q,
                    column_name: g,
                    'k': k,
                    'skew': skew_val
                })

    df_skew_long = pd.DataFrame(skew_records)

    # 4) Add overall proportions
    df_prop = overall_prop.reset_index()
    df_plot = df_skew_long.merge(df_prop, on=['query', column_name], how='left')

    # 5) Plot scatter per group
    groups = df_plot[column_name].unique()
    for g in groups:
        sub = df_plot[df_plot[column_name] == g]
        fig, ax = plt.subplots(figsize=(6, 4))
        for k in ks:
            sk = sub[sub['k'] == k]
            ax.scatter(sk['skew'], sk['p_q'], label=f'k={k}', alpha=0.7)
        ax.set_xlabel('Skew@k')
        ax.set_ylabel('Overall Proportion')
        ax.set_title(f'Skew vs. Prop for {column_name.title()} = {g}')
        ax.legend(title='k')
        plt.tight_layout()
        fig.savefig(f'./plots/overall/pdf_versions/skew_vs_prop_manual_{column_name}_{g}_{max_missing_ratio}.pdf', dpi=300, format='pdf')
        plt.close(fig)




def churn_heatmap_complete(df,column_title, bucket_size = 25, max_missing_ratio = 0.15, save_path = "./plots/overall/"):
    df['date'] = pd.to_datetime(df['date'])
    earliest = df.groupby('query')['date'].min()
    df_e = df.merge(earliest, on=['query', 'date'])
    if max_missing_ratio is not None:
        counts = df_e.groupby('query').size()
        max_ranks = df_e.groupby('query')['rank'].max()
        missing_ratio = (max_ranks - counts) / max_ranks
        keep_q = missing_ratio[missing_ratio <= max_missing_ratio].index
        df = df[df['query'].isin(keep_q)]
        # print(f"num queries remaining: {len(keep_q)} from {len(query_order)}")
        if df_e.empty:
            raise RuntimeError("No queries remain after missing-ratio filter")
    eligible = []
    for q, sub in df.groupby('query'):
        if sub['rank'].max() >= 200 and sub['date'].nunique() >= 5:
            eligible.append(q)
    df_filtered = df[df['query'].isin(eligible)]
    print(f"total queries {df['query'].nunique()},after missing {df_e['query'].nunique()} ,remaining {df_filtered['query'].nunique()}")
    churn_dfs = df_filtered.groupby('query').apply(lambda g: compute_churn_for_query(g,bucket_size, column_title, max_rank=200)).reset_index()
    days = sorted(churn_dfs['end_day'].unique())
    queries = list(dict.fromkeys(churn_dfs['query'].tolist()))
    thresholds = sorted(churn_dfs['threshold'].unique())
    if column_title == 'gender':
        attributes = ['F', 'M']
    elif column_title == 'race':
        attributes = ['nh_white', 'other']

    # 2) build a matrix per gender
    mats = {}
    all_vals = []
    for g in attributes:
        for d in days:
            pivot = churn_dfs[
                (churn_dfs['gender'] == g) & (churn_dfs['end_day'] == d)
                ].pivot(
                index='query',
                columns='threshold',
                values='churn_rate'
            )
            mat = pivot.reindex(index=queries, columns=thresholds).values.astype(float)
            mats[(g, d)] = mat
            all_vals.append(mat.flatten())
    all_vals = np.concatenate(all_vals)
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)

    # 4) plot
    fig, axes = plt.subplots(
        nrows=len(attributes),
        ncols=len(days),
        figsize=(4 * len(days), 0.4 * len(queries) * len(attributes)),
        sharey=True, sharex=True
    )
    if len(attributes) == 1:
        axes = [axes]
    cmap = plt.get_cmap('viridis')
    for i, g in enumerate(attributes):
        for j, d in enumerate(days):
            ax = axes[i, j]
            mat = mats[(g, d)]
            im = ax.imshow(mat, aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap)
            # Titles on top row
            if i == 0:
                ax.set_title(f"Day 1 → {d}")
            # Y-axis labels on first column
            if j == 0:
                ax.set_yticks(np.arange(len(queries)))
                ax.set_yticklabels(queries)
                ax.set_ylabel(f"Gender: {g}")
            else:
                ax.set_yticks([])  # hide yticks on other columns
            # X-axis ticks and labels
            ax.set_xticks(np.arange(len(thresholds)))
            ax.set_xticklabels(thresholds, rotation=45)

        # Common colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Churn Rate')

    plt.tight_layout()
    plt.savefig(f"{save_path}churn_{column_title}_{max_missing_ratio}.png", dpi=300)
    plt.close(fig)

def churn_proportion_correlation(df,column_title, end_day_number=2,bucket_size = 25, max_missing_ratio = 0.15, max_rank_in_plot = 75,save_path = "./plots/overall/"):
    df['date'] = pd.to_datetime(df['date'])
    earliest = df.groupby('query')['date'].min()
    df_e = df.merge(earliest, on=['query', 'date'])
    if max_missing_ratio is not None:
        counts = df_e.groupby('query').size()
        max_ranks = df_e.groupby('query')['rank'].max()
        missing_ratio = (max_ranks - counts) / max_ranks
        keep_q = missing_ratio[missing_ratio <= max_missing_ratio].index
        df = df[df['query'].isin(keep_q)]
        # print(f"num queries remaining: {len(keep_q)} from {len(query_order)}")
        if df_e.empty:
            raise RuntimeError("No queries remain after missing-ratio filter")
    eligible = []
    for q, sub in df.groupby('query'):
        if sub['rank'].max() >= max_rank_in_plot and sub['date'].nunique() >= end_day_number:
            eligible.append(q)
    df_filtered = df[df['query'].isin(eligible)]
    print(
        f"total queries {df['query'].nunique()},after missing {df_e['query'].nunique()} ,remaining {df_filtered['query'].nunique()}")
    churn_dfs = df_filtered.groupby('query').apply(
        lambda g: compute_churn_for_query(g, bucket_size, column_title, max_rank=max_rank_in_plot, number_of_days=2)).reset_index()
    thresholds = sorted(churn_dfs['threshold'].unique())
    if column_title == 'gender':
        attributes = ['F', 'M']
    elif column_title == 'race':
        attributes = ['nh_white', 'other']
    total_per_query = df_filtered.groupby('query').size()
    count_per_group = df_filtered.groupby(['query', column_title]).size()
    overall_prop = (count_per_group / total_per_query) \
        .rename('overall_prop') \
        .reset_index()  # columns: ['query', column_title, 'overall_prop']

    # merge into churn_dfs
    plot_df = pd.merge(
        churn_dfs,
        overall_prop,
        on=['query', column_title],
        how='left'
    )
    for attr in attributes:
        sub = plot_df[(plot_df[column_title] == attr) & (plot_df['end_day'] == end_day_number)]
        fig, ax = plt.subplots(figsize=(6, 4))
        for thr in thresholds:
            thr_df = sub[sub['threshold'] == thr]
            ax.scatter(
                thr_df['churn_rate'],
                thr_df['overall_prop'],
                label=f"{thr}",
                alpha=0.7
            )
        ax.set_xlabel("Churn Rate")
        ax.set_ylabel("Overall Proportion")
        ax.set_title(f"{column_title.title()} = {attr}")
        ax.legend(title="Threshold")
        plt.tight_layout()
        fig.savefig(f"{save_path}churn_vs_prop_{column_title}_{attr}.pdf", dpi=300, format='pdf')
        plt.close(fig)

def compute_churn_for_query(query_df, k, column_title, max_rank = None, number_of_days = 5):
    query_dates = sorted(query_df['date'].unique())[-5:]
    start_data = query_df[query_df['date'] == query_dates[0]]
    rows = []
    for end_date in range(1,number_of_days):
        end_data = query_df[query_df['date'] == query_dates[end_date]]
        # compute overall proportions
        start_props = start_data[column_title].value_counts(normalize=True).to_dict()
        end_props = end_data[column_title].value_counts(normalize=True).to_dict()

        # All possible attribute values seen in either snapshot
        print('UNIQUE', end_data[column_title].unique(), start_data[column_title].unique())
        if column_title == 'gender':
            attributes = ['F','M']
        elif column_title == 'race':
            attributes = ['nh_white','other']

        # figure out how many buckets we need
        if max_rank is None:
            max_rank = max(start_data['rank'].max(), end_data['rank'].max())
        num_bins = math.ceil(max_rank / k)


        for i in range(1, num_bins + 1):
            cutoff = i * k

            # Ensure the last cutoff is exactly the number of items in the ranking
            cutoff = min(cutoff, max_rank)
            s_slice = start_data[start_data['rank'] <= cutoff]
            e_slice = end_data[end_data['rank'] <= cutoff]

            for group in attributes:
                s_ids = set(s_slice.loc[s_slice[column_title] == group, 'full_name'])
                e_ids = set(e_slice.loc[e_slice[column_title] == group, 'full_name'])

                retained = s_ids & e_ids
                lost = s_ids - e_ids
                gained = e_ids - s_ids

                start_count = len(s_ids)
                end_count = len(e_ids)
                churn_rate = (len(lost) / start_count) if start_count else np.nan

                rows.append({
                    column_title: group,
                    'threshold': cutoff,
                    'start_count': start_count,
                    'end_count': end_count,
                    'retained': len(retained),
                    'lost': len(lost),
                    'gained': len(gained),
                    'churn_rate': churn_rate,
                    f'num_{group}': len(s_ids),
                    'end_day': end_date+1
                })

    churn_df = pd.DataFrame(rows).sort_values(['threshold', column_title]).reset_index(drop=True)
    return churn_df

def compute_churn_for_query_all_days(query_df, k, column_title, max_rank=None, number_of_days=5):
    """
    Compute churn rates for all day pairs (start_day, end_day) within a query.
    Includes churn for 1→2, 1→3, ..., 4→5 and all intermediate gaps.
    Returns a DataFrame with start_day, end_day, and day_distance.
    """

    query_dates = sorted(query_df['date'].unique())[-number_of_days:]
    rows = []

    for start_idx in range(len(query_dates) - 1):
        start_date = query_dates[start_idx]
        start_data = query_df[query_df['date'] == start_date]

        for end_idx in range(start_idx + 1, len(query_dates)):
            end_date = query_dates[end_idx]
            end_data = query_df[query_df['date'] == end_date]

            # define attribute values
            if column_title == 'gender':
                attributes = ['F', 'M']
            elif column_title == 'race':
                attributes = ['nh_white', 'other']
            else:
                attributes = sorted(query_df[column_title].unique())

            # determine rank cutoff buckets
            if max_rank is None:
                max_rank = max(start_data['rank'].max(), end_data['rank'].max())
            num_bins = math.ceil(max_rank / k)

            for i in range(1, num_bins + 1):
                cutoff = min(i * k, max_rank)
                s_slice = start_data[start_data['rank'] <= cutoff]
                e_slice = end_data[end_data['rank'] <= cutoff]

                for group in attributes:
                    s_ids = set(s_slice.loc[s_slice[column_title] == group, 'full_name'])
                    e_ids = set(e_slice.loc[e_slice[column_title] == group, 'full_name'])

                    retained = s_ids & e_ids
                    lost = s_ids - e_ids
                    gained = e_ids - s_ids

                    start_count = len(s_ids)
                    end_count = len(e_ids)
                    churn_rate = (len(lost) / start_count) if start_count else np.nan

                    rows.append({
                        column_title: group,
                        'threshold': cutoff,
                        'start_day': start_idx + 1,
                        'end_day': end_idx + 1,
                        'day_distance': (end_idx - start_idx),
                        'start_count': start_count,
                        'end_count': end_count,
                        'retained': len(retained),
                        'lost': len(lost),
                        'gained': len(gained),
                        'churn_rate': churn_rate,
                    })

    churn_df = pd.DataFrame(rows).sort_values(
        ['day_distance', 'threshold', column_title]
    ).reset_index(drop=True)

    return churn_df

def summarize_deviation_all_days(df, p_star_method='actual value', max_missing_ratio=None, column_name='gender'):
    """
    Calculates mean and std deviation at k = 25, 50, 75, 100 for each group (F, M, etc.)
    across all queries and all days.

    Returns a dictionary: {group: {k: {'mean': val, 'std': val, 'count': val}}}
    """
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['query'] != "Materials engineer"]

    # Apply missing ratio filter if set
    if max_missing_ratio is not None:
        counts = df.groupby(['query', 'date']).size()
        missing_ratio = (200 - counts) / 200
        keep = missing_ratio[missing_ratio <= max_missing_ratio].index
        df = df[df.set_index(['query', 'date']).index.isin(keep)]

    # Calculate overall proportions (p*) from actual values
    if p_star_method == 'actual value':
        total_per_qd = df.groupby(['query', 'date']).size()
        count_per_group = df.groupby(['query', 'date', column_name]).size()
        overall_prop = (count_per_group / total_per_qd).rename('overall_prop')
    else:
        raise NotImplementedError("Only 'actual value' supported.")

    if column_name == 'gender':
        groups = ['F', 'M']
    elif column_name == 'race':
        groups = ['nh_white', 'other']
    else:
        raise ValueError("Invalid column_name.")

    ks = [25, 50, 75, 100]
    summary = {g: {k: [] for k in ks} for g in groups}

    for (query, date), group_df in df.groupby(['query', 'date']):
        group_df = group_df.dropna(subset=['rank']).sort_values('rank')
        available_ranks = group_df['rank'].max()

        for g in groups:
            p = overall_prop.get((query, date, g), 0.0)
            for k in ks:
                if k > available_ranks:
                    continue
                top_k_count = ((group_df[column_name] == g) & (group_df['rank'] <= k)).sum()
                deviation = p - (top_k_count / k)
                summary[g][k].append(deviation)

    # Convert to summary statistics
    result = {
        g: {
            k: {
                'mean': float(np.mean(vals)) if vals else None,
                'std': float(np.std(vals)) if vals else None,
                'count': len(vals)
            }
            for k, vals in k_dict.items()
        }
        for g, k_dict in summary.items()
    }

    return result


def format_latex_table(summary):
    ks = [25, 50, 75, 100]
    groups = list(summary.keys())

    header = r"\begin{tabular}{lcccc}" + "\n"
    header += r"\toprule" + "\n"
    header += "Group & $k=25$ & $k=50$ & $k=75$ & $k=100$ \\\\" + "\n"
    header += r"\midrule" + "\n"

    rows = []
    for g in groups:
        row = f"{g}"
        for k in ks:
            data = summary[g][k]
            if data['count'] > 0:
                cell = f"{data['mean']:.3f} $\\pm$ {data['std']:.3f}"
            else:
                cell = "--"
            row += f" & {cell}"
        row += r" \\"
        rows.append(row)

    footer = r"\bottomrule" + "\n" + r"\end{tabular}"

    return header + "\n".join(rows) + "\n" + footer


def randomize_ranks(df):
    df_shuffled = df.copy()
    df_shuffled['rank'] = (
        df_shuffled
        .groupby(['query', 'date'])['rank']
        .transform(lambda x: np.random.permutation(x.values))
    )
    return df_shuffled

def summarize_skewk(df, max_missing_ratio=0.01, k_values=[25, 50, 75, 100],
                    groups=('F', 'M'),
                    normalize_against_best_possible_skew=False,
                    decimals=3):
    """
    Compute statistical summaries (mean ± std) of skew_k across all queries and days,
    for selected k values and groups, based on gender or race depending on the input groups.
    """

    df['date'] = pd.to_datetime(df['date'])
    earliest = df.groupby('query')['date'].min()
    df_e = df.merge(earliest, on=['query', 'date'])

    # ---- Detect group column automatically ----
    if all(g in df_e['gender'].unique() for g in groups):
        group_col = 'gender'
    elif 'race' in df_e.columns and all(g in df_e['race'].unique() for g in groups):
        group_col = 'race'
    else:
        raise ValueError("Groups do not match values in either 'gender' or 'race' column.")

    # ---- Filter queries by missing ratio ----
    counts = df_e.groupby('query').size()
    max_ranks = df_e.groupby('query')['rank'].max()
    missing_ratio = (max_ranks - counts) / max_ranks
    keep_q = missing_ratio[missing_ratio <= max_missing_ratio].index
    df_e = df_e[df_e['query'].isin(keep_q)]
    print(f"num queries remaining: {len(keep_q)}")
    if df_e.empty:
        raise RuntimeError("No queries remain after missing-ratio filter")

    queries = sorted(df_e['query'].unique())

    # ---- Compute target proportions p_qg ----
    total_per_query = df_e.groupby('query').size()
    count_per_group = df_e.groupby(['query', group_col]).size()
    overall_prop = (count_per_group / total_per_query).rename('p_q')

    # ---- Compute skew_k for each query and group ----
    records = []
    for q in queries:
        sub = df_e[df_e['query'] == q].sort_values('rank')
        query_max_rank = sub['rank'].max()
        p_qg = {g: overall_prop.get((q, g), np.nan) for g in groups}

        for k in k_values:
            if k > query_max_rank:
                continue
            for g in groups:
                p_star = p_qg[g]
                if pd.isna(p_star) or p_star == 0:
                    continue
                t = ((sub[group_col] == g) & (sub['rank'] <= k)).sum()
                p_rk = t / k
                if p_rk == 0:
                    skew_val = np.nan
                else:
                    if normalize_against_best_possible_skew:
                        possible_skew_values = []
                        for cnt in (math.floor(p_star * k), math.ceil(p_star * k)):
                            if cnt > 0:
                                possible_skew_values.append(np.log((cnt / k) / p_star))
                        best_skew_value = max(possible_skew_values) if possible_skew_values else np.nan
                        skew_val = np.log(p_rk / p_star) - best_skew_value
                    else:
                        skew_val = np.log(p_rk / p_star)
                records.append({'query': q, 'group': g, 'k': k, 'skew_k': skew_val})

    skew_df = pd.DataFrame(records)

    # ---- Compute mean ± std ----
    summary = skew_df.groupby(['group', 'k'])['skew_k'].agg(['mean', 'std']).round(decimals)

    # ---- LaTeX Table ----
    print("\n=== LaTeX Table ===\n")
    header = "Group & " + " & ".join([f"$k={k}$" for k in k_values]) + " \\\\"
    print("\\begin{tabular}{l" + "c" * len(k_values) + "}")
    print("\\toprule")
    print(header)
    print("\\midrule")

    for g in groups:
        row = [g]
        for k in k_values:
            if (g, k) in summary.index:
                m, s = summary.loc[(g, k)]
                cell = f"{m:.3f} $\\pm$ {s:.3f}"
            else:
                cell = "--"
            row.append(cell)
        print(" & ".join(row) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

    return summary


def summarize_churn_by_distance(churn_dfs, column_title='gender',
                                k_values=[25, 50, 75, 100],
                                day_max=5, decimals=3):
    """
    Summarize churn rates over all queries grouped by day distance (1-4 days apart).

    Parameters
    ----------
    churn_dfs : pd.DataFrame
        Output of compute_churn_for_query() for all queries, concatenated.
        Must have columns: ['query', 'start_day', 'end_day', 'threshold', column_title, 'churn_rate'].
    column_title : str
        Either 'gender' or 'race'.
    k_values : list[int]
        Threshold values to include (e.g., [25, 50, 75, 100]).
    day_max : int
        Maximum day index (usually 5 for days 1–5).
    decimals : int
        Rounding precision for mean/std in the table.

    Returns
    -------
    dict of pd.DataFrames
        One summary DataFrame per day distance.
    """

    # Determine which attribute values exist
    attributes = churn_dfs[column_title].unique().tolist()

    # Compute day distance
    churn_dfs = churn_dfs.copy()
    churn_dfs['day_distance'] = churn_dfs['end_day'] - churn_dfs['start_day']

    summaries = {}

    for dist in range(1, day_max):
        sub = churn_dfs[churn_dfs['day_distance'] == dist]
        if sub.empty:
            continue

        # Compute mean ± std by group and threshold
        summary = sub.groupby([column_title, 'threshold'])['churn_rate'].agg(['mean', 'std']).round(decimals)

        # Save summary for this distance
        summaries[dist] = summary

        # ---- Print LaTeX Table ----
        print(f"\n=== Day Distance: {dist} day(s) apart ===\n")
        print("\\begin{tabular}{l" + "c" * len(k_values) + "}")
        print("\\toprule")
        print("Group & " + " & ".join([f"$k={k}$" for k in k_values]) + " \\\\")
        print("\\midrule")

        for g in attributes:
            row = [g]
            for k in k_values:
                if (g, k) in summary.index:
                    m, s = summary.loc[(g, k)]
                    cell = f"{m:.3f} $\\pm$ {s:.3f}"
                else:
                    cell = "--"
                row.append(cell)
            print(" & ".join(row) + " \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")

    return summaries



if __name__ == '__main__':
    database = Database("DATABASE_ADDR")
    query_batch_list = [1,2,3,4,5,6,7,8]
    dfs = []
    attribute = 'gender'
    for query_batch_number in query_batch_list:
        # dfs.append(add_LI_members_to_missing_ranks(database.get_ranking_data_for_particular_batch(query_batch_number),attribute))
        dfs.append(database.get_ranking_data_for_particular_batch(query_batch_number))
    combined_df = pd.concat(dfs, ignore_index=True)
    query_order = sorted(dfs[0]['query'].unique()) + sorted(dfs[1]['query'].unique())
    combined_df['race'] = np.where(
        combined_df['race'] == 'nh_white',
        'nh_white',
        "other"
    )
    #Example usage for creating all the plots in the paper is below:
    plot_skewk(combined_df, max_missing_ratio=0.01,save_path="./plots/overall/pdf_versions/skew_200.pdf", query_order = query_order, normalize_against_best_possible_skew=True)
    #For random baseline
    # random_df = randomize_ranks(combined_df)
    plot_skew_vs_prop_scatter(combined_df,'gender')
    plot_minskewk(
        combined_df,
        max_missing_ratio=0.01,
        k_intervals=None,
        save_path="./plots/overall/x.pdf",
        query_order=query_order,
        normalize_against_best_possible_skew=False,
        plot_only_min=True)
    plot_skew_vs_prop_scatter(combined_df)
    churn_heatmap_daypairs(combined_df, 'race', bucket_size=25, max_missing_ratio=0.15, save_path="./plots/overall/",
                           query_order=None)
    #
    churn_proportion_correlation(combined_df,
    'gender', end_day_number=2, bucket_size=25, max_missing_ratio=0.15,
                                 max_rank_in_plot=75, save_path="./plots/overall/")
    churn_heatmap_daypairs(combined_df, 'race',max_missing_ratio=0.15,query_order = query_order)
    churn_dfs = churn_heatmap_groups(combined_df, 'gender', max_missing_ratio=0.15, query_order=query_order, max_rank_in_plot=100)
    summarize_churn_by_distance(churn_dfs, column_title='gender')
    churn_dfs = churn_heatmap_groups(combined_df, 'race', max_missing_ratio=0.15, query_order=query_order,
                                     max_rank_in_plot=100)
    summarize_churn_by_distance(churn_dfs, column_title='race')
    plot_skewk(combined_df, save_path="./plots/overall/x.pdf", query_order = query_order, normalize_against_best_possible_skew=False)
    plot_deviation(combined_df, p_star_method="actual value", max_missing_ratio=0.15, column_name = "race")

