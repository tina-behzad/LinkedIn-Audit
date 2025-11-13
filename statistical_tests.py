import math
import statsmodels.formula.api as smf
from scipy import stats
import numpy as np
import pandas as pd

from full_plots import compute_churn_for_query
from src.utils.load_data import Database
def analyze_skew_at_k(
    df_raw,
    column_name,
    k,
    max_missing_ratio=0.01,
    normalize_against_best_possible_skew=False,
    test_value=-0.011
):
    """
    For a single k, computes min-skew across genders per query/day, fits an
    intercept-only mixed model, and tests whether the overall mean differs
    from test_value.

    Returns:
      - fitted MixedLMResults (mdf)
      - t_statistic
      - p_value
    """

    # 1) Compute full skew DataFrame for that k
    df_skew = compute_skewk(
        df_raw,
        column_name,
        max_missing_ratio=max_missing_ratio,
        # k_intervals=[k],
        normalize_against_best_possible_skew=normalize_against_best_possible_skew
    )

    # 2) Extract and rename the skew column
    kcol = f"skew{k}"
    df_min = (
        df_skew
        .groupby(['query', 'day'], as_index=False)[kcol]
        .min()
        .rename(columns={kcol: 'min_skew'})
    )

    # 3) Fit intercept-only mixed model: min_skew ~ 1 + (1|query)
    md = smf.mixedlm("min_skew ~ 1", df_min, groups=df_min["query"])
    mdf = md.fit(reml=True)

    # 4) Wald test of intercept vs test_value
    beta0 = mdf.params["Intercept"]
    se0 = mdf.bse["Intercept"]
    t_stat = (beta0 - test_value) / se0
    df_resid = mdf.df_resid
    p_val = 2 * stats.t.sf(abs(t_stat), df_resid)

    return mdf, t_stat, p_val


def compute_churn_df(df,column_title='gender', bucket_size = 25, max_missing_ratio = 0.15, max_rank_in_plot = 100):
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
        lambda g: compute_churn_for_query(g, bucket_size, column_title, max_rank=max_rank_in_plot)).reset_index()
    churn_dfs['end_day'] = churn_dfs['end_day'] - 2
    return churn_dfs[['query',column_title,'threshold','churn_rate','end_day']]


def analyze_churn_by_k(
    df,
    column_title,
    k,
    test_value=0.0,
):
    """
    For a given threshold k, fits a mixed-effects model:
      churn_rate ~ end_day + C(column_title) + (1 | query)
    and tests whether the group effect differs from test_value.

    Args:
        df: DataFrame with ['query', column_title, 'threshold',
                            'churn_rate', 'end_day'].
        column_title: string name of the group column (e.g. 'gender').
        k: integer threshold to filter on.
        test_value: null hypothesis value for the group coefficient.
        max_missing_ratio: if set, drop queries whose missing ratio > this.

    Returns:
        mdf: fitted MixedLMResults object
        param_name: name of the tested parameter
        beta: estimated coefficient
        se: standard error of coefficient
        t_stat: Wald t-value
        p_val: two-sided p-value
    """
    # Filter by threshold
    df_k = df[df['threshold'] == k].copy()

    # Ensure end_day numeric
    df_k['end_day'] = pd.to_numeric(df_k['end_day'])

    formula = f"churn_rate ~ end_day + C({column_title})"
    # Fit mixed-effects model
    md = smf.mixedlm(formula, df_k, groups=df_k['query'])
    mdf = md.fit(reml=True)

    # Identify the parameter for the non-baseline level
    param_candidates = [p for p in mdf.params.index if p.startswith(f"C({column_title})[T.")]
    if not param_candidates:
        raise ValueError(f"No C({column_title}) parameter found in model.")
    param_name = param_candidates[0]

    beta = mdf.params[param_name]
    se = mdf.bse[param_name]
    # Wald t-test against test_value
    t_stat = (beta - test_value) / se
    df_resid = mdf.df_resid
    p_val = 2 * stats.t.sf(abs(t_stat), df_resid)

    return mdf, param_name, beta, se, t_stat, p_val

def compute_skewk(df, column_name,max_missing_ratio = 0.01, k_intervals = 25,normalize_against_best_possible_skew = False, max_rank_per_query = 100):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df.groupby('query')['date'].transform(lambda dt: (dt - dt.min()).dt.days + 1)
    print(df['query'].nunique())
    max_rank_per_qd = df.groupby(['query', 'day'])['rank'].transform('max')

    # 2) Keep only rows where that per-query-day max exceeds your threshold
    df = df[max_rank_per_qd > max_rank_per_query]
    print(df['query'].nunique())
    # 2) filter queries by missing ratio
    if max_missing_ratio is not None:
        counts = df.groupby('query').size()
        max_ranks = df.groupby('query')['rank'].max()
        missing_ratio = (max_ranks - counts) / max_ranks
        keep_q = missing_ratio[missing_ratio <= max_missing_ratio].index
        df = df[df['query'].isin(keep_q)]
        print(df['query'].nunique())


    # 3) set ks
    # if k_intervals is None:
    k_intervals = np.arange(25, 201, 25)
    ks = np.array(k_intervals, dtype=int)

    # 4) groups
    if column_name == 'gender':
        groups = ['F','M']
    else:
        groups = df[column_name].unique()

    # 5) overall prop per (query, day, group)
    total = df.groupby(['query', 'day']).size()
    counts = df.groupby(['query', 'day', column_name]).size()
    overall_prop = (counts / total).rename('p_star')

    # 6) iterate and build rows
    rows = []
    for (q, d), sub_qd in df.groupby(['query', 'day']):
        max_rank = sub_qd['rank'].max()
        # compute p_star for each group in this q,d
        p_star_dict = {g: overall_prop.get((q, d, g), np.nan) for g in groups}

        for g in groups:
            row = {'query': q, 'day': d, column_name: g}
            p_star = p_star_dict.get(g, np.nan)
            # compute skew@k for each k
            for k in ks:
                if k <= max_rank and p_star > 0:
                    t = ((sub_qd[column_name] == g) & (sub_qd['rank'] <= k)).sum()
                    p_rk = t / k
                    if p_rk > 0:
                        actual_skew = math.log(p_rk / p_star)
                    else:
                        actual_skew = np.nan
                    # optional normalization
                    if normalize_against_best_possible_skew and p_star > 0:
                        opts = []
                        for cnt in (math.floor(p_star * k), math.ceil(p_star * k)):
                            if cnt > 0:
                                opts.append(math.log((cnt / k) / p_star))
                        best_skew = min(opts) if opts else np.nan
                        skew_val = actual_skew - best_skew
                    else:
                        skew_val = actual_skew
                else:
                    skew_val = np.nan
                row[f'skew{k}'] = skew_val
            rows.append(row)

    return pd.DataFrame(rows)


def skew_statistical_test(combined_df):
    mdf25, t25, p25 = analyze_skew_at_k(combined_df, 'gender', 25, max_missing_ratio=0.15)
    print(mdf25.summary())
    print(f"k=25 intercept vs -0.011: t={t25:.2f}, p={p25}")
    # skew_df = compute_skewk(combined_df,'gender',max_missing_ratio=0.15)
    mdf50, t50, p50 = analyze_skew_at_k(combined_df, 'gender', 50, max_missing_ratio=0.15)
    print(mdf50.summary())
    print(f"k=50 intercept vs -0.011: t={t50:.2f}, p={p50}")

    mdf75, t75, p75 = analyze_skew_at_k(combined_df, 'gender', 75, max_missing_ratio=0.15)
    print(mdf75.summary())
    print(f"k=75 intercept vs -0.011: t={t75:.2f}, p={p75}")

    mdf100, t100, p100 = analyze_skew_at_k(combined_df, 'gender', 100, max_missing_ratio=0.15)
    print(mdf100.summary())
    print(f"k=100 intercept vs -0.011: t={t100:.2f}, p={p100}")


if __name__ == '__main__':
    database = Database("DATABASE_ADDR")
    query_batch_list = [1,2,3,4,5,7,8]
    dfs = []
    attribute = 'gender'
    for query_batch_number in query_batch_list:
        dfs.append(database.get_ranking_data_for_particular_batch(query_batch_number))
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df['race'] = np.where(
        combined_df['race'] == 'nh_white',
        'nh_white',
        "other"
    )
    # skew_statistical_test(combined_df)
    churn_df = compute_churn_df(combined_df)
    mdf25, param, beta, se, t_stat, p_val = analyze_churn_by_k(churn_df, 'gender', 25)
    print("########25")
    print(mdf25.summary())
    print(f"{param}: β={beta:.3f} (SE={se:.3f}), t={t_stat:.2f}, p={p_val:.3f}")
    mdf50, param, beta, se, t_stat, p_val = analyze_churn_by_k(churn_df, 'gender', 50)
    print("########50")
    print(mdf50.summary())
    print(f"{param}: β={beta:.3f} (SE={se:.3f}), t={t_stat:.2f}, p={p_val:.3f}")
    mdf75, param, beta, se, t_stat, p_val = analyze_churn_by_k(churn_df, 'gender', 75)
    print("########75")
    print(mdf75.summary())
    print(f"{param}: β={beta:.3f} (SE={se:.3f}), t={t_stat:.2f}, p={p_val:.3f}")
    mdf100, param, beta, se, t_stat, p_val = analyze_churn_by_k(churn_df, 'gender', 100)
    print("########100")
    print(mdf100.summary())
    print(f"{param}: β={beta:.3f} (SE={se:.3f}), t={t_stat:.2f}, p={p_val:.3f}")
