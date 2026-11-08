"""H1: lullabies are rated more soothing (primary), less arousing, and not less pleasant.

Primary model:  soothing ~ lullaby + (1|participant) + (1|pair) + (1|song)
fitted as a linear mixed model with crossed random effects (statsmodels MixedLM, variance components).
With only 10 pairs, Wald z-tests are too liberal (13 % false positives in 60 null simulations), so the lullaby
term's p-value and CI use a t distribution with df = pairs - 1 (8 %, consistent with 5 %), confirmed by the exact
permutation test over the 10 pair differences. Robustness: per-participant paired difference (t-test and
Wilcoxon), per-pair differences with a sign test.
"""
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .common import holm

VC = {"participant": "0 + C(pid)", "pair": "0 + C(pair)", "song": "0 + C(track_id)"}


def mixed(df: pd.DataFrame, outcome: str, covariates: str = "", term: str = "lullaby") -> dict:
    """Fit outcome ~ lullaby [+ covariates] with crossed random intercepts; return the `term` estimate."""
    import statsmodels.formula.api as smf
    d = df.dropna(subset=[outcome]).copy()
    formula = f"{outcome} ~ lullaby" + (f" + {covariates}" if covariates else "")
    vc = {k: v for k, v in VC.items() if d[{"participant": "pid", "pair": "pair", "song": "track_id"}[k]].nunique() > 1}
    out = dict(outcome=outcome, model=formula, n_trials=len(d), n_participants=d["pid"].nunique(),
               n_songs=d["track_id"].nunique())
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.mixedlm(formula, d, groups=np.ones(len(d)), re_formula="0", vc_formula=vc).fit(reml=True)
        ci = m.conf_int().loc[term]
        out.update(estimate=m.params[term], se=m.bse[term], ci_low=ci[0], ci_high=ci[1], p=m.pvalues[term],
                   converged=bool(m.converged), method="MixedLM crossed")
    except Exception as e:                               # tiny or degenerate data: fall back to cluster-robust OLS
        m = smf.ols(formula, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["pid"])[0]})
        ci = m.conf_int().loc[term]
        out.update(estimate=m.params[term], se=m.bse[term], ci_low=ci[0], ci_high=ci[1], p=m.pvalues[term],
                   converged=False, method=f"OLS cluster(pid) fallback: {type(e).__name__}")
    if term == "lullaby":                                # stimulus-level inference: df = pairs - 1
        df_t = max(d["pair"].nunique() - 1, 1)
        q = stats.t.ppf(0.975, df_t)
        out.update(p_z=out["p"], df=df_t, p=2 * stats.t.sf(abs(out["estimate"] / out["se"]), df_t),
                   ci_low=out["estimate"] - q * out["se"], ci_high=out["estimate"] + q * out["se"])
    sd = d[outcome].std()
    out["std_effect"] = out["estimate"] / sd if sd > 0 else np.nan     # in SDs of the outcome
    return out


def pair_permutation(df: pd.DataFrame, outcome: str = "soothing") -> dict:
    """Exact sign-flip permutation test on the per-pair lullaby minus non-lullaby means."""
    s = df.groupby(["pair", "condition"])[outcome].mean().unstack()
    d = (s.get("lullaby") - s.get("non_lullaby")).dropna().to_numpy()
    if len(d) < 2:
        return dict(n_pairs=len(d))
    flips = np.array(list(itertools.product([1, -1], repeat=len(d))))
    return dict(n_pairs=len(d), mean_pair_diff=d.mean(),
                p_perm_pairs=(np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-12).mean())


def participant_diffs(df: pd.DataFrame, outcome: str) -> pd.Series:
    """Per participant: mean over lullaby trials minus mean over non-lullaby trials."""
    m = df.pivot_table(index="pid", columns="lullaby", values=outcome, aggfunc="mean")
    return (m.get(1) - m.get(0)).dropna()


def paired_tests(df: pd.DataFrame, outcome: str) -> dict:
    d = participant_diffs(df, outcome)
    t = stats.ttest_1samp(d, 0) if len(d) > 1 else None
    w = stats.wilcoxon(d) if len(d) > 5 and (d != 0).any() else None
    return dict(outcome=outcome, n=len(d), mean_diff=d.mean(), sd_diff=d.std(),
                dz=d.mean() / d.std() if d.std() > 0 else np.nan,
                t=t.statistic if t else np.nan, p_t=t.pvalue if t else np.nan,
                p_wilcoxon=w.pvalue if w else np.nan, share_positive=(d > 0).mean())


def tost(d: pd.Series, bound: float) -> dict:
    """Two one-sided tests: is the mean difference within +/- bound?"""
    n, m, se = len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    p_lo = 1 - stats.t.cdf((m + bound) / se, n - 1)
    p_hi = stats.t.cdf((m - bound) / se, n - 1)
    return dict(mean_diff=m, bound=bound, p_lower=p_lo, p_upper=p_hi, p_tost=max(p_lo, p_hi), n=n)


def per_pair(df: pd.DataFrame, outcome: str = "soothing") -> tuple[pd.DataFrame, dict]:
    g = df.groupby(["pair", "condition"])[outcome].agg(["mean", "count"]).unstack("condition")
    t = pd.DataFrame({"pair": g.index,
                      "lullaby_mean": g[("mean", "lullaby")].values, "lullaby_n": g[("count", "lullaby")].values,
                      "non_lullaby_mean": g[("mean", "non_lullaby")].values,
                      "non_lullaby_n": g[("count", "non_lullaby")].values})
    t["difference"] = t["lullaby_mean"] - t["non_lullaby_mean"]
    t["same_singer"] = t["pair"].isin({"P3", "P5", "P6", "P9", "P10"})
    t = t.sort_values("pair", key=lambda s: s.str[1:].astype(int))
    k = int((t["difference"] > 0).sum()); n = int(t["difference"].notna().sum())
    sign = dict(pairs_positive=k, pairs=n, p_sign=stats.binomtest(k, n, 0.5).pvalue if n else np.nan)
    return t, sign


def run(R: pd.DataFrame, gems: list) -> dict:
    res = {}
    # primary
    res["primary"] = pd.DataFrame([{**mixed(R, "soothing"), **pair_permutation(R)}])
    res["primary_paired"] = pd.DataFrame([paired_tests(R, "soothing")])
    # secondary outcomes, Holm-corrected together
    sec = [mixed(R, o) for o in ["arousal", "valence"] + gems if o in R]
    sec = pd.DataFrame(sec)
    sec["p_holm"] = holm(sec["p"])
    res["secondary"] = sec
    res["valence_equivalence"] = pd.DataFrame([tost(participant_diffs(R, "valence"), 0.5)])
    # per pair
    pp, sign = per_pair(R)
    res["per_pair"] = pp
    res["sign_test"] = pd.DataFrame([sign])
    # sensitivity (pre-registered)
    sens = []
    def add(label, d, cov=""):
        if d["pid"].nunique() > 2 and d["lullaby"].nunique() == 2:
            sens.append({"analysis": label, **mixed(d, "soothing", cov)})
    add("primary", R)
    add("+ familiarity + lyric understanding", R, "familiar_n + lyrics_n")
    add("excluding songs the listener knows well", R[R["familiar"] != "know_well"])
    add("same-singer pairs only (P3, P5, P6, P9, P10)", R[R["same_singer"]])
    cov = " + ".join(c for c in ("years_indian_training", "mood_valence") if c in R and R[c].notna().any())
    if cov:
        add(f"+ {cov.replace(' + ', ' + ')}", R, cov)
    res["sensitivity"] = pd.DataFrame(sens)
    # H4 (exploratory): lullaby x age
    if "age" in R and R["age"].notna().sum() > 10 and R["age"].nunique() > 2:
        d = R.copy(); d["age_c"] = (d["age"] - d["age"].mean()) / 10          # per decade
        res["h4_age"] = pd.DataFrame([mixed(d, "soothing", "age_c + lullaby:age_c", term="lullaby:age_c")])
    # per-snippet means (used to link H1 and H3)
    res["snippet_means"] = (R.groupby(["pair", "condition", "track_id", "token"])
                            [["soothing", "arousal", "valence", "liking"] + gems].mean().reset_index()
                            .assign(n=R.groupby(["pair", "condition", "track_id", "token"]).size().values))
    # free words per condition
    if "words" in R:
        w = (R.dropna(subset=["words"]).assign(word=lambda x: x["words"].str.lower().str.split(r"[,;/]| and "))
             .explode("word"))
        w["word"] = w["word"].str.strip().str.strip(".")
        w = w[w["word"].str.len() > 1]
        res["words"] = (w.groupby(["condition", "word"]).size().rename("count").reset_index()
                        .sort_values(["condition", "count"], ascending=[True, False]))
    return res
