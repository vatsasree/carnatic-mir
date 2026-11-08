"""Run every pre-registered analysis and write tables, figures and a summary.

    python -m experiment.export          # first: responses -> study/data/export/
    python -m analysis.report            # -> study/data/results/ (summary.md, *.csv, *.png)
    python -m analysis.report --export <dir> --out <dir> [--quick]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import h1, h2, h3
from .common import EXPORT, RESULTS, fmt_p, load, save


def figures(out: Path, R: pd.DataFrame, res1: dict, res2: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    blue, grey = "#2f5d8a", "#9a9a9a"

    # H1: per-pair means, lullaby vs non-lullaby
    pp = res1["per_pair"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(len(pp))
    for i, r in enumerate(pp.itertuples()):
        ax.plot([i - 0.12, i + 0.12], [r.non_lullaby_mean, r.lullaby_mean], color=grey, lw=1, zorder=1)
    ax.scatter(x - 0.12, pp["non_lullaby_mean"], color=grey, label="non-lullaby", zorder=2)
    ax.scatter(x + 0.12, pp["lullaby_mean"], color=blue, label="lullaby", zorder=2)
    ax.set_xticks(x, [p + ("*" if s else "") for p, s in zip(pp["pair"], pp["same_singer"])])
    ax.set_ylabel("Soothing (mean of Peacefulness, Tenderness; 1-5)")
    ax.set_title("Soothing rating per pair (* = same singer)", loc="left")
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    fig.tight_layout(); fig.savefig(out / "fig_h1_pairs.png", dpi=200); plt.close(fig)

    # H1: per-participant difference
    d = h1.participant_diffs(R, "soothing")
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.hist(d, bins=np.arange(-3, 3.01, 0.25), color=blue, alpha=.8)
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("Lullaby minus non-lullaby soothing, per listener"); ax.set_ylabel("Listeners")
    fig.tight_layout(); fig.savefig(out / "fig_h1_listeners.png", dpi=200); plt.close(fig)

    if not res2:
        return
    # H2: event-locked time courses
    co = res2["event_courses"]
    if len(co):
        names = [e for e in h2.EVENT_TYPES if e in set(co["event"])]
        lk = res2["event_locked"].set_index("event")
        fig, axs = plt.subplots(1, len(names), figsize=(2.0 * len(names), 2.4), sharey=True)
        for ax, n in zip(np.atleast_1d(axs), names):
            c = co[co["event"] == n]
            ax.plot(c["lag_s"], c["share"], color=blue)
            ax.axvline(0, color="k", lw=.6); ax.axvspan(0, h2.POST, color=blue, alpha=.06)
            r = lk.loc[n]
            ax.set_title(f"{n.replace('_', ' ')}\nn={int(r.n_events)}, p={fmt_p(r.p_perm)}", fontsize=8)
            ax.set_xlabel("s from event")
        np.atleast_1d(axs)[0].set_ylabel("Share holding 'soothing'")
        fig.tight_layout(); fig.savefig(out / "fig_h2_event_locked.png", dpi=200); plt.close(fig)
    # H2: press curves per snippet with held-note markers
    pc, se = res2["press_curves"], res2["snippet_events"]
    toks = pc.drop_duplicates("token").sort_values(["pair", "condition"], key=lambda s: s.map(
        lambda v: int(v[1:]) if isinstance(v, str) and v.startswith("P") and v[1:].isdigit() else v))
    n = len(toks); cols = 4; rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(11, 1.6 * rows), sharey=True, squeeze=False)
    for ax, r in zip(axs.flat, toks.itertuples()):
        c = pc[pc["token"] == r.token]
        ax.fill_between(c["t"], c["share"], color=blue if r.condition == "lullaby" else grey, alpha=.6, lw=0)
        e = se[(se["token"] == r.token) & se["event"].str.startswith("elongation")]
        ax.vlines(e["t"], 0, 1, color="k", lw=.4, alpha=.5)
        ax.set_title(f"{r.pair} {r.condition.replace('_', '-')} (n={int(c['n_listeners'].iloc[0])})", fontsize=8)
        ax.set_ylim(0, 1)
    for ax in axs.flat[n:]:
        ax.axis("off")
    fig.suptitle("Share of listeners holding 'calm or soothing' (lines: held-note onsets)", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(out / "fig_h2_press_curves.png", dpi=170); plt.close(fig)


def summary(out: Path, S, res1, res2, res3) -> str:
    P = S.participants
    L = [f"# Results summary\n", f"Generated {time.strftime('%Y-%m-%d %H:%M')} from {len(P)} participant records.\n",
         "## Participants\n"]
    flow = P["reason"].replace("", "kept").value_counts()
    L.append("| Status | n |\n| --- | --- |\n" + "\n".join(f"| {k} | {v} |" for k, v in flow.items()) + "\n")
    k = P[~P["excluded"]]
    if "profile.age" in k and len(k):
        L.append(f"Kept: {len(k)} (Set A {int((k['set_id'] == 'A').sum())}, Set B {int((k['set_id'] == 'B').sum())}); "
                 f"age median {k['profile.age'].median():.0f} (range {k['profile.age'].min():.0f}-"
                 f"{k['profile.age'].max():.0f}); median session {k['duration_min'].median():.0f} min.\n")
    p = res1["primary"].iloc[0]; pt = res1["primary_paired"].iloc[0]; sg = res1["sign_test"].iloc[0]
    L += ["## H1 (primary): soothing ~ lullaby\n",
          f"- Mixed model ({p['method']}): lullaby effect **{p['estimate']:+.2f}** scale points "
          f"(95 % CI {p['ci_low']:+.2f} to {p['ci_high']:+.2f}; {p['std_effect']:+.2f} SD), p = {fmt_p(p['p'])} "
          f"(t, df = {int(p['df'])}); exact permutation over {int(p['n_pairs'])} pairs p = {fmt_p(p['p_perm_pairs'])}; "
          f"{int(p['n_trials'])} ratings, {int(p['n_participants'])} listeners, {int(p['n_songs'])} songs.",
          f"- Per listener: mean difference {pt['mean_diff']:+.2f} (dz = {pt['dz']:.2f}), t-test p = {fmt_p(pt['p_t'])}, "
          f"Wilcoxon p = {fmt_p(pt['p_wilcoxon'])}; {pt['share_positive']:.0%} of listeners rated lullabies higher.",
          f"- Per pair: lullaby higher in {int(sg['pairs_positive'])} of {int(sg['pairs'])} pairs (sign test p = {fmt_p(sg['p_sign'])}).\n",
          "### Secondary outcomes (Holm-corrected)\n", "| Outcome | Effect | 95 % CI | p | p (Holm) |\n| --- | --- | --- | --- | --- |"]
    for r in res1["secondary"].itertuples():
        L.append(f"| {r.outcome} | {r.estimate:+.2f} | {r.ci_low:+.2f} to {r.ci_high:+.2f} | {fmt_p(r.p)} | {fmt_p(r.p_holm)} |")
    v = res1["valence_equivalence"].iloc[0]
    L.append(f"\nValence equivalence (+/- {v['bound']}): mean difference {v['mean_diff']:+.2f}, TOST p = {fmt_p(v['p_tost'])}.\n")
    L += ["### Sensitivity\n", "| Analysis | Effect | 95 % CI | p | ratings |\n| --- | --- | --- | --- | --- |"]
    for r in res1["sensitivity"].itertuples():
        L.append(f"| {r.analysis} | {r.estimate:+.2f} | {r.ci_low:+.2f} to {r.ci_high:+.2f} | {fmt_p(r.p)} | {r.n_trials} |")
    if "h4_age" in res1:
        a = res1["h4_age"].iloc[0]
        L.append(f"\nH4 (exploratory): lullaby x age, per decade {a['estimate']:+.2f} (95 % CI {a['ci_low']:+.2f} to "
                 f"{a['ci_high']:+.2f}), p = {fmt_p(a['p'])}.\n")
    if res2:
        L += ["\n## H2: moments followed by soothing presses\n",
              f"Change in share holding, {h2.POST:.0f} s after minus {h2.PRE:.0f} s before each event, vs random times "
              f"({h2.N_PERM} permutations).\n", "| Event | n | Change | z | p |\n| --- | --- | --- | --- | --- |"]
        for r in res2["event_locked"].itertuples():
            if getattr(r, "n_events", 0) and not pd.isna(getattr(r, "delta_share", np.nan)):
                L.append(f"| {r.event} | {int(r.n_events)} | {r.delta_share:+.3f} | {r.z:+.1f} | {fmt_p(r.p_perm)} |")
        if len(res2["lagged_regression"]):
            lag = res2["lagged_regression"]["lag_s"].iloc[0]
            L += [f"\nLagged regression (exploratory, best lag {lag:.2f} s):\n",
                  "| Feature | Coef | SE | p |\n| --- | --- | --- | --- |"]
            L += [f"| {r.feature} | {r.coef:+.3f} | {r.se:.3f} | {fmt_p(r.p)} |" for r in res2["lagged_regression"].itertuples()]
    L += ["\n## H3: lullaby minus non-lullaby, acoustic features (paired over pairs)\n",
          "| Level | Feature | Predicted | Lullaby | Non-lullaby | Diff | dz | p | As predicted |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in res3["h3"].itertuples():
        if pd.isna(getattr(r, "mean_diff", np.nan)):
            continue
        ap = getattr(r, "direction_as_predicted", None)
        L.append(f"| {r.level} | {r.feature} | {r.predicted} | {r.lullaby_mean:.3g} | {r.non_lullaby_mean:.3g} | "
                 f"{r.mean_diff:+.3g} | {r.dz:+.2f} | {fmt_p(r.p_perm)} | {'yes' if ap is True else 'no' if ap is False else '-'} |")
    L.append("\nFigures: fig_h1_pairs.png, fig_h1_listeners.png, fig_h2_event_locked.png, fig_h2_press_curves.png.")
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", default=str(EXPORT))
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--quick", action="store_true", help="200 permutations instead of 2000")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.quick:
        h2.N_PERM = 200
    S = load(Path(a.export))
    save(S.participants, "participants_with_exclusions.csv", out)
    res1 = h1.run(S.ratings, S.gems)
    res2 = h2.run(S.phase2, S.presses, S.snippets)
    res3 = h3.run(S.snippets, res1["snippet_means"])
    for prefix, res in (("h1", res1), ("h2", res2), ("h3", res3)):
        for k, v in res.items():
            if isinstance(v, pd.DataFrame):
                save(v, f"{prefix}_{k}.csv", out)
    figures(out, S.ratings, res1, res2)
    print(summary(out, S, res1, res2, res3))
    print(f"results in {out}")


if __name__ == "__main__":
    main()
