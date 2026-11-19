"""Statistical comparison and decoding across conditions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..context import Ctx
from ..errors import DataError, DependencyError, UsageError
from ..registry import Param, command
from ..utils import resolve_picks
from ..result import sig


def _epochs_param() -> Param:
    return Param("epochs", "str", "epochs handle with at least two conditions", required=True, example="epochs_0001")


@command(
    "epochs.stats",
    "Compare two conditions sample-by-sample: uncorrected t-test or cluster permutation.",
    params=[
        _epochs_param(),
        Param("a", "str", "first condition name (as in event_id)", required=True, example="audio"),
        Param("b", "str", "second condition name", required=True, example="visual"),
        Param("mode", "choice", "uncorrected t-test or cluster-based permutation", default="cluster", choices=["ttest", "cluster"]),
        Param("picks", "csv", "channels to analyse"),
        Param("n-permutations", "int", "permutations for the cluster test", default=1024),
        Param("threshold", "float", "cluster-forming threshold (t value)", default=3.0),
        Param("tail", "int", "0 two-sided, 1 greater, -1 lower", default=0),
        Param("out", "path", "optional .npz with t/p maps or cluster masks"),
    ],
    returns="{mode, t_range, n_significant, min_p, clusters?}",
    examples=[
        {"args": "epochs.stats --epochs epochs_0001 --a audio --b visual --mode cluster"},
        {"args": "epochs.stats --epochs epochs_0001 --a audio --b visual --mode ttest"},
    ],
    notes="Cluster permutation is the recommended test for EEG/MEG; it controls family-wise error.",
    tags=("stats",),
)
def cmd_epochs_stats(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    epochs = ctx.object("epochs", p["epochs"])
    names = resolve_picks(epochs.info, p.get("picks"))
    for cond in (p["a"], p["b"]):
        if cond not in epochs.event_id:
            raise DataError(
                f"condition '{cond}' not found; available: {', '.join(epochs.event_id)}",
                hint="Use `epochs info` to see condition names.",
            )
    # Constant channels (stim/misc) make the t-test blow up to NaN, so default
    # to the data channels unless the caller asked for specific ones.
    subset = epochs.copy().pick(names) if names else epochs.copy().pick("data")
    data_a = subset[p["a"]].get_data(copy=True)
    data_b = subset[p["b"]].get_data(copy=True)
    if len(data_a) < 2 or len(data_b) < 2:
        raise DataError("need at least two trials per condition")

    if p.get("mode") == "ttest":
        from scipy.stats import ttest_ind

        t_vals, p_vals = ttest_ind(data_a, data_b, axis=0, equal_var=True)
        significant = p_vals < 0.05
        payload = {
            "mode": "ttest",
            "shape": list(t_vals.shape),
            "t_range": [sig(float(np.nanmin(t_vals)), 4), sig(float(np.nanmax(t_vals)), 4)],
            "n_significant": int(significant.sum()),
            "min_p": round(float(np.nanmin(p_vals)), 6),
            "n_tests": int(p_vals.size),
        }
        if p.get("out"):
            out = Path(p["out"]).expanduser()
            np.savez(str(out), t=t_vals, p=p_vals, ch_names=np.array(subset.ch_names), times=subset.times)
            ctx.add_artifact("npz", out, "ttest maps")
            payload["path"] = str(out.resolve())
        return payload

    import mne
    from mne.stats import permutation_cluster_test, spatio_temporal_cluster_test
    from ..utils import ch_type_counts

    adjacency = None
    counts = ch_type_counts(subset.info)
    ch_type = "eeg" if counts.get("eeg") else ("meg" if counts.get("meg") else None)
    if ch_type:
        try:
            adjacency, _ = mne.channels.find_ch_adjacency(subset.info, ch_type)
        except Exception as exc:  # noqa: BLE001
            ctx.warn(f"no channel adjacency available ({exc}); clustering will be time-only")
    else:
        ctx.warn("no EEG/MEG adjacency; clustering will be time-only")

    n_perm = int(p.get("n_permutations") or 1024)
    threshold = float(p.get("threshold") or 3.0)
    tail = int(p.get("tail") or 0)

    if adjacency is not None:
        # (n_obs, n_times, n_channels) with channel adjacency = spatio-temporal clusters
        X = [data_a.transpose(0, 2, 1), data_b.transpose(0, 2, 1)]
        T_obs, clusters, cluster_p, _ = spatio_temporal_cluster_test(
            X,
            n_permutations=n_perm,
            threshold=threshold,
            tail=tail,
            adjacency=adjacency,
            out_type="indices",
            verbose="ERROR",
        )
        spatial_first = False
    else:
        # (n_channels, n_times, n_obs), no adjacency = time-only clusters
        X = [data_a.transpose(1, 2, 0), data_b.transpose(1, 2, 0)]
        T_obs, clusters, cluster_p, _ = permutation_cluster_test(
            X,
            n_permutations=n_perm,
            threshold=threshold,
            tail=tail,
            out_type="indices",
            verbose="ERROR",
        )
        spatial_first = True

    times = subset.times
    ch_names = subset.ch_names
    rows = []
    for idx, (cluster, cp) in enumerate(zip(clusters, cluster_p)):
        if spatial_first:
            ch_idx, time_idx = cluster
        else:
            time_idx, ch_idx = cluster
        ch_idx = np.atleast_1d(ch_idx)
        time_idx = np.atleast_1d(time_idx)
        unique_ch = sorted(set(int(i) for i in ch_idx))
        peak_map = T_obs[ch_idx, time_idx] if spatial_first else T_obs[time_idx, ch_idx]
        rows.append(
            {
                "index": idx,
                "p": round(float(cp), 6),
                "n_points": int(peak_map.size),
                "channels": [ch_names[i] for i in unique_ch[:10]],
                "n_channels": int(len(unique_ch)),
                "tmin_s": round(float(times[time_idx.min()]), 4),
                "tmax_s": round(float(times[time_idx.max()]), 4),
                "peak_t": round(float(peak_map.max() if tail >= 0 else peak_map.min()), 3),
            }
        )

    rows.sort(key=lambda r: r["p"])
    payload = {
        "mode": "cluster",
        "n_clusters": len(rows),
        "n_significant": int(sum(1 for r in rows if r["p"] < 0.05)),
        "min_p": round(float(np.min(cluster_p)), 6) if len(cluster_p) else None,
        "clusters": rows[:5],
        "clusters_note": f"showing the 5 most significant of {len(rows)} clusters",
    }
    if p.get("out"):
        out = Path(p["out"]).expanduser()
        np.savez(str(out), t_obs=T_obs, clumps=[np.array(c) for c in clusters] or np.array([]), clump_p=cluster_p)
        ctx.add_artifact("npz", out, "cluster stats")
        payload["path"] = str(out.resolve())
    return payload


@command(
    "epochs.decode",
    "Decode conditions from single trials (CSP + LDA by default).",
    params=[
        _epochs_param(),
        Param("pipeline", "choice", "classifier family", default="csp-lda", choices=["csp-lda", "csp-svm", "time-lda"]),
        Param("n-components", "int", "CSP components", default=4),
        Param("cv", "int", "cross-validation folds", default=5),
        Param("scoring", "choice", "metric", default="accuracy", choices=["accuracy", "roc_auc"]),
        Param("picks", "csv", "channels to use"),
        Param("seed", "int", "random seed", default=42),
    ],
    returns="{accuracy_mean, accuracy_std, fold_scores, chance, n_epochs, classes}",
    examples=[
        {"args": "epochs.decode --epochs epochs_0001"},
        {"args": "epochs.decode --epochs epochs_0001 --pipeline time-lda --scoring roc_auc"},
    ],
    notes="Needs scikit-learn: pip install \"mne-cli[decoding]\".",
    tags=("decoding", "stats"),
)
def cmd_epochs_decode(ctx: Ctx, p: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.svm import SVC
    except ImportError:
        raise DependencyError("scikit-learn is required for decoding", hint='pip install "mne-cli[decoding]"') from None

    epochs = ctx.object("epochs", p["epochs"])
    names = resolve_picks(epochs.info, p.get("picks"))
    # Constant channels (stim/misc) make the t-test blow up to NaN, so default
    # to the data channels unless the caller asked for specific ones.
    subset = epochs.copy().pick(names) if names else epochs.copy().pick("data")
    X = subset.get_data(copy=True)
    y = subset.events[:, 2]
    classes = sorted(set(int(v) for v in y))
    if len(classes) < 2:
        raise DataError("decoding needs at least two conditions", hint="Check `epochs info`.")

    pipeline = p.get("pipeline") or "csp-lda"
    if pipeline.startswith("csp"):
        from mne.decoding import CSP

        if len(classes) > 2:
            raise DataError("CSP decoding here supports exactly two classes", hint="Use --pipeline time-lda for multiclass data.")
        csp = CSP(n_components=int(p.get("n_components") or 4), reg="ledoit_wolf", log=True, norm_trace=False)
        clf = SVC(kernel="linear") if pipeline == "csp-svm" else LinearDiscriminantAnalysis()
        model = make_pipeline(csp, clf)
    else:
        from mne.decoding import Vectorizer

        model = make_pipeline(Vectorizer(), LinearDiscriminantAnalysis())

    cv = StratifiedKFold(n_splits=int(p.get("cv") or 5), shuffle=True, random_state=int(p.get("seed") or 42))
    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring=p.get("scoring") or "accuracy", n_jobs=1)
    except ValueError as exc:
        raise UsageError(
            f"cross-validation failed: {exc}",
            hint="Reduce --cv or --n-components; every class needs at least --cv trials.",
        ) from None

    chance = 1.0 / len(classes)
    return {
        "pipeline": pipeline,
        "n_epochs": int(len(y)),
        "n_channels": len(subset.ch_names),
        "classes": classes,
        "fold_scores": [round(float(s), 4) for s in scores],
        "accuracy_mean": round(float(np.mean(scores)), 4),
        "accuracy_std": round(float(np.std(scores)), 4),
        "chance": round(chance, 4),
        "above_chance": bool(float(np.mean(scores)) - float(np.std(scores)) > chance),
    }
